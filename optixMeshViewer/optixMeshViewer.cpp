//
// Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// 

#include <glad/glad.h> // Needs to be included before gl_interop
#include <fenv.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <sampleConfig.h>

#include <cuda/whitted.h>
#include <cuda/Light.h>

#include <sutil/Camera.h>
#include <sutil/Trackball.h>
#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Exception.h>
#include <sutil/GLDisplay.h>
#include <sutil/Matrix.h>
#include <sutil/Scene.h>
#include <sutil/sutil.h>
#include <sutil/vec_math.h>

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"
#include "imgui/imgui_stdlib.h"

#include "gaussian.hpp"
#include "OctreeGaussian.hpp"


//#define USE_IAS // WAR for broken direct intersection of GAS on non-RTX cards

bool              resize_dirty  = false;
bool              minimized     = false;

// Camera state
bool              camera_changed = true;
sutil::Camera     camera;
sutil::Trackball  trackball;

// Mouse state
int32_t           mouse_button = -1;

int32_t           samples_per_launch = 1;

whitted::LaunchParams*  d_params = nullptr;
whitted::LaunchParams   params   = {};
int32_t                 width    = 1280;
int32_t                 height   = 720;

float   fov = 0.1f;

glm::vec3 g_position;
glm::vec3 g_rotation;
glm::vec3 g_scale;
glm::mat4 proj;
glm::mat4 model;

float near = 0.1;
float far = 1000.f;


std::string snapLabel = "snapshot";
const std::string snapFolder = "snapshots/";
int snapCounter = 0;
bool takeSnap = false;

bool keepY = false;
float yAxis = 0.f;

oct::OctreeGaussian *octree;
oct::OctreeGaussian *octreeLow;
 
//------------------------------------------------------------------------------
//
// GLFW callbacks
//
//------------------------------------------------------------------------------


void snapshotImage(sutil::CUDAOutputBuffer<uchar4>& output_buffer ) {
    sutil::ImageBuffer buffer;
    buffer.data         = output_buffer.getHostPointer();
    buffer.width        = output_buffer.width();
    buffer.height       = output_buffer.height();
    buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;

    std::string file = snapFolder;
    file.append(snapLabel + "-" + std::to_string(snapCounter) + ".png");
    snapCounter++;
    sutil::saveImage( file.c_str(), buffer, false );
}


static void mouseButtonCallback( GLFWwindow* window, int button, int action, int mods )
{
    double xpos, ypos;
    glfwGetCursorPos( window, &xpos, &ypos );

    if( action == GLFW_PRESS )
    {
        mouse_button = button;
        trackball.startTracking(static_cast<int>( xpos ), static_cast<int>( ypos ));
    }
    else
    {
        mouse_button = -1;
    }
}


static void cursorPosCallback( GLFWwindow* window, double xpos, double ypos )
{
    if( mouse_button == GLFW_MOUSE_BUTTON_LEFT )
    {
        trackball.setViewMode( sutil::Trackball::LookAtFixed );
        trackball.updateTracking( static_cast<int>( xpos ), static_cast<int>( ypos ), width, height );
        camera_changed = true;
    }
    else if( mouse_button == GLFW_MOUSE_BUTTON_RIGHT )
    {
        trackball.setViewMode( sutil::Trackball::EyeFixed );
        trackball.updateTracking( static_cast<int>( xpos ), static_cast<int>( ypos ), width, height );
        camera_changed = true;
    }
}


static void windowSizeCallback( GLFWwindow* window, int32_t res_x, int32_t res_y )
{
    // Keep rendering at the current resolution when the window is minimized.
    if( minimized )
        return;

    // Output dimensions must be at least 1 in both x and y.
    sutil::ensureMinimumSize( res_x, res_y );

    width   = res_x;
    height  = res_y;
    camera_changed = true;
    resize_dirty   = true;
}


static void windowIconifyCallback( GLFWwindow* window, int32_t iconified )
{
    minimized = ( iconified > 0 );
}


static void keyCallback( GLFWwindow* window, int32_t key, int32_t /*scancode*/, int32_t action, int32_t /*mods*/ )
{
    if( action == GLFW_PRESS )
    {
        if( key == GLFW_KEY_Q ||
            key == GLFW_KEY_ESCAPE )
        {
            glfwSetWindowShouldClose( window, true );
        }
    }
    else if( key == GLFW_KEY_G )
    {
        // toggle UI draw
    }
}


static void scrollCallback( GLFWwindow* window, double xscroll, double yscroll )
{
    if(trackball.wheelEvent((int)yscroll))
        camera_changed = true;
}


//------------------------------------------------------------------------------
//
// Helper functions
// TODO: some of these should move to sutil or optix util header
//
//------------------------------------------------------------------------------

void printUsageAndExit( const char* argv0 )
{
    std::cerr <<  "Usage  : " << argv0 << " [options]\n";
    std::cerr <<  "Options: --file | -f <filename>      File for image output\n";
    std::cerr << "          --dim=<width>x<height>      Set image dimensions; defaults to 768x768\n";
    std::cerr <<  "         --launch-samples | -s       Number of samples per pixel per launch (default 16)\n";
    std::cerr <<  "         --no-gl-interop             Disable GL interop for display\n";
    std::cerr <<  "         --model <model.gltf>        Specify model to render (required)\n";
    std::cerr <<  "         --help | -h                 Print this usage message\n";
    exit( 0 );
}


void initLaunchParams( const sutil::Scene& scene ) {
    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &params.accum_buffer ),
                width*height*sizeof(float4)
                ) );
    params.frame_buffer = nullptr; // Will be set when output buffer is mapped

    params.subframe_index = 0u;

    const float loffset = scene.aabb().maxExtent();

    // TODO: add light support to sutil::Scene
    std::vector<Light> lights( 1);
    lights[0].type            = Light::Type::POINT;
    lights[0].point.color     = {1.0f, 1.0f, 1.0f};
    lights[0].point.intensity = .8f;
    lights[0].point.position  = make_float3(1.4,4.6,3);
    lights[0].point.falloff   = Light::Falloff::QUADRATIC; //{26.545f, 12.8f, 2.4f},

    // lights[1].type            = Light::Type::POINT;
    // lights[1].point.color     = {1.0f, 1.0f, 1.0f};
    // lights[1].point.intensity = 10.0f;
    // lights[1].point.position  = make_float3(26.545f, 12.8f, 2.4f);
    // lights[1].point.falloff   = Light::Falloff::QUADRATIC;

    // lights[2].type            = Light::Type::POINT;
    // lights[2].point.color     = {1.0f, 1.0f, 0.8f};
    // lights[2].point.intensity = 2.0f;
    // lights[2].point.position  = make_float3(4,4.9,-3);
    // lights[2].point.falloff   = Light::Falloff::QUADRATIC;

    // lights[3].type            = Light::Type::POINT;
    // lights[3].point.color     = {1.0f, 1.0f, 0.8f};
    // lights[3].point.intensity = 2.0f;
    // lights[3].point.position  = make_float3(-4,4.9,-3);
    // lights[3].point.falloff   = Light::Falloff::QUADRATIC;

    params.lights.count  = static_cast<uint32_t>( lights.size() );
    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &params.lights.data ),
                lights.size() * sizeof( Light )
                ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( params.lights.data ),
                lights.data(),
                lights.size() * sizeof( Light ),
                cudaMemcpyHostToDevice
                ) );

    params.miss_color   = make_float3( 0.1f );

    //CUDA_CHECK( cudaStreamCreate( &stream ) );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_params ), sizeof( whitted::LaunchParams ) ) );

    params.handle = scene.traversableHandle();
}

void updateModel( ) {
    
    // params.modelMatrix = sutil::Matrix4x4({1.0f});
    // params.g_position = make_float3(g_position.x, g_position.y, g_position.z);
    // sutil::Matrix4x4 m1 = sutil::Matrix4x4({1.0f});

    // params.modelMatrix = m1.scale(g_scale);
    // params.modelMatrix *= m1.translate(params.g_position);
    // params.modelMatrix *= m1.rotate(glm::radians(g_rotation.x),{1.0f, 0.0f, 0.0f});
    // params.modelMatrix *= m1.rotate(glm::radians(g_rotation.y),{0.0f, 1.0f, 0.0f});
    // params.modelMatrix *= m1.rotate(glm::radians(g_rotation.z),{0.0f, 0.0f, 1.0f});

    glm::mat4 m1 = glm::mat4(1.0f);
    // params.g_position = make_float3(g_position.x, g_position.y, g_position.z);

    params.modelMatrix = glm::rotate(m1, glm::radians(g_rotation.x), {1.0f, 0.0f, 0.0f});
    params.modelMatrix = glm::rotate(params.modelMatrix, glm::radians(g_rotation.y), {0.0f, 1.0f, 0.0f});
    params.modelMatrix = glm::rotate(params.modelMatrix, glm::radians(g_rotation.z), {0.0f, 0.0f, 1.0f});
    params.modelMatrix = glm::translate(params.modelMatrix, g_position);
    params.modelMatrix = glm::scale(params.modelMatrix, g_scale);
    
}


void handleCameraUpdate( whitted::LaunchParams& params )
{
    if( !camera_changed )
        return;
    camera_changed = false;

    camera.setAspectRatio( static_cast<float>( width ) / static_cast<float>( height ) );
    if (keepY) {
        float3 newEye = camera.eye();
        newEye.y = yAxis;
        camera.setEye(newEye);
    }
    
    params.eye = camera.eye();
    camera.UVWFrame( params.U, params.V, params.W );
    updateModel();

    // std::cout << "Camera Eye: " << camera.eye

}


void handleResize( sutil::CUDAOutputBuffer<uchar4>& output_buffer )
{
    if( !resize_dirty )
        return;
    resize_dirty = false;

    output_buffer.resize( width, height );

    // Realloc accumulation buffer
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( params.accum_buffer ) ) );
    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &params.accum_buffer ),
                width*height*sizeof(float4)
                ) );
}




void updateState( sutil::CUDAOutputBuffer<uchar4>& output_buffer, whitted::LaunchParams& params )
{

    // Update params on device
    if( camera_changed || resize_dirty )
        params.subframe_index = 0;

    handleCameraUpdate( params );
    handleResize( output_buffer );
}


void launchSubframe( sutil::CUDAOutputBuffer<uchar4>& output_buffer, const sutil::Scene& scene )
{

    // Launch
    uchar4* result_buffer_data = output_buffer.map();
    params.frame_buffer        = result_buffer_data;
    CUDA_CHECK( cudaMemcpyAsync( reinterpret_cast<void*>( d_params ),
                &params,
                sizeof( whitted::LaunchParams ),
                cudaMemcpyHostToDevice,
                0 // stream
                ) );

    OPTIX_CHECK( optixLaunch(
                scene.pipeline(),
                0,             // stream
                reinterpret_cast<CUdeviceptr>( d_params ),
                sizeof( whitted::LaunchParams ),
                scene.sbt(),
                width,  // launch width
                height, // launch height
                1       // launch depth
                ) );
    output_buffer.unmap();
    CUDA_SYNC_CHECK(); 
}


void displaySubframe(
        sutil::CUDAOutputBuffer<uchar4>&  output_buffer,
        sutil::GLDisplay&                 gl_display,
        GLFWwindow*                       window )
{
    // Display
    int framebuf_res_x = 0;   // The display's resolution (could be HDPI res)
    int framebuf_res_y = 0;   //
    glfwGetFramebufferSize( window, &framebuf_res_x, &framebuf_res_y );
    gl_display.display(
            output_buffer.width(),
            output_buffer.height(),
            framebuf_res_x,
            framebuf_res_y,
            output_buffer.getPBO()
            );
}


void initCameraState( const sutil::Scene& scene )
{
    camera = scene.camera();
    camera_changed = true;

    trackball.setCamera( &camera );
    trackball.setMoveSpeed( 10.0f );
    trackball.setReferenceFrame( make_float3( 1.0f, 0.0f, 0.0f ), make_float3( 0.0f, 0.0f, 1.0f ), make_float3( 0.0f, 1.0f, 0.0f ) );
    trackball.setGimbalLock(true);
}


void cleanup()
{
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( params.accum_buffer    ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( params.lights.data     ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_params               ) ) );
}




void updateProjectionMatrix() {
    float aspectRatio = whitted::WIDTH / float(whitted::HEIGHT);
    float fovy = glm::radians(fov); // Convertendo graus para radianos
    
    auto pm = glm::perspectiveFov(fovy,(float) whitted::WIDTH,(float) whitted::HEIGHT, near, far);
    // pm[1][1] *= -1;  // Inverting Y for Vulkan (not needed with perspectiveVK).
    // pm[2] *= -1;

    // for (int i = 0; i < 4; i++)
    //     for (int j=0; j< 4; j++)
    //         params.projMatrix[(i*4) +j] = pm[i][j];

    params.projMatrix = pm;



    float tan_fovy = tan(fovy * 0.5f);
    float tan_fovx = tan_fovy * aspectRatio;

    const float focal_y = whitted::HEIGHT/ (2.0f * tan_fovy);
	const float focal_x = whitted::WIDTH / (2.0f * tan_fovx);

    params.focal = make_float2(focal_x, focal_y);
    params.tan_fovx = tan_fovx;
    params.tan_fovy = tan_fovy;
    params.fov = fov;
    params.near = near;

    proj = pm;

}


void printGui(double frameTime) {

    ImGui::Begin("Tab");

    if (ImGui::InputInt("gn", &params.gn))
        camera_changed = true;
    if (ImGui::InputInt("gs", &params.gs))
        camera_changed = true;

    ImGui::InputText("file",&snapLabel);
    
    if (ImGui::Button("Save"))
        takeSnap = true;
    ImGui::SameLine();

    ImGui::Text("Camera Properties");
    if (ImGui::SliderFloat("Near", &near, 0.1f, 1.f)) {
        updateProjectionMatrix();
        camera_changed = true;
    }
    if (ImGui::SliderFloat("Far", &far, 10.f, 1000.f)) {
        updateProjectionMatrix();
        camera_changed = true;
    }
    if (ImGui::SliderFloat("Field of View", &fov, 1.f, 60.f)) {
        updateProjectionMatrix();
        camera_changed = true;
    }
    std::string eye = "Eye X:" + std::to_string(camera.eye().x) + " Y:"+ std::to_string(camera.eye().y) + " Z:" + std::to_string(camera.eye().z);
    ImGui::Text(eye.c_str());
    if (ImGui::Checkbox("Keep Y", &keepY)) {
        yAxis = camera.eye().y;
    }

    if (ImGui::Button("LaunchRay")) {
        float3 origin = camera.eye();
        float3 direction = camera.direction();
        // octree->lancarRaio(origin, direction);
    }

    ImGui::Separator();

    ImGui::Text("Mode");
    if (ImGui::RadioButton("Ray traced", &params.mode, 0))
        camera_changed = true;
    ImGui::SameLine();
    if (ImGui::RadioButton("Gaussian", &params.mode, 1))
        camera_changed = true;
    if (ImGui::RadioButton("Octree", &params.mode, 2))
        camera_changed = true;
    if (ImGui::RadioButton("Depth", &params.mode, 3))
        camera_changed = true;
    
    ImGui::Separator();

    ImGui::SliderFloat("Scale Factor", &params.scaleFactor, 0.1f, 3.0f);


    // ImGui::Separator();
    // if (ImGui::Checkbox("High Gaussian", &params.highGaussian))
    //     camera_changed = true;

    
    ImGui::Separator();

    ImGui::Text("Transform Matrix");
    if (ImGui::SliderFloat("Rotation X", &g_rotation.x, -180.f, 180.f))
        camera_changed = true;
    if (ImGui::SliderFloat("Rotation Y", &g_rotation.y, -180.f, 180.f))
        camera_changed = true;
    if (ImGui::SliderFloat("Rotation Z", &g_rotation.z, -180.f, 180.f))
        camera_changed = true;

    if (ImGui::SliderFloat("Position X", &g_position.x, -50.f, 50.f))
        camera_changed = true;
    if (ImGui::SliderFloat("Position Y", &g_position.y, -20.f, 20.f))
        camera_changed = true;
    if (ImGui::SliderFloat("Position Z", &g_position.z, -50.f, 50.f))
        camera_changed = true;

    if (ImGui::SliderFloat("Scale X", &g_scale.x, -1.f, 1.f))
        camera_changed = true;
    if (ImGui::SliderFloat("Scale Y", &g_scale.y, -1.f, 1.f))
        camera_changed = true;
    if (ImGui::SliderFloat("Scale Z", &g_scale.z, -1.f, 1.f))
        camera_changed = true;
    
    ImGui::Text("Frame time %.1f ms", frameTime);
    
    ImGui::End();
}


//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

int main( int argc, char* argv[] )
{
    sutil::CUDAOutputBufferType output_buffer_type = sutil::CUDAOutputBufferType::GL_INTEROP;

    // feenableexcept(FE_INVALID | FE_OVERFLOW);

    //
    // Parse command line options
    //
    std::string outfile;
    std::string infile;
    std::string gaussianFile1;
    std::string gaussianFile2;
    
    
    

    // output_buffer_type = sutil::CUDAOutputBufferType::CUDA_DEVICE;
    for( int i = 1; i < argc; ++i )
    {
        const std::string arg = argv[i];
        if( arg == "--help" || arg == "-h" )
        {
            printUsageAndExit( argv[0] );
        }
        else if( arg == "--no-gl-interop" )
        {
            output_buffer_type = sutil::CUDAOutputBufferType::CUDA_DEVICE;
        }
        else if( arg == "--model" || arg == "-m")
        {
            if( i >= argc - 1 )
                printUsageAndExit( argv[0] );
            infile = argv[++i];
        }
        else if( arg == "-g1")
        {
            if( i >= argc - 1 )
                printUsageAndExit( argv[0] );
            gaussianFile1 = argv[++i];
        }
        else if( arg == "-g2")
        {
            if( i >= argc - 1 )
                printUsageAndExit( argv[0] );
            gaussianFile2 = argv[++i];
        }
        else if( arg == "--file" || arg == "-f" )
        {
            if( i >= argc - 1 )
                printUsageAndExit( argv[0] );
            outfile = argv[++i];
        }
        else if( arg.substr( 0, 6 ) == "--dim=" )
        {
            const std::string dims_arg = arg.substr( 6 );
            sutil::parseDimensions( dims_arg.c_str(), width, height );
        }
        else if( arg == "--launch-samples" || arg == "-s" )
        {
            if( i >= argc - 1 )
                printUsageAndExit( argv[0] );
            samples_per_launch = atoi( argv[++i] );
        }
        else
        {
            std::cerr << "Unknown option '" << argv[i] << "'\n";
            printUsageAndExit( argv[0] );
        }
    }

    if( infile.empty() )
    {
        std::cerr << "--model argument required" << std::endl;
        printUsageAndExit( argv[0] );
    }


    try
    {

        // std::string infile = sutil::sampleDataFilePath( "teapot-only_rough.glb" ); 
        Gaussian gaussian((uint) width,(uint) height,
            sutil::sampleDataFilePath(gaussianFile1.c_str()), 
            3, false, 0);

        params.g_cov3d = gaussian.cov3d_cuda;
        params.g_opacity = gaussian.opacity_cuda;
        params.g_pos = gaussian.pos_cuda;
        params.g_shs = gaussian.shs_cuda;
        params.gcount = gaussian.count;
        // params.g_scale = gaussian.scale_cuda;
        // params.g_rotation = gaussian.rot_cuda;
        std::cout<<"Count Gaussians 1:" << gaussian.count << std::endl;

        Gaussian gaussianLow((uint) width,(uint) height,
                sutil::sampleDataFilePath(gaussianFile2.c_str()), 
                3, false, 0);
        params.g_cov3d_low = gaussianLow.cov3d_cuda;
        params.g_opacity_low = gaussianLow.opacity_cuda;
        params.g_pos_low = gaussianLow.pos_cuda;
        params.g_shs_low = gaussianLow.shs_cuda;
        // params.g_scale_low = gaussianLow.scale_cuda;
        // params.g_rotation_low = gaussianLow.rot_cuda;

        params.scaleFactor = 1.0f;

        updateProjectionMatrix();
        
        g_position = glm::vec3(0, 0, 0); //Zerado
        // g_position = glm::vec3(4.416, 0.5101, 0.13); //Sponza_claro
        // g_position = glm::vec3(4.8, 1.7, -1.43); //Quarto
        // g_rotation = glm::vec3(-180, -150, -9.3);
        g_rotation = glm::vec3(0, 0, 0); // Sponza_claro
        
        g_scale =    glm::vec3(1.f, 1.f, 1.f);
        // g_scale =    make_float3(-0.8f, 0.8f, 0.8f);
        params.gn = 1;
        params.mode = 0;
        updateModel();

        octree = new oct::OctreeGaussian(gaussian, gaussianLow);
        params.octree = octree->deviceOctree;
        camera.setFovY(60.f);

        sutil::Scene scene;
        sutil::loadScene( sutil::sampleDataFilePath(infile.c_str()), scene );
        scene.finalize();

        OPTIX_CHECK( optixInit() ); // Need to initialize function table
        initCameraState( scene );
        initLaunchParams( scene );


        if( outfile.empty() )
        {
            GLFWwindow* window = sutil::initUI( "GaussianReflection", width, height );
            glfwSetMouseButtonCallback  ( window, mouseButtonCallback   );
            glfwSetCursorPosCallback    ( window, cursorPosCallback     );
            glfwSetWindowSizeCallback   ( window, windowSizeCallback    );
            glfwSetWindowIconifyCallback( window, windowIconifyCallback );
            glfwSetKeyCallback          ( window, keyCallback           );
            glfwSetScrollCallback       ( window, scrollCallback        );
            glfwSetWindowUserPointer    ( window, &params               );


            //
            // Render loop
            //
            {


                sutil::CUDAOutputBuffer<uchar4> output_buffer( output_buffer_type, width, height );
                sutil::GLDisplay gl_display;

                std::chrono::duration<double> state_update_time( 0.0 );
                std::chrono::duration<double> render_time( 0.0 );
                std::chrono::duration<double> display_time( 0.0 );
                do
                {
                    auto t0 = std::chrono::steady_clock::now();
                    glfwPollEvents();

                    
                    updateState( output_buffer, params );
                    auto t1 = std::chrono::steady_clock::now();
                    state_update_time += t1 - t0;
                    t0 = t1;

                    launchSubframe( output_buffer, scene );
                    t1 = std::chrono::steady_clock::now();
                    render_time = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
                    t0 = t1;


                    displaySubframe( output_buffer, gl_display, window );
                    t1 = std::chrono::steady_clock::now();
                    display_time += std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);

                    if (takeSnap ) {
                        snapshotImage(output_buffer);
                        takeSnap = false;
                        
                    }

                    
                    // std::cout <<"frameTime: " << render_time.count() * 60 << "ms" << std::endl;

                    sutil::beginFrameImGui(); 
                    printGui(render_time.count() * 1000);
                    sutil::endFrameImGui();
                    
                    glfwSwapBuffers(window);
                    
                    ++params.subframe_index;
                }
                while( !glfwWindowShouldClose( window ) );
                CUDA_SYNC_CHECK();
            }

            sutil::cleanupUI( window );

        }
        else
        {
			if( output_buffer_type == sutil::CUDAOutputBufferType::GL_INTEROP )
			{
				sutil::initGLFW(); // For GL context
				sutil::initGL();
			}

            {
                // this scope is for output_buffer, to ensure the destructor is called bfore glfwTerminate()

                sutil::CUDAOutputBuffer<uchar4> output_buffer( output_buffer_type, width, height );
                handleCameraUpdate( params );
                handleResize( output_buffer );
                launchSubframe( output_buffer, scene );

                sutil::ImageBuffer buffer;
                buffer.data         = output_buffer.getHostPointer();
                buffer.width        = output_buffer.width();
                buffer.height       = output_buffer.height();
                buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;

                sutil::saveImage( outfile.c_str(), buffer, false );
            }

            if( output_buffer_type == sutil::CUDAOutputBufferType::GL_INTEROP )
            {
                glfwTerminate();
            }
        }

        cleanup();

    }
    catch( std::exception& e )
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
