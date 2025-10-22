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
#include <sutil/Quaternion.h>
#include <sutil/GaussianScene.h>

// --- INÍCIO DA CORREÇÃO DEFINITIVA ---

// 1. Força o GLFW a usar o backend X11 diretamente no código.
#define _GLFW_X11

// 2. Inclui os headers do GLFW usando caminhos padrão.
//    O CMake será responsável por apontar para a pasta correta.
#define _GLFW_X11
#define GLFW_EXPOSE_NATIVE_X11
#define GLFW_EXPOSE_NATIVE_GLX

// 2. Headers principais do GLFW
#include <GLFW/glfw3.h>

// 3. Headers nativos do X11/GLX (antes do glfw3native.h)
#include <X11/Xlib.h>
#include <GL/glx.h>

// 4. Header nativo do GLFW (deve vir por último)
#include <GLFW/glfw3native.h>


// --- FIM DA CORREÇÃO DEFINITIVA ---


#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <chrono> 
#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"
#include "imgui/imgui_stdlib.h"
#include <glm/gtc/matrix_transform.hpp>

// --- INÍCIO DAS ADIÇÕES PARA VR ---
// Agora, defina as macros do OpenXR.
#define XR_USE_PLATFORM_XLIB
#define XR_USE_GRAPHICS_API_OPENGL
#include <openxr/openxr.h>

// Ordem de inclusão correta para garantir que todos os tipos sejam definidos:
// 1. openxr.h (tipos base)
// 2. openxr_reflection.h (tipos de evento e strings)
// 3. openxr_platform.h (tipos específicos da plataforma)
#include <openxr/openxr_reflection.h> // Define XR_TYPE_SESSION_STATE_CHANGED
#include <openxr/openxr_platform.h>   // Define XrGraphicsBindingOpenGLXlibKHR

// Helper para verificar resultados de chamadas OpenXR
inline void xr_check(XrResult result, const std::string& message)
{
    if (XR_FAILED(result))
    {
        char resultString[XR_MAX_RESULT_STRING_SIZE];
        xrResultToString(XR_NULL_HANDLE, result, resultString);
        throw std::runtime_error(message + ": " + resultString);
    }
}

// Struct para manter o estado do OpenXR
struct VRState
{
    XrInstance instance = XR_NULL_HANDLE;
    XrSystemId systemId = XR_NULL_SYSTEM_ID;
    XrSession  session  = XR_NULL_HANDLE;
    // Mais handles serão adicionados aqui depois (space, swapchains, etc.)
};
// --- FIM DAS ADIÇÕES PARA VR ---

// --- INÍCIO DA CORREÇÃO DE LOCOMOÇÃO VR ---
glm::quat g_headset_orientation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f); // Orientação do headset
// --- FIM DA CORREÇÃO DE LOCOMOÇÃO VR ---

#include "gaussian.hpp"
#include "OctreeGaussian.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include <support/tinygltf/stb_image.h>



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

float   fov = 60;

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

 
// Adicione esta função em optixMeshViewer.cpp, antes do main()

sutil::Matrix4x4 createProjectionMatrixFov(const XrFovf& fov, float nearZ, float farZ)
{
    const float tanAngleLeft = tanf(fov.angleLeft);
    const float tanAngleRight = tanf(fov.angleRight);
    const float tanAngleDown = tanf(fov.angleDown);
    const float tanAngleUp = tanf(fov.angleUp);

    const float tanAngleWidth = tanAngleRight - tanAngleLeft;
    const float tanAngleHeight = tanAngleUp - tanAngleDown;

    sutil::Matrix4x4 mat;
    mat.setRow(0, {2.0f / tanAngleWidth, 0.0f, (tanAngleRight + tanAngleLeft) / tanAngleWidth, 0.0f});
    mat.setRow(1, {0.0f, 2.0f / tanAngleHeight, (tanAngleUp + tanAngleDown) / tanAngleHeight, 0.0f});
    mat.setRow(2, {0.0f, 0.0f, -(farZ + nearZ) / (farZ - nearZ), -2.0f * farZ * nearZ / (farZ - nearZ)});
    mat.setRow(3, {0.0f, 0.0f, -1.0f, 0.0f});
    
    return mat;
}

// Função para carregar um arquivo .jpg e retornar um buffer de float4
std::vector<float4> loadJPG(const std::string& filename, int& width, int& height)
{
    int channels;
    unsigned char* data = stbi_load(filename.c_str(), &width, &height, &channels, 4);
    if (!data)
    {
        std::cerr << "ERRO ao carregar JPG: " << filename << std::endl;
        return {};
    }

    std::vector<float4> buffer(width * height);
    for (int i = 0; i < width * height; ++i)
    {
        // Converte os valores de 0-255 para 0.0-1.0
        buffer[i] = make_float4(
            data[4 * i + 0] / 255.0f,
            data[4 * i + 1] / 255.0f,
            data[4 * i + 2] / 255.0f,
            data[4 * i + 3] / 255.0f
        );
    }

    stbi_image_free(data);
    return buffer;
}

cudaTextureObject_t createReflectionTexture(const std::string& jpg_path)
{
    int width = 0;
    int height = 0;
    std::vector<float4> reflection_buffer = loadJPG(jpg_path, width, height);

    if (reflection_buffer.empty())
    {
        return 0;
    }

    // Criar um array CUDA para a textura
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    cudaArray_t cuArray;
    CUDA_CHECK(cudaMallocArray(&cuArray, &channelDesc, width, height));

    // Copiar o buffer do host para o array CUDA
    CUDA_CHECK(cudaMemcpy2DToArray(cuArray, 0, 0, reflection_buffer.data(), width * sizeof(float4), width * sizeof(float4), height, cudaMemcpyHostToDevice));

    // Descrever o recurso de textura
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    // Descrever a textura
    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    // Criar e retornar o objeto de textura
    cudaTextureObject_t cudaTex;
    CUDA_CHECK(cudaCreateTextureObject(&cudaTex, &resDesc, &texDesc, nullptr));

    return cudaTex;
}

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

// Adicione esta função antes do loop principal
void handleKeyboardInput(GLFWwindow* window, bool is_vr_mode)
{
    float currentSpeed = trackball.moveSpeed();
    
    // Lógica original para modo desktop
    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        trackball.moveForward(currentSpeed);
        camera_changed = true;
    }
    if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        trackball.moveBackward(currentSpeed);
        camera_changed = true;
    }
    if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
        trackball.moveLeft(currentSpeed);
        camera_changed = true;
    }
    if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
        trackball.moveRight(currentSpeed);
        camera_changed = true;
    }
    // Adiciona movimento vertical com Q e E
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
        trackball.moveUp(currentSpeed);
    }
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
        trackball.moveDown(currentSpeed);
    }
}

//------------------------------------------------------------------------------
//
// GLFW callbacks
//
//------------------------------------------------------------------------------

static void mouseButtonCallback( GLFWwindow* window, int button, int action, int mods )
{
    // Adicione esta verificação no início da função
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse)
    {
        // Se o ImGui quer o mouse, não faça nada para a câmera e retorne.
        // O backend do ImGui para GLFW já cuidará de passar o evento para o ImGui.
        return;
    }

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
    // Adicione esta verificação também aqui
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse)
    {
        // O mouse está sobre uma janela do ImGui, então não mova a câmera.
        return;
    }

    if( mouse_button == GLFW_MOUSE_BUTTON_RIGHT )
    {
        trackball.setViewMode( sutil::Trackball::LookAtFixed );
        trackball.updateTracking( static_cast<int>( xpos ), static_cast<int>( ypos ), width, height );
        camera_changed = true;
    }
    else if( mouse_button == GLFW_MOUSE_BUTTON_LEFT )
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
    // É uma boa prática adicionar a verificação para o teclado também
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureKeyboard)
    {
        return;
    }

    if( action == GLFW_PRESS )
    {
        if( key == GLFW_KEY_ESCAPE )
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
    // E finalmente, adicione a verificação para o scroll
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse)
    {
        return;
    }

    if(trackball.wheelEvent((int)yscroll))
        camera_changed = true;
}


//------------------------------------------------------------------------------
//
// Helper functions
// TODO: some of these should move to sutil or optix util header
//
// --- INÍCIO DA CORREÇÃO: Declaração antecipada ---
void updateProjectionMatrix();
// --- FIM DA CORREÇÃO ---
//------------------------------------------------------------------------------

void printUsageAndExit( const char* argv0 )
{
    std::cerr <<  "Usage  : " << argv0 << " [options]\n";
    std::cerr <<  "Options: --file | -f <filename>      File for image output\n";
    std::cerr << "          --dim=<width>x<height>      Set image dimensions; defaults to 1280x728\n";
    std::cerr <<  "         --launch-samples | -s       Number of samples per pixel per launch (default 16)\n";
    std::cerr <<  "         --no-gl-interop             Disable GL interop for display\n";
    std::cerr <<  "         --model <model.gltf>        Specify model to render (required)\n";
    // --- INÍCIO DAS ADIÇÕES PARA VR ---
    std::cerr <<  "         --vr                        Enable OpenXR VR mode\n";
    // --- FIM DAS ADIÇÕES PARA VR ---
    std::cerr <<  "         --help | -h                 Print this usage message\n";
    exit( 0 );
}


void initLaunchParams( const sutil::GaussianScene& gscene, const sutil::Scene& scene ) {
    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &params.accum_buffer ),
                width*height*sizeof(float4)
                ) );
    params.frame_buffer_ptr = nullptr; // Unificado para ambos os modos

    params.subframe_index = 0u;

    // const float loffset = scene.aabb().maxExtent(); // AVISO: Variável não utilizada

    // TODO: add light support to sutil::Scene
    std::vector<Light> lights( 1);
    lights[0].type            = Light::Type::POINT;
    lights[0].point.color     = {1.0f, 1.0f, 1.0f};
    lights[0].point.intensity = .8f;
    lights[0].point.position  = make_float3(1.4,4.6,3);
    lights[0].point.falloff   = Light::Falloff::QUADRATIC; //{26.545f, 12.8f, 2.4f},

    // lights[1].type            = Light::Type::POINT;
    // lights[1].point.color     = {1.0f, 1.0f, 1.0f};
    // lights[1].point.intensity = 5.0f;
    // lights[1].point.position  = make_float3(0.0f, 2.8f, 0.0f);
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

    params.miss_color   = make_float3( 0.5f );

    //CUDA_CHECK( cudaStreamCreate( &stream ) );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_params ), sizeof( whitted::LaunchParams ) ) );

    params.ghandle = gscene.traversableHandle1();
    params.ghandle2 = gscene.traversableHandle2();
    params.handle =  scene.traversableHandle();
}

void updateModel( ) {

    // Construir a matriz do modelo usando sutil::Matrix4x4
    params.modelMatrix = sutil::Matrix4x4::translate( make_float3(g_position.x, g_position.y, g_position.z) );
    params.modelMatrix *= sutil::Matrix4x4::rotate( glm::radians(g_rotation.z), make_float3(0,0,1) );
    params.modelMatrix *= sutil::Matrix4x4::rotate( glm::radians(g_rotation.y), make_float3(0,1,0) );
    params.modelMatrix *= sutil::Matrix4x4::rotate( glm::radians(g_rotation.x), make_float3(1,0,0) );
    params.modelMatrix *= sutil::Matrix4x4::scale( make_float3(g_scale.x, g_scale.y, g_scale.z) );
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
    // --- INÍCIO DA CORREÇÃO DEFINITIVA: Restaurando a lógica do modo desktop ---
    uchar4* result_buffer_data = output_buffer.map();
    params.frame_buffer_ptr    = result_buffer_data;
    // params.frame_buffer_surf   = 0; // Não é mais usado
    
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
    // --- FIM DA CORREÇÃO DEFINITIVA ---
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
    trackball.setMoveSpeed( .03f );
    // trackball.setReferenceFrame( make_float3( 1.0f, 0.0f, 0.0f ), make_float3( 0.0f, 0.0f, 1.0f ), make_float3( 0.0f, 1.0f, 0.0f ) );
    // trackb
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

    params.projMatrix = sutil::Matrix4x4(glm::value_ptr(pm)).transpose();


    params.fov = fov;
    params.near = near;

    proj = pm;

}


void printGui(double frameTime) {

    ImGui::Begin("Tab");

    if (ImGui::SliderFloat("roughness", &params.roughness , 0.0f, 0.35f)) {
        camera_changed = true;
    }

    ImGui::InputText("file",&snapLabel);
    
    if (ImGui::Button("Save"))
        takeSnap = true;
    ImGui::SameLine();

    ImGui::Text("Camera Properties");
    if (ImGui::SliderFloat("Near", &near, 0.1f, 1.f)) {
        camera_changed = true;
    }
    if (ImGui::SliderFloat("Far", &far, 10.f, 1000.f)) {
        camera_changed = true;
    }
    if (ImGui::SliderFloat("Field of View", &fov, 1.f, 90.f)) {
        camera_changed = true;
    }

    std::string eye = "Eye X:" + std::to_string(camera.eye().x) + " Y:"+ std::to_string(camera.eye().y) + " Z:" + std::to_string(camera.eye().z);
    ImGui::Text(eye.c_str());

    std::string center = "Center X:" + std::to_string(camera.direction().x) + " Y:"+ std::to_string(camera.direction().y) + " Z:" + std::to_string(camera.direction().z);
    ImGui::Text(center.c_str());

    if (ImGui::Checkbox("Keep Y", &keepY)) {
        yAxis = camera.eye().y;
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
    
    // --- INÍCIO DA CORREÇÃO: Atualizar a matriz de projeção se a câmera mudou na GUI ---
    if (camera_changed)
    {
        // updateProjectionMatrix();
    }
    // --- FIM DA CORREÇÃO ---
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
    std::string envFile;
    
    // --- INÍCIO DAS ADIÇÕES PARA VR ---
    bool use_vr = false;
    // --- FIM DAS ADIÇÕES PARA VR ---

    envFile = "shperical_map.jpg";
    

    // output_buffer_type = sutil::CUDAOutputBufferType::CUDA_DEVICE;
    for( int i = 1; i < argc; ++i )
    {
        const std::string arg = argv[i];
        if( arg == "--help" || arg == "-h" )
        {
            printUsageAndExit( argv[0] );
        }
        // --- INÍCIO DAS ADIÇÕES PARA VR ---
        else if( arg == "--vr" )
        {
            use_vr = true;
        }
        // --- FIM DAS ADIÇÕES PARA VR ---
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
        else if( arg == "--env")
        {
            if( i >= argc - 1 )
                printUsageAndExit( argv[0] );
            envFile = argv[++i];
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


    // --- INÍCIO DA LÓGICA DE SELEÇÃO DE MODO ---
    if (use_vr)
    {
        std::cout << "Iniciando em modo OpenXR VR..." << std::endl;
        VRState vrState = {};
        GLFWwindow* window = nullptr;

        try
        {
            // --- INÍCIO DA CORREÇÃO: Ordem de Inicialização ---
            // 1. Inicializar um contexto OpenGL via GLFW
            window = sutil::initUI( "VR Gaussians", width, height );

            // --- INÍCIO DA ADIÇÃO: Registrar callbacks de mouse e teclado para a janela VR ---
            glfwSetMouseButtonCallback  ( window, mouseButtonCallback   );
            glfwSetCursorPosCallback    ( window, cursorPosCallback     );
            glfwSetWindowSizeCallback   ( window, windowSizeCallback    );
            glfwSetWindowIconifyCallback( window, windowIconifyCallback );
            glfwSetKeyCallback          ( window, keyCallback           );
            glfwSetScrollCallback       ( window, scrollCallback        );
            // --- FIM DA ADIÇÃO ---


            Gaussian gaussian((uint) width,(uint) height, sutil::sampleDataFilePath(gaussianFile1.c_str()), 3, false, 0);
            Gaussian gaussian2((uint) width,(uint) height, sutil::sampleDataFilePath(gaussianFile2.c_str()), 3, false, 0);
            sutil::GaussianScene gscene;
            gscene.addGaussians(gaussian.pos, gaussian.hsize);
            gscene.addGaussiansLow(gaussian2.pos, gaussian2.hsize);
            gscene.finalize();
            sutil::Scene scene;
            sutil::loadScene( sutil::sampleDataFilePath(infile.c_str()), scene );
            scene.finalize();
            // octree = new oct::OctreeGaussian(gaussian);
            cudaTextureObject_t reflection_texture = createReflectionTexture(envFile);

            // Inicializa os parâmetros de lançamento
            std::cout << "Iniciado o LaunchParams" << std::endl;
            initLaunchParams( gscene, scene );
            params.reflection_texture = reflection_texture;
            g_position = glm::vec3(0, 0, 0);
            g_rotation = glm::vec3(0, 0, 0);
            g_scale = glm::vec3(1.f, 1.f, 1.f);
            params.mode = 0;
            updateModel();
            params.g_pos = gaussian.pos_cuda;
            params.g_opacity = gaussian.opacity_cuda;
            params.g_shs = gaussian.shs_cuda;
            params.g_cov3d9 = gaussian.cov3d9_cuda;

            params.g2_pos = gaussian2.pos_cuda;
            params.g2_opacity = gaussian2.opacity_cuda;
            params.g2_shs = gaussian2.shs_cuda;
            params.g2_cov3d9 = gaussian2.cov3d9_cuda;
            
            // 2. Inicializar OpenXR - Criar a Instância
            std::vector<const char*> extensions = { XR_KHR_OPENGL_ENABLE_EXTENSION_NAME };
            XrInstanceCreateInfo instanceCreateInfo = {};
            instanceCreateInfo.type = XR_TYPE_INSTANCE_CREATE_INFO;
            strcpy(instanceCreateInfo.applicationInfo.applicationName, "OptiX Mesh Viewer");
            instanceCreateInfo.applicationInfo.applicationVersion = 1;
            strcpy(instanceCreateInfo.applicationInfo.engineName, "OptiX");
            instanceCreateInfo.applicationInfo.engineVersion = 1;
            instanceCreateInfo.applicationInfo.apiVersion = XR_CURRENT_API_VERSION;
            instanceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
            instanceCreateInfo.enabledExtensionNames = extensions.data();

            xr_check(xrCreateInstance(&instanceCreateInfo, &vrState.instance), "Falha ao criar instância OpenXR");

            // 3. Obter o SystemId (o headset)
            XrSystemGetInfo systemGetInfo = {};
            systemGetInfo.type = XR_TYPE_SYSTEM_GET_INFO;
            systemGetInfo.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
            xr_check(xrGetSystem(vrState.instance, &systemGetInfo, &vrState.systemId), "Falha ao obter sistema OpenXR");

            // ********************* INÍCIO DA NOVA CORREÇÃO ***************************
            // Obter a função xrGetOpenGLGraphicsRequirementsKHR
            PFN_xrGetOpenGLGraphicsRequirementsKHR xrGetOpenGLGraphicsRequirementsKHR = nullptr;
            xr_check(xrGetInstanceProcAddr(vrState.instance,
                "xrGetOpenGLGraphicsRequirementsKHR",
                reinterpret_cast<PFN_xrVoidFunction*>(&xrGetOpenGLGraphicsRequirementsKHR)),
                "Falha ao obter xrGetOpenGLGraphicsRequirementsKHR");

            // Preencher a estrutura de requisitos gráficos OpenGL
            XrGraphicsRequirementsOpenGLKHR glRequirements = { XR_TYPE_GRAPHICS_REQUIREMENTS_OPENGL_KHR };
            xr_check(xrGetOpenGLGraphicsRequirementsKHR(vrState.instance, vrState.systemId, &glRequirements),
                "Falha ao obter requisitos gráficos OpenGL");

            std::cout << "Versão mínima requerida do OpenGL: "
                      << XR_VERSION_MAJOR(glRequirements.minApiVersionSupported) << "."
                      << XR_VERSION_MINOR(glRequirements.minApiVersionSupported) << "\n";
            // ********************* FIM DA NOVA CORREÇÃO *****************************

            // 4. Criar a Sessão, ligando o contexto OpenGL
            #if defined(WIN32)
                XrGraphicsBindingOpenGLWin32KHR graphicsBinding = {};
                graphicsBinding.type = XR_TYPE_GRAPHICS_BINDING_OPENGL_WIN32_KHR;
                graphicsBinding.hDC = wglGetCurrentDC();
                graphicsBinding.hGLRC = wglGetCurrentContext();
            #else // Linux
                XrGraphicsBindingOpenGLXlibKHR graphicsBinding = {};
                graphicsBinding.type = XR_TYPE_GRAPHICS_BINDING_OPENGL_XLIB_KHR;
                graphicsBinding.xDisplay = glfwGetX11Display();
                graphicsBinding.glxContext = glfwGetGLXContext(window);
                graphicsBinding.glxDrawable = glfwGetGLXWindow(window);
            #endif

            XrSessionCreateInfo sessionCreateInfo = {};
            sessionCreateInfo.type = XR_TYPE_SESSION_CREATE_INFO;
            sessionCreateInfo.next = &graphicsBinding;
            sessionCreateInfo.systemId = vrState.systemId;

            xr_check(xrCreateSession(vrState.instance, &sessionCreateInfo, &vrState.session), "Falha ao criar sessão OpenXR");

            std::cout << "Sessão OpenXR criada com sucesso!" << std::endl;


            // --- STEP 5: Criar XrReferenceSpace ────────────────────────────────────────
            XrReferenceSpaceCreateInfo spaceCreateInfo = {};
            spaceCreateInfo.type = XR_TYPE_REFERENCE_SPACE_CREATE_INFO;
            spaceCreateInfo.poseInReferenceSpace.orientation = {0, 0, 0, 1}; // Identidade
            spaceCreateInfo.poseInReferenceSpace.position = {0, 0, 0};       // Origem
            spaceCreateInfo.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_LOCAL; // ou XR_REFERENCE_SPACE_TYPE_STAGE se preferir

            XrSpace referenceSpace = XR_NULL_HANDLE;
            xr_check(xrCreateReferenceSpace(vrState.session, &spaceCreateInfo, &referenceSpace),
                "Falha ao criar XrReferenceSpace");

            // --- STEP 6: Criar Swapchains ───────────────────────────────────────────────
            // Primeiro: consultar o número de views (normalmente 2 para estereoscopia)
            uint32_t viewCount = 0;
            xr_check(xrEnumerateViewConfigurationViews(vrState.instance, vrState.systemId,
                XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, 0, &viewCount, nullptr),
                "Falha ao enumerar views");

            std::vector<XrViewConfigurationView> viewConfigs(viewCount, { XR_TYPE_VIEW_CONFIGURATION_VIEW });
            xr_check(xrEnumerateViewConfigurationViews(vrState.instance, vrState.systemId,
                XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, viewCount, &viewCount, viewConfigs.data()),
                "Falha ao obter detalhes das views");

            // Criar um vetor para armazenar os handles dos swapchains (um por view)
            std::vector<XrSwapchain> swapchains(viewCount, XR_NULL_HANDLE);

            // Configurar as propriedades do swapchain baseadas na primeira view (geralmente ambas têm propriedades semelhantes)
            XrSwapchainCreateInfo swapchainCreateInfo = {};
            swapchainCreateInfo.type = XR_TYPE_SWAPCHAIN_CREATE_INFO;
            swapchainCreateInfo.format = GL_SRGB8_ALPHA8; // Escolha um formato compatível com OpenGL
            swapchainCreateInfo.sampleCount = viewConfigs[0].recommendedSwapchainSampleCount;
            swapchainCreateInfo.width = viewConfigs[0].recommendedImageRectWidth;
            swapchainCreateInfo.height = viewConfigs[0].recommendedImageRectHeight;
            swapchainCreateInfo.faceCount = 1;
            swapchainCreateInfo.arraySize = 1;
            swapchainCreateInfo.mipCount = 1;
            swapchainCreateInfo.usageFlags = XR_SWAPCHAIN_USAGE_SAMPLED_BIT | XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT;

            std::cout << "Criando swapchains com resolução " 
                      << swapchainCreateInfo.width << "x" << swapchainCreateInfo.height 
                      << " e formato " << swapchainCreateInfo.format << std::endl;

            // Criar o swapchain para cada view
            for (uint32_t i = 0; i < viewCount; ++i) {
                xr_check(xrCreateSwapchain(vrState.session, &swapchainCreateInfo, &swapchains[i]),
                         "Falha ao criar swapchain para view");
            }
            std::cout << "Swapchains criados com sucesso!" << std::endl;

            std::cout << "Tamanho de whitted::LaunchParams: " << sizeof(whitted::LaunchParams) << std::endl;

            // Precisamos de um vetor de vetores para armazenar as imagens de cada swapchain.
            std::vector<std::vector<XrSwapchainImageOpenGLKHR>> swapchainImages(viewCount);
            std::vector<std::vector<GLuint>> fbos(viewCount);
            std::vector<GLuint> pbos(viewCount);
            std::vector<cudaGraphicsResource_t> pbo_resources(viewCount);

            for (uint32_t i = 0; i < viewCount; ++i)
            {
                uint32_t imageCount = 0;
                xr_check(xrEnumerateSwapchainImages(swapchains[i], 0, &imageCount, nullptr), "Falha ao enumerar imagens do swapchain (contagem)");
    
                // Alocar espaço e inicializar as estruturas
                swapchainImages[i].resize(imageCount, {XR_TYPE_SWAPCHAIN_IMAGE_OPENGL_KHR});
    
                xr_check(xrEnumerateSwapchainImages(swapchains[i], imageCount, &imageCount,
                                                    reinterpret_cast<XrSwapchainImageBaseHeader*>(swapchainImages[i].data())),
                         "Falha ao enumerar imagens do swapchain (dados)");
    
                // Criar Framebuffers para cada imagem do swapchain
                fbos[i].resize(swapchainImages[i].size());
                for (size_t j = 0; j < swapchainImages[i].size(); ++j)
                {
                    glGenFramebuffers(1, &fbos[i][j]);
                    glBindFramebuffer(GL_FRAMEBUFFER, fbos[i][j]);
                    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, swapchainImages[i][j].image, 0);
                    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
                    if (status != GL_FRAMEBUFFER_COMPLETE) throw std::runtime_error("Framebuffer incompleto!");
                }
            }

            // Criar PBOs para renderização OptiX
            glGenBuffers(static_cast<GLsizei>(viewCount), pbos.data());
            for (uint32_t i = 0; i < viewCount; ++i)
            {
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbos[i]);
                glBufferData(GL_PIXEL_UNPACK_BUFFER, swapchainCreateInfo.width * swapchainCreateInfo.height * sizeof(uchar4), nullptr, GL_DYNAMIC_DRAW);
                CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&pbo_resources[i], pbos[i], cudaGraphicsRegisterFlagsWriteDiscard));
            }            
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            
            bool session_running = false;
            bool exit_loop = false;

            // Loop para processar eventos do OpenXR até a sessão estar pronta ou ser encerrada.
            while (!exit_loop) {
                XrEventDataBuffer eventBuffer = { XR_TYPE_EVENT_DATA_BUFFER };
                XrResult result = xrPollEvent(vrState.instance, &eventBuffer);

                if (result == XR_SUCCESS) {
                    switch (eventBuffer.type) {
                        case XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED: {
                            XrEventDataSessionStateChanged* stateEvent = reinterpret_cast<XrEventDataSessionStateChanged*>(&eventBuffer);
                            std::cout << "Estado da sessão OpenXR mudou para: " << stateEvent->state << std::endl;

                            if (stateEvent->state == XR_SESSION_STATE_READY) {
                                XrSessionBeginInfo beginInfo = { XR_TYPE_SESSION_BEGIN_INFO };
                                beginInfo.primaryViewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
                                xr_check(xrBeginSession(vrState.session, &beginInfo), "Falha ao iniciar sessão");
                                session_running = true;
                                std::cout << "Sessão iniciada." << std::endl;
                            }
                            if (stateEvent->state == XR_SESSION_STATE_STOPPING) {
                                xr_check(xrEndSession(vrState.session), "Falha ao encerrar sessão");
                                session_running = false;
                                std::cout << "Sessão encerrada." << std::endl;
                            }
                            if (stateEvent->state == XR_SESSION_STATE_EXITING || stateEvent->state == XR_SESSION_STATE_LOSS_PENDING) {
                                exit_loop = true; // Sinaliza para sair do loop principal
                                session_running = false;
                            }
                            break;
                        }
                        case XR_TYPE_EVENT_DATA_INSTANCE_LOSS_PENDING: {
                            exit_loop = true;
                            session_running = false;
                            break;
                        }
                    }
                } else if (result == XR_EVENT_UNAVAILABLE) {
                    // Se a sessão já está rodando, podemos sair deste loop e ir para o de renderização.
                    if (session_running) {
                        break;
                    }
                    // Se não, esperamos um pouco para não sobrecarregar a CPU.
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                } else {
                    xr_check(result, "Falha ao pesquisar eventos (xrPollEvent)");
                }
            }

            // Copiar parâmetros para dispositivo com alinhamento adequado
            CUDA_CHECK(cudaSetDevice(0));

            whitted::LaunchParams* d_aligned_params;
            CUDA_CHECK( cudaMalloc( &d_aligned_params, sizeof(whitted::LaunchParams) ) );

            std::chrono::duration<double> render_time(0.0);

            initCameraState( scene );
            trackball.setMoveSpeed( .1f );

            while (session_running && !glfwWindowShouldClose(window))
            {

                updateModel(); // Atualiza a matriz do modelo (rotação, etc.)
                handleKeyboardInput(window, true /* is_vr_mode */); // Passa o flag de VR

                if (camera_changed) {
                    params.subframe_index = 0;
                }

                glfwPollEvents(); // Mantém a janela do desktop responsiva
                sutil::beginFrameImGui();
                XrEventDataBuffer eventBuffer;
                while (true) {
                    // Sempre inicialize o buffer antes de chamar xrPollEvent.
                    eventBuffer = { XR_TYPE_EVENT_DATA_BUFFER };
                    XrResult result = xrPollEvent(vrState.instance, &eventBuffer);

                    if (result == XR_EVENT_UNAVAILABLE) {
                        // Não há mais eventos na fila, podemos prosseguir para a renderização.
                        break;
                    }
                    if (result != XR_SUCCESS) {
                        // Um erro ocorreu durante a pesquisa de eventos.
                        xr_check(result, "Falha ao pesquisar eventos no loop de renderização");
                        break;
                    }

                    // Processa o evento recebido.
                    switch (eventBuffer.type) {
                        case XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED: {
                            XrEventDataSessionStateChanged* stateEvent = reinterpret_cast<XrEventDataSessionStateChanged*>(&eventBuffer);
                            if (stateEvent->state == XR_SESSION_STATE_STOPPING) {
                                std::cout << "Sessão parando..." << std::endl;
                                xr_check(xrEndSession(vrState.session), "Falha ao encerrar sessão");
                                session_running = false;
                            }
                            break;
                        }
                        // Adicione outros casos de evento aqui se necessário.
                    }
                }

                if (!session_running) {
                    // Se a sessão parou, saia do loop de renderização imediatamente.
                    break;
                }

                // 1. Esperar o próximo frame
                XrFrameState frameState = { XR_TYPE_FRAME_STATE };
                xr_check(xrWaitFrame(vrState.session, nullptr, &frameState),
                         "Falha em xrWaitFrame");

                // 2. Iniciar o frame
                xr_check(xrBeginFrame(vrState.session, nullptr), "Falha em xrBeginFrame");

                // 3. Preparar as views para cada olho
                std::vector<XrView> views(viewCount, { XR_TYPE_VIEW });
                XrViewState viewState = { XR_TYPE_VIEW_STATE };
                XrViewLocateInfo viewLocateInfo = { XR_TYPE_VIEW_LOCATE_INFO };
                viewLocateInfo.viewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
                viewLocateInfo.displayTime = frameState.predictedDisplayTime;
                viewLocateInfo.space = referenceSpace;
                uint32_t viewCountOutput = 0;
                xr_check(xrLocateViews(vrState.session, &viewLocateInfo, &viewState,
                            viewCount, &viewCountOutput, views.data()),
                         "Falha em xrLocateViews");

                // 4. Para cada olho (view), obter a imagem do swapchain, renderizar e liberar
    
                std::vector<XrCompositionLayerProjectionView> projectionViews(viewCount, {XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW});

                auto t0 = std::chrono::steady_clock::now();
               
                for (uint32_t i = 0; i < viewCount; ++i)
                {
                    // Adquirir uma imagem do swapchain para renderizar
                    uint32_t imageIndex;
                    XrSwapchainImageAcquireInfo acquireInfo = {XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
                    xr_check(xrAcquireSwapchainImage(swapchains[i], &acquireInfo, &imageIndex), "Falha ao adquirir imagem do swapchain");

                    // Esperar a imagem estar pronta para ser usada
                    XrSwapchainImageWaitInfo waitInfo = {XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO};
                    waitInfo.timeout = XR_INFINITE_DURATION;
                    xr_check(xrWaitSwapchainImage(swapchains[i], &waitInfo), "Falha ao esperar pela imagem do swapchain");

                    const uint32_t vr_width = swapchainCreateInfo.width;
                    const uint32_t vr_height = swapchainCreateInfo.height;

                    const XrPosef& pose = views[i].pose;
                    const glm::quat xr_orientation(
                        pose.orientation.w, 
                        pose.orientation.x,  // Mantém X
                        pose.orientation.y,  // Mantém Y
                        -pose.orientation.z  // Inverte Z para corrigir o sentido de rotação
                    );

                    g_headset_orientation = xr_orientation; // Salva para uso em outras partes
                    
                    // 2. Obter base da câmera de navegação
                    float3 orig_u, orig_v, orig_w;
                    camera.UVWFrame(orig_u, orig_v, orig_w);
                    
                    // 3. Converte vetores da base para glm
                    glm::vec3 base_right(orig_u.x, orig_u.y, orig_u.z);
                    glm::vec3 base_up(orig_v.x, orig_v.y, orig_v.z);
                    glm::vec3 base_forward(orig_w.x, orig_w.y, orig_w.z);
                    
                    // 4. Aplica rotação do headset aos vetores da base
                    glm::vec3 rotated_right = xr_orientation * base_right;
                    glm::vec3 rotated_up = xr_orientation * base_up;
                    glm::vec3 rotated_forward = xr_orientation * base_forward;
                    
                    // 5. Converte de volta para float3 e normaliza
                    float3 final_right = normalize(make_float3(rotated_right.x, rotated_right.y, rotated_right.z));
                    float3 final_up = normalize(make_float3(rotated_up.x, rotated_up.y, rotated_up.z));
                    float3 final_forward = normalize(make_float3(rotated_forward.x, rotated_forward.y, rotated_forward.z));
                    
                    // 6. Atualiza os vetores da câmera mantendo as magnitudes originais
                    params.U = length(orig_u) * final_right;
                    params.V = length(orig_v) * final_up;
                    params.W = length(orig_w) * final_forward;

                    trackball.setVRMoveDirection(final_forward);
                    
                    // Calcula posição final do olho: deslocamento fornecido pelo XR rotacionado e somado à posição de navegação
                    const float3 eye_position = make_float3(pose.position.x, pose.position.y, pose.position.z);
                    glm::vec3 eye_offset = xr_orientation * glm::vec3(eye_position.x, eye_position.y, eye_position.z);
                    params.eye = camera.eye() + make_float3(eye_offset.x, eye_offset.y, eye_offset.z);
                    
                    if(camera_changed) params.subframe_index = 0;

                    // 2. Mapear o PBO para obter um CUdeviceptr
                    CUDA_CHECK(cudaGraphicsMapResources(1, &pbo_resources[i], 0));
                    void* pbo_void_ptr = nullptr;
                    size_t pbo_size = 0;
                    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(&pbo_void_ptr, &pbo_size, pbo_resources[i]));
                    CUdeviceptr pbo_ptr = (CUdeviceptr)pbo_void_ptr;

                    // 5. Configurar os parâmetros de lançamento e executar o kernel
                    // A lógica do kernel agora é a mesma do desktop.
                    params.frame_buffer_ptr = reinterpret_cast<uchar4*>(pbo_ptr);
                    params.width = vr_width;
                    params.height = vr_height;

                    
                    CUDA_CHECK( cudaMemcpyAsync( 
                        d_aligned_params,
                        &params,
                        sizeof(whitted::LaunchParams),
                        cudaMemcpyHostToDevice,
                        0
                    ) );

                    // Launch do OptiX
                    OPTIX_CHECK( optixLaunch(
                        scene.pipeline(),
                        0,
                        reinterpret_cast<CUdeviceptr>(d_aligned_params),
                        sizeof(whitted::LaunchParams),
                        scene.sbt(),
                        vr_width,
                        vr_height,
                        1
                    ) );

                    // Sincronizar e liberar
                    CUDA_SYNC_CHECK();
                    
                    auto t1 = std::chrono::steady_clock::now();
                    render_time += std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
                    t0 = t1;

                    // 6. Desmapear o PBO
                    CUDA_CHECK(cudaGraphicsUnmapResources(1, &pbo_resources[i], 0));

                    // 7. Copiar do PBO para a textura do swapchain usando OpenGL
                    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbos[i]);
                    glBindTexture(GL_TEXTURE_2D, swapchainImages[i][imageIndex].image);
                    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, vr_width, vr_height, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
                    glBindTexture(GL_TEXTURE_2D, 0);
                    // --- FIM DA CORREÇÃO ---
                    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
                    // --- FIM DA ALTERAÇÃO ---

                    // Liberar a imagem, sinalizando que a renderização está completa
                    XrSwapchainImageReleaseInfo releaseInfo = {XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
                    xr_check(xrReleaseSwapchainImage(swapchains[i], &releaseInfo), "Falha ao liberar imagem do swapchain");

                    // 7. Preencher a estrutura de projeção para esta view
                    // Preenche a estrutura de projeção para esta view.
                    projectionViews[i].pose = views[i].pose;
                    projectionViews[i].fov = views[i].fov;
                    projectionViews[i].subImage.swapchain = swapchains[i];
                    projectionViews[i].subImage.imageArrayIndex = 0;
                    projectionViews[i].subImage.imageRect.offset = { 0, 0 };
                    projectionViews[i].subImage.imageRect.extent = { (int32_t)vr_width, (int32_t)vr_height };
                }

                // 5. Compor as views para o ambiente VR
                XrCompositionLayerProjection layer = { XR_TYPE_COMPOSITION_LAYER_PROJECTION };
                layer.space = referenceSpace;
                
                // --- INÍCIO DA CORREÇÃO ---
                // Define a contagem de views e aponta para os dados que acabamos de preencher.
                layer.viewCount = viewCount;
                layer.views = projectionViews.data();
                // --- FIM DA CORREÇÃO ---

                // 6. Finalizar o frame
                XrFrameEndInfo frameEndInfo = { XR_TYPE_FRAME_END_INFO };
                frameEndInfo.displayTime = frameState.predictedDisplayTime;
                frameEndInfo.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
                frameEndInfo.layerCount = 1;
                const XrCompositionLayerBaseHeader* layers[] = { reinterpret_cast<const XrCompositionLayerBaseHeader*>(&layer) };
                frameEndInfo.layers = layers;
                xr_check(xrEndFrame(vrState.session, &frameEndInfo),
                         "Falha em xrEndFrame");

                glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
                glClear(GL_COLOR_BUFFER_BIT);

                // --- INÍCIO DA ADIÇÃO: ImGui no loop VR ---
                printGui(render_time.count() * 1000.0 / viewCount); // Média por olho
                sutil::endFrameImGui();
                // --- FIM DA ADIÇÃO ---

                glfwSwapBuffers(window);
                // --- FIM DA CORREÇÃO ---
            }
            CUDA_CHECK( cudaFree(d_aligned_params) );

        }
        catch( std::exception& e )
        {
            std::cerr << "Caught exception in VR mode: " << e.what() << "\n";
            // Limpeza em caso de erro
            if (vrState.session) xrDestroySession(vrState.session);
            if (vrState.instance) xrDestroyInstance(vrState.instance);
            if (window) sutil::cleanupUI(window);
            return 1;
        }

        // Limpeza normal
        std::cout << "Fechando a aplicação VR..." << std::endl;
        if (vrState.session) xrDestroySession(vrState.session);
        if (vrState.instance) xrDestroyInstance(vrState.instance);
        sutil::cleanupUI(window);
    }
    else // Modo Local (código original)
    {
        try
        {
            // --- INÍCIO DA CORREÇÃO: Lógica de inicialização para modo desktop ---
            // Carrega os dados da cena antes de criar a janela e o contexto GL.
            Gaussian gaussian((uint) width,(uint) height, sutil::sampleDataFilePath(gaussianFile1.c_str()), 3, false, 0);
            Gaussian gaussian2((uint) width,(uint) height, sutil::sampleDataFilePath(gaussianFile2.c_str()), 3, false, 0);
            sutil::GaussianScene gscene;
            gscene.addGaussians(gaussian.pos, gaussian.hsize);
            gscene.addGaussiansLow(gaussian2.pos, gaussian2.hsize);
            gscene.finalize();
            sutil::Scene scene;
            sutil::loadScene( sutil::sampleDataFilePath(infile.c_str()), scene );
            scene.finalize();
            // octree = new oct::OctreeGaussian(gaussian);
            cudaTextureObject_t reflection_texture = createReflectionTexture(envFile);

            // Inicializa os parâmetros de lançamento
            initLaunchParams( gscene, scene );
            params.reflection_texture = reflection_texture;
            g_position = glm::vec3(0, 0, 0);
            g_rotation = glm::vec3(0, 0, 0);
            g_scale = glm::vec3(1.f, 1.f, 1.f);
            params.mode = 0;
            updateModel();
            params.g_pos = gaussian.pos_cuda;
            params.g_opacity = gaussian.opacity_cuda;
            params.g_shs = gaussian.shs_cuda;
            params.g_cov3d9 = gaussian.cov3d9_cuda;

            params.g2_pos = gaussian2.pos_cuda;
            params.g2_opacity = gaussian2.opacity_cuda;
            params.g2_shs = gaussian2.shs_cuda;
            params.g2_cov3d9 = gaussian2.cov3d9_cuda;
            initCameraState( scene );
            camera.setFovY(60.f);



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
                    std::cout << "Vamos entrar no loop" << std::endl;
                    do
                    {
                        auto t0 = std::chrono::steady_clock::now();
                        glfwPollEvents();
                        
                        handleKeyboardInput(window, false /* is_vr_mode */);
                        
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
    }
    // --- FIM DA LÓGICA DE SELEÇÃO DE MODO ---

    return 0;
}
