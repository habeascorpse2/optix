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
#include <ctime>
#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"
#include "imgui/imgui_stdlib.h"
#include <glm/gtc/matrix_transform.hpp>

#define XR_USE_PLATFORM_XLIB
#define XR_USE_GRAPHICS_API_OPENGL
#include <openxr/openxr.h>

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

    // --- Adições para Passthrough ---
    PFN_xrCreatePassthroughFB xrCreatePassthroughFB = nullptr;
    PFN_xrDestroyPassthroughFB xrDestroyPassthroughFB = nullptr;
    PFN_xrPassthroughStartFB xrPassthroughStartFB = nullptr;
    PFN_xrPassthroughPauseFB xrPassthroughPauseFB = nullptr;
    PFN_xrCreatePassthroughLayerFB xrCreatePassthroughLayerFB = nullptr;
    PFN_xrDestroyPassthroughLayerFB xrDestroyPassthroughLayerFB = nullptr;

    XrPassthroughFB passthroughFeature = XR_NULL_HANDLE;
    // O XrPassthroughLayerFB será criado no loop de renderização.

    // Input Actions
    XrActionSet actionSet = XR_NULL_HANDLE;
    XrAction moveAction = XR_NULL_HANDLE;
    XrAction triggerAction = XR_NULL_HANDLE; // Ação para o gatilho
    XrAction handPoseAction = XR_NULL_HANDLE;
    XrSpace handSpace = XR_NULL_HANDLE;
};


#include "gaussian.hpp"
#include "OctreeGaussian.hpp"
#include "VRLogger.hpp"

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
float g_scale;
glm::mat4 proj;
glm::mat4 model;


glm::quat g_headset_orientation;
float nav_scale = 1.0f;   
float3 initial_headset_pos = make_float3(0.f, 0.f, 0.f); // Posição inicial do headset
bool initial_pos_captured = false; // Flag para capturar a posição apenas uma vez


float3 nav_offset = make_float3(0.f, 0.f, 2.0f); // Começa 2 metros atrás para ver o objeto

float near = 0.1;
float far = 1000.f;

// --- VARIÁVEIS GLOBAIS PARA INSTANCIAMENTO DINÂMICO ---
std::vector<OptixInstance> g_instances;       // Lista de instâncias na CPU
OptixTraversableHandle g_scene_gas = 0;       // Handle da geometria original (protótipo)
int g_held_instance_idx = -1;                 // Índice da instância sendo segurada (-1 = nenhuma)
bool g_trigger_was_pressed = false;           // Estado anterior do gatilho

// Buffers persistentes para o IAS (para evitar realocação constante)
CUdeviceptr d_instances_buffer = 0;
CUdeviceptr d_ias_output_buffer = 0;
CUdeviceptr d_ias_scratch_buffer = 0;
size_t      d_ias_output_size = 0;
size_t      d_ias_scratch_size = 0;


std::string snapLabel = "snapshot";
const std::string snapFolder = "snapshots/";
int snapCounter = 0;
bool takeSnap = false;

bool keepY = false;
float yAxis = 0.f;
bool g_is_logging = false;
std::string g_logFilename = "tracking_log";
VRLogger* g_logger = nullptr;

 
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

// --- FUNÇÃO PARA CONSTRUIR/ATUALIZAR O IAS ---
void buildDynamicIAS(const sutil::Scene& scene) {
    if (g_instances.empty()) {
        params.handle = g_scene_gas; // Se não houver instâncias, renderiza a cena original estática
        return;
    }

    // 1. Copiar instâncias para a GPU
    size_t instances_size_in_bytes = sizeof(OptixInstance) * g_instances.size();
    
    // Realocar se necessário (simples: libera e aloca de novo se crescer, ou apenas aloca na primeira vez)
    // Para produção, ideal seria gerenciar capacidade vs tamanho.
    if (d_instances_buffer) CUDA_CHECK(cudaFree((void*)d_instances_buffer));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_instances_buffer), instances_size_in_bytes));
    CUDA_CHECK(cudaMemcpy((void*)d_instances_buffer, g_instances.data(), instances_size_in_bytes, cudaMemcpyHostToDevice));

    // 2. Configurar Build Input
    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    build_input.instanceArray.instances = d_instances_buffer;
    build_input.instanceArray.numInstances = static_cast<unsigned int>(g_instances.size());

    // 3. Configurar Opções de Build
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;

    // 4. Calcular Tamanhos de Memória
    OptixAccelBufferSizes buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(scene.context(), &accel_options, &build_input, 1, &buffer_sizes));

    // Re-alocar buffers de saída/scratch se o tamanho necessário aumentou
    if (buffer_sizes.outputSizeInBytes > d_ias_output_size) {
        if (d_ias_output_buffer) CUDA_CHECK(cudaFree((void*)d_ias_output_buffer));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_ias_output_buffer), buffer_sizes.outputSizeInBytes));
        d_ias_output_size = buffer_sizes.outputSizeInBytes;
    }
    if (buffer_sizes.tempSizeInBytes > d_ias_scratch_size) {
        if (d_ias_scratch_buffer) CUDA_CHECK(cudaFree((void*)d_ias_scratch_buffer));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_ias_scratch_buffer), buffer_sizes.tempSizeInBytes));
        d_ias_scratch_size = buffer_sizes.tempSizeInBytes;
    }

    // 5. Construir o IAS
    OptixTraversableHandle ias_handle;
    OPTIX_CHECK(optixAccelBuild(scene.context(), 0, &accel_options, &build_input, 1, d_ias_scratch_buffer, d_ias_scratch_size, d_ias_output_buffer, d_ias_output_size, &ias_handle, nullptr, 0));

    // 6. Atualizar o handle global usado no Raygen
    params.handle = ias_handle;
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

// Função para processar input do controle VR
void handleControllerInput(VRState& vrState, const XrSpace& referenceSpace, XrTime displayTime)
{
    if (vrState.session == XR_NULL_HANDLE) return;

    const XrActiveActionSet activeActionSet = {vrState.actionSet, XR_NULL_PATH};
    XrActionsSyncInfo syncInfo = {XR_TYPE_ACTIONS_SYNC_INFO};
    syncInfo.countActiveActionSets = 1;
    syncInfo.activeActionSets = &activeActionSet;
    
    if (XR_FAILED(xrSyncActions(vrState.session, &syncInfo))) return;

    XrActionStateGetInfo getInfo = {XR_TYPE_ACTION_STATE_GET_INFO};
    getInfo.action = vrState.moveAction;
    
    XrActionStateVector2f moveState = {XR_TYPE_ACTION_STATE_VECTOR2F};
    if (XR_FAILED(xrGetActionStateVector2f(vrState.session, &getInfo, &moveState))) return;

    if (moveState.isActive && (fabsf(moveState.currentState.x) > 0.1f || fabsf(moveState.currentState.y) > 0.1f)) {
        const float move_speed = 0.05f;
        
        // Direção do headset projetada no chão
        glm::vec3 forward = g_headset_orientation * glm::vec3(0.0f, 0.0f, -1.0f);
        forward.y = 0.0f;
        if (glm::length(forward) > 0.01f) forward = glm::normalize(forward);

        glm::vec3 right = g_headset_orientation * glm::vec3(1.0f, 0.0f, 0.0f);
        right.y = 0.0f;
        if (glm::length(right) > 0.01f) right = glm::normalize(right);

        // Input do analógico (Y é frente/trás, X é esquerda/direita)
        float dx = moveState.currentState.x;
        float dy = moveState.currentState.y;

        float3 move_vec = make_float3(
            forward.x * dy + right.x * dx,
            forward.y * dy + right.y * dx,
            forward.z * dy + right.z * dx
        );
        nav_offset += move_speed * move_vec;
    }

    // --- LÓGICA DE SPAWN & GRAB (Gatilho) ---
    XrActionStateGetInfo triggerInfo = {XR_TYPE_ACTION_STATE_GET_INFO};
    triggerInfo.action = vrState.triggerAction;
    XrActionStateFloat triggerState = {XR_TYPE_ACTION_STATE_FLOAT};
    if (XR_SUCCEEDED(xrGetActionStateFloat(vrState.session, &triggerInfo, &triggerState))) {
        bool is_pressed = triggerState.currentState > 0.5f; // Limiar de ativação

        // Obter posição da mão
        XrSpaceLocation handLocation = {XR_TYPE_SPACE_LOCATION};
        xrLocateSpace(vrState.handSpace, referenceSpace, displayTime, &handLocation);

        if (handLocation.locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT) {
            glm::vec3 hand_pos = glm::vec3(handLocation.pose.position.x, handLocation.pose.position.y, handLocation.pose.position.z);
            glm::quat hand_rot = glm::quat(handLocation.pose.orientation.w, handLocation.pose.orientation.x, handLocation.pose.orientation.y, handLocation.pose.orientation.z);
            
            // Calcular posição virtual da mão (Mundo VR)
            glm::vec3 virtual_hand_pos = glm::vec3(nav_offset.x, nav_offset.y, nav_offset.z) + (hand_pos * nav_scale);
            float gltf_scale = .1f;

            // Construir matriz de transformação (GLM Col-Major)
            glm::mat4 mat = glm::mat4(1.0f);
            mat = glm::translate(mat, virtual_hand_pos);
            mat = mat * glm::mat4_cast(hand_rot);
            mat = glm::scale(mat, glm::vec3(gltf_scale));

            // Converter para formato OptiX (Row-Major 3x4)
            float transform[12];
            const float* m = (const float*)&mat;
            // Transpor: GLM[col][row] -> OptiX[row][col]
            transform[0] = m[0*4+0]; transform[1] = m[1*4+0]; transform[2] = m[2*4+0]; transform[3] = m[3*4+0];
            transform[4] = m[0*4+1]; transform[5] = m[1*4+1]; transform[6] = m[2*4+1]; transform[7] = m[3*4+1];
            transform[8] = m[0*4+2]; transform[9] = m[1*4+2]; transform[10] = m[2*4+2]; transform[11] = m[3*4+2];

            // 1. Rising Edge (Apertou agora): Criar nova instância
            if (is_pressed && !g_trigger_was_pressed) {
                OptixInstance new_instance = {};
                memcpy(new_instance.transform, transform, sizeof(float)*12);
                new_instance.instanceId = g_instances.size(); // ID único
                new_instance.visibilityMask = 255;
                new_instance.sbtOffset = 0; 
                new_instance.flags = OPTIX_INSTANCE_FLAG_NONE;
                new_instance.traversableHandle = g_scene_gas; // Aponta para o modelo original
                
                g_instances.push_back(new_instance);
                g_held_instance_idx = g_instances.size() - 1; // Segura o novo objeto
            }
            
            // 2. Holding (Segurando): Atualizar posição da instância segurada
            else if (is_pressed && g_held_instance_idx != -1) {
                if (g_held_instance_idx < g_instances.size()) {
                    memcpy(g_instances[g_held_instance_idx].transform, transform, sizeof(float)*12);
                }
            }

            // 3. Falling Edge (Soltou): Parar de segurar
            else if (!is_pressed && g_trigger_was_pressed) {
                g_held_instance_idx = -1;
            }
        }
        g_trigger_was_pressed = is_pressed;
    }
}

// Adicione esta função antes do loop principal
void handleKeyboardInput(GLFWwindow* window, bool is_vr_mode)
{
    const float move_speed = 0.05f;  // A velocidade já é ajustada no trackball
    if (is_vr_mode) {
        // Lógica de locomoção para o modo VR
        // Calcula a direção do movimento no plano horizontal (XZ)
        glm::vec3 forward = g_headset_orientation * glm::vec3(0.0f, 0.0f, -1.0f);
        forward.y = 0; // Projeta no plano horizontal
        forward = glm::normalize(forward);

        glm::vec3 right = g_headset_orientation * glm::vec3(1.0f, 0.0f, 0.0f);
        right.y = 0; // Projeta no plano horizontal
        right = glm::normalize(right);

        if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
            nav_offset += move_speed * make_float3(forward.x, forward.y, forward.z);
        if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            nav_offset -= move_speed * make_float3(forward.x, forward.y, forward.z);
        if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
            nav_offset -= move_speed * make_float3(right.x, right.y, right.z);
        if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
            nav_offset += move_speed * make_float3(right.x, right.y, right.z);
        if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
            nav_offset.y += move_speed; // Movimento vertical para cima
        if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
            nav_offset.y -= move_speed; // Movimento vertical para baixo

    
    } else {
        // Lógica original para modo desktop
        if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
            trackball.moveForward(move_speed, true);
            camera_changed = true;
        }
        if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
            trackball.moveBackward(move_speed, true);
            camera_changed = true;
        }
        if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
            trackball.moveLeft(move_speed, true);
            camera_changed = true;
        }
        if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
            trackball.moveRight(move_speed, true);
            camera_changed = true;
        }
        // Adiciona movimento vertical com Q e E
        if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
            trackball.moveUp(move_speed, true);
            camera_changed = true;
        }
        if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
            trackball.moveDown(move_speed, true);
            camera_changed = true;
        }
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

    std::vector<Light> lights( 1);
    lights[0].type            = Light::Type::POINT;
    lights[0].point.color     = {1.0f, 1.0f, 1.0f};
    lights[0].point.intensity = .8f;
    lights[0].point.position  = make_float3(1.4,4.6,3);
    lights[0].point.falloff   = Light::Falloff::QUADRATIC; //{26.545f, 12.8f, 2.4f},
    
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
    
    // Se estiver em modo VR com ALVR, usamos verde-limão para chroma key.
    // Caso contrário, usamos a cor de fundo normal.
    params.miss_color   = make_float3( 1.0f, 0.0f, 1.0f );

    //CUDA_CHECK( cudaStreamCreate( &stream ) );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_params ), sizeof( whitted::LaunchParams ) ) );


    params.ghandle = gscene.traversableHandle1();
    params.ghandle2 = gscene.traversableHandle2();
    
    // Salvar o handle da geometria original para instanciar depois
    // FIX: Usar o GAS da primeira malha para evitar aninhamento de IAS (IAS->IAS) que falha em pipelines de nível único.
    if (!scene.meshes().empty()) {
        g_scene_gas = scene.meshes()[0]->gas_handle;
    } else {
        g_scene_gas = scene.traversableHandle();
    }
    params.handle = g_scene_gas; // Inicializa com a cena estática visível
}

void updateModel( ) {

    // Construir a matriz do modelo usando sutil::Matrix4x4
    params.gaussianModelMatrix = sutil::Matrix4x4::translate( make_float3(g_position.x, g_position.y, g_position.z) );
    params.gaussianModelMatrix *= sutil::Matrix4x4::rotate( glm::radians(g_rotation.z), make_float3(0,0,1) );
    params.gaussianModelMatrix *= sutil::Matrix4x4::rotate( glm::radians(g_rotation.y), make_float3(0,1,0) );
    params.gaussianModelMatrix *= sutil::Matrix4x4::rotate( glm::radians(g_rotation.x), make_float3(1,0,0) );
    params.gaussianModelMatrix *= sutil::Matrix4x4::scale( make_float3(g_scale) );

}


void handleCameraUpdate( whitted::LaunchParams& params )
{
    if( !camera_changed )
        return;

    // Se o movimento foi pelo mouse, reseta a matriz do objeto.
    if (mouse_button != -1) {
    } else { // Se o movimento foi pelo teclado, atualiza a matriz do objeto.
        float3 eye_prev = make_float3(params.eye.x, params.eye.y, params.eye.z);
        float3 eye_curr = camera.eye();
        float3 move_vec = eye_curr - eye_prev;
    }
    
    camera.setAspectRatio( static_cast<float>( width ) / static_cast<float>( height ) );
    if (keepY) {
        float3 newEye = camera.eye();
        newEye.y = yAxis;
        camera.setEye(newEye);
    }
    
    params.eye = camera.eye();
    camera.UVWFrame( params.U, params.V, params.W );
    updateModel();
    
    camera_changed = false;

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

    // params.projMatrix = sutil::Matrix4x4(glm::value_ptr(pm)).transpose();


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

    ImGui::InputText("Log", &g_logFilename);
    ImGui::SameLine();
    if (g_is_logging) {
        if (ImGui::Button("Stop")) {
            g_is_logging = false;
            if (g_logger) {
                delete g_logger;
                g_logger = nullptr;
            }
            std::cout << "Logging stopped." << std::endl;
        }
    } else {
        if (ImGui::Button("Record")) {
            std::string finalName = g_logFilename;
            if (finalName.find(".json") == std::string::npos)
                finalName += ".json";
            g_logger = new VRLogger(finalName);
            g_is_logging = true;
            std::cout << "Logging started." << std::endl;
        }
    }

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

    if (ImGui::SliderFloat("HMD scale position", &nav_scale, 1.f, 5.f)) {
        // camera_changed = true;
    }

    std::string eye = "Eye X:" + std::to_string(camera.eye().x) + " Y:"+ std::to_string(camera.eye().y) + " Z:" + std::to_string(camera.eye().z);
    ImGui::Text(eye.c_str());

    std::string center = "Center X:" + std::to_string(camera.direction().x) + " Y:"+ std::to_string(camera.direction().y) + " Z:" + std::to_string(camera.direction().z);
    ImGui::Text(center.c_str());

    if (ImGui::Checkbox("Keep Y", &keepY)) {
        yAxis = camera.eye().y;
    }

    ImGui::Separator();

    ImGui::Text("Environment");
    if (ImGui::ColorEdit3("Miss Color", &params.miss_color.x, ImGuiColorEditFlags_DisplayRGB)) {
        camera_changed = true; // Reseta acumulação para aplicar a nova cor
    }

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

    if (ImGui::SliderFloat("Scale ", &g_scale, 0.1f, 6.f))
        camera_changed = true;
    // if (ImGui::SliderFloat("Scale Y", &g_scale.y, -1.f, 4.f))
    //     camera_changed = true;
    // if (ImGui::SliderFloat("Scale Z", &g_scale.z, -1.f, 4.f))
    //     camera_changed = true;
    
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
    
    bool use_vr = false;

    envFile = "shperical_map.jpg";
    gaussianFile1 = "quarto3_100.ply";
    gaussianFile2 = "quarto3_10.ply";
    

    output_buffer_type = sutil::CUDAOutputBufferType::CUDA_DEVICE;
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
        infile = "quarto2.glb";
        std::cerr << "--model argument required" << std::endl;
        // printUsageAndExit( argv[0] );
    }

    if (use_vr)
    {
        std::cout << "Iniciando em modo OpenXR VR..." << std::endl;
        VRState vrState = {};
        GLFWwindow* window = nullptr;
        params.is_vr = true;

        try
        {
            // 1. Inicializar um contexto OpenGL via GLFW
            window = sutil::initUI( "VR Gaussians", width, height );

            // Registrar callbacks de mouse e teclado para a janela VR ---
            glfwSetMouseButtonCallback  ( window, mouseButtonCallback   );
            glfwSetCursorPosCallback    ( window, cursorPosCallback     );
            glfwSetWindowSizeCallback   ( window, windowSizeCallback    );
            glfwSetWindowIconifyCallback( window, windowIconifyCallback );
            glfwSetKeyCallback          ( window, keyCallback           );
            glfwSetScrollCallback       ( window, scrollCallback        );
            // --- 


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
            g_scale = 5.3f; //glm::vec3(1.f, 1.f, 1.f);
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
            
            // 2. Inicializar OpenXR - Verificar e Criar a Instância
            std::vector<const char*> extensions = { XR_KHR_OPENGL_ENABLE_EXTENSION_NAME };

            // --- INÍCIO PASSO 1: HABILITAR EXTENSÃO PASSTHROUGH ---
            // Verificar se a extensão de Passthrough está disponível
            uint32_t extCount = 0;
            xrEnumerateInstanceExtensionProperties(nullptr, 0, &extCount, nullptr);
            std::vector<XrExtensionProperties> extProps(extCount, {XR_TYPE_EXTENSION_PROPERTIES});
            xrEnumerateInstanceExtensionProperties(nullptr, extCount, &extCount, extProps.data());

            bool passthrough_supported = false;
            for (const auto& prop : extProps) {
                if (strcmp(prop.extensionName, XR_FB_PASSTHROUGH_EXTENSION_NAME) == 0) {
                    passthrough_supported = true;
                    extensions.push_back(XR_FB_PASSTHROUGH_EXTENSION_NAME);
                    std::cout << "Extensao XR_FB_passthrough encontrada e habilitada." << std::endl;
                    break;
                }
            }
            // --- FIM PASSO 1 ---

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

            // --- INÍCIO PASSO 1: OBTER PONTEIROS DE FUNÇÃO ---
            if (passthrough_supported) {
                xr_check(xrGetInstanceProcAddr(vrState.instance, "xrCreatePassthroughFB", (PFN_xrVoidFunction*)(&vrState.xrCreatePassthroughFB)), "Falha ao obter xrCreatePassthroughFB");
                xr_check(xrGetInstanceProcAddr(vrState.instance, "xrDestroyPassthroughFB", (PFN_xrVoidFunction*)(&vrState.xrDestroyPassthroughFB)), "Falha ao obter xrDestroyPassthroughFB");
                xr_check(xrGetInstanceProcAddr(vrState.instance, "xrPassthroughStartFB", (PFN_xrVoidFunction*)(&vrState.xrPassthroughStartFB)), "Falha ao obter xrPassthroughStartFB");
                xr_check(xrGetInstanceProcAddr(vrState.instance, "xrPassthroughPauseFB", (PFN_xrVoidFunction*)(&vrState.xrPassthroughPauseFB)), "Falha ao obter xrPassthroughPauseFB");
                xr_check(xrGetInstanceProcAddr(vrState.instance, "xrCreatePassthroughLayerFB", (PFN_xrVoidFunction*)(&vrState.xrCreatePassthroughLayerFB)), "Falha ao obter xrCreatePassthroughLayerFB");
                xr_check(xrGetInstanceProcAddr(vrState.instance, "xrDestroyPassthroughLayerFB", (PFN_xrVoidFunction*)(&vrState.xrDestroyPassthroughLayerFB)), "Falha ao obter xrDestroyPassthroughLayerFB");
                std::cout << "Funcoes da extensao Passthrough carregadas com sucesso." << std::endl;
            } else {
                std::cout << "AVISO: Extensao XR_FB_passthrough nao suportada pelo runtime." << std::endl;
            }
            // --- FIM PASSO 1 ---

            // --- CONFIGURAÇÃO DE INPUT (ACTIONS) ---
            XrActionSetCreateInfo actionSetInfo = {XR_TYPE_ACTION_SET_CREATE_INFO};
            strcpy(actionSetInfo.actionSetName, "gameplay");
            strcpy(actionSetInfo.localizedActionSetName, "Gameplay");
            xr_check(xrCreateActionSet(vrState.instance, &actionSetInfo, &vrState.actionSet), "Falha ao criar ActionSet");

            XrActionCreateInfo actionInfo = {XR_TYPE_ACTION_CREATE_INFO};
            actionInfo.actionType = XR_ACTION_TYPE_VECTOR2F_INPUT;
            strcpy(actionInfo.actionName, "move");
            strcpy(actionInfo.localizedActionName, "Move");
            xr_check(xrCreateAction(vrState.actionSet, &actionInfo, &vrState.moveAction), "Falha ao criar Action Move");

            // Action para o Gatilho (Trigger)
            actionInfo.actionType = XR_ACTION_TYPE_FLOAT_INPUT;
            strcpy(actionInfo.actionName, "trigger");
            strcpy(actionInfo.localizedActionName, "Trigger");
            xr_check(xrCreateAction(vrState.actionSet, &actionInfo, &vrState.triggerAction), "Falha ao criar Action Trigger");

            // Action para a Pose da Mão
            actionInfo.actionType = XR_ACTION_TYPE_POSE_INPUT;
            strcpy(actionInfo.actionName, "hand_pose");
            strcpy(actionInfo.localizedActionName, "Hand Pose");
            xr_check(xrCreateAction(vrState.actionSet, &actionInfo, &vrState.handPoseAction), "Falha ao criar Action Hand Pose");

            // Sugerir bindings para Oculus Touch (Quest)
            XrPath oculusTouchProfilePath;
            xrStringToPath(vrState.instance, "/interaction_profiles/oculus/touch_controller", &oculusTouchProfilePath);
            
            XrPath movePath;
            xrStringToPath(vrState.instance, "/user/hand/left/input/thumbstick", &movePath);

            XrPath triggerPath;
            xrStringToPath(vrState.instance, "/user/hand/right/input/trigger/value", &triggerPath);

            XrPath handPosePath;
            xrStringToPath(vrState.instance, "/user/hand/right/input/grip/pose", &handPosePath);

            std::vector<XrActionSuggestedBinding> bindings;
            bindings.push_back({vrState.moveAction, movePath});
            bindings.push_back({vrState.triggerAction, triggerPath});
            bindings.push_back({vrState.handPoseAction, handPosePath});

            XrInteractionProfileSuggestedBinding suggestedBindings = {XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING};
            suggestedBindings.interactionProfile = oculusTouchProfilePath;
            suggestedBindings.suggestedBindings = bindings.data();
            suggestedBindings.countSuggestedBindings = (uint32_t)bindings.size();
            xr_check(xrSuggestInteractionProfileBindings(vrState.instance, &suggestedBindings), "Falha ao sugerir bindings");

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

            // Anexar ActionSet à sessão
            XrSessionActionSetsAttachInfo attachInfo = {XR_TYPE_SESSION_ACTION_SETS_ATTACH_INFO};
            attachInfo.countActionSets = 1;
            attachInfo.actionSets = &vrState.actionSet;
            xr_check(xrAttachSessionActionSets(vrState.session, &attachInfo), "Falha ao anexar ActionSets");

            // Criar Espaço para a Mão (baseado na Action)
            XrActionSpaceCreateInfo actionSpaceInfo = {XR_TYPE_ACTION_SPACE_CREATE_INFO};
            actionSpaceInfo.action = vrState.handPoseAction;
            actionSpaceInfo.poseInActionSpace.orientation = {0,0,0,1};
            actionSpaceInfo.poseInActionSpace.position = {0,0,0};
            xr_check(xrCreateActionSpace(vrState.session, &actionSpaceInfo, &vrState.handSpace), "Falha ao criar Hand Space");

            // --- INÍCIO PASSO 1: CRIAR E INICIAR O PASSTHROUGH ---
            if (passthrough_supported) {
                XrPassthroughCreateInfoFB passthroughCreateInfo = {XR_TYPE_PASSTHROUGH_CREATE_INFO_FB};
                // passthroughCreateInfo.flags pode ser usado para, por exemplo, solicitar um layer por cima de tudo
                xr_check(vrState.xrCreatePassthroughFB(vrState.session, &passthroughCreateInfo, &vrState.passthroughFeature),
                         "Falha ao criar o recurso de Passthrough (xrCreatePassthroughFB)");

                xr_check(vrState.xrPassthroughStartFB(vrState.passthroughFeature),
                         "Falha ao iniciar o Passthrough (xrPassthroughStartFB)");
                
                std::cout << "Recurso de Passthrough criado e iniciado." << std::endl;
            }
            // --- FIM PASSO 1 ---


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

            camera_changed = true;

          

            while (session_running && !glfwWindowShouldClose(window))
            {

                updateModel(); // Atualiza a matriz do modelo (rotação, etc.)
                handleKeyboardInput(window, true /* is_vr_mode */); // Passa o flag de VR
                
                // Atualiza input e lógica de instâncias
                // Precisamos de um tempo de display previsto aproximado para a lógica, ou usar o do frame anterior
                // Aqui usaremos o tempo atual apenas para lógica, o render usa o predictedDisplayTime correto
                XrTime logicTime = 0; // Será atualizado dentro do loop de frame se necessário, mas para input simples ok
                
                // Nota: handleControllerInput movido para dentro do xrBeginFrame/WaitFrame loop para ter o tempo correto

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

                // Processar Input com o tempo correto do frame
                handleControllerInput(vrState, referenceSpace, frameState.predictedDisplayTime);

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

                if (viewCountOutput >= 2) {
                    EyePose leftEye = {
                        {views[0].pose.position.x, views[0].pose.position.y, views[0].pose.position.z},
                        {views[0].pose.orientation.x, views[0].pose.orientation.y, views[0].pose.orientation.z, views[0].pose.orientation.w}
                    };
                    EyePose rightEye = {
                        {views[1].pose.position.x, views[1].pose.position.y, views[1].pose.position.z},
                        {views[1].pose.orientation.x, views[1].pose.orientation.y, views[1].pose.orientation.z, views[1].pose.orientation.w}
                    };
                    if (g_is_logging && g_logger)
                        g_logger->log(frameState.predictedDisplayTime, leftEye, rightEye);
                }

                // Atualizar o IAS na GPU antes de renderizar
                buildDynamicIAS(scene);

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
                        pose.orientation.x,
                        pose.orientation.y, 
                        pose.orientation.z 
                    );
                    g_headset_orientation = xr_orientation;

                    // --- INÍCIO DA CORREÇÃO DE ESCALA DE MOVIMENTO ---
                    const float3 current_headset_pos = make_float3(pose.position.x, pose.position.y, pose.position.z);


                    params.eye = nav_offset + current_headset_pos * nav_scale;
                    

                    const float w_len = 2.0f;
                    // Calculamos as dimensões do plano de imagem baseadas no FOV do OpenXR
                    const float tanAngleLeft = tanf(views[i].fov.angleLeft);
                    const float tanAngleRight = tanf(views[i].fov.angleRight);
                    const float tanAngleDown = tanf(views[i].fov.angleDown);
                    const float tanAngleUp = tanf(views[i].fov.angleUp);

                    const float v_len = w_len * (tanAngleUp - tanAngleDown) * 0.5f;
                    const float u_len = w_len * (tanAngleRight - tanAngleLeft) * 0.5f;
                    
                    // Correção para frustum assimétrico (drift rotacional) e centro óptico
                    const float center_x = w_len * (tanAngleRight + tanAngleLeft) * 0.5f;
                    const float center_y = w_len * (tanAngleUp + tanAngleDown) * 0.5f;

                    glm::vec3 u_vec = xr_orientation * glm::vec3(u_len, 0.0f, 0.0f);
                    glm::vec3 v_vec = xr_orientation * glm::vec3(0.0f, v_len, 0.0f);
                    glm::vec3 w_vec = xr_orientation * glm::vec3(center_x, center_y, -w_len);

                    params.U = make_float3(u_vec.x, u_vec.y, u_vec.z);
                    params.V = make_float3(v_vec.x, v_vec.y, v_vec.z);
                    params.W = make_float3(w_vec.x, w_vec.y, w_vec.z);
                    

                    params.subframe_index = 0;

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
                // Para Chroma Key com ALVR, o ambiente é opaco do ponto de vista da aplicação.
                // O ALVR no headset fará a mágica.
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
            if (g_logger) {
                delete g_logger;
                g_logger = nullptr;
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
        sutil::cleanupUI(window);
    }
    else // Modo Local (código original)
    {
        try
        {
            params.is_vr = false;
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
            g_scale = 1.0f; //glm::vec3(1.f, 1.f, 1.f);
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

            // Criar uma instância padrão para o modo Desktop
            if (g_instances.empty()) {
                OptixInstance default_instance = {};
                float transform[12] = {
                    1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0
                };

                // Preserva a transformação (escala/rotação) definida no arquivo do modelo, se houver.
                const float* m = nullptr;
                if (!scene.instances().empty()) m = scene.instances()[0]->transform.getData();

                if (m) {
                    // Converte de Column-Major (GL) para Row-Major (OptiX) e preserva a translação
                    transform[0] = m[0];  transform[1] = m[4];  transform[2] = m[8];  transform[3] = m[12];
                    transform[4] = m[1];  transform[5] = m[5];  transform[6] = m[9];  transform[7] = m[13];
                    transform[8] = m[2];  transform[9] = m[6];  transform[10]= m[10]; transform[11]= m[14];
                }

                memcpy(default_instance.transform, transform, sizeof(float) * 12);
                default_instance.instanceId = 0;
                default_instance.visibilityMask = 255;
                default_instance.traversableHandle = g_scene_gas;
                
                g_instances.push_back(default_instance);
                buildDynamicIAS(scene);
            }
            
            // Inicializar matrizes de objeto como identidade
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

    return 0;
}
