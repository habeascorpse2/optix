#pragma once

// Definições de plataforma para OpenXR e GLFW
#define _GLFW_X11
#define GLFW_EXPOSE_NATIVE_X11
#define GLFW_EXPOSE_NATIVE_GLX
#define XR_USE_PLATFORM_XLIB
#define XR_USE_GRAPHICS_API_OPENGL

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#include <X11/Xlib.h>
#include <GL/glx.h>

#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>
#include <openxr/openxr_reflection.h>

#include <optix.h>
#include <sutil/vec_math.h>
#include <sutil/Matrix.h>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <vector>
#include <string>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <chrono>
#include <cstring>
#include <cmath>

class OpenXRHandler {
public:
    struct SwapchainInfo {
        XrSwapchain swapchain;
        uint32_t width;
        uint32_t height;
        std::vector<XrSwapchainImageOpenGLKHR> images;
        std::vector<GLuint> fbos;
    };

    OpenXRHandler() = default;
    ~OpenXRHandler() { cleanup(); }

    void init(GLFWwindow* window) {
        std::vector<const char*> extensions = { XR_KHR_OPENGL_ENABLE_EXTENSION_NAME };

        // Verificar suporte a Passthrough
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

        // Criar Instância
        XrInstanceCreateInfo instanceCreateInfo = {XR_TYPE_INSTANCE_CREATE_INFO};
        strcpy(instanceCreateInfo.applicationInfo.applicationName, "OptiX Mesh Viewer");
        instanceCreateInfo.applicationInfo.applicationVersion = 1;
        strcpy(instanceCreateInfo.applicationInfo.engineName, "OptiX");
        instanceCreateInfo.applicationInfo.engineVersion = 1;
        instanceCreateInfo.applicationInfo.apiVersion = XR_CURRENT_API_VERSION;
        instanceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        instanceCreateInfo.enabledExtensionNames = extensions.data();

        check(xrCreateInstance(&instanceCreateInfo, &m_instance), "Falha ao criar instância OpenXR");

        // Carregar funções de extensão
        if (passthrough_supported) {
            loadExtensionFunction("xrCreatePassthroughFB", (PFN_xrVoidFunction*)&xrCreatePassthroughFB);
            loadExtensionFunction("xrDestroyPassthroughFB", (PFN_xrVoidFunction*)&xrDestroyPassthroughFB);
            loadExtensionFunction("xrPassthroughStartFB", (PFN_xrVoidFunction*)&xrPassthroughStartFB);
            loadExtensionFunction("xrPassthroughPauseFB", (PFN_xrVoidFunction*)&xrPassthroughPauseFB);
            loadExtensionFunction("xrCreatePassthroughLayerFB", (PFN_xrVoidFunction*)&xrCreatePassthroughLayerFB);
            loadExtensionFunction("xrDestroyPassthroughLayerFB", (PFN_xrVoidFunction*)&xrDestroyPassthroughLayerFB);
        }

        // Obter SystemId
        XrSystemGetInfo systemGetInfo = {XR_TYPE_SYSTEM_GET_INFO};
        systemGetInfo.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
        check(xrGetSystem(m_instance, &systemGetInfo, &m_systemId), "Falha ao obter sistema OpenXR");

        // Verificar requisitos OpenGL
        PFN_xrGetOpenGLGraphicsRequirementsKHR xrGetOpenGLGraphicsRequirementsKHR = nullptr;
        loadExtensionFunction("xrGetOpenGLGraphicsRequirementsKHR", (PFN_xrVoidFunction*)&xrGetOpenGLGraphicsRequirementsKHR);
        XrGraphicsRequirementsOpenGLKHR glRequirements = { XR_TYPE_GRAPHICS_REQUIREMENTS_OPENGL_KHR };
        check(xrGetOpenGLGraphicsRequirementsKHR(m_instance, m_systemId, &glRequirements), "Falha ao obter requisitos OpenGL");

        // Criar Sessão
        XrGraphicsBindingOpenGLXlibKHR graphicsBinding = {XR_TYPE_GRAPHICS_BINDING_OPENGL_XLIB_KHR};
        graphicsBinding.xDisplay = glfwGetX11Display();
        graphicsBinding.glxContext = glfwGetGLXContext(window);
        graphicsBinding.glxDrawable = glfwGetGLXWindow(window);

        XrSessionCreateInfo sessionCreateInfo = {XR_TYPE_SESSION_CREATE_INFO};
        sessionCreateInfo.next = &graphicsBinding;
        sessionCreateInfo.systemId = m_systemId;
        check(xrCreateSession(m_instance, &sessionCreateInfo, &m_session), "Falha ao criar sessão OpenXR");

        // Configurar Input
        createActions();

        // Criar Espaços
        XrReferenceSpaceCreateInfo spaceCreateInfo = {XR_TYPE_REFERENCE_SPACE_CREATE_INFO};
        spaceCreateInfo.poseInReferenceSpace = {{0,0,0,1}, {0,0,0}};
        spaceCreateInfo.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_LOCAL;
        check(xrCreateReferenceSpace(m_session, &spaceCreateInfo, &m_referenceSpace), "Falha ao criar ReferenceSpace");

        // Configurar Passthrough
        if (passthrough_supported) {
            XrPassthroughCreateInfoFB passthroughCreateInfo = {XR_TYPE_PASSTHROUGH_CREATE_INFO_FB};
            check(xrCreatePassthroughFB(m_session, &passthroughCreateInfo, &m_passthroughFeature), "Falha ao criar Passthrough");
            check(xrPassthroughStartFB(m_passthroughFeature), "Falha ao iniciar Passthrough");
        }

        createSwapchains();
    }

    void cleanup() {
        if (m_session) xrDestroySession(m_session);
        if (m_instance) xrDestroyInstance(m_instance);
        m_session = XR_NULL_HANDLE;
        m_instance = XR_NULL_HANDLE;
    }

    void pollEvents(bool& exitLoop, bool& sessionRunning) {
        XrEventDataBuffer eventBuffer = { XR_TYPE_EVENT_DATA_BUFFER };
        while (xrPollEvent(m_instance, &eventBuffer) == XR_SUCCESS) {
            switch (eventBuffer.type) {
                case XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED: {
                    auto* stateEvent = reinterpret_cast<XrEventDataSessionStateChanged*>(&eventBuffer);
                    if (stateEvent->state == XR_SESSION_STATE_READY) {
                        XrSessionBeginInfo beginInfo = { XR_TYPE_SESSION_BEGIN_INFO };
                        beginInfo.primaryViewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
                        xrBeginSession(m_session, &beginInfo);
                        sessionRunning = true;
                    } else if (stateEvent->state == XR_SESSION_STATE_STOPPING) {
                        xrEndSession(m_session);
                        sessionRunning = false;
                    } else if (stateEvent->state == XR_SESSION_STATE_EXITING) {
                        exitLoop = true;
                    }
                    break;
                }
                case XR_TYPE_EVENT_DATA_INSTANCE_LOSS_PENDING:
                    exitLoop = true;
                    break;
            }
            eventBuffer = { XR_TYPE_EVENT_DATA_BUFFER };
        }
    }

    bool beginFrame(XrTime& displayTime) {
        XrFrameWaitInfo waitInfo = {XR_TYPE_FRAME_WAIT_INFO};
        if (XR_FAILED(xrWaitFrame(m_session, &waitInfo, &m_frameState))) return false;

        XrFrameBeginInfo beginInfo = {XR_TYPE_FRAME_BEGIN_INFO};
        if (XR_FAILED(xrBeginFrame(m_session, &beginInfo))) return false;

        displayTime = m_frameState.predictedDisplayTime;
        return true;
    }

    void endFrame(const std::vector<XrCompositionLayerProjectionView>& projectionViews) {
        XrCompositionLayerProjection layer = { XR_TYPE_COMPOSITION_LAYER_PROJECTION };
        layer.space = m_referenceSpace;
        layer.viewCount = static_cast<uint32_t>(projectionViews.size());
        layer.views = projectionViews.data();

        XrFrameEndInfo frameEndInfo = { XR_TYPE_FRAME_END_INFO };
        frameEndInfo.displayTime = m_frameState.predictedDisplayTime;
        frameEndInfo.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
        frameEndInfo.layerCount = 1;
        const XrCompositionLayerBaseHeader* layers[] = { reinterpret_cast<const XrCompositionLayerBaseHeader*>(&layer) };
        frameEndInfo.layers = layers;

        xrEndFrame(m_session, &frameEndInfo);
    }

    void updateInput(XrTime displayTime, const glm::quat& headset_orientation, float& nav_scale, float3& nav_offset, 
                     std::vector<OptixInstance>& instances, OptixTraversableHandle gas_handle, 
                     int& held_instance_idx, bool& trigger_was_pressed) {
        
        const XrActiveActionSet activeActionSet = {m_actionSet, XR_NULL_PATH};
        XrActionsSyncInfo syncInfo = {XR_TYPE_ACTIONS_SYNC_INFO};
        syncInfo.countActiveActionSets = 1;
        syncInfo.activeActionSets = &activeActionSet;
        xrSyncActions(m_session, &syncInfo);

        // Movimento
        XrActionStateGetInfo getInfo = {XR_TYPE_ACTION_STATE_GET_INFO};
        getInfo.action = m_moveAction;
        XrActionStateVector2f moveState = {XR_TYPE_ACTION_STATE_VECTOR2F};
        xrGetActionStateVector2f(m_session, &getInfo, &moveState);

        if (moveState.isActive && (fabsf(moveState.currentState.x) > 0.1f || fabsf(moveState.currentState.y) > 0.1f)) {
            const float move_speed = 0.05f;
            glm::vec3 forward = headset_orientation * glm::vec3(0.0f, 0.0f, -1.0f);
            forward.y = 0.0f;
            if (glm::length(forward) > 0.01f) forward = glm::normalize(forward);

            glm::vec3 right = headset_orientation * glm::vec3(1.0f, 0.0f, 0.0f);
            right.y = 0.0f;
            if (glm::length(right) > 0.01f) right = glm::normalize(right);

            float3 move_vec = make_float3(
                forward.x * moveState.currentState.y + right.x * moveState.currentState.x,
                forward.y * moveState.currentState.y + right.y * moveState.currentState.x,
                forward.z * moveState.currentState.y + right.z * moveState.currentState.x
            );
            nav_offset += move_speed * move_vec;
        }

        // Gatilho e Spawn
        getInfo.action = m_triggerAction;
        XrActionStateFloat triggerState = {XR_TYPE_ACTION_STATE_FLOAT};
        if (XR_SUCCEEDED(xrGetActionStateFloat(m_session, &getInfo, &triggerState))) {
            bool is_pressed = triggerState.currentState > 0.5f;

            XrSpaceLocation handLocation = {XR_TYPE_SPACE_LOCATION};
            xrLocateSpace(m_handSpace, m_referenceSpace, displayTime, &handLocation);

            if (handLocation.locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT) {
                glm::vec3 hand_pos = glm::vec3(handLocation.pose.position.x, handLocation.pose.position.y, handLocation.pose.position.z);
                glm::quat hand_rot = glm::quat(handLocation.pose.orientation.w, handLocation.pose.orientation.x, handLocation.pose.orientation.y, handLocation.pose.orientation.z);
                
                glm::vec3 virtual_hand_pos = glm::vec3(nav_offset.x, nav_offset.y, nav_offset.z) + (hand_pos * nav_scale);
                float gltf_scale = .1f;

                glm::mat4 mat = glm::translate(glm::mat4(1.0f), virtual_hand_pos) * glm::mat4_cast(hand_rot) * glm::scale(glm::mat4(1.0f), glm::vec3(gltf_scale));

                float transform[12];
                const float* m = (const float*)&mat;
                transform[0] = m[0*4+0]; transform[1] = m[1*4+0]; transform[2] = m[2*4+0]; transform[3] = m[3*4+0];
                transform[4] = m[0*4+1]; transform[5] = m[1*4+1]; transform[6] = m[2*4+1]; transform[7] = m[3*4+1];
                transform[8] = m[0*4+2]; transform[9] = m[1*4+2]; transform[10] = m[2*4+2]; transform[11] = m[3*4+2];

                if (is_pressed && !trigger_was_pressed) {
                    OptixInstance new_instance = {};
                    memcpy(new_instance.transform, transform, sizeof(float)*12);
                    new_instance.instanceId = static_cast<unsigned int>(instances.size());
                    new_instance.visibilityMask = 255;
                    new_instance.traversableHandle = gas_handle;
                    instances.push_back(new_instance);
                    held_instance_idx = static_cast<int>(instances.size()) - 1;
                } else if (is_pressed && held_instance_idx != -1 && held_instance_idx < instances.size()) {
                    memcpy(instances[held_instance_idx].transform, transform, sizeof(float)*12);
                } else if (!is_pressed && trigger_was_pressed) {
                    held_instance_idx = -1;
                }
            }
            trigger_was_pressed = is_pressed;
        }
    }

    void locateViews(XrTime displayTime, std::vector<XrView>& views) {
        uint32_t viewCountOutput = 0;
        XrViewLocateInfo viewLocateInfo = {XR_TYPE_VIEW_LOCATE_INFO};
        viewLocateInfo.viewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
        viewLocateInfo.displayTime = displayTime;
        viewLocateInfo.space = m_referenceSpace;
        XrViewState viewState = {XR_TYPE_VIEW_STATE};
        xrLocateViews(m_session, &viewLocateInfo, &viewState, (uint32_t)views.size(), &viewCountOutput, views.data());
    }

    uint32_t acquireImage(uint32_t viewIndex) {
        uint32_t imageIndex;
        XrSwapchainImageAcquireInfo acquireInfo = {XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
        xrAcquireSwapchainImage(m_swapchains[viewIndex].swapchain, &acquireInfo, &imageIndex);
        XrSwapchainImageWaitInfo waitInfo = {XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO};
        waitInfo.timeout = XR_INFINITE_DURATION;
        xrWaitSwapchainImage(m_swapchains[viewIndex].swapchain, &waitInfo);
        return imageIndex;
    }

    void releaseImage(uint32_t viewIndex) {
        XrSwapchainImageReleaseInfo releaseInfo = {XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
        xrReleaseSwapchainImage(m_swapchains[viewIndex].swapchain, &releaseInfo);
    }

    const SwapchainInfo& getSwapchainInfo(uint32_t i) const { return m_swapchains[i]; }
    uint32_t getViewCount() const { return static_cast<uint32_t>(m_swapchains.size()); }

    // static sutil::Matrix4x4 createProjectionMatrix(const XrFovf& fov, float nearZ, float farZ) {
    //     const float tanL = tanf(fov.angleLeft), tanR = tanf(fov.angleRight);
    //     const float tanD = tanf(fov.angleDown), tanU = tanf(fov.angleUp);
    //     const float w = tanR - tanL, h = tanU - tanD;
    //     sutil::Matrix4x4 mat;
    //     mat.setRow(0, {2.0f / w, 0.0f, (tanR + tanL) / w, 0.0f});
    //     mat.setRow(1, {0.0f, 2.0f / h, (tanU + tanD) / h, 0.0f});
    //     mat.setRow(2, {0.0f, 0.0f, -(farZ + nearZ) / (farZ - nearZ), -2.0f * farZ * nearZ / (farZ - nearZ)});
    //     mat.setRow(3, {0.0f, 0.0f, -1.0f, 0.0f});
    //     return mat;
    // }

private:
    XrInstance m_instance = XR_NULL_HANDLE;
    XrSystemId m_systemId = XR_NULL_SYSTEM_ID;
    XrSession m_session = XR_NULL_HANDLE;
    XrSpace m_referenceSpace = XR_NULL_HANDLE;
    
    PFN_xrCreatePassthroughFB xrCreatePassthroughFB = nullptr;
    PFN_xrDestroyPassthroughFB xrDestroyPassthroughFB = nullptr;
    PFN_xrPassthroughStartFB xrPassthroughStartFB = nullptr;
    PFN_xrPassthroughPauseFB xrPassthroughPauseFB = nullptr;
    PFN_xrCreatePassthroughLayerFB xrCreatePassthroughLayerFB = nullptr;
    PFN_xrDestroyPassthroughLayerFB xrDestroyPassthroughLayerFB = nullptr;
    XrPassthroughFB m_passthroughFeature = XR_NULL_HANDLE;

    XrActionSet m_actionSet = XR_NULL_HANDLE;
    XrAction m_moveAction = XR_NULL_HANDLE;
    XrAction m_triggerAction = XR_NULL_HANDLE;
    XrAction m_handPoseAction = XR_NULL_HANDLE;
    XrSpace m_handSpace = XR_NULL_HANDLE;

    std::vector<SwapchainInfo> m_swapchains;
    XrFrameState m_frameState = {XR_TYPE_FRAME_STATE};

    void check(XrResult result, const std::string& msg) {
        if (XR_FAILED(result)) {
            char resStr[XR_MAX_RESULT_STRING_SIZE];
            xrResultToString(m_instance, result, resStr);
            throw std::runtime_error(msg + ": " + resStr);
        }
    }

    void loadExtensionFunction(const char* name, PFN_xrVoidFunction* function) {
        xrGetInstanceProcAddr(m_instance, name, function);
    }

    void createActions() {
        XrActionSetCreateInfo actionSetInfo = {XR_TYPE_ACTION_SET_CREATE_INFO};
        strcpy(actionSetInfo.actionSetName, "gameplay");
        strcpy(actionSetInfo.localizedActionSetName, "Gameplay");
        check(xrCreateActionSet(m_instance, &actionSetInfo, &m_actionSet), "Falha ActionSet");

        XrActionCreateInfo actionInfo = {XR_TYPE_ACTION_CREATE_INFO};
        actionInfo.actionType = XR_ACTION_TYPE_VECTOR2F_INPUT;
        strcpy(actionInfo.actionName, "move");
        strcpy(actionInfo.localizedActionName, "Move");
        check(xrCreateAction(m_actionSet, &actionInfo, &m_moveAction), "Falha Action Move");

        actionInfo.actionType = XR_ACTION_TYPE_FLOAT_INPUT;
        strcpy(actionInfo.actionName, "trigger");
        strcpy(actionInfo.localizedActionName, "Trigger");
        check(xrCreateAction(m_actionSet, &actionInfo, &m_triggerAction), "Falha Action Trigger");

        actionInfo.actionType = XR_ACTION_TYPE_POSE_INPUT;
        strcpy(actionInfo.actionName, "hand_pose");
        strcpy(actionInfo.localizedActionName, "Hand Pose");
        check(xrCreateAction(m_actionSet, &actionInfo, &m_handPoseAction), "Falha Action HandPose");

        XrPath profilePath, movePath, triggerPath, handPath;
        xrStringToPath(m_instance, "/interaction_profiles/oculus/touch_controller", &profilePath);
        xrStringToPath(m_instance, "/user/hand/left/input/thumbstick", &movePath);
        xrStringToPath(m_instance, "/user/hand/right/input/trigger/value", &triggerPath);
        xrStringToPath(m_instance, "/user/hand/right/input/grip/pose", &handPath);

        std::vector<XrActionSuggestedBinding> bindings = {
            {m_moveAction, movePath}, {m_triggerAction, triggerPath}, {m_handPoseAction, handPath}
        };
        XrInteractionProfileSuggestedBinding suggestedBindings = {XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING};
        suggestedBindings.interactionProfile = profilePath;
        suggestedBindings.suggestedBindings = bindings.data();
        suggestedBindings.countSuggestedBindings = (uint32_t)bindings.size();
        check(xrSuggestInteractionProfileBindings(m_instance, &suggestedBindings), "Falha Bindings");

        XrSessionActionSetsAttachInfo attachInfo = {XR_TYPE_SESSION_ACTION_SETS_ATTACH_INFO};
        attachInfo.countActionSets = 1;
        attachInfo.actionSets = &m_actionSet;
        check(xrAttachSessionActionSets(m_session, &attachInfo), "Falha Attach ActionSet");

        XrActionSpaceCreateInfo spaceInfo = {XR_TYPE_ACTION_SPACE_CREATE_INFO};
        spaceInfo.action = m_handPoseAction;
        spaceInfo.poseInActionSpace = {{0,0,0,1}, {0,0,0}};
        check(xrCreateActionSpace(m_session, &spaceInfo, &m_handSpace), "Falha HandSpace");
    }

    void createSwapchains() {
        uint32_t viewCount = 0;
        xrEnumerateViewConfigurationViews(m_instance, m_systemId, XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, 0, &viewCount, nullptr);
        std::vector<XrViewConfigurationView> configViews(viewCount, {XR_TYPE_VIEW_CONFIGURATION_VIEW});
        xrEnumerateViewConfigurationViews(m_instance, m_systemId, XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, viewCount, &viewCount, configViews.data());

        m_swapchains.resize(viewCount);
        for (uint32_t i = 0; i < viewCount; ++i) {
            XrSwapchainCreateInfo info = {XR_TYPE_SWAPCHAIN_CREATE_INFO};
            info.format = GL_SRGB8_ALPHA8;
            info.sampleCount = configViews[i].recommendedSwapchainSampleCount;
            info.width = configViews[i].recommendedImageRectWidth;
            info.height = configViews[i].recommendedImageRectHeight;
            info.faceCount = 1; info.arraySize = 1; info.mipCount = 1;
            info.usageFlags = XR_SWAPCHAIN_USAGE_SAMPLED_BIT | XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT;
            
            check(xrCreateSwapchain(m_session, &info, &m_swapchains[i].swapchain), "Falha Swapchain");
            m_swapchains[i].width = info.width;
            m_swapchains[i].height = info.height;

            uint32_t imgCount = 0;
            xrEnumerateSwapchainImages(m_swapchains[i].swapchain, 0, &imgCount, nullptr);
            m_swapchains[i].images.resize(imgCount, {XR_TYPE_SWAPCHAIN_IMAGE_OPENGL_KHR});
            xrEnumerateSwapchainImages(m_swapchains[i].swapchain, imgCount, &imgCount, (XrSwapchainImageBaseHeader*)m_swapchains[i].images.data());
        }
    }
};