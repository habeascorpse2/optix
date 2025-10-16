# 1. Encontra as dependências externas.
find_package(OpenXR REQUIRED)
find_package(X11 REQUIRED)

# Força a localização do loader do OpenXR explicitamente
find_library(OPENXR_LOADER_LIBRARY openxr_loader REQUIRED)

# 2. Define o executável com a macro do SDK.
OPTIX_add_sample_executable( optixMeshViewer target_name
  optixMeshViewer.cpp
  OPTIONS -rdc true
)

# 3. Adiciona os diretórios de include necessários.
target_include_directories(optixMeshViewer PRIVATE
    ${OpenXR_INCLUDE_DIRS}
)

# 4. Adiciona as bibliotecas.
target_link_libraries(optixMeshViewer
    ${OPENXR_LOADER_LIBRARY}   # Linka explicitamente com libopenxr_loader
    ${X11_LIBRARIES}
    ${CUDA_LIBRARIES}
)

# 5. Adiciona as definições de compilação.
target_compile_definitions(optixMeshViewer PRIVATE
  _GLFW_X11
  XR_USE_PLATFORM_XLIB
)
