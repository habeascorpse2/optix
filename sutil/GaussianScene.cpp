#include <optix.h>
#include <optix_stubs.h>

#include <cuda/whitted.h>
#include <sutil/CuBuffer.h>
#include <sutil/Exception.h>
#include <sutil/Matrix.h>
#include <sutil/Quaternion.h>
#include <sutil/Record.h>
#include <sutil/GaussianScene.h>
#include <sutil/sutil.h>

#include <iostream>

#ifndef OPTIX_AABB_BUFFER_BYTE_ALIGNMENT
    #define OPTIX_AABB_BUFFER_BYTE_ALIGNMENT 16u
#endif

#define LOG(message) std::cerr << message << std::endl

namespace sutil
{

// struct EmptyRecord 
// { 
//     alignas(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE]; 
// };


using namespace whitted;
void context_log_cb(unsigned int level, const char* tag, const char* message, void*)
{
    std::cerr << "[" << level << "][" << tag << "]: " << message << "\n";
}

GaussianScene::GaussianScene() 
{
    m_gaussian_group1 = std::make_shared<GaussianGroup>();
    m_gaussian_group2 = std::make_shared<GaussianGroup>();
}

GaussianScene::~GaussianScene()
{
    cleanup();
}

void GaussianScene::addCamera(const Camera& camera)
{
    m_cameras.push_back(camera);
}

void GaussianScene::addGaussians(
    const std::vector<Pos>& positions,
    const std::vector<Pos>& half_sizes)
{
    const size_t count = positions.size();
    m_gaussian_group1->gaussians.reserve(count);

    for(size_t i = 0; i < count; ++i)
    {
        Gaussian g;
        g.center = positions[i];
        g.half_size = half_sizes[i];// * 0.5f; // Ajuste para half_size
        g.id = i;
        m_gaussian_group1->gaussians.push_back(g);
        
        // Update scene AABB
        m_scene_aabb1.include(sutil::Aabb(
            make_float3(g.center.x() - g.half_size.x(),
                         g.center.y() - g.half_size.y(),
                         g.center.z() - g.half_size.z()),
            make_float3(g.center.x() + g.half_size.x(),
                         g.center.y() + g.half_size.y(),
                         g.center.z() + g.half_size.z())
        ));
    }
}

void GaussianScene::addGaussiansLow(
    const std::vector<Pos>& positions,
    const std::vector<Pos>& half_sizes)
{
    const size_t count = positions.size();
    m_gaussian_group2->gaussians.reserve(count);

    for(size_t i = 0; i < count; ++i)
    {
        Gaussian g;
        g.center = positions[i];
        g.half_size = half_sizes[i];// * 0.5f; // Ajuste para half_size
        g.id = i;
        m_gaussian_group2->gaussians.push_back(g);
        
        // Update scene AABB
        m_scene_aabb2.include(sutil::Aabb(
            make_float3(g.center.x() - g.half_size.x(),
                         g.center.y() - g.half_size.y(),
                         g.center.z() - g.half_size.z()),
            make_float3(g.center.x() + g.half_size.x(),
                         g.center.y() + g.half_size.y(),
                         g.center.z() + g.half_size.z())
        ));
    }
}

void GaussianScene::finalize()
{
    createContext();
    buildGaussianAccels(m_gaussian_group1);
    buildInstanceAccel(m_gaussian_group1, m_ias_handle1, m_d_ias_output_buffer1);
    buildGaussianAccels(m_gaussian_group2);
    buildInstanceAccel(m_gaussian_group2, m_ias_handle2, m_d_ias_output_buffer2);

}

void GaussianScene::cleanup()
{

    // if(m_sbt.raygenRecord) CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_sbt.raygenRecord)));
    // if(m_sbt.missRecordBase) CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_sbt.missRecordBase)));
    // if(m_sbt.hitgroupRecordBase) CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_sbt.hitgroupRecordBase)));

    if(m_pipeline)
        OPTIX_CHECK(optixPipelineDestroy(m_pipeline));
    
    if(m_raygen_prog_group)
        OPTIX_CHECK(optixProgramGroupDestroy(m_raygen_prog_group));
    
    if(m_miss_group)
        OPTIX_CHECK(optixProgramGroupDestroy(m_miss_group));
    
    if(m_radiance_hit_group)
        OPTIX_CHECK(optixProgramGroupDestroy(m_radiance_hit_group));
    
    if(m_context)
        OPTIX_CHECK(optixDeviceContextDestroy(m_context));
    
    if(m_gaussian_group1->d_aabb_buffer)
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_gaussian_group1->d_aabb_buffer)));
    
    if(m_gaussian_group1->d_gas_output)
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_gaussian_group1->d_gas_output)));

    if(m_gaussian_group2->d_aabb_buffer)
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_gaussian_group2->d_aabb_buffer)));
    
    if(m_gaussian_group2->d_gas_output)
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_gaussian_group2->d_gas_output)));
    
    if(m_d_ias_output_buffer1)
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_d_ias_output_buffer1)));
    if(m_d_ias_output_buffer2)
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_d_ias_output_buffer2)));
}

Camera GaussianScene::camera() const
{
    if(!m_cameras.empty())
        return m_cameras.front();
    
    Camera default_cam;
    // default_cam.setEye(m_scene_aabb.center());
    default_cam.setEye(make_float3(0));
    default_cam.setLookat(m_scene_aabb1.center());
    default_cam.setUp(make_float3(0, 1, 0));
    default_cam.setFovY(60.0f);
    return default_cam;
}

// ------ OptiX Setup Methods ------

void GaussianScene::createContext()
{
    // Initialize CUDA
    CUDA_CHECK( cudaFree( nullptr ) );

    CUcontext          cuCtx = nullptr;  // zero means take the current context
    OPTIX_CHECK( optixInit() );
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction       = &context_log_cb;
    options.logCallbackLevel          = 4;
#ifndef NDEBUG
    // This may incur significant performance cost and should only be done during development.
    options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#endif
    OPTIX_CHECK( optixDeviceContextCreate( cuCtx, &options, &m_context ) );
}

void GaussianScene::buildGaussianAccels(std::shared_ptr<GaussianGroup> m_gaussian_group)
{
    if (m_gaussian_group->gaussians.empty()) return;

    // 1) Agrupa em AABBs no host
    std::vector<OptixAabb> aabbs;
    aabbs.reserve(m_gaussian_group->gaussians.size());
    for (const auto& g : m_gaussian_group->gaussians)
    {
        OptixAabb aabb;
        aabb.minX = g.center.x() - g.half_size.x();
        aabb.minY = g.center.y() - g.half_size.y();
        aabb.minZ = g.center.z() - g.half_size.z();
        aabb.maxX = g.center.x() + g.half_size.x();
        aabb.maxY = g.center.y() + g.half_size.y();
        aabb.maxZ = g.center.z() + g.half_size.z();
        aabbs.push_back(aabb);
    }

    // 2) Aloca e copia para GPU
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&m_gaussian_group->d_aabb_buffer),
        aabbs.size() * sizeof(OptixAabb)
    ));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(m_gaussian_group->d_aabb_buffer),
        aabbs.data(),
        aabbs.size() * sizeof(OptixAabb),
        cudaMemcpyHostToDevice
    ));

    // 3) Flags para as primitivas (desabilita any-hit)
    std::vector<unsigned int> build_flags(
        m_gaussian_group->gaussians.size(),
        OPTIX_GEOMETRY_FLAG_NONE
    );

    // 4) Opções de build
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags    = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_options.operation     = OPTIX_BUILD_OPERATION_BUILD;

    // 5) Input para o custom primitives GAS
    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    build_input.customPrimitiveArray.aabbBuffers             = &m_gaussian_group->d_aabb_buffer;
    build_input.customPrimitiveArray.numPrimitives           = static_cast<uint32_t>(aabbs.size());
    build_input.customPrimitiveArray.strideInBytes           = sizeof(OptixAabb);
    build_input.customPrimitiveArray.numSbtRecords           = 1;
    build_input.customPrimitiveArray.flags                   = build_flags.data();

    // 6) Consulta de tamanho de buffers
    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        m_context,
        &accel_options,
        &build_input,
        1,
        &gas_buffer_sizes
    ));

    // 7) Aloca buffers temporário e de saída
    CUdeviceptr d_temp_buffer = 0;
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_temp_buffer),
        gas_buffer_sizes.tempSizeInBytes
    ));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&m_gaussian_group->d_gas_output),
        gas_buffer_sizes.outputSizeInBytes
    ));

    // 8) Emissão da propriedade de tamanho compactado
    OptixAccelEmitDesc emit_property = {};
    emit_property.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    CUdeviceptr compacted_size = 0;
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&compacted_size),
        sizeof(size_t)
    ));
    emit_property.result = compacted_size;

    // 9) Build do GAS
    OPTIX_CHECK(optixAccelBuild(
        m_context,
        0,  // CUDA stream
        &accel_options,
        &build_input,
        1,                             // num build inputs
        d_temp_buffer,
        gas_buffer_sizes.tempSizeInBytes,
        m_gaussian_group->d_gas_output,
        gas_buffer_sizes.outputSizeInBytes,
        &m_gaussian_group->gas_handle,
        &emit_property,
        1                              // num emit descs
    ));

    // 10) Cleanup
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(compacted_size)));
}

void GaussianScene::buildInstanceAccel(std::shared_ptr<GaussianGroup> m_gaussian_group, OptixTraversableHandle& m_ias_handle, CUdeviceptr m_d_ias_output_buffer)
{
    OptixInstance instance = {};
    instance.transform[0] = instance.transform[5] = instance.transform[10] = 1.0f;
    instance.visibilityMask = 1;
    instance.traversableHandle = m_gaussian_group->gas_handle;
    instance.sbtOffset = 0;

    CUdeviceptr d_instances;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_instances), sizeof(OptixInstance)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_instances), &instance,
                         sizeof(OptixInstance), cudaMemcpyHostToDevice));

    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    build_input.instanceArray.instances = d_instances;
    build_input.instanceArray.numInstances = 1;

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes ias_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(m_context, 
                                            &accel_options,
                                            &build_input,
                                            1,
                                            &ias_buffer_sizes));

    CUdeviceptr d_temp_buffer;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer),
               ias_buffer_sizes.tempSizeInBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_d_ias_output_buffer),
               ias_buffer_sizes.outputSizeInBytes));

    OPTIX_CHECK(optixAccelBuild(m_context, 
                               0,
                               &accel_options,
                               &build_input,
                               1,
                               d_temp_buffer,
                               ias_buffer_sizes.tempSizeInBytes,
                               m_d_ias_output_buffer,
                               ias_buffer_sizes.outputSizeInBytes,
                               &m_ias_handle,
                               nullptr,
                               0));

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_instances)));
}



} // namespace sutil