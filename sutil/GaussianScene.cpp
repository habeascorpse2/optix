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
    m_gaussian_group = std::make_shared<GaussianGroup>();
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
    m_gaussian_group->gaussians.reserve(count);

    for(size_t i = 0; i < count; ++i)
    {
        Gaussian g;
        g.center = positions[i];
        g.half_size = half_sizes[i];// * 0.5f; // Ajuste para half_size
        g.id = i;
        m_gaussian_group->gaussians.push_back(g);
        
        // Update scene AABB
        m_scene_aabb.include(sutil::Aabb(
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
    buildGaussianAccels();
    buildInstanceAccel();
    // createPTXModule();
    // std::cout << "PTX Modules created..." << std::endl;
    // createProgramGroups();
    // std::cout << "Program Groups Created..." << std::endl;
    // createPipeline();
    // std::cout << "Pipeline Created..." << std::endl;
    // createSBT();
    // std::cout << "SBT Created..." << std::endl;
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
    
    if(m_gaussian_group->d_aabb_buffer)
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_gaussian_group->d_aabb_buffer)));
    
    if(m_gaussian_group->d_gas_output)
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_gaussian_group->d_gas_output)));
    
    if(m_d_ias_output_buffer)
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_d_ias_output_buffer)));
}

Camera GaussianScene::camera() const
{
    if(!m_cameras.empty())
        return m_cameras.front();
    
    Camera default_cam;
    // default_cam.setEye(m_scene_aabb.center());
    default_cam.setEye(make_float3(0));
    default_cam.setLookat(m_scene_aabb.center());
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

void GaussianScene::buildGaussianAccels()
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

void GaussianScene::buildInstanceAccel()
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

void GaussianScene::createPTXModule()
{
    OptixModuleCompileOptions module_compile_options = {};
#if !defined( NDEBUG )
    module_compile_options.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif

    m_pipeline_compile_options = {};
    m_pipeline_compile_options.usesMotionBlur            = false;
    m_pipeline_compile_options.traversableGraphFlags     = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    m_pipeline_compile_options.numPayloadValues          = whitted::NUM_PAYLOAD_VALUES;
    m_pipeline_compile_options.numAttributeValues        = 2; // TODO
    m_pipeline_compile_options.exceptionFlags            = OPTIX_EXCEPTION_FLAG_NONE; // should be OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
    m_pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

    size_t      inputSize = 0;
    const char* input     = sutil::getInputData( nullptr, nullptr, "whitted.cu", inputSize );

    m_ptx_module  = {};
    OPTIX_CHECK_LOG( optixModuleCreate(
                m_context,
                &module_compile_options,
                &m_pipeline_compile_options,
                input,
                inputSize,
                LOG, &LOG_SIZE,
                &m_ptx_module
                ) );
}

void GaussianScene::createProgramGroups()
{
    OptixProgramGroupOptions program_group_options = {};

    //
    // Ray generation
    //
    {

        OptixProgramGroupDesc raygen_prog_group_desc = {};
        raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_prog_group_desc.raygen.module            = m_ptx_module;
        raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__pinhole";

        OPTIX_CHECK_LOG( optixProgramGroupCreate(
                    m_context,
                    &raygen_prog_group_desc,
                    1,                             // num program groups
                    &program_group_options,
                    LOG, &LOG_SIZE,
                    &m_raygen_prog_group
                    )
                );
    }

    //
    // Miss
    //
    {
        OptixProgramGroupDesc miss_prog_group_desc = {};
        miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module            = m_ptx_module;
        miss_prog_group_desc.miss.entryFunctionName = "__miss__radiance";
        OPTIX_CHECK_LOG( optixProgramGroupCreate(
                    m_context,
                    &miss_prog_group_desc,
                    1,                             // num program groups
                    &program_group_options,
                    LOG, &LOG_SIZE,
                    &m_miss_group
                    )
                );

    }

    //
    // Hit group
    //
    {
        OptixProgramGroupDesc hit_prog_group_desc = {};
        hit_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hit_prog_group_desc.hitgroup.moduleCH            = m_ptx_module;
        hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
        hit_prog_group_desc.hitgroup.moduleAH            = m_ptx_module;
        hit_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";
        hit_prog_group_desc.hitgroup.moduleIS            = m_ptx_module;
        hit_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__";
        OPTIX_CHECK_LOG( optixProgramGroupCreate(
                         m_context,
                         &hit_prog_group_desc,
                         1,                             // num program groups
                         &program_group_options,
                         LOG, &LOG_SIZE,
                         &m_radiance_hit_group
                         )
                );
    }
}

void GaussianScene::createPipeline()
{
    OptixProgramGroup groups[] = {m_raygen_prog_group, m_miss_group, m_radiance_hit_group};
    
    OptixPipelineLinkOptions link_options = {};
    link_options.maxTraceDepth = 2;  // Aumentado para suportar primitivas custom
    // link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;  // Ativar debug
    
    char log[4096];
    size_t sizeof_log = sizeof(log);
    
    OPTIX_CHECK(optixPipelineCreate(
        m_context,
        &m_pipeline_compile_options,
        &link_options,
        groups,
        sizeof(groups)/sizeof(groups[0]),
        log,
        &sizeof_log,
        &m_pipeline
    ));

                                   
}

void GaussianScene::createSBT()
{
    // 1. Raygen Record
    {
        const size_t raygen_record_size = sizeof( EmptyRecord );
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &m_sbt.raygenRecord ), raygen_record_size ) );

        EmptyRecord rg_sbt;
        OPTIX_CHECK( optixSbtRecordPackHeader( m_raygen_prog_group, &rg_sbt ) );
        CUDA_CHECK( cudaMemcpy(
                    reinterpret_cast<void*>( m_sbt.raygenRecord ),
                    &rg_sbt,
                    raygen_record_size,
                    cudaMemcpyHostToDevice
                    ) );
    }

    // 2. Miss Records
    {
        EmptyRecord miss_record;
        OPTIX_CHECK(optixSbtRecordPackHeader(m_miss_group, &miss_record));

        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&m_sbt.missRecordBase),
            sizeof(EmptyRecord) * whitted::RAY_TYPE_COUNT
        ));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(m_sbt.missRecordBase),
            &miss_record,
            sizeof(EmptyRecord),
            cudaMemcpyHostToDevice
        ));
        m_sbt.missRecordStrideInBytes = sizeof(EmptyRecord);
        m_sbt.missRecordCount = whitted::RAY_TYPE_COUNT;
    }

    // 3. Hitgroup Records (Crítico para Gaussianas)
    {
        std::vector<HitRecord> hit_records;
        hit_records.reserve(m_gaussian_group->gaussians.size());

        for(const auto& gaussian : m_gaussian_group->gaussians)
        {
            HitRecord rec;
            OPTIX_CHECK(optixSbtRecordPackHeader(m_radiance_hit_group, &rec));
            rec.gaussian_id = gaussian.id;  // Atribuição direta
            hit_records.push_back(rec);
        }

        // Alocar e copiar para GPU
        const size_t hit_record_size = sizeof(HitRecord) * hit_records.size();
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&m_sbt.hitgroupRecordBase),
            hit_record_size
        ));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(m_sbt.hitgroupRecordBase),
            hit_records.data(),
            hit_record_size,
            cudaMemcpyHostToDevice
        ));
        
        m_sbt.hitgroupRecordStrideInBytes = sizeof(HitRecord);
        m_sbt.hitgroupRecordCount = static_cast<uint32_t>(hit_records.size());
    }

    // 4. Callable Records (Vazio se não usado)
    // m_sbt.callablesRecordBase = 0;
    // m_sbt.callablesRecordCount = 0;
    // m_sbt.callablesRecordStrideInBytes = 0;
}

} // namespace sutil