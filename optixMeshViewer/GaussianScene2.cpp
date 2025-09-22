#include "GaussianScene.hpp"
#include <sutil/Exception.h>  // for OPTIX_CHECK_LOG, CUDA_CHECK
#include <optix_stack_size.h>
#include <vector>

namespace sutil {

GaussianScene::GaussianScene(OptixDeviceContext context, CUstream stream)
: m_context(context), m_stream(stream)
{
    // Pipeline compile options
    m_pipeline_compile_options.usesMotionBlur        = false;
    m_pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    m_pipeline_compile_options.numPayloadValues      = 2;
    m_pipeline_compile_options.numAttributeValues    = 2;
    m_pipeline_compile_options.exceptionFlags        = OPTIX_EXCEPTION_FLAG_NONE;
    m_pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
    
    m_pipeline_link_options.maxTraceDepth = 1;
}

GaussianScene::~GaussianScene() { cleanup(); }

void GaussianScene::init(const std::vector<Pos>& pos, const std::vector<Pos>& hsize) {
    m_num_gaussians = static_cast<uint32_t>(pos.size());
    createModules();
    createProgramGroups();
    createPipeline();
    buildAccel(pos, hsize);
    buildSBT();
}

void GaussianScene::cleanup() {
    if (m_d_ias_output_buffer) { CUDA_CHECK(cudaFree((void*)m_d_ias_output_buffer)); m_d_ias_output_buffer = 0; }
    if (m_d_aabb_buffer)       { CUDA_CHECK(cudaFree((void*)m_d_aabb_buffer));       m_d_aabb_buffer       = 0; }
    if (m_pipeline)            OPTIX_CHECK(optixPipelineDestroy(m_pipeline));
    if (m_hitgroup_prog_group) OPTIX_CHECK(optixProgramGroupDestroy(m_hitgroup_prog_group));
    if (m_miss_prog_group)     OPTIX_CHECK(optixProgramGroupDestroy(m_miss_prog_group));
    if (m_raygen_prog_group)   OPTIX_CHECK(optixProgramGroupDestroy(m_raygen_prog_group));
    if (m_module)              OPTIX_CHECK(optixModuleDestroy(m_module));
}

void GaussianScene::createModules() {
    const char* ptxCode = R"ptx(
// <insert your PTX here>
)ptx";
    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_compile_options.optLevel          = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    module_compile_options.debugLevel        = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
    
    char LOG[2048]; size_t LOG_SIZE = sizeof(LOG);
    OPTIX_CHECK_LOG(optixModuleCreate(
        m_context,
        &module_compile_options,
        &m_pipeline_compile_options,
        ptxCode,
        strlen(ptxCode),
        LOG, &LOG_SIZE,
        &m_module
    ));
}

void GaussianScene::createProgramGroups() {
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc    pgDesc    = {};
    char LOG[2048]; size_t LOG_SIZE = sizeof(LOG);

    // Raygen
    pgDesc.kind                      = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgDesc.raygen.module            = m_module;
    pgDesc.raygen.entryFunctionName = "__raygen__rg";
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        m_context, &pgDesc, 1, &pgOptions, LOG, &LOG_SIZE, &m_raygen_prog_group
    ));

    // Miss
    pgDesc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pgDesc.miss.module            = m_module;
    pgDesc.miss.entryFunctionName = "__miss__ms";
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        m_context, &pgDesc, 1, &pgOptions, LOG, &LOG_SIZE, &m_miss_prog_group
    ));

    // Hitgroup
    pgDesc.kind                          = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pgDesc.hitgroup.moduleCH             = m_module;
    pgDesc.hitgroup.entryFunctionNameCH  = "__closesthit__ch";
    pgDesc.hitgroup.moduleAH             = nullptr;
    pgDesc.hitgroup.entryFunctionNameAH  = nullptr;
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        m_context, &pgDesc, 1, &pgOptions, LOG, &LOG_SIZE, &m_hitgroup_prog_group
    ));
}

void GaussianScene::createPipeline() {
    // Program groups
    OptixProgramGroup program_groups[] = {
        m_raygen_prog_group,
        m_miss_prog_group,
        m_hitgroup_prog_group
    };

    // Create pipeline
    char LOG[2048]; size_t LOG_SIZE = sizeof(LOG);
    OPTIX_CHECK_LOG(optixPipelineCreate(
        m_context,
        &m_pipeline_compile_options,
        &m_pipeline_link_options,
        program_groups,
        sizeof(program_groups)/sizeof(program_groups[0]),
        LOG, &LOG_SIZE,
        &m_pipeline
    ));

    // Compute and set stack sizes using raygen group
    OptixStackSizes stack_sizes = {};
    OPTIX_CHECK(optixUtilAccumulateStackSizes(
        m_raygen_prog_group,
        &stack_sizes,
        m_pipeline
    ));
    OPTIX_CHECK(optixPipelineSetStackSize(
        m_pipeline,
        stack_sizes.cssRG,
        stack_sizes.cssMS,
        stack_sizes.cssCH,
        stack_sizes.dssDC
    ));
}

void GaussianScene::buildAccel(const std::vector<Pos>& pos, const std::vector<Pos>& hsize) {
    std::vector<OptixAabb> aabbs(m_num_gaussians);
    for (uint32_t i = 0; i < m_num_gaussians; ++i) {
        const Pos& c = pos[i]; const Pos& h = hsize[i];
        aabbs[i].minX = c.x()-h.x(); aabbs[i].minX = c.x()-h.x(); aabbs[i].minY = c.y()-h.y(); aabbs[i].minZ = c.z()-h.z();
        aabbs[i].maxX = c.x()+h.x(); aabbs[i].maxY = c.y()+h.y(); aabbs[i].maxZ = c.z()+h.z();
    }
    size_t size = sizeof(OptixAabb)*m_num_gaussians;
    CUDA_CHECK(cudaMalloc((void**)&m_d_aabb_buffer, size));
    CUDA_CHECK(cudaMemcpy((void*)m_d_aabb_buffer, aabbs.data(), size, cudaMemcpyHostToDevice));

    OptixBuildInput build_input = {};
    build_input.type                         = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    build_input.customPrimitiveArray.aabbBuffers   = &m_d_aabb_buffer;
    build_input.customPrimitiveArray.numPrimitives = m_num_gaussians;
    build_input.customPrimitiveArray.strideInBytes = sizeof(OptixAabb);
    build_input.customPrimitiveArray.numSbtRecords  = m_num_gaussians;

    OptixAccelBuildOptions build_opts = {};
    build_opts.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    build_opts.operation  = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes buf_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(m_context, &build_opts, &build_input, 1, &buf_sizes));
    CUdeviceptr dtemp;
    CUDA_CHECK(cudaMalloc((void**)&dtemp, buf_sizes.tempSizeInBytes));
    CUDA_CHECK(cudaMalloc((void**)&m_d_ias_output_buffer, buf_sizes.outputSizeInBytes));
    OPTIX_CHECK(optixAccelBuild(
        m_context, m_stream,
        &build_opts,
        &build_input, 1,
        dtemp, buf_sizes.tempSizeInBytes,
        m_d_ias_output_buffer, buf_sizes.outputSizeInBytes,
        &m_ias_handle,
        nullptr,0
    ));
    CUDA_CHECK(cudaFree((void*)dtemp));
}

void GaussianScene::buildSBT() {
    CUdeviceptr dR;
    CUDA_CHECK(cudaMalloc((void**)&dR,sizeof(RayGenSbtRecord)));
    optixSbtRecordPackHeader(m_raygen_prog_group,(void*)dR);
    m_sbt.raygenRecord=dR;
    CUdeviceptr dM;
    CUDA_CHECK(cudaMalloc((void**)&dM,sizeof(MissSbtRecord)));
    optixSbtRecordPackHeader(m_miss_prog_group,(void*)dM);
    m_sbt.missRecordBase=dM;
    m_sbt.missRecordStrideInBytes=sizeof(MissSbtRecord);
    m_sbt.missRecordCount=1;
    size_t hitSz=sizeof(HitGroupSbtRecord);
    CUdeviceptr dH;
    CUDA_CHECK(cudaMalloc((void**)&dH,hitSz*m_num_gaussians));
    std::vector<HitGroupSbtRecord> hits(m_num_gaussians);
    for(uint32_t i=0;i<m_num_gaussians;++i){
        optixSbtRecordPackHeader(m_hitgroup_prog_group,&hits[i]);
        hits[i].gaussianID=i;
    }
    CUDA_CHECK(cudaMemcpy((void*)dH,hits.data(),hitSz*m_num_gaussians,cudaMemcpyHostToDevice));
    m_sbt.hitgroupRecordBase=dH;
    m_sbt.hitgroupRecordStrideInBytes=hitSz;
    m_sbt.hitgroupRecordCount=m_num_gaussians;
}

} // namespace sutil