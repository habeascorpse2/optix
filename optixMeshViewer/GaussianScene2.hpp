// GaussianScene.h
#pragma once

#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <vector>
#include "eigen/Eigen/Dense"

namespace sutil {

typedef Eigen::Vector3f Pos;

struct RayGenSbtRecord { char header[OPTIX_SBT_RECORD_HEADER_SIZE]; };
struct MissSbtRecord   { char header[OPTIX_SBT_RECORD_HEADER_SIZE]; };
struct HitGroupSbtRecord {
    char    header[OPTIX_SBT_RECORD_HEADER_SIZE];
    uint32_t gaussianID;
};

class GaussianScene {
public:
    GaussianScene(OptixDeviceContext context, CUstream stream = 0);
    ~GaussianScene();

    /**
     * Initialize the scene with 3D Gaussians. Builds GPU structures.
     * @param pos Vector of centers of Gaussians in world space
     * @param hsize Vector of half-sizes (extent) of each Gaussian's AABB
     */
    void init(const std::vector<Pos>& pos, const std::vector<Pos>& hsize);

    /**
     * Release all GPU and Optix resources
     */
    void cleanup();

    OptixTraversableHandle getIASHandle() const { return m_ias_handle; }
    CUdeviceptr           getIASOutputBuffer() const { return m_d_ias_output_buffer; }

private:
    void createModules();
    void createProgramGroups();
    void createPipeline();
    void buildAccel(const std::vector<Pos>& pos, const std::vector<Pos>& hsize);
    void buildSBT();

    OptixDeviceContext                   m_context               = nullptr;
    CUstream                             m_stream                = 0;

    // Pipeline options
    OptixPipelineCompileOptions          m_pipeline_compile_options = {};
    OptixPipelineLinkOptions             m_pipeline_link_options    = {};

    // Module
    OptixModule                          m_module                = nullptr;

    // Program groups
    OptixProgramGroup                    m_raygen_prog_group     = nullptr;
    OptixProgramGroup                    m_miss_prog_group       = nullptr;
    OptixProgramGroup                    m_hitgroup_prog_group   = nullptr;

    // Pipeline and SBT
    OptixPipeline                        m_pipeline              = nullptr;
    OptixShaderBindingTable              m_sbt                   = {};

    // Acceleration structures
    OptixTraversableHandle               m_ias_handle            = 0;
    CUdeviceptr                          m_d_ias_output_buffer   = 0;

    // AABB buffer
    CUdeviceptr                          m_d_aabb_buffer         = 0;
    uint32_t                             m_num_gaussians         = 0;
};

} // namespace sutil