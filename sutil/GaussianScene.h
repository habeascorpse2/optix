#pragma once


#include <cuda/BufferView.h>
#include <cuda/MaterialData.h>
#include <cuda/whitted.h>
#include <sutil/Aabb.h>
#include <sutil/Camera.h>
#include <sutil/Matrix.h>
#include <sutil/Preprocessor.h>
#include <sutil/sutilapi.h>

#include <cuda_runtime.h>

#include <optix.h>

#include <memory>
#include <string>
#include <vector>
#include "../optixMeshViewer/GaussianInclude.hpp"

namespace sutil
{

class SUTILAPI GaussianScene
{
public:
    GaussianScene();
    ~GaussianScene();

    struct Gaussian
    {
        Pos center;
        Pos half_size;
        uint32_t id;
    };

    struct GaussianGroup
    {
        CUdeviceptr d_aabb_buffer = 0;
        std::vector<Gaussian> gaussians;
        OptixTraversableHandle gas_handle = 0;
        CUdeviceptr d_gas_output = 0;
    };

    SUTILAPI void addCamera(const Camera& camera);
    SUTILAPI void addGaussians(const std::vector<Pos>& positions, 
                              const std::vector<Pos>& half_sizes);
    SUTILAPI void finalize();
    SUTILAPI void cleanup();
    
    // Getters
    SUTILAPI Camera camera() const;
    SUTILAPI OptixPipeline pipeline() const { return m_pipeline; }
    SUTILAPI const OptixShaderBindingTable* sbt() const { return &m_sbt; }
    SUTILAPI OptixTraversableHandle traversableHandle() const { return m_ias_handle; }
    SUTILAPI OptixDeviceContext context() const { return m_context; }
    SUTILAPI sutil::Aabb                                    aabb() const              { return m_scene_aabb; }
    SUTILAPI CUdeviceptr getAABB_Buffer() { return m_gaussian_group->d_aabb_buffer; }

private:
    void createContext();
    void buildGaussianAccels();
    void buildInstanceAccel();
    void createPTXModule();
    void createProgramGroups();
    void createPipeline();
    void createSBT();

    // Scene data
    std::vector<Camera> m_cameras;
    std::shared_ptr<GaussianGroup> m_gaussian_group;
    sutil::Aabb m_scene_aabb;
    
    // OptiX structures
    OptixDeviceContext m_context = 0;
    OptixShaderBindingTable m_sbt = {};
    OptixPipeline m_pipeline = 0;
    OptixModule m_ptx_module = 0;
    OptixProgramGroup m_raygen_prog_group = 0;
    OptixProgramGroup m_miss_group = 0;
    OptixProgramGroup                    m_radiance_hit_group       = 0;
    OptixTraversableHandle m_ias_handle = 0;
    CUdeviceptr m_d_ias_output_buffer = 0;
    
    // Pipeline options
    OptixPipelineCompileOptions m_pipeline_compile_options = {};
};

} // namespace sutil