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
    SUTILAPI void addGaussiansLow(const std::vector<Pos>& positions, 
                              const std::vector<Pos>& half_sizes);

    SUTILAPI void finalize();
    SUTILAPI void cleanup();
    
    // Getters
    SUTILAPI Camera camera() const;
    
    SUTILAPI OptixTraversableHandle traversableHandle1() const { return m_ias_handle1; }
    SUTILAPI OptixTraversableHandle traversableHandle2() const { return m_ias_handle2; }
    SUTILAPI OptixDeviceContext context() const { return m_context; }
    SUTILAPI sutil::Aabb                                    aabb1() const              { return m_scene_aabb1; }
    SUTILAPI CUdeviceptr getAABB_Buffer1() { return m_gaussian_group1->d_aabb_buffer; }
    SUTILAPI sutil::Aabb                                    aabb2() const              { return m_scene_aabb2; }
    SUTILAPI CUdeviceptr getAABB_Buffer2() { return m_gaussian_group2->d_aabb_buffer; }

private:
    void createContext();
    void buildGaussianAccels(std::shared_ptr<GaussianGroup> m_gaussian_group);
    void buildInstanceAccel(std::shared_ptr<GaussianGroup> m_gaussian_group, OptixTraversableHandle& m_ias_handle, CUdeviceptr m_d_ias_output_buffer);


    // Scene data
    std::vector<Camera> m_cameras;
    std::shared_ptr<GaussianGroup> m_gaussian_group1;
    sutil::Aabb m_scene_aabb1;

    std::shared_ptr<GaussianGroup> m_gaussian_group2;
    sutil::Aabb m_scene_aabb2;
    
    // OptiX structures
    OptixDeviceContext m_context = 0;
    OptixShaderBindingTable m_sbt = {};
    OptixPipeline m_pipeline = 0;
    OptixModule m_ptx_module = 0;
    OptixProgramGroup m_raygen_prog_group = 0;
    OptixProgramGroup m_miss_group = 0;
    OptixProgramGroup                    m_radiance_hit_group       = 0;
    OptixTraversableHandle m_ias_handle1 = 0;
    CUdeviceptr m_d_ias_output_buffer1 = 0;

    OptixTraversableHandle m_ias_handle2 = 0;
    CUdeviceptr m_d_ias_output_buffer2 = 0;
    
    // Pipeline options
    OptixPipelineCompileOptions m_pipeline_compile_options = {};
};

} // namespace sutil