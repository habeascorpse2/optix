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
#pragma once

#include <vector_types.h>

#include <cuda/BufferView.h>
#include <cuda/GeometryData.h>
#include <cuda/Light.h>
#include <cuda/MaterialData.h>
#include <sutil/Matrix.h>
#include <sutil/vec_math.h>
#include <glm/glm.hpp>


// Funções auxiliares para calcular o mínimo e o máximo entre dois float3
__host__ __device__ glm::vec3 min(const float3& a, const float3& b) {
    return glm::vec3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

__host__ __device__ glm::vec3 max(const float3& a, const float3& b) {
    return glm::vec3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

struct Cube {
    glm::vec3 center;
    glm::vec3 half_size;
    int index;
    // glm::vec3 vertices[8];  // O cubo tem 8 vértices no total

    __host__ __device__
    Cube() {
        index = -1;
    }

    // Construtor: cria um cubo com os vértices fornecidos ou default
    __host__ __device__
    Cube(glm::vec3 ccenter, glm::vec3 half_size, int index): center(ccenter), half_size(half_size), index(index) {
    }



};

    __host__ __device__
    Cube applyCenterTransformation(const Cube& scube, const glm::mat4& matrix) {
        Cube cube;
        cube.center = glm::vec3(matrix * glm::vec4(scube.center, 1.0f));
        return cube;
    }


struct OctreeNodeD {
    int *cubes_0;
    int *cubes_1;
    int children[8]; // Índices dos filhos
    Cube boundary;
    // bool is_leaf;
    int numCubes_0;
    int numCubes_1;
    int branchCubes;

    __host__ __device__
    OctreeNodeD() : numCubes_0(0),numCubes_1(0), branchCubes(0) {
        // is_leaf = false;
        for (int i = 0; i < 8; ++i) {
            children[i] = -1;
        }
    }
    
};

namespace whitted
{

const unsigned int NUM_ATTRIBUTE_VALUES = 4u;
const unsigned int NUM_PAYLOAD_VALUES   = 6u;
const unsigned int MAX_TRACE_DEPTH      = 2u;

const unsigned int GSM_MAX_SIZE = 120;
// const unsigned int gaussian_block = 200;
const unsigned int WIDTH = 800;
const unsigned int HEIGHT = 600;
#define BLOCK_X  16
#define BLOCK_Y  16

struct HitGroupData
{
    GeometryData geometry_data;
    MaterialData material_data;
};


enum RayType
{
    RAY_TYPE_RADIANCE  = 0,
    RAY_TYPE_OCCLUSION = 1,
    RAY_TYPE_COUNT = 2
};


struct LaunchParams
{
    unsigned int             width;
    unsigned int             height;
    unsigned int             subframe_index;
    float4*                  accum_buffer;
    uchar4*                  frame_buffer;
    int                      max_depth;
    float                    scene_epsilon;

    float3                   eye;
    float3                   U;
    float3                   V;
    float3                   W;

    BufferView<Light>        lights;
    float3                   miss_color;
    OptixTraversableHandle   handle;

    int gcount;
    int gn;
    int gs;
    int K;
    float* g_pos;
    float* g_opacity;
    float* g_shs;
    float* g_cov3d;
    float* g_hsize;
    // float* g_scale;
    // float* g_rotation;

    OctreeNodeD*  octree;
    OctreeNodeD* octreeLow;

    float* g_pos_low;
    float* g_opacity_low;
    float* g_shs_low;
    float* g_cov3d_low;
    // float* g_scale_low;
    // float* g_rotation_low;

    
    // int* cubes;
    // int     numNodes;
    int mode;
    // bool highGaussian;

    glm::mat4 projMatrix;
    glm::mat4 modelMatrix;
    float scaleFactor;

    float2 focal;
    float near;
    float fov;

    float tan_fovx;
    float tan_fovy;
};


struct PayloadRadiance
{
    float3 result;
    float  importance;
    
    float3       attenuation;
    unsigned int seed;
    int          depth;

    float3       emitted;
    float3       radiance;

    unsigned int highGaussian;
};


struct PayloadOcclusion
{
    float3 radiance;
};

// struct DepthGaussian {
//     //   unsigned char r, g,b,a, z;
//     float z;
//     float4 c;
// };
struct DepthGaussian {
    //   unsigned char r, g,b,a, z;
    float z;
    // int index;
    float4 c;
    // float3 ponto;
};

// struct GSM_tree {
//   DepthGaussian tree[20];
//   uint size;
// };


} // end namespace whitted

