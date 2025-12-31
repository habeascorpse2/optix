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
inline __host__ __device__ glm::vec3 min(const float3& a, const float3& b) {
    return glm::vec3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

inline __host__ __device__ glm::vec3 max(const float3& a, const float3& b) {
    return glm::vec3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

struct octnode{
    float z;
    int index;
};

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

    __host__ __device__
    // Calcula a interseção entre um raio e o AABB do nó.
    // Retorna true se houver interseção, preenchendo tmin e tmax.
    bool intersectRay(const glm::vec3& rayOrigin, const glm::vec3& rayDir, float &tmin, float &tmax) const {
        glm::vec3 min = center - half_size;
        glm::vec3 max = center + half_size;

        // Interseção no eixo X
        float tx1 = (min.x - rayOrigin.x) / rayDir.x;
        float tx2 = (max.x - rayOrigin.x) / rayDir.x;
        tmin = fmin(tx1, tx2);
        tmax = fmax(tx1, tx2);

        // Interseção no eixo Y
        float ty1 = (min.y - rayOrigin.y) / rayDir.y;
        float ty2 = (max.y - rayOrigin.y) / rayDir.y;
        float tymin = fmin(ty1, ty2);
        float tymax = fmax(ty1, ty2);

        // Verifica se há separação entre os intervalos de X e Y
        if ((tmin > tymax) || (tymin > tmax))
            return false;
        if (tymin > tmin)
            tmin = tymin;
        if (tymax < tmax)
            tmax = tymax;

        // Interseção no eixo Z
        float tz1 = (min.z - rayOrigin.z) / rayDir.z;
        float tz2 = (max.z - rayOrigin.z) / rayDir.z;
        float tzmin = fmin(tz1, tz2);
        float tzmax = fmax(tz1, tz2);

        // Verifica se há separação entre os intervalos já atualizados e o eixo Z
        if ((tmin > tzmax) || (tzmin > tmax))
            return false;
        if (tzmin > tmin)
            tmin = tzmin;
        if (tzmax < tmax)
            tmax = tzmax;

        return true;
    }



};

inline    __host__ __device__
    Cube applyCenterTransformation(const Cube& scube, const glm::mat4& matrix) {
        Cube cube;
        cube.center = glm::vec3(matrix * glm::vec4(scube.center, 1.0f));
        return cube;
    }


struct OctreeNodeD {
    int *cubes_0;
    // int *cubes_1;
    int children[8]; // Índices dos filhos
    Cube boundary;
    // bool is_leaf;
    int numCubes_0;
    // int numCubes_1;
    int branchCubes;

    __host__ __device__
    OctreeNodeD() : numCubes_0(0), branchCubes(0) {
        // is_leaf = false;
        for (int i = 0; i < 8; ++i) {
            children[i] = -1;
        }
    }
    
};

namespace whitted
{

const unsigned int NUM_ATTRIBUTE_VALUES = 4u;
const unsigned int NUM_PAYLOAD_VALUES   = 10u;
const unsigned int MAX_TRACE_DEPTH      = 2u;

const unsigned int GSM_MAX_SIZE = 40;
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

struct HitRecord 
{
    alignas(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    uint32_t gaussian_id;  // Membro direto sem struct aninhado

    __host__ __device__ HitRecord() : gaussian_id(0) {}
};


enum RayType
{
    RAY_TYPE_RADIANCE  = 0,
    RAY_TYPE_OCCLUSION = 1,
    RAY_TYPE_COUNT = 2
};


struct LaunchParams
{
    // --- INÍCIO DA CORREÇÃO DE ALINHAMENTO ---
    // Agrupar membros por tamanho (maior para o menor) para garantir layout consistente.

    sutil::Matrix4x4 gaussianModelMatrix;
    
    // Membros de 8 bytes (64 bits) - Ponteiros e Handles
    float4*                  accum_buffer;
    uchar4*                  frame_buffer_ptr;
    BufferView<Light>        lights;
    CUdeviceptr              aabb_buffer;
    OptixTraversableHandle   handle;
    OptixTraversableHandle   ghandle;
    OptixTraversableHandle   ghandle2;
    cudaTextureObject_t      reflection_texture;

    float*                   g_pos;
    float*                   g_opacity;
    float*                   g_shs;
    float*                   g_cov3d;
    float*                   g_cov3d9;
    float*                   g_hsize;
    float*                   g2_pos;
    float*                   g2_opacity;
    float*                   g2_shs;
    float*                   g2_cov3d9;
    float*                   g2_hsize;
    OctreeNodeD*             octree;

    // Membros de 12 bytes
    float3                   eye;
    float3                   U;
    float3                   V;
    float3                   W;
    float3                   miss_color;
    
    unsigned int             width;
    unsigned int             height;
    unsigned int             subframe_index;
    int                      max_depth;
    float                    scene_epsilon;
    // int gcount;
    int mode;
    float roughness;
    float                    near;
    float                    fov;
    bool                     is_vr;

};
struct PayloadRadiance
{
    float3 result;
    float  importance;
    float  alpha;
    
    float3       attenuation;
    unsigned int seed;
    int          depth;

    float3       emitted;
    float3       radiance;
    float3       ray_dir;
    float3       ray_origin;
    int         htype; // 0 = RADIANCE, 1 = gaussians
    bool fov = true;

    octnode gstack[GSM_MAX_SIZE];
    int stackSize = 0;
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
    float z;
    float4 c;
};

// struct GSM_tree {
//   DepthGaussian tree[20];
//   uint size;
// };


} // end namespace whitted
