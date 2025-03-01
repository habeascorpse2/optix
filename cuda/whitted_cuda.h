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
#ifndef WHITTED_CUDA_H
#define WHITTED_CUDA_H

#include <sutil/vec_math.h>
#include <sutil/Matrix.h>

#include "whitted.h"

#include <cuda.h>
#include "cuda_runtime.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>


namespace whitted {

extern "C" {
__constant__ whitted::LaunchParams params;
}

using namespace sutil;
//------------------------------------------------------------------------------
//
// GGX/smith shading helpers
// TODO: move into header so can be shared by path tracer and bespoke renderers
//
//------------------------------------------------------------------------------

__device__ __forceinline__ float3 schlick( const float3 spec_color, const float V_dot_H )
{
    return spec_color + ( make_float3( 1.0f ) - spec_color ) * powf( 1.0f - V_dot_H, 5.0f );
}

__device__ __forceinline__ float vis( const float N_dot_L, const float N_dot_V, const float alpha )
{
    const float alpha_sq = alpha*alpha;

    const float ggx0 = N_dot_L * sqrtf( N_dot_V*N_dot_V * ( 1.0f - alpha_sq ) + alpha_sq );
    const float ggx1 = N_dot_V * sqrtf( N_dot_L*N_dot_L * ( 1.0f - alpha_sq ) + alpha_sq );

    return 2.0f * N_dot_L * N_dot_V / (ggx0+ggx1);
}


__device__ __forceinline__ float ggxNormal( const float N_dot_H, const float alpha )
{
    const float alpha_sq   = alpha*alpha;
    const float N_dot_H_sq = N_dot_H*N_dot_H;
    const float x          = N_dot_H_sq*( alpha_sq - 1.0f ) + 1.0f;
    return alpha_sq/( M_PIf*x*x );
}


__device__ __forceinline__ float3 linearize( float3 c )
{
    return make_float3(
            powf( c.x, 2.2f ),
            powf( c.y, 2.2f ),
            powf( c.z, 2.2f )
            );
}


//------------------------------------------------------------------------------
//
//
//
//------------------------------------------------------------------------------


static __forceinline__ __device__ void traceRadiance(
        OptixTraversableHandle      handle,
        float3                      ray_origin,
        float3                      ray_direction,
        float                       tmin,
        float                       tmax,
        whitted::PayloadRadiance*   payload
        )
{
    unsigned int u0 = 0; // output only
    unsigned int u1 = 0; // output only
    unsigned int u2 = 0; // output only
    unsigned int u3 = payload->depth;
    unsigned int u4 = payload->seed;
    unsigned int u5 = payload->highGaussian;
    optixTrace(
            handle,
            ray_origin, ray_direction,
            tmin,
            tmax,
            0.0f,                     // rayTime
            OptixVisibilityMask( 1 ),
            OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
            whitted::RAY_TYPE_RADIANCE,        // SBT offset
            whitted::RAY_TYPE_COUNT,           // SBT stride
            whitted::RAY_TYPE_RADIANCE,        // missSBTIndex
            u0, u1, u2, u3, u4, u5 );

     payload->result.x = __uint_as_float( u0 );
     payload->result.y = __uint_as_float( u1 );
     payload->result.z = __uint_as_float( u2 );
     payload->depth    = 0; // input only
     payload->seed = 0;
    //  payload->highGaussian = u5;
}

__forceinline__ __device__ unsigned int getPayloadDepth()
{
    return optixGetPayload_3();
}

__forceinline__ __device__ unsigned int getPayloadSeed()
{
    return optixGetPayload_4();
}

__forceinline__ __device__ unsigned int getPayloadHighGaussian()
{
    return optixGetPayload_5();
}

static __forceinline__ __device__ float traceOcclusion(
        OptixTraversableHandle handle,
        float3                 ray_origin,
        float3                 ray_direction,
        float                  tmin,
        float                  tmax
        )
{
    // Introduce the concept of 'pending' and 'committed' attenuation.
    // This avoids the usage of closesthit shaders and allows the usage of the OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT flag.
    // The attenuation is marked as pending with a positive sign bit and marked committed by switching the sign bit.
    // Attenuation magnitude can be changed in anyhit programs and stays pending.
    // The final attenuation gets committed in the miss shader (by setting the sign bit).
    // If no miss shader is invoked (traversal was terminated due to an opaque hit)
    // the attenuation is not committed and the ray is deemed fully occluded.
    unsigned int attenuation = __float_as_uint(1.f);
    optixTrace(
            handle,
            ray_origin,
            ray_direction,
            tmin,
            tmax,
            0.0f,                    // rayTime
            OptixVisibilityMask( 1 ),
            OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
            whitted::RAY_TYPE_OCCLUSION,      // SBT offset
            whitted::RAY_TYPE_COUNT,          // SBT stride
            whitted::RAY_TYPE_OCCLUSION,      // missSBTIndex
            attenuation );

    // committed attenuation is negated
    return fmaxf(0, -__uint_as_float(attenuation));
}

__forceinline__ __device__ void setPayloadResult( float3 p )
{
    optixSetPayload_0( __float_as_uint( p.x ) );
    optixSetPayload_1( __float_as_uint( p.y ) );
    optixSetPayload_2( __float_as_uint( p.z ) );
}

__forceinline__ __device__ float getPayloadOcclusion()
{
    return __uint_as_float( optixGetPayload_0() );
}

__forceinline__ __device__ void setPayloadOcclusion( float attenuation )
{
    optixSetPayload_0( __float_as_uint( attenuation ) );
}

__forceinline__ __device__ void setPayloadOcclusionCommit()
{
    // set the sign
    optixSetPayload_0( optixGetPayload_0() | 0x80000000 );
}

} // namespace whitted

__forceinline__ __device__ float3 transformPoint4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
	};
	return transformed;
}

__forceinline__ __device__ float4 transformPoint4x4(const float3& p, const float* matrix)
{
	float4 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
		matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
	};
	return transformed;
}
__forceinline__ __device__ float3 transformPoint4x3(float3 p, sutil::Matrix4x4& matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[1 * 4 + 0] * p.y + matrix[2 * 4 + 0] * p.z + matrix[3 * 4 + 0],
		matrix[0 * 4 + 1] * p.x + matrix[1 * 4 + 1] * p.y + matrix[2 * 4 + 1] * p.z + matrix[3 * 4 + 1],
		matrix[0 * 4 + 2] * p.x + matrix[1 * 4 + 2] * p.y + matrix[2 * 4 + 2] * p.z + matrix[3 * 4 + 2],
	};
	return transformed;
}

__forceinline__ __device__ glm::vec3 transformPoint4x3(glm::vec3 p, glm::mat4& matrix)
{
	glm::vec3 transformed = {
		matrix[0][0] * p.x + matrix[1][0] * p.y + matrix[2][0] * p.z + matrix[3][0],
		matrix[0][1] * p.x + matrix[1][1] * p.y + matrix[2][1] * p.z + matrix[3][1],
		matrix[0][2] * p.x + matrix[1][2] * p.y + matrix[2][2] * p.z + matrix[3][2],
	};
	return transformed;
}

__forceinline__ __device__ float4 transformPoint4x4(float3 p, sutil::Matrix4x4& matrix)
{
	float4 transformed = {
		matrix[0 * 4 + 0] * p.x + matrix[1 * 4 + 0] * p.y + matrix[2 * 4 + 0] * p.z + matrix[3 * 4 + 0],
		matrix[0 * 4 + 1] * p.x + matrix[1 * 4 + 1] * p.y + matrix[2 * 4 + 1] * p.z + matrix[3 * 4 + 1],
		matrix[0 * 4 + 2] * p.x + matrix[1 * 4 + 2] * p.y + matrix[2 * 4 + 2] * p.z + matrix[3 * 4 + 2],
		matrix[0 * 4 + 3] * p.x + matrix[1 * 4 + 3] * p.y + matrix[2 * 4 + 3] * p.z + matrix[3 * 4 + 3]
	};
	return transformed;
}

__forceinline__ __device__ glm::vec4 transformPoint4x4(glm::vec3 p, glm::mat4& matrix)
{
	glm::vec4 transformed = {
		matrix[0][0] * p.x + matrix[1][0] * p.y + matrix[2][0] * p.z + matrix[3][0],
		matrix[0][1] * p.x + matrix[1][1] * p.y + matrix[2][1] * p.z + matrix[3][1],
		matrix[0][2] * p.x + matrix[1][2] * p.y + matrix[2][2] * p.z + matrix[3][2],
		matrix[0][3] * p.x + matrix[1][3] * p.y + matrix[2][3] * p.z + matrix[3][3]
	};
	return transformed;
}

__device__ const float SH_C0 = 0.28209479177387814;
__device__ const float SH_C1 = 0.4886025119029199;
__device__ const float SH_C2[5] = {
  1.0925484305920792,
  -1.0925484305920792,
  0.31539156525252005,
  -1.0925484305920792,
  0.5462742152960396
};
__device__ const float SH_C3[7] = {
  -0.5900435899266435,
  2.890611442640554,
  -0.4570457994644658,
  0.3731763325901154,
  -0.4570457994644658,
  1.445305721320277,
  -0.5900435899266435
};




// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}


// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, sutil::Matrix4x4& viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(glm::vec3& t, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, glm::mat4& viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0][0], viewmatrix[1][0], viewmatrix[2][0],
		viewmatrix[0][1], viewmatrix[1][1], viewmatrix[2][1],
		viewmatrix[0][2], viewmatrix[1][2], viewmatrix[2][2]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;
    // glm::mat3 cov = glm::transpose(T) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// __device__ float3 computeCov2D(glm::vec3& t, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, glm::mat4& viewmatrix) {

// 	// Clamping dos valores de t.x e t.y (como no código original)
// 	const float limx = 1.3f * tan_fovx;
// 	const float limy = 1.3f * tan_fovy;
// 	const float txtz = t.x / t.z;
// 	const float tytz = t.y / t.z;
// 	t.x = min(limx, max(-limx, txtz)) * t.z;
// 	t.y = min(limy, max(-limy, tytz)) * t.z;

// 	// Calcula o Jacobiano J da projeção (2x3, representado como uma mat3 com a última linha zero)
// 	glm::mat3 J = glm::mat3(
// 	focal_x / t.z,      0.0f,  -(focal_x * t.x) / (t.z * t.z),
// 	0.0f,      focal_y / t.z,  -(focal_y * t.y) / (t.z * t.z),
// 	0.0f,           0.0f,           0.0f
// 	);

// 	// Extrai a parte rotacional da view matrix para transformar a covariância do espaço mundo para view space.
//     glm::mat3 W = glm::mat3(
//     viewmatrix[0][0], viewmatrix[1][0], viewmatrix[2][0],
//     viewmatrix[0][1], viewmatrix[1][1], viewmatrix[2][1],
//     viewmatrix[0][2], viewmatrix[1][2], viewmatrix[2][2]);

// 	// Reconstrói a covariância 3D simétrica a partir de cov3D
// 	glm::mat3 Vrk = glm::mat3(
// 		cov3D[0], cov3D[1], cov3D[2],
// 		cov3D[1], cov3D[3], cov3D[4],
// 		cov3D[2], cov3D[4], cov3D[5]
// 	);

// 	// Transforma a covariância para o espaço de visualização:
// 	glm::mat3 V_view = W * Vrk * glm::transpose(W);

// 	// Agora, projeta para o espaço 2D usando o Jacobiano:
// 	// Σ₂D = J * V_view * Jᵀ.
// 	// Como J é efetivamente 2x3 (última linha zero), podemos computar manualmente a matriz 2x2.
// 	float c00 = 0.0f, c01 = 0.0f, c11 = 0.0f;
// 	for (int k = 0; k < 3; k++){
// 		for (int l = 0; l < 3; l++){
// 			float a0 = J[0][k];  // componente (0,k) de J
// 			float a1 = J[1][k];  // componente (1,k) de J
// 			float v = V_view[k][l];
// 			float b0 = J[0][l];
// 			float b1 = J[1][l];
// 			c00 += a0 * v * b0;
// 			c01 += a0 * v * b1;
// 			c11 += a1 * v * b1;
// 		}
// 	}
// 	// Retorna os componentes necessários: cov[0][0], cov[0][1] e cov[1][1]
// 	return make_float3(c00, c01, c11);
// }

__forceinline__ __device__ void getRect(const float2 p, int max_radius, uint2& rect_min, uint2& rect_max, dim3 grid)
{
	rect_min = {
		min(grid.x, max((int)0, (int)((p.x - max_radius) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y - max_radius) / BLOCK_Y)))
	};
	rect_max = {
		min(grid.x, max((int)0, (int)((p.x + max_radius + BLOCK_X - 1) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y + max_radius + BLOCK_Y - 1) / BLOCK_Y)))
	};
}

__forceinline__ __device__ void getRect(const float2 p, int2 ext_rect, uint2& rect_min, uint2& rect_max, uint2 grid)
{
	rect_min = {
		min(grid.x, max((int)0, (int)((p.x - ext_rect.x) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y - ext_rect.y) / BLOCK_Y)))
	};
	rect_max = {
		min(grid.x, max((int)0, (int)((p.x + ext_rect.x + BLOCK_X - 1) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y + ext_rect.y + BLOCK_Y - 1) / BLOCK_Y)))
	};
}

__forceinline__ __device__ float3 get_GaussianRGB(float3 d, int i, int& level) {
    int idx = i * 48;
    float* s;

    if (level == 0)
        s = whitted::params.g_shs;
    else
        s = whitted::params.g_shs_low;
    float3 rgb = SH_C0 * make_float3(s[idx], s[idx + 1], s[idx + 2]);

        rgb +=
            - SH_C1 * d.y * make_float3(s[idx + 3 * 1 + 0], s[ idx + 3 * 1 +  1], s[idx + 3 * 1 + 2])
            + SH_C1 * d.z * make_float3(s[idx + 3 * 2 + 0], s[ idx + 3 * 2 +  1], s[idx + 3 * 2 + 2])
            - SH_C1 * d.x * make_float3(s[idx + 3 * 3 + 0], s[ idx + 3 * 3 +  1], s[idx + 3 * 3 + 2]);
 
        float xx = d.x * d.x;
        float yy = d.y * d.y;
        float zz = d.z * d.z;
        float xy = d.x * d.y;
        float yz = d.y * d.z;
        float xz = d.x * d.z;
        rgb +=
            SH_C2[0] * xy * make_float3(s[idx + 3 * 4 + 0], s[ idx + 3 * 4 +  1], s[idx + 3 * 4 + 2]) +
            SH_C2[1] * yz * make_float3(s[idx + 3 * 5 + 0], s[ idx + 3 * 5 +  1], s[idx + 3 * 5 + 2]) +
            SH_C2[2] * (2.0 * zz - xx - yy) * make_float3(s[idx + 3 * 6 + 0], s[ idx + 3 * 6 +  1], s[idx + 3 * 6 + 2]) +
            SH_C2[3] * xz * make_float3(s[idx + 3 * 7 + 0], s[ idx + 3 * 7 +  1], s[idx + 3 * 7 + 2]) +
            SH_C2[4] * (xx - yy) * make_float3(s[idx + 3 * 8 + 0], s[ idx + 3 * 8 +  1], s[idx + 3 * 8 + 2]);

        
        rgb +=
            SH_C3[0] * d.y * (3.0 * xx - yy) * make_float3(s[idx + 3 * 9 + 0], s[ idx + 3 * 9 +  1], s[idx + 3 * 9 + 2]) +
            SH_C3[1] * d.z * xy * make_float3(s[idx + 3 * 10 + 0], s[ idx + 3 * 10 +  1], s[idx + 3 * 10 + 2]) +
            SH_C3[2] * d.y * (4.0 * zz - xx - yy) * make_float3(s[idx + 3 * 11 + 0], s[ idx + 3 * 11 +  1], s[idx + 3 * 11 + 2]) +
            SH_C3[3] * d.z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * make_float3(s[idx + 3 * 12 + 0], s[ idx + 3 * 12 +  1], s[idx + 3 * 12 + 2]) +
            SH_C3[4] * d.x * (4.0 * zz - xx - yy) * make_float3(s[idx + 3 * 13 + 0], s[ idx + 3 * 13 +  1], s[idx + 3 * 13 + 2]) +
            SH_C3[5] * d.z * (xx - yy) * make_float3(s[idx + 3 * 14 + 0], s[ idx + 3 * 14 +  1], s[idx + 3 * 14 + 2]) +
            SH_C3[6] * d.x * (xx - 3.0 * yy) * make_float3(s[idx + 3 * 15 + 0], s[ idx + 3 * 15 +  1], s[idx + 3 * 15 + 2]);
    
    rgb.x += 0.5f;
    rgb.y += 0.5f;
    rgb.z += 0.5f;
    // return rgb;  
    return clamp(rgb, 0.0, 1.0);
}

__forceinline__ __device__ whitted::Matrix4x4 createLookAt(float3 eye, float3 center, float3 up)
{
    float3 f = normalize(center - eye);
    float3 r = normalize(cross(up, f));
    float3 u = cross(f, r);

    whitted::Matrix4x4 result = whitted::Matrix4x4({
        r.x, r.y, r.z, 0.0,
        u.x, u.y, u.z, 0.0,
        -f.x, -f.y, -f.z, 0.0,
        0.0, 0.0, 0.0, 1.0
    });

    result = result.transpose();
    result[3 * 4 + 0] = -eye.x;
    result[3 * 4 + 1] = -eye.y;
    result[3 * 4 + 2] = eye.z;
    result[3 * 4 + 3] = 1.0f;

    return result;
}

__forceinline__ __device__ glm::mat4 createLookAt(glm::vec3 eye, glm::vec3 center, glm::vec3 up)
{
    glm::vec3 f = glm::normalize(center - eye);
    glm::vec3 r = glm::normalize(cross(up, f));
    glm::vec3 u = glm::cross(f, r);

    glm::mat4 result = glm::mat4(
        r.x, r.y, r.z, 0.0,
        u.x, u.y, u.z, 0.0,
        -f.x, -f.y, -f.z, 0.0,
        0.0, 0.0, 0.0, 1.0
    );

    result = glm::transpose(result);
    result[3][0] = -eye.x;
    result[3][1] = -eye.y;
    result[3][2] = eye.z;
    result[3][3] = 1.0f;

    return result;
}

__device__ glm::mat4 lookAtGLM(const glm::vec3& eye, glm::vec3 center, const glm::vec3 up)
	{
		glm::vec3 f(glm::normalize(center - eye));
		glm::vec3 s(glm::normalize(glm::cross(f, up)));
		glm::vec3 u(glm::cross(s, f));

		glm::mat4 Result(1);
		Result[0][0] = s.x;
		Result[1][0] = s.y;
		Result[2][0] = s.z;
		Result[0][1] = u.x;
		Result[1][1] = u.y;
		Result[2][1] = u.z;
		Result[0][2] =-f.x;
		Result[1][2] =-f.y;
		Result[2][2] =-f.z;
		Result[3][0] =-dot(s, eye);
		Result[3][1] =-dot(u, eye);
		Result[3][2] = dot(f, eye);
		return Result;
	}


__device__ glm::vec3 f2c(float3 a) {
    return glm::vec3(a.x, a.y, a.z);
}
__device__ float3 c2f(glm::vec3 a) {
    return make_float3(a.x, a.y, a.z);
}

__forceinline__ __device__ whitted::Matrix4x4 customLookAt(float3 position, float3 direction, float3 up) {
  // Normaliza o vetor de direção
    float3 forward = normalize(direction);

    // Calcula o vetor "right" usando o produto cruzado entre o vetor de direção e o vetor "up"
    float3 right = normalize(cross(up, forward));

    // Calcula o vetor "up" corrigido
    float3 correctedUp = cross(forward, right);
    whitted::Matrix4x4 one({
        1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
        });

    // Cria e retorna a matriz de visualização
    return whitted::Matrix4x4({
        right.x, right.y, right.z, 0.0f,
        correctedUp.x, correctedUp.y, correctedUp.z, 0.0f,
        -forward.x, forward.y, forward.z, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    }) * one.translate(-position);
}

__forceinline__ __device__ float ndc2Pix(float v, float S) {
    return ((v + 1.) * S - 1.) * .5;
}

__forceinline__ __device__ float4 applyGaussianFilter(float3 gc, float alpha, float3 conic, float2 xy, float x, float y) {

    float2 coordxy = make_float2(xy.x - x ,xy.y - y) ;
    
    float power = -0.5f * (conic.x * coordxy.x * coordxy.x + conic.z * coordxy.y * coordxy.y) - conic.y * coordxy.x * coordxy.y;
    if (!(power > 0.f)) {
      float opacity = min(0.99f, alpha * exp(power));
      if (!(opacity < 1.f / 255.f)) {
        // Normalizar os resultados
        return make_float4(gc.x, gc.y, gc.z, opacity);
      }
    }
    return make_float4(0);
}

__forceinline__ __device__ float getOpacity(float alpha, float3 conic, float2 xy, float x, float y) {

    float2 coordxy = make_float2(xy.x - x ,xy.y - y) ;
    
    float power = -0.5f * (conic.x * coordxy.x * coordxy.x + conic.z * coordxy.y * coordxy.y) - conic.y * coordxy.x * coordxy.y;
    // float power = -0.5f * (conic.x + conic.z ) - conic.y;
    if (!(power > 0.f)) {
      float opacity = min(0.99f, alpha * exp(power));
      if (!(opacity < 1.f / 255.f)) {
        // Normalizar os resultados
        return opacity;
      }
    }
    return 0;
}

__forceinline__ __device__ float getOpacity(float alpha, float3 conic, float2 xy) {

  int midle_x = (whitted::WIDTH ) / 2;
  int midle_y = (whitted::HEIGHT) / 2;
    
  float2 coordxy = make_float2(xy.x - midle_x ,xy.y - midle_y) ;
    
    float power = -0.5f * (conic.x * coordxy.x * coordxy.x + conic.z * coordxy.y * coordxy.y) - conic.y * coordxy.x * coordxy.y;
    if (!(power > 0.f)) {
      float opacity = min(0.99f, alpha * exp(power));
      if (!(opacity < 1.f / 255.f)) {
        // Normalizar os resultados
        return opacity;
      }
    }
    return 0;
}

__forceinline__ __device__ float getOpacity(float alpha, float3 conic) {

    
    float power = -0.5f * (conic.x + conic.z ) - conic.y;
    if (!(power > 0.f)) {
      float opacity = min(0.99f, alpha * exp(power));
      if (!(opacity < 1.f / 255.f)) {
        // Normalizar os resultados
        return opacity;
      }
    }
    return 0;
}




// Função para converter um componente de cor sRGB para linear
__device__ float sRGBToLinear(float c) {
    return (c <= 0.04045) ? (c / 12.92) : pow((c + 0.055) / 1.055, 2.4);
}

// Função para converter uma cor RGBA do espaço de cores OpenGL para sRGBA
__device__ float3 convertSRGBToRGB(float3 rgb) {
    // Convertendo os componentes de cores RGB para sRGB
    return make_float3(sRGBToLinear(rgb.x), sRGBToLinear(rgb.y), sRGBToLinear(rgb.z));
     
}

__device__ float calculateDistance(float3 point1, float3 point2) {
    float3 diff = point2 - point1;
    return sqrt(dot(diff, diff));
}

// Função para criar uma matriz de projeção em perspectiva
__device__ sutil::Matrix4x4 createPerspectiveMatrix(float fov, float aspectRatio, float nearPlane, float farPlane) {
    assert(aspectRatio > 0);
    assert(nearPlane < farPlane);

    // sutil::Matrix4x4 mat({0});
    // const float rad = fov * M_PI / 180.0f;
    // const float tanHalfFOV = std::tan(rad / 2.0f);

    // mat[0] = 1.0f / (aspectRatio * tanHalfFOV);
    // mat[5] = 1.0f / tanHalfFOV;
    // mat[10] = -(farPlane + nearPlane) / (farPlane - nearPlane);
    // mat[11] = -1.0f;
    // mat[14] = -(2.0f * farPlane * nearPlane) / (farPlane - nearPlane);
    // mat[15] = 0.0f;  // Alteração importante para projeção

    // return mat;

    const float rad = fov * M_PI / 180.0f;
    const float h = cos(0.5 * rad) / glm::sin(0.5 * rad);
    const float w = h * aspectRatio; ///todo max(width , Height) / min(width , Height)?

    sutil::Matrix4x4 Result({0});
    Result[0] = w;
    Result[5] = h * -1;
    Result[10] = - (farPlane + nearPlane) / (farPlane - nearPlane);
    Result[11] = - 1;
    Result[14] = - (2 * farPlane * nearPlane) / (farPlane - nearPlane);
    return Result;
}

#endif