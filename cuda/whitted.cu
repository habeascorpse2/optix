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
#include <optix.h>  

#include <cuda/LocalGeometry.h>
#include <cuda/LocalShading.h>
#include <cuda/helpers.h>
#include <cuda/random.h>
#include <sutil/vec_math.h>

#include "whitted_cuda.h"
#include "../optixMeshViewer/octree.cu"
#include "minstack.cu"

//------------------------------------------------------------------------------
//
//
//
//------------------------------------------------------------------------------

extern "C" __global__ void __raygen__pinhole()
{
    const uint3  launch_idx     = optixGetLaunchIndex();
    const uint3  launch_dims    = optixGetLaunchDimensions();
    const float3 eye            = whitted::params.eye;
    const float3 U              = whitted::params.U;
    const float3 V              = whitted::params.V;
    const float3 W              = whitted::params.W;
    const int    subframe_index = whitted::params.subframe_index;

    //
    // Generate camera ray
    //
    unsigned int seed = tea<4>( launch_idx.y * launch_dims.x + launch_idx.x, subframe_index );

    // The center of each pixel is at fraction (0.5,0.5)
    const float2 subpixel_jitter =
        subframe_index == 0 ? make_float2( 0.5f, 0.5f ) :  make_float2( rnd( seed ), rnd( seed ) );
    // const float2 subpixel_jitter = make_float2( 0.5f, 0.5f );

    
    const float2 d =
        2.0f
            * make_float2( ( static_cast<float>( launch_idx.x ) + subpixel_jitter.x ) / static_cast<float>( launch_dims.x ),
                           ( static_cast<float>( launch_idx.y ) + subpixel_jitter.y ) / static_cast<float>( launch_dims.y ) )
        - 1.0f;
    const float3 ray_direction = normalize( d.x * U + d.y * V + W );
    const float3 ray_origin    = eye;
    float3 result = make_float3(0.f);

    

    if (whitted::params.mode != 1) {
        //
        // Trace camera ray
        //
        whitted::PayloadRadiance payload;
        payload.result     = make_float3( 0.0f );
        payload.depth      = 0;
        payload.seed = seed;


        float hy = launch_dims.y / 2;
        float hx = launch_dims.x / 2;
        float fov_a = hy * 0.3f;
        float pointy = launch_idx.y - hy;
        float pointx = launch_idx.x - hx;

        if (pointx < 0)
            pointx *= -1;
        if (pointy < 0)
            pointy *= -1;

        float point = sqrtf((pointx * pointx) + (pointy * pointy));

        if (point - fov_a >= 0.f) {
            payload.highGaussian = 0;
        }
        else {
            float fov_b = fov_a * (.3f);
            if (point - fov_a + fov_b >= 0) {
                float opacity = (point - fov_a + fov_b) / fov_b ;
                rnd(seed) > opacity ? payload.highGaussian = 1: payload.highGaussian = 0;
            }
            else
                payload.highGaussian = 1;
        }

        traceRadiance( whitted::params.handle, ray_origin, ray_direction,
                    0.00f,  // tmin
                    1e16f,  // tmax
                    &payload );

        result = payload.result;

    }
    

    //
    // Update results
    // TODO: timview mode
    //
    const unsigned int image_index = launch_idx.y * launch_dims.x + launch_idx.x;
    float3             accum_color = result;

    if( subframe_index > 0 )
    {
        const float  a                = 1.0f / static_cast<float>( subframe_index + 1 );
        const float3 accum_color_prev = make_float3( whitted::params.accum_buffer[image_index] );
        accum_color                   = lerp( accum_color_prev, accum_color, a );
    }
    whitted::params.accum_buffer[image_index] = make_float4( accum_color, 1.0f );
    whitted::params.frame_buffer[image_index] = make_color( accum_color );
}

extern "C" __global__ void __anyhit__radiance()
{
    const whitted::HitGroupData* hit_group_data = reinterpret_cast< whitted::HitGroupData* >( optixGetSbtDataPointer() );
    if( hit_group_data->material_data.pbr.base_color_tex )
    {
        const LocalGeometry geom       = getLocalGeometry( hit_group_data->geometry_data );
        const float         base_alpha = sampleTexture<float4>( hit_group_data->material_data.pbr.base_color_tex, geom ).w;
        // force mask mode, even for blend mode, as we don't do recursive traversal.
        if( base_alpha < hit_group_data->material_data.alpha_cutoff )
            optixIgnoreIntersection(); 
    }
}

extern "C" __global__ void __anyhit__occlusion()
{
    const whitted::HitGroupData* hit_group_data = reinterpret_cast< whitted::HitGroupData* >( optixGetSbtDataPointer() );
    if( hit_group_data->material_data.pbr.base_color_tex )
    {
        const LocalGeometry geom       = getLocalGeometry( hit_group_data->geometry_data );
        const float         base_alpha = sampleTexture<float4>( hit_group_data->material_data.pbr.base_color_tex, geom ).w;

        if( hit_group_data->material_data.alpha_mode != MaterialData::ALPHA_MODE_OPAQUE )
        {
            if( hit_group_data->material_data.alpha_mode == MaterialData::ALPHA_MODE_MASK )
            {
                if( base_alpha < hit_group_data->material_data.alpha_cutoff )
                    optixIgnoreIntersection();
            }

            float attenuation = whitted::getPayloadOcclusion() * (1.f - base_alpha);
            if( attenuation > 0.f )
            {
                whitted::setPayloadOcclusion( attenuation );
                optixIgnoreIntersection();
            }
        }
    }
}

extern "C" __global__ void __miss__constant_radiance()
{
    whitted::setPayloadResult( whitted::params.miss_color );
}

extern "C" __global__ void __miss__occlusion()
{
    whitted::setPayloadOcclusionCommit();
}

extern "C" __global__ void __closesthit__radiance()
{
    const whitted::HitGroupData* hit_group_data = reinterpret_cast<whitted::HitGroupData*>( optixGetSbtDataPointer() );
    const LocalGeometry          geom           = getLocalGeometry( hit_group_data->geometry_data );
    const float3 ray_dir         = optixGetWorldRayDirection();
    float3 P    = optixGetWorldRayOrigin() + optixGetRayTmax()*ray_dir;

    //
    // Retrieve material data
    //
    float4 base_color = hit_group_data->material_data.pbr.base_color * geom.color;
    if( hit_group_data->material_data.pbr.base_color_tex )
    {
        const float4 base_color_tex = sampleTexture<float4>( hit_group_data->material_data.pbr.base_color_tex, geom );

        // don't gamma correct the alpha channel.
        const float3 base_color_tex_linear = whitted::linearize( make_float3( base_color_tex ) );

        base_color *= make_float4( base_color_tex_linear.x, base_color_tex_linear.y, base_color_tex_linear.z, base_color_tex.w );
    }

    float  metallic  = hit_group_data->material_data.pbr.metallic;
    float  roughness = hit_group_data->material_data.pbr.roughness;
    float4 mr_tex    = make_float4( 1.0f );
    if( hit_group_data->material_data.pbr.metallic_roughness_tex )
        // MR tex is (occlusion, roughness, metallic )
        mr_tex = sampleTexture<float4>( hit_group_data->material_data.pbr.metallic_roughness_tex, geom );
    roughness *= mr_tex.y;
    metallic *= mr_tex.z;

    //
    // Convert to material params
    //
    const float  F0         = 0.04f;
    float3 diff_color = make_float3( base_color ) * ( 1.0f - F0 ) * ( 1.0f - metallic );
    float3 spec_color = lerp( make_float3( F0 ), make_float3( base_color ), metallic );
    const float  alpha      = roughness * roughness;

    float3 result = make_float3( 0.0f );

    //
    // compute emission
    //

    float3 emissive_factor = hit_group_data->material_data.emissive_factor;
    float4 emissive_tex = make_float4( 1.0f );
    if( hit_group_data->material_data.emissive_tex )
        emissive_tex = sampleTexture<float4>( hit_group_data->material_data.emissive_tex, geom );
    result += emissive_factor * make_float3( emissive_tex );

    //
    // compute direct lighting
    //

    float3 N = geom.N;
    if( hit_group_data->material_data.normal_tex )
    {
        const int texcoord_idx = hit_group_data->material_data.normal_tex.texcoord;
        const float4 NN =
            2.0f * sampleTexture<float4>( hit_group_data->material_data.normal_tex, geom ) - make_float4( 1.0f );

        // Transform normal from texture space to rotated UV space.
        const float2 rotation = hit_group_data->material_data.normal_tex.texcoord_rotation;
        const float2 NN_proj  = make_float2( NN.x, NN.y );
        const float3 NN_trns  = make_float3( 
            dot( NN_proj, make_float2( rotation.y, -rotation.x ) ), 
            dot( NN_proj, make_float2( rotation.x,  rotation.y ) ),
            NN.z );

        N = normalize( NN_trns.x * normalize( geom.texcoord[texcoord_idx].dpdu ) + NN_trns.y * normalize( geom.texcoord[texcoord_idx].dpdv ) + NN_trns.z * geom.N );
    }

    // Flip normal to the side of the incomming ray
    if( dot( N, optixGetWorldRayDirection() ) > 0.f )
        N = -N;


    // Implementação do 3D Gaussian
    unsigned int seed = whitted::getPayloadSeed();
    if ((rnd(seed) > roughness) && (rnd(seed) < hit_group_data->material_data.pbr.metallic)  ) {
    // if (metallic > 0.99f) {
        float3 roughNormal = N;
        roughNormal.x -= (rnd(seed)/2 * roughness) - (rnd(seed)/2 * roughness);
        roughNormal.y -= (rnd(seed)/2 * roughness) - (rnd(seed)/2 * roughness);
        roughNormal.z -= (rnd(seed)/2 * roughness) - (rnd(seed)/2 * roughness);
        glm::mat4 modelMatrix = glm::inverse(whitted::params.modelMatrix);
        glm::mat4 projMatrix = whitted::params.projMatrix;

        
        glm::vec3 R = f2c(reflect(ray_dir,roughNormal));
        glm::vec3 Rn = glm::normalize(glm::vec3(modelMatrix * glm::vec4(R, 0)));
        glm::vec3 Pn = f2c(P);
        Pn = glm::vec3(modelMatrix * glm::vec4(Pn, 1));

        //Colmap to CUDA positions
        // Pn.x *= -1;
        // Rn.x *= -1;
        // Pn.y *= -1;
        // Rn.y *= -1;
        // Pn.z *= -1;
        // Rn.z *= -1;

        // //Projection fix
        projMatrix[0] *= -1;
        projMatrix[1] *= -1;

        
        glm::vec3 up = {0, 1, 0};
        glm::mat4 viewMat = lookAtGLM(Pn, Pn + Rn, up);
        unsigned int highGaussian = whitted::getPayloadHighGaussian();

        int x = 0;
        int y = 0;
        if (whitted::params.mode == 0) {

            highGaussian=1;
            int aux = whitted::params.gs * whitted::params.gn;
            whitted::DepthGaussian dtree[whitted::GSM_MAX_SIZE];
            int size = 0;
            for (int idx = aux; idx < whitted::params.gn + aux; idx++) {
                glm::vec3 ponto;
                    ponto = glm::vec3(whitted::params.g_pos[idx * 3], whitted::params.g_pos[(idx*3)+1], whitted::params.g_pos[(idx*3)+2]);

                glm::vec3 p_view = transformPoint4x3(ponto, viewMat);
                
                if (p_view.z > 0.2f) {

                    glm::vec4 p_hom = transformPoint4x4(p_view, projMatrix);
                    float p_w = 1.0f / (p_hom.w + 0.0000001f);
                    glm::vec3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

                    float cov3D[6];
                    if (highGaussian > 0) {
                        cov3D[0] = whitted::params.g_cov3d[idx * 6];
                        cov3D[1] = whitted::params.g_cov3d[idx * 6 + 1];
                        cov3D[2] = whitted::params.g_cov3d[idx * 6 + 2];
                        cov3D[3] = whitted::params.g_cov3d[idx * 6 + 3];
                        cov3D[4] = whitted::params.g_cov3d[idx * 6 + 4];
                        cov3D[5] = whitted::params.g_cov3d[idx * 6 + 5];
                    }
                    else {
                        cov3D[0] = whitted::params.g_cov3d_low[idx * 6];
                        cov3D[1] = whitted::params.g_cov3d_low[idx * 6 + 1];
                        cov3D[2] = whitted::params.g_cov3d_low[idx * 6 + 2];
                        cov3D[3] = whitted::params.g_cov3d_low[idx * 6 + 3];
                        cov3D[4] = whitted::params.g_cov3d_low[idx * 6 + 4];
                        cov3D[5] = whitted::params.g_cov3d_low[idx * 6 + 5];
                    }

                    float3 cov =  computeCov2D(p_view, whitted::params.focal.x, whitted::params.focal.y, whitted::params.tan_fovx, whitted::params.tan_fovy, &cov3D[0], viewMat);
                    float det = (cov.x * cov.z - cov.y * cov.y);

                    constexpr float h_var = 0.3f;
                    cov.x += h_var;
                    cov.z += h_var;
                    const float det_cov_plus_h_cov = cov.x * cov.z - cov.y * cov.y;
                    float h_convolution_scaling = sqrt(max(0.000025f, det / det_cov_plus_h_cov)); // max for numerical stability

                    if (det != 0.0f) {

                        float det_inv = 1.f / det;
                        float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };
                        float2 point_image = { ndc2Pix(p_proj.x, whitted::WIDTH), ndc2Pix(p_proj.y, whitted::HEIGHT) };
                        
                        
                        float3 dir = c2f(glm::normalize(ponto - Pn));
                        dir = dir / length(dir);
                        float opacity;
                        if (highGaussian > 0)
                            opacity = getOpacity( whitted::params.g_opacity[idx] * h_convolution_scaling, conic, point_image);
                        else
                            opacity = getOpacity( whitted::params.g_opacity_low[idx] * h_convolution_scaling, conic, point_image);
                        if (opacity > 0 ) {      
                            whitted::DepthGaussian d;
                            int level = highGaussian -1;
                            d.c = make_float4(get_GaussianRGB(dir, idx, level), opacity);
                            d.z = p_view.z;
                            GSM_insert(d, &dtree[0], size);
                        }
                        
                    }
                }   
            }
            
            float T = 1.0f;
            while (size > 0) {
                float test_T = T * (1 - dtree[0].c.w);
                if (test_T < 0.0001f)
                {
                    break;
                }
                
                float3 color = make_float3(dtree[0].c.x, dtree[0].c.y, dtree[0].c.z);
                result += color  * dtree[0].c.w * T;
                GSM_removeMin(&dtree[0], size);

                T = test_T;
            }
            result = convertSRGBToRGB(result); 

            whitted::setPayloadResult( result );
            return;
        }
        else if( whitted::params.mode == 2 || whitted::params.mode == 3) { //Octree Mode

            OctreeNodeD* nodes = whitted::params.octree;
            minstack::octnode octStack[OCTNODE_SIZE];
            
            int count = 0;
            int level = 1;
            if (highGaussian > 0)
                level = 0;
            else
                level = 1;

            highGaussian=1;
            level = 0;
            
            count = searchIntersectingNodes(nodes, Pn, Rn, &octStack[0], viewMat, level);
            whitted::DepthGaussian dtree[whitted::GSM_MAX_SIZE];


            bool end = false;
            int size = 0;
            float T = 1.0f;

            int k   = 10; //K First Gaussians
            int k_i = 0;  // count K

            while (count > 0 && whitted::params.mode == 2) { 

                OctreeNodeD node = nodes[octStack[0].index];
                
                int numCubes;
                if (level == 0)
                    numCubes = node.numCubes_0;
                else
                    numCubes = node.numCubes_1;

                removeMin(octStack, count);
                for (int i = 0; i < numCubes; i++) {
                    
                    int idx;
                    glm::vec3 ponto;
                     if (level == 0) {
                        idx = node.cubes_0[i];
                        ponto = glm::vec3(whitted::params.g_pos[idx * 3], whitted::params.g_pos[(idx*3)+1], whitted::params.g_pos[(idx*3)+2]);
                     }
                    else {
                        idx = node.cubes_1[i];
                        ponto = glm::vec3(whitted::params.g_pos_low[idx * 3], whitted::params.g_pos_low[(idx*3)+1], whitted::params.g_pos_low[(idx*3)+2]);
                    }
                    glm::vec3 p_view = transformPoint4x3(ponto, viewMat);
                    
                    if (p_view.z > 0.2f) {

                        glm::vec4 p_hom = transformPoint4x4(p_view, projMatrix);
                        float p_w = 1.0f / (p_hom.w + 0.0000001f);
                        glm::vec3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

                        float cov3D[6];

                        // if (whitted::params.mode == 3) {
                        //     computeCov3D(glm::vec3(0.005f), whitted::params.scaleFactor, glm::vec4(1.0f), cov3D);
                        // }
                        // else {
                            if (level == 0) {
                                cov3D[0] = whitted::params.g_cov3d[idx * 6];
                                cov3D[1] = whitted::params.g_cov3d[idx * 6 + 1];
                                cov3D[2] = whitted::params.g_cov3d[idx * 6 + 2];
                                cov3D[3] = whitted::params.g_cov3d[idx * 6 + 3];
                                cov3D[4] = whitted::params.g_cov3d[idx * 6 + 4];
                                cov3D[5] = whitted::params.g_cov3d[idx * 6 + 5];
                            }
                            else {
                                cov3D[0] = whitted::params.g_cov3d_low[idx * 6];
                                cov3D[1] = whitted::params.g_cov3d_low[idx * 6 + 1];
                                cov3D[2] = whitted::params.g_cov3d_low[idx * 6 + 2];
                                cov3D[3] = whitted::params.g_cov3d_low[idx * 6 + 3];
                                cov3D[4] = whitted::params.g_cov3d_low[idx * 6 + 4];
                                cov3D[5] = whitted::params.g_cov3d_low[idx * 6 + 5];
                            }
                        // }

                        float3 cov =  computeCov2D(p_view, whitted::params.focal.x, whitted::params.focal.y, whitted::params.tan_fovx, whitted::params.tan_fovy, &cov3D[0], viewMat);
                        float det = (cov.x * cov.z - cov.y * cov.y);

                        constexpr float h_var = 0.3f;
                        cov.x += h_var;
                        cov.z += h_var;
                        const float det_cov_plus_h_cov = cov.x * cov.z - cov.y * cov.y;
                        float h_convolution_scaling = sqrt(max(0.000025f, det / det_cov_plus_h_cov)); // max for numerical stability

                        if (det != 0.0f) {

                            float det_inv = 1.f / det;
                            float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };
                            float2 point_image = { ndc2Pix(p_proj.x, whitted::WIDTH), ndc2Pix(p_proj.y, whitted::HEIGHT) };
                            
                            
                            float3 dir = c2f(glm::normalize(ponto - Pn));
                            dir = dir / length(dir);
                            float opacity;
                            if (level == 0)
                                opacity = getOpacity( whitted::params.g_opacity[idx] * h_convolution_scaling, conic, point_image);
                            else
                                opacity = getOpacity( whitted::params.g_opacity_low[idx] * h_convolution_scaling, conic, point_image);
                            // float opacity = whitted::params.g_opacity[idx] * 0.1;
                            if (opacity > 0 ) {   
                                
                                if (k_i >= k) {
                                    int lvl = highGaussian -1;
                                    float test_T = T * (1 - opacity);
                                    if (test_T < 0.0001f) {
                                        end = true;
                                        break;
                                    }
                                    
                                    float3 color = get_GaussianRGB(dir, idx, lvl);
                                    result += color  * opacity * T;

                                    T = test_T;
                                }
                                else {
                                    whitted::DepthGaussian d;
                                    int lvl = highGaussian -1;
                                    d.c = make_float4(get_GaussianRGB(dir, idx, lvl), opacity);
                                    d.z = p_view.z;
                                    GSM_insert(d, &dtree[0], size);
                                }
                            }
                            
                        }
                    }

                }
                

                while (size > 0) {
                    k_i++;
                    float test_T = T * (1 - dtree[0].c.w);
                    if (test_T < 0.0001f)
                    {
                        end = true;
                        break;
                    }
                    
                    float3 color = make_float3(dtree[0].c.x, dtree[0].c.y, dtree[0].c.z);
                    result += color  * dtree[0].c.w * T;
                    GSM_removeMin(&dtree[0], size);

                    T = test_T;
                }

                if (end)
                    break;

            }
            if (whitted::params.mode == 3) {

                result = make_float3(count/25, count/25, count/25);
            }

            // result = (1.0 - (metallic + (roughness))) * sm.albedo;
            result = (make_float3(base_color) * (1 - metallic)) + (result * metallic);
            result = convertSRGBToRGB(result); 

            whitted::setPayloadResult( result );
            return;
        }
        else {
            
            float z = P.z;
            if (z < 0)
                z *= -1;
            float x = P.x;
            if (x < 0)
                x *= -1;
            float y = P.y;
            if (y < 0)
                y *= -1;
            result = make_float3(whitted::WIDTH / x ,0, 0);

            whitted::setPayloadResult( result );
            return;
        }
        
    }

    unsigned int depth = whitted::getPayloadDepth() + 1;

    bool indirect = false;
    
    // int i = rnd(seed) * whitted::params.lights.count;
    for (int i= 0; i < whitted::params.lights.count; i++) {
    Light light = whitted::params.lights[i];
    if( light.type == Light::Type::POINT )
        {
            if( depth < whitted::MAX_TRACE_DEPTH )
            {
                // TODO: optimize
                const float  L_dist  = length( light.point.position - geom.P );
                const float3 L       = ( light.point.position - geom.P ) / L_dist;
                const float3 V       = -normalize( optixGetWorldRayDirection() );
                const float3 H       = normalize( L + V );
                const float  N_dot_L = dot( N, L );
                const float  N_dot_V = dot( N, V );
                const float  N_dot_H = dot( N, H );
                const float  V_dot_H = dot( V, H );

                if( N_dot_L > 0.0f && N_dot_V > 0.0f )
                {
                    const float tmin        = 0.001f;           // TODO
                    const float tmax        = L_dist - 0.001f;  // TODO
                    const float attenuation = whitted::traceOcclusion( whitted::params.handle, geom.P, L, tmin, tmax );
                    if( attenuation > 0.f )
                    {
                        const float3 F     = whitted::schlick( spec_color, V_dot_H );
                        const float  G_vis = whitted::vis( N_dot_L, N_dot_V, alpha );
                        const float  D     = whitted::ggxNormal( N_dot_H, alpha );

                        const float3 diff = ( 1.0f - F ) * diff_color / M_PIf;
                        const float3 spec = F * G_vis * D;

                        result += light.point.color * attenuation * light.point.intensity * N_dot_L * ( diff + spec );
                    }
                    else {
                        indirect = true;
                    }
                }
                else {
                    indirect = true;
                }
                
            }
        }
        else if( light.type == Light::Type::AMBIENT )
        {
            result += light.ambient.color * make_float3( base_color );
        }
    }

    if( hit_group_data->material_data.alpha_mode == MaterialData::ALPHA_MODE_BLEND )
    {
        result *= base_color.w;
                
        if( depth < whitted::MAX_TRACE_DEPTH )
        {
            whitted::PayloadRadiance alpha_payload;
            alpha_payload.result = make_float3( 0.0f );
            alpha_payload.depth  = depth;
            whitted::traceRadiance( 
                whitted::params.handle, 
                optixGetWorldRayOrigin(), 
                optixGetWorldRayDirection(),
                optixGetRayTmax(),  // tmin
                1e16f,              // tmax
                &alpha_payload );

            result += alpha_payload.result * make_float3( 1.f - base_color.w );
        }
    }

    // if (indirect) {
        if (whitted::params.mode == 2) {

            float3 result2 = make_float3(0.0f);
            glm::vec3 Pn = f2c(P); //f2c(optixGetWorldRayOrigin());
            glm::vec3 Rn = f2c(optixGetWorldRayDirection());

            glm::mat4 modelMatrix = whitted::params.modelMatrix;
            glm::mat4 projMatrix = whitted::params.projMatrix;
            
            //Colmap to CUDA positions
            Pn.y *= -1;
            Pn.x *= -1;
            Pn.z *= -1;

            //Projection fix
            projMatrix[0] *= -1;
            projMatrix[1] *= -1;

            glm::vec3 up = {0, 1, 0};
            glm::mat4 viewMat = lookAtGLM(Pn, Pn + Rn, up);
            unsigned int highGaussian = whitted::getPayloadHighGaussian();

            OctreeNodeD* nodes = whitted::params.octree;
            int results[MAX_RESULTS];
            int count = 0;
            int level = 1;
            if (highGaussian > 0)
                level = 0;
            else
                level = 1;

            highGaussian=1;
            level = 0;

            count = searchInsideNode(nodes, Pn, &results[0], level);
            whitted::DepthGaussian dtree[whitted::GSM_MAX_SIZE];

            bool end = false;
            int size = 0;
            float T = 1.0f;
            for (int j=0; j< count; j++) {
                OctreeNodeD node = nodes[results[j]];
                
                int numCubes;
                if (level == 0)
                    numCubes = node.numCubes_0;
                else
                    numCubes = node.numCubes_1;

                for (int i = 0; i < numCubes; i++) {
                    
                    int idx;
                    glm::vec3 ponto;
                        if (level == 0) {
                        idx = node.cubes_0[i];
                        ponto = glm::vec3(whitted::params.g_pos[idx * 3], whitted::params.g_pos[(idx*3)+1], whitted::params.g_pos[(idx*3)+2]);
                        }
                    else {
                        idx = node.cubes_1[i];
                        ponto = glm::vec3(whitted::params.g_pos_low[idx * 3], whitted::params.g_pos_low[(idx*3)+1], whitted::params.g_pos_low[(idx*3)+2]);
                    }

                    ponto.x *= -1;
                    
                    glm::vec3 p_view = transformPoint4x3(ponto, viewMat);
                    
                    if (p_view.z > 0.2f) {

                        glm::vec4 p_hom = transformPoint4x4(p_view, projMatrix);
                        float p_w = 1.0f / (p_hom.w + 0.0000001f);
                        glm::vec3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

                        float cov3D[6];

                        if (whitted::params.mode == 3) {
                            computeCov3D(glm::vec3(0.005f), whitted::params.scaleFactor, glm::vec4(1.0f), cov3D);
                        }
                        else
                            if (level == 0) {
                                cov3D[0] = whitted::params.g_cov3d[idx * 6];
                                cov3D[1] = whitted::params.g_cov3d[idx * 6 + 1];
                                cov3D[2] = whitted::params.g_cov3d[idx * 6 + 2];
                                cov3D[3] = whitted::params.g_cov3d[idx * 6 + 3];
                                cov3D[4] = whitted::params.g_cov3d[idx * 6 + 4];
                                cov3D[5] = whitted::params.g_cov3d[idx * 6 + 5];
                            }
                            else {
                                cov3D[0] = whitted::params.g_cov3d_low[idx * 6];
                                cov3D[1] = whitted::params.g_cov3d_low[idx * 6 + 1];
                                cov3D[2] = whitted::params.g_cov3d_low[idx * 6 + 2];
                                cov3D[3] = whitted::params.g_cov3d_low[idx * 6 + 3];
                                cov3D[4] = whitted::params.g_cov3d_low[idx * 6 + 4];
                                cov3D[5] = whitted::params.g_cov3d_low[idx * 6 + 5];
                            }

                        float3 cov =  computeCov2D(p_view, whitted::params.focal.x, whitted::params.focal.y, whitted::params.tan_fovx, whitted::params.tan_fovy, &cov3D[0], viewMat);
                        float det = (cov.x * cov.z - cov.y * cov.y);

                        constexpr float h_var = 0.3f;
                        cov.x += h_var;
                        cov.z += h_var;
                        const float det_cov_plus_h_cov = cov.x * cov.z - cov.y * cov.y;
                        float h_convolution_scaling = sqrt(max(0.000025f, det / det_cov_plus_h_cov)); // max for numerical stability

                        if (det != 0.0f) {

                            float det_inv = 1.f / det;
                            float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };
                            float2 point_image = { ndc2Pix(p_proj.x, whitted::WIDTH), ndc2Pix(p_proj.y, whitted::HEIGHT) };
                            
                            
                            float3 dir = c2f(glm::normalize(ponto - Pn));
                            dir = dir / length(dir);
                            float opacity;
                            if (level == 0)
                                opacity = getOpacity( whitted::params.g_opacity[idx] * h_convolution_scaling, conic, point_image);
                            else
                                opacity = getOpacity( whitted::params.g_opacity_low[idx] * h_convolution_scaling, conic, point_image);
                            if (opacity > 0 ) {      
                                whitted::DepthGaussian d;
                                int lvl = highGaussian -1;
                                d.c = make_float4(get_GaussianRGB(dir, idx, lvl), opacity);
                                d.z = p_view.z;
                                GSM_insert(d, &dtree[0], size);
                            }
                            
                        }
                    }

                }
                

                while (size > 0) {
                    float test_T = T * (1 - dtree[0].c.w);
                    if (test_T < 0.0001f)
                    {
                        end = true;
                        break;
                    }
                    
                    float3 color = make_float3(dtree[0].c.x, dtree[0].c.y, dtree[0].c.z);
                    result2 += color  * dtree[0].c.w * T;
                    GSM_removeMin(&dtree[0], size);

                    T = test_T;
                }
                if (end)
                    break;
            }
            result2 = convertSRGBToRGB(result2);
            if (!indirect)
                result = result2 * 0.5f + (0.5 * result);
            else
                result = result2 * 1.3 * (make_float3(base_color));
            whitted::setPayloadResult( result );
            return;
        }
    // }

    whitted::setPayloadResult( result );
}



// test_T = (1 - imgc.w);
// float3 color= make_float3(imgc.x, imgc.y, imgc.z) * imgc.w;
// if (lastMinorZ > p_view.z) {  // Se o ponto for mais próximo
//     result = (result * test_T)  + (outColor * imgc.w);
//     lastMinorZ = p_view.z; 
//     lastW = imgc.w;
//     lastColor = outColor;
//     last_t = test_T;
    
// }                            
// else { // caso o ponto seja mais distante
//     result = (result * lastW) + outColor  * imgc.w * last_t;
    
// }

// result = (lastColor * lastW) + result * (test_T);
    // result += make_float3(1) * T;