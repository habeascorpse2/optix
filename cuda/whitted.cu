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

    float3 result = make_float3(0.f);
    float alpha = 0.0f;

    const int samples_per_frame = 1;

    for (int i = 0; i < samples_per_frame; ++i) {
        // The center of each pixel is at fraction (0.5,0.5)
        float2 subpixel_jitter = make_float2( 0.5f, 0.5f );
        if (samples_per_frame > 1 || subframe_index > 0)
            subpixel_jitter = make_float2( rnd( seed ), rnd( seed ) );

        const float2 d =
            2.0f
                * make_float2( ( static_cast<float>( launch_idx.x ) + subpixel_jitter.x ) / static_cast<float>( launch_dims.x ),
                               ( static_cast<float>( launch_idx.y ) + subpixel_jitter.y ) / static_cast<float>( launch_dims.y ) )
            - 1.0f;
        
        float3 ray_direction_world = normalize( d.x * U + d.y * V + W );
        float3 ray_origin_world    = eye;

        // --- TRANSFORMAÇÃO DO RAIO PARA ESPAÇO DO OBJETO ---
        // Ray_Obj = Inv_M * Ray_World
        // O objeto GLTF está estático na origem (0,0,0) na estrutura de aceleração.
        // Para movê-lo virtualmente, transformamos o raio pelo inverso da matriz da mão.
        const float3 ray_origin = make_float3(whitted::params.inverse_object_transform * make_float4(ray_origin_world, 1.0f));
        const float3 ray_direction = make_float3(whitted::params.inverse_object_transform * make_float4(ray_direction_world, 0.0f));

        float3 sample_result = make_float3(0.f);
        float sample_alpha = 1.0f;

        if (whitted::params.mode != 1) {
            whitted::PayloadRadiance payload;
            payload.result     = make_float3( 0.0f );
            payload.depth      = 0;
            payload.seed = seed;
            payload.htype = 0;
            payload.alpha = 1.0f;

            float hy = launch_dims.y / 2;
            float hx = launch_dims.x / 2;
            float fov_a = hy * 0.5f;
            float pointy = launch_idx.y - hy;
            float pointx = launch_idx.x - hx;

            if (pointx < 0)
                pointx *= -1;
            if (pointy < 0)
                pointy *= -1;

            float point = sqrtf((pointx * pointx) + (pointy * pointy));

            if (point - fov_a >= 0.f) {
                payload.fov = false;
            }
            else {
                float fov_b = fov_a * (.5f);
                if (point - fov_a + fov_b >= 0) {
                    float opacity = (point - fov_a + fov_b) / fov_b ;
                    rnd(seed) > opacity ? payload.fov = true: payload.fov = false;
                }
                else
                    payload.fov = true;
            }

            // payload.fov = true;

            traceRadiance( whitted::params.handle, ray_origin, ray_direction,
                        0.00f,  // tmin
                        1e16f,  // tmax
                        &payload );

            sample_result = payload.result;
            sample_alpha = payload.alpha;

        }
        result += sample_result;
        alpha += sample_alpha;
    }

    result /= static_cast<float>(samples_per_frame);
    alpha /= static_cast<float>(samples_per_frame);

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
    if (whitted::params.is_vr == true)
        // Para o modo VR com Chroma Key, garantimos que o alfa seja 1.0 (opaco).
        // A cor RGB já será verde (do miss_color) ou a cor da cena.
        whitted::params.frame_buffer_ptr[image_index] = make_uchar4(
            static_cast<unsigned char>( __saturatef( accum_color.x ) * 255.99f ),
            static_cast<unsigned char>( __saturatef( accum_color.y ) * 255.99f ),
            static_cast<unsigned char>( __saturatef( accum_color.z ) * 255.99f ),
            static_cast<unsigned char>( __saturatef( alpha ) * 255.99f )
        );
    else {
        whitted::params.frame_buffer_ptr[image_index] = make_color(accum_color);
        whitted::params.accum_buffer[image_index] = make_float4( accum_color, 1.0f );
    }
        

}

extern "C" __global__ void __intersection__()
{
    // 1. Obter o índice da primitiva atual
    const unsigned int prim_idx = optixGetPrimitiveIndex();
    uint64_t payload_ptr = (uint64_t)(optixGetPayload_5()) | ((uint64_t)(optixGetPayload_6()) << 32);
    whitted::PayloadRadiance* payload = reinterpret_cast<whitted::PayloadRadiance*>(payload_ptr);

    
    const float3 g_mean = getPos(prim_idx,payload->fov)[0];
    // Acessa a matriz correta usando aritmética de ponteiros
    // const float* inv_transform = whitted::params.g_cov3d9[prim_idx * 9];
    float* inv_cov3d = getCov9(prim_idx,payload->fov);
    const float Q = 6.25f;

    // 3. Obter o raio (funções do OptiX já retornam float3)
    const float3 ray_origin = optixGetWorldRayOrigin();
    const float3 ray_dir = optixGetWorldRayDirection();
    const float ray_tmin = optixGetRayTmin();

    // 4. Chamar a função de interseção precisa
    float t_hit;
    if (intersectRayGaussianEllipsoid(ray_origin, ray_dir, ray_tmin, g_mean, inv_cov3d, Q, t_hit))
    {
        // 5. Se acertou, reporte o hit para o OptiX com a distância 't' exata
        optixReportIntersection(
            t_hit,
            0, // hit kind
            0, 0 // atributos (se necessário)
        );
    }
}

extern "C" __global__ void __anyhit__radiance()
{
    uint64_t payload_ptr = (uint64_t)(optixGetPayload_5()) | ((uint64_t)(optixGetPayload_6()) << 32);
    whitted::PayloadRadiance* payload = reinterpret_cast<whitted::PayloadRadiance*>(payload_ptr);

    if (payload->htype == 1) {

        // // 1. Calcular distância da origem (t = optixGetRayTmax())
        const int gaussian_id = optixGetPrimitiveIndex(); // ID da Gaussiana
        octnode node;
        node.index = gaussian_id;
        node.z = optixGetRayTmax(); // Distância da câmera (ray origin) ao ponto de interseção

        // float3 ray_orig = payload->ray_origin;
        // float3 g_mean = make_float3(whitted::params.g_pos[gaussian_id * 3], whitted::params.g_pos[gaussian_id * 3 + 1], whitted::params.g_pos[gaussian_id * 3 + 2]);
        // node.z = calculateDistance(ray_orig, g_mean); // Distância da câmera ao ponto de interseção
        if (node.z > .55f) // near culling
            minstack::insert(node, &payload->gstack[0], payload->stackSize );

        // Permite que o raio continue para outras interseções
        optixIgnoreIntersection();
    }
    else {
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

extern "C" __global__ void __miss__radiance()
{
    // const whitted::HitRecord* hitRecord = reinterpret_cast<whitted::HitRecord*>( optixGetSbtDataPointer() );
    uint64_t payload_ptr = (uint64_t)(optixGetPayload_5()) | ((uint64_t)(optixGetPayload_6()) << 32);
    whitted::PayloadRadiance* payload = reinterpret_cast<whitted::PayloadRadiance*>(payload_ptr);

    if (payload->htype == 1) {
        if (payload->stackSize == 0) {
            // whitted::setPayloadResult(whitted::params.miss_color);
            whitted::setPayloadResult(make_float3(0.0f, 0.0f, 0.0f));
            return;
        }

        float3 result = make_float3(0.0f);
        float T = 1.0f;
        
        float3 ray_dir         = payload->ray_dir;

        float3 ray_origin = payload->ray_origin;



        while (payload->stackSize > 0) { 

            octnode node = payload->gstack[0];
            minstack::removeMin(&payload->gstack[0], payload->stackSize);
            int idx = node.index;
            
            float* inv_cov3d = getCov9(idx, payload->fov);

            // 3. Carregar a posição central da gaussiana
            float3 g_mean = getPos(idx,payload->fov)[0];


            // Passo A: Transformar a origem e a direção do raio para o espaço local da Gaussiana.
            // Esta é a parte (o' = Σ⁻¹(o-m)) e (d' = Σ⁻¹d) da derivação do paper.
            const float3 o_prime = transform(inv_cov3d, ray_origin - g_mean);
            const float3 d_prime = transform(inv_cov3d, ray_dir);

            // Passo B: Calcular o quadrado da distância mínima entre a linha do raio e o centro (0,0,0)
            // no espaço local. A fórmula para a distância mínima ao quadrado é: ||o'||² - (<o',d'>)² / ||d'||²
            const float dot_o_d = dot(o_prime, d_prime);
            const float dot_d_d = dot(d_prime, d_prime);

            // Evitar divisão por zero se a direção transformada for nula.
            if (fabsf(dot_d_d) < 1e-8f) {
                continue;
            }

            // O ponto mais próximo na linha ocorre em t = -<o',d'> / ||d'||²
            float t_closest = -dot_o_d / dot_d_d;

            float min_dist_sq;
            // O paper especifica max(t>=0). Se o ponto mais próximo estiver atrás do raio (t<0),
            // a distância mínima relevante é no início do raio (t=0), que é simplesmente o comprimento de o'.
            if (t_closest < 0.0f) {
                min_dist_sq = dot(o_prime, o_prime);
            } else {
                // A distância mínima ao quadrado é ||o' + t_closest * d'||²
                // Uma forma mais estável e rápida de calcular: ||o'||² - t_closest * <o',d'>
                min_dist_sq = dot(o_prime, o_prime) + t_closest * dot_o_d;
            }
            
            // O paper tem um expoente de -1/2 * distância², esta é a "influência" da gaussiana.
            float power = -0.5f * min_dist_sq;
            float opacity = expf(power) * getGOpacity(idx, payload->fov);

        
            float3 dir = normalize(g_mean - ray_origin);
            dir = dir / length(dir);
            // dir = -ray_dir; // confirmar

            if (opacity > 0) { // Se a opacidade for significativa) {      
                float3 color = get_GaussianRGB(dir, idx, payload->fov);
                
                float test_T = T * (1 - opacity);
                if (test_T < 0.001f)
                {
                    break;
                }
        
                result += color  * opacity * T;
                T = test_T;
            }
        }

        result = convertSRGBToRGB(result);
        whitted::setPayloadResult( result );
        return;
    }
    else {
        whitted::setPayloadResult( whitted::params.miss_color );
        payload->alpha = 0.0f;
    }
}


extern "C" __global__ void __miss__occlusion()
{
    whitted::setPayloadOcclusionCommit();
}

extern "C" __global__ void __closesthit__radiance()
{
    const whitted::HitGroupData* hit_group_data = reinterpret_cast<whitted::HitGroupData*>( optixGetSbtDataPointer() );
    const LocalGeometry          geom           = getLocalGeometry( hit_group_data->geometry_data );
    
    // O raio que atingiu aqui está no Espaço do Objeto.
    // Precisamos calcular P e N no Espaço do Mundo para iluminação correta.
    
    const float3 ray_dir_obj = optixGetWorldRayDirection();
    const float3 ray_orig_obj = optixGetWorldRayOrigin();
    float3 P_obj = ray_orig_obj + optixGetRayTmax() * ray_dir_obj;

    // Transformar P e N de volta para World Space
    float3 P = make_float3(whitted::params.object_transform * make_float4(P_obj, 1.0f));
    // Para normais, usamos apenas a rotação (w=0)
    float3 N_local = geom.N;
    float3 N = normalize(make_float3(whitted::params.object_transform * make_float4(N_local, 0.0f)));
    
    // Recalcula direção do raio no mundo (Vetor V)
    float3 ray_dir = normalize(P - whitted::params.eye); 
    

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

    // float3 N = normalize(N_hmd);
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

        N_local = normalize( NN_trns.x * normalize( geom.texcoord[texcoord_idx].dpdu ) + NN_trns.y * normalize( geom.texcoord[texcoord_idx].dpdv ) + NN_trns.z * geom.N );
        N = normalize(make_float3(whitted::params.object_transform * make_float4(N_local, 0.0f)));
    }

    // Flip normal to the side of the incomming ray
    if( dot( N, ray_dir ) > 0.f )
        N = -N;


    // Implementação do 3D Gaussian
    unsigned int seed = whitted::getPayloadSeed();

    // roughness = whitted::params.roughness;
    if ( metallic > 0.5f) {
    // if (metallic > 0.99f) {
        float3 roughNormal = N;
        roughNormal.x -= (rnd(seed)/2 * roughness) - (rnd(seed)/2 * roughness);
        roughNormal.y -= (rnd(seed)/2 * roughness) - (rnd(seed)/2 * roughness);
        roughNormal.z -= (rnd(seed)/2 * roughness) - (rnd(seed)/2 * roughness);
        sutil::Matrix4x4 modelMatrix(whitted::params.gaussianModelMatrix);

        uint64_t payload_ptr = (uint64_t)(optixGetPayload_5()) | ((uint64_t)(optixGetPayload_6()) << 32);
        whitted::PayloadRadiance* payload = reinterpret_cast<whitted::PayloadRadiance*>(payload_ptr);

        // View vector correto
        float3 V = normalize(ray_dir);
        float3 R = normalize(reflect(V, roughNormal));
        float3 Pn = make_float3(modelMatrix * make_float4(P, 1.f));
        float3 Rn = normalize(make_float3(modelMatrix * make_float4(R, 0.f)));
        Rn.y *= -1; // Correção de coordenadas para o sistema de textura    
        Pn.y  *= -1; // Correção de coordenadas para o sistema de textura  
        Rn.z *= -1; // Correção de coordenadas para o sistema de textura
        Pn.z  *= -1; // Correção de coordenadas para o sistema de textura

        //
        //  Traça o raio para as Gaussianas
        //
        if (whitted::params.mode == 0) {

            //Traça o raio de reflexão de Gaussiana
            float3 origin = Pn;
            float3 direction = Rn;

            // --- INÍCIO DA CORREÇÃO ---
            // Crie um novo payload para o raio secundário para evitar corromper o estado do primário.
            whitted::PayloadRadiance secondary_payload;
            secondary_payload.depth = payload->depth + 1; // Incrementar profundidade
            secondary_payload.seed = payload->seed; // Reutilizar a semente ou gerar uma nova
            secondary_payload.htype = 1; // Indica que é um raio de reflexão de Gaussiana
            secondary_payload.alpha = 1.0f;
            secondary_payload.ray_origin = origin;
            secondary_payload.ray_dir = direction;
            
            // A decisão de qual handle usar é baseada no FOV do raio que nos atingiu (o primário).
            // E também definimos o valor de FOV para o payload do raio secundário.

            const uint3  launch_idx     = optixGetLaunchIndex();
            bool use_fov_handle = payload->fov;
            secondary_payload.fov = use_fov_handle;


            if (secondary_payload.fov)
                traceRadiance( whitted::params.ghandle, origin, direction,
                        0.00f,  // tmin
                        1e16f,  // tmax
                        &secondary_payload ); // Use o novo payload
            else
                traceRadiance( whitted::params.ghandle2, origin, direction,
                    0.00f,  // tmin
                    1e16f,  // tmax
                    &secondary_payload ); // Use o novo payload
            
            // O resultado do raio secundário é o resultado final para este hit.
            whitted::setPayloadResult( secondary_payload.result );
            payload->alpha = secondary_payload.alpha;
            

            spec_color = secondary_payload.result;
            // return;

        }
        else if( whitted::params.mode == 2 || whitted::params.mode == 3) { //Octree Mode
            // Converter o vetor de reflexão para coordenadas UV
            float3 origin = P;
            float3 direction = R;
            float u = 0.5f + atan2f(direction.z, direction.x) / (2.0f * M_PIf);
            float v = 0.5f - asinf(direction.y) / M_PIf;

            // Usar a função tex2D do CUDA para amostrar a textura
            // O OptiX 7 não tem uma função de amostragem própria, mas usa as do CUDA.
            // As coordenadas u e v devem estar no intervalo [0, 1].
            float4 texture_color = tex2D<float4>(whitted::params.reflection_texture, u, v);

            // Set the ray payload with the color from the environment map
            
            // float3 result_color = convertSRGBToRGB(make_float3(texture_color.x, texture_color.y, texture_color.z));
            spec_color = convertSRGBToRGB(make_float3(texture_color.x, texture_color.y, texture_color.z));

            // whitted::setPayloadResult( result_color );
            // return;
        }
        result = diff_color + spec_color;
    }
    else   
        result = make_float3(base_color);



    unsigned int depth = whitted::getPayloadDepth() + 1;

    

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


    whitted::setPayloadResult( result );
}
