glm::vec3 p_view = transformPoint4x3(ponto, viewMat);
                        if (p_view.z > 0.2f) {

                            // glm::mat3 covariance = glm::make_mat3( whitted::params.g_cov3d9[idx * 9] );
                            glm::mat3 covariance(whitted::params.g_cov3d9[idx * 9+0], whitted::params.g_cov3d9[idx * 9+1], whitted::params.g_cov3d9[idx * 9+2],
                                                whitted::params.g_cov3d9[idx * 9+3], whitted::params.g_cov3d9[idx * 9+4], whitted::params.g_cov3d9[idx * 9+5],
                                                whitted::params.g_cov3d9[idx * 9+6], whitted::params.g_cov3d9[idx * 9+7], whitted::params.g_cov3d9[idx * 9+8]);

                            // Determina o parâmetro t que minimiza a distância entre o ponto no raio e o centro da gaussiana.
                            // Esse é o ponto da reta mais próximo do centro.
                            float t = glm::dot(ponto - Pn, Rn);
                            glm::vec3 closestPoint = Pn + t * Rn;
                            
                            // Calcula a diferença entre o ponto mais próximo e o centro da gaussiana
                            glm::vec3 diff = closestPoint - ponto;
                            
                            // Calcula a inversa da matriz de covariância
                            glm::mat3 invCov = glm::inverse(covariance);
                            
                            // Computa a distância de Mahalanobis ao quadrado: diff^T * invCov * diff
                            float mahalanobis = glm::dot(diff, invCov * diff);
                            float opacity = whitted::params.g_opacity[idx];
                            
                            // Avalia a função gaussiana (sem constante de normalização) no ponto
                            opacity = opacity * std::exp(-0.5f * mahalanobis);

                            if (opacity > 1/255 ) {   
                                float3 dir = c2f(glm::normalize(ponto - Pn));
                                        
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