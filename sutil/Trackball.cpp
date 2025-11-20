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

#include <sutil/Camera.h>
#include <sutil/Trackball.h>
#include <sutil/vec_math.h>
#include <cmath>
#include <algorithm>

namespace sutil
{

namespace
{
float radians(float degrees)
{
    return degrees * M_PIf / 180.0f;
}
float degrees(float radians)
{
    return radians * M_1_PIf * 180.0f;
}

} // namespace

void Trackball::startTracking(int x, int y)
{
    m_prevPosX = x;
    m_prevPosY = y;
    m_performTracking = true;
}

void Trackball::updateTracking(int x, int y, int /*canvasWidth*/, int /*canvasHeight*/)
{
    if(!m_performTracking)
    {
        startTracking(x, y);
        return;
    }

    int deltaX = x - m_prevPosX;
    int deltaY = y - m_prevPosY;

    m_prevPosX = x;
    m_prevPosY = y;
    m_latitude = radians(std::min(89.0f, std::max(-89.0f, degrees(m_latitude) + 0.5f*deltaY)));
    m_longitude = radians(fmod(degrees(m_longitude) - 0.5f*deltaX, 360.0f));

    updateCamera();

    if(!m_gimbalLock) {
        reinitOrientationFromCamera();
        m_camera->setUp(m_w);
    }
}

void Trackball::updateCamera()
{
    // use latlon for view definition
    float3 localDir;
    localDir.x = cos(m_latitude)*sin(m_longitude);
    localDir.y = cos(m_latitude)*cos(m_longitude);
    localDir.z = sin(m_latitude);

    float3 dirWS = m_u * localDir.x + m_v * localDir.y + m_w * localDir.z;

    if(m_viewMode == EyeFixed)
    {
        const float3& eye = m_camera->eye();
        m_camera->setLookat(eye - dirWS * m_cameraEyeLookatDistance);
    }
    else // LookAtFixed
    {
        const float3& lookat = m_camera->lookat();
        m_camera->setEye(lookat + dirWS * m_cameraEyeLookatDistance);
    }
}

void Trackball::setReferenceFrame(const float3& u, const float3& v, const float3& w)
{
    m_u = u;
    m_v = v;
    m_w = w;
    float3 dirWS = -normalize(m_camera->lookat() - m_camera->eye());
    float3 dirLocal;
    dirLocal.x = dot(dirWS, u);
    dirLocal.y = dot(dirWS, v);
    dirLocal.z = dot(dirWS, w);
    m_longitude = atan2(dirLocal.x, dirLocal.y);
    m_latitude = asin(dirLocal.z);
}

void Trackball::zoom(int direction)
{
    float zoom = (direction > 0) ? 1 / m_zoomMultiplier : m_zoomMultiplier;
    m_cameraEyeLookatDistance *= zoom;
    const float3& lookat = m_camera->lookat();
    const float3& eye = m_camera->eye();
    m_camera->setEye(lookat + (eye - lookat) * zoom);
}

void Trackball::reinitOrientationFromCamera()
{
    m_camera->UVWFrame(m_u, m_v, m_w);
    m_u = normalize(m_u);
    m_v = normalize(m_v);
    m_w = normalize(-m_w);
    std::swap(m_v, m_w);
    m_latitude = 0.0f;
    m_longitude = 0.0f;
    m_cameraEyeLookatDistance = length(m_camera->lookat() - m_camera->eye());
}

float3 Trackball::moveForward(float speed, bool move_cam)
{
    float3 dirWS = (m_vr_forward_direction.x != 0.0f || m_vr_forward_direction.y != 0.0f || m_vr_forward_direction.z != 0.0f)
                       ? m_vr_forward_direction
                       : normalize(m_camera->lookat() - m_camera->eye());
    float3 move_vec = dirWS * speed;
    if(move_cam)
    {
        m_camera->setEye(m_camera->eye() + move_vec);
        m_camera->setLookat(m_camera->lookat() + move_vec);
    }
    return move_vec;
}
float3 Trackball::moveBackward(float speed, bool move_cam)
{
    float3 dirWS = (m_vr_forward_direction.x != 0.0f || m_vr_forward_direction.y != 0.0f || m_vr_forward_direction.z != 0.0f)
                       ? m_vr_forward_direction
                       : normalize(m_camera->lookat() - m_camera->eye());
    float3 move_vec = -dirWS * speed;
    if(move_cam)
    {
        m_camera->setEye(m_camera->eye() + move_vec);
        m_camera->setLookat(m_camera->lookat() + move_vec);
    }
    return move_vec;
}
float3 Trackball::moveLeft(float speed, bool move_cam)
{
    float3 dirWS = (m_vr_forward_direction.x != 0.0f || m_vr_forward_direction.y != 0.0f || m_vr_forward_direction.z != 0.0f)
                       ? m_vr_forward_direction
                       : normalize(m_camera->lookat() - m_camera->eye());

    // O vetor 'up' do mundo é (0,1,0). O vetor 'right' é o produto vetorial de 'forward' e 'up'.
    float3 world_up = make_float3(0.0f, 1.0f, 0.0f);
    float3 rightWS = normalize(cross(dirWS, world_up));
    float3 move_vec = -rightWS * speed;
    if(move_cam)
    {
        m_camera->setEye(m_camera->eye() + move_vec);
        m_camera->setLookat(m_camera->lookat() + move_vec);
    }
    return move_vec;
}
float3 Trackball::moveRight(float speed, bool move_cam)
{
    float3 dirWS = (m_vr_forward_direction.x != 0.0f || m_vr_forward_direction.y != 0.0f || m_vr_forward_direction.z != 0.0f)
                       ? m_vr_forward_direction
                       : normalize(m_camera->lookat() - m_camera->eye());

    // O vetor 'up' do mundo é (0,1,0). O vetor 'right' é o produto vetorial de 'forward' e 'up'.
    float3 world_up = make_float3(0.0f, 1.0f, 0.0f);
    float3 rightWS = normalize(cross(dirWS, world_up));
    float3 move_vec = rightWS * speed;
    if(move_cam)
    {
        m_camera->setEye(m_camera->eye() + move_vec);
        m_camera->setLookat(m_camera->lookat() + move_vec);
    }
    return move_vec;
}
float3 Trackball::moveUp(float speed, bool move_cam)
{
    float3 u, v, w;
    m_camera->UVWFrame(u, v, w);
    v = normalize(v);
    float3 move_vec = v * speed;
    if(move_cam)
    {
        m_camera->setEye(m_camera->eye() + move_vec);
        m_camera->setLookat(m_camera->lookat() + move_vec);
    }
    return move_vec;
}
float3 Trackball::moveDown(float speed, bool move_cam)
{
    float3 u, v, w;
    m_camera->UVWFrame(u, v, w);
    v = normalize(v);
    float3 move_vec = -v * speed;
    if(move_cam)
    {
        m_camera->setEye(m_camera->eye() + move_vec);
        m_camera->setLookat(m_camera->lookat() + move_vec);
    }
    return move_vec;
}

void Trackball::rollLeft(float speed)
{
    float3 u, v, w;
    m_camera->UVWFrame(u, v, w);
    u = normalize(u);
    v = normalize(v);

    m_camera->setUp(u * cos(radians(90.0f + speed)) + v * sin(radians(90.0f + speed)));
}

void Trackball::rollRight(float speed)
{
    float3 u, v, w;
    m_camera->UVWFrame(u, v, w);
    u = normalize(u);
    v = normalize(v);

    m_camera->setUp(u * cos(radians(90.0f - speed)) + v * sin(radians(90.0f - speed)));
}

bool Trackball::wheelEvent(int dir)
{
    zoom(dir);
    return true;
}

bool Trackball::handleKeyEvent(unsigned char key)
{
    float currentSpeed = m_moveSpeed;
    
    switch(key)
    {
        // Arrow keys
        case 37: // Left arrow
        case 'A':
        case 'a':
            moveLeft(currentSpeed);
            return true;
            
        case 38: // Up arrow
        case 'W':
        case 'w':
            moveForward(currentSpeed);
            return true;
            
        case 39: // Right arrow
        case 'D':
        case 'd':
            moveRight(currentSpeed);
            return true;
            
        case 40: // Down arrow
        case 'S':
        case 's':
            moveBackward(currentSpeed);
            return true;
    }
    
    return false;
}

// Define a direção da câmera diretamente a partir de um vetor.
// Útil para alinhar a câmera com a orientação do headset VR.
void Trackball::setDirection(const float3& dir)
{
    if (!m_camera) return;

    // A direção da câmera é do olho para o lookat.
    float3 new_lookat = m_camera->eye() + dir * m_cameraEyeLookatDistance;
    m_camera->setLookat(new_lookat);
    reinitOrientationFromCamera();
}

// --- INÍCIO DA CORREÇÃO DE LOCOMOÇÃO VR ---
void Trackball::setVRMoveDirection(const float3& forward)
{
    // Apenas armazena a direção para uso nos métodos de movimento.
    // Normaliza e ignora a componente Y para movimento no plano.
    m_vr_forward_direction = normalize(make_float3(forward.x, 0.0f, forward.z));
}
// --- FIM DA CORREÇÃO DE LOCOMOÇÃO VR ---

} // namespace sutil
