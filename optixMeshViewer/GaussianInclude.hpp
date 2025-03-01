#ifndef GAUSSIAN_INCLUDE_HPP
#define GAUSSIAN_INCLUDE_HPP

#include "eigen/Eigen/Eigen"
#include <vector>

typedef Eigen::Vector3f Pos;
template<int D>
struct SHs
{
	float shs[(D+1)*(D+1)*3];
};
struct Scale
{
	float scale[3];
};
struct Rot
{
	float rot[4];
};
template<int D>
struct RichPoint
{
	Pos pos;
	float n[3];
	SHs<D> shs;
	float opacity;
	Scale scale;
	Rot rot;
};

struct RichPointPtr {
	void* pos;
	void* n[3];
	void *shs;
	void* scale;
	void* rot;
	void* cov3d; 
	void* opacity;
	void* depthIndex;
	int count;
};


/*
float3 transformPoint4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
	};
	return transformed;
}

float4 transformPoint4x4(const float3& p, const float* matrix)
{
	float4 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
		matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
	};
	return transformed;
}

// Forward version of 2D covariance matrix computation
glm::vec3 computeCov2D(const glm::vec3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, glm::mat4 viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	glm::vec4 t = viewmatrix * glm::vec4(mean, 1.0f);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = glm::min(limx, glm::max(-limx, txtz)) * t.z;
	t.y = glm::min(limy, glm::max(-limy, tytz)) * t.z;

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

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return glm::vec3(float(cov[0][0]), float(cov[0][1]), float(cov[1][1]));
}
*/
float ndc2Pix(float v, float S) {
    return ((v + 1.) * S - 1.) * .5;
}
float sRGBToLinear(float c) {
    return (c <= 0.04045) ? (c / 12.92) : pow((c + 0.055) / 1.055, 2.4);
}

float3 convertSRGBToRGB(float3 rgb) {
    // Convertendo os componentes de cores RGB para sRGB
    return make_float3(sRGBToLinear(rgb.x), sRGBToLinear(rgb.y), sRGBToLinear(rgb.z));
     
}


#endif