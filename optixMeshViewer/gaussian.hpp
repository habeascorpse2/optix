#include <vector>
#include <iostream>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cerrno>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "GaussianInclude.hpp"
// Define the types and sizes that make up the contents of each Gaussian 
// in the trained model.

#ifndef gaussian_include
#define gaussian_include

float sigmoid(const float m1)
{
	return 1.0f / (1.0f + exp(-m1));
}

float inverse_sigmoid(const float m1)
{
	return log(m1 / (1.0f - m1));
}

# define CUDA_SAFE_CALL_ALWAYS(A) \
A; \
cudaDeviceSynchronize(); \
if (cudaPeekAtLastError() != cudaSuccess) \
	std::cerr << cudaGetErrorString(cudaGetLastError()) << std::endl;

#if DEBUG || _DEBUG
# define CUDA_SAFE_CALL(A) CUDA_SAFE_CALL_ALWAYS(A)
#else
# define CUDA_SAFE_CALL(A) A
#endif


std::function<char* (size_t N)> resizeFunctional(void** ptr, size_t& S) {
	auto lambda = [ptr, &S](size_t N) {
		if (N > S)
		{
			if (*ptr)
				CUDA_SAFE_CALL(cudaFree(*ptr));
			CUDA_SAFE_CALL(cudaMalloc(ptr, 2 * N));
			S = 2 * N;
		}
		return reinterpret_cast<char*>(*ptr);
	};
	return lambda;
}

const float SH_C0 = 0.28209479177387814;
const float SH_C1 = 0.4886025119029199;

glm::vec3 get_rgb(SHs<3> shs, glm::vec3 d) {
    glm::vec3 rgb = glm::vec3(0.5);

    rgb += SH_C0 * shs.shs[0];

        rgb +=
            - SH_C1 * d.y * shs.shs[1]
            + SH_C1 * d.z * shs.shs[2]
            - SH_C1 * d.x * shs.shs[3];
        

    return rgb;
}

class Gaussian {

	public:

		void* fallbackBufferCuda = nullptr;
		cudaGraphicsResource* cudaResource;
		RichPointPtr splattingPoints;

		int count;

		float* pos_cuda;
		float* rot_cuda;
		float* scale_cuda;
		float* opacity_cuda;
		float* shs_cuda;
		float* cov3d_cuda;
		float* cov3d9_cuda;
		float* hsize_cuda;
		

	Gaussian() {

	}

	~Gaussian() {
		// Cleanup
		cudaFree(pos_cuda);
		cudaFree(rot_cuda);
		cudaFree(scale_cuda);
		cudaFree(opacity_cuda);
		cudaFree(shs_cuda);
		cudaFree(cov3d_cuda);
		cudaFree(hsize_cuda);
		cudaFree(cov3d9_cuda);
		cudaFree(depthIndexCuda);

		cudaFree(view_cuda);
		cudaFree(proj_cuda);
		cudaFree(cam_pos_cuda);
		cudaFree(background_cuda);
		cudaFree(rect_cuda);
		cudaFree(fallbackBufferCuda);

		if (geomPtr)
			cudaFree(geomPtr);
		if (binningPtr)
			cudaFree(binningPtr);
		if (imgPtr)
			cudaFree(imgPtr);
	}

	Gaussian(uint render_w, uint render_h, const char* file, int sh_degree,
			bool white_bg, int device) :
		_sh_degree(sh_degree)
	{
		int num_devices;
		CUDA_SAFE_CALL_ALWAYS(cudaGetDeviceCount(&num_devices));
		_device = device;
		if (device >= num_devices)
		{
			if (num_devices == 0)
				std::cerr << "No CUDA devices detected!";
			else
				std::cerr << "Provided device index exceeds number of available CUDA devices!";
		}
		CUDA_SAFE_CALL_ALWAYS(cudaSetDevice(device));
		cudaDeviceProp prop;
		CUDA_SAFE_CALL_ALWAYS(cudaGetDeviceProperties(&prop, device));
		if (prop.major < 7)
		{
			std::cerr << "Sorry, need at least compute capability 7.0+!";
		}

		
		if (sh_degree == 0)
		{
			count = loadPly<0>(file, pos, shs, opacity, scale, rot, _scenemin, _scenemax);
		}
		else if (sh_degree == 1)
		{
			count = loadPly<1>(file, pos, shs, opacity, scale, rot, _scenemin, _scenemax);
		}
		else if (sh_degree == 2)
		{
			count = loadPly<2>(file, pos, shs, opacity, scale, rot, _scenemin, _scenemax);
		}
		else if (sh_degree == 3)
		{
			count = loadPly<3>(file, pos, shs, opacity, scale, rot, _scenemin, _scenemax);
		}

		_boxmin = _scenemin;
		_boxmax = _scenemax;
		
		int P = count;

		for (int i=0; i< count; i++)
		{
			cov3d9.push_back(convertMatrix3fToArray(ComputeCov3D(i)));
		}

		// Allocate and fill the GPU data
		CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&pos_cuda, sizeof(Pos) * P));
		CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(pos_cuda, pos.data(), sizeof(Pos) * P, cudaMemcpyHostToDevice));
		//Bounding Box
		CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&hsize_cuda, sizeof(Pos) * P));
		CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(hsize_cuda, hsize.data(), sizeof(Pos) * P, cudaMemcpyHostToDevice));
		// CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&rot_cuda, sizeof(Rot) * P));
		// CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(rot_cuda, rot.data(), sizeof(Rot) * P, cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&shs_cuda, sizeof(SHs<3>) * P));
		CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(shs_cuda, shs.data(), sizeof(SHs<3>) * P, cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&opacity_cuda, sizeof(float) * P));
		CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(opacity_cuda, opacity.data(), sizeof(float) * P, cudaMemcpyHostToDevice));
		// CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&scale_cuda, sizeof(Scale) * P));
		// CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(scale_cuda, scale.data(), sizeof(Scale) * P, cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&cov3d_cuda, sizeof(float) * 6 * P));
		CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(cov3d_cuda, cov3d.data(), sizeof(float) * 6 * P, cudaMemcpyHostToDevice));
		// CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&cov3d9_cuda, sizeof(float) * 9 * P));
		// CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(cov3d9_cuda, cov3d9.data(), sizeof(float) * 9 * P, cudaMemcpyHostToDevice));

		

		float bg[3] = { white_bg ? 1.f : 0.f, white_bg ? 1.f : 0.f, white_bg ? 1.f : 0.f };

		resolution_x = render_w;
		resolution_y = render_h;
		_scalingModifier = 1.0f;

		
		geomBufferFunc = resizeFunctional(&geomPtr, allocdGeom);
		binningBufferFunc = resizeFunctional(&binningPtr, allocdBinning);
		imgBufferFunc = resizeFunctional(&imgPtr, allocdImg);
	}

	std::array<float, 9> convertMatrix3fToArray(const Eigen::Matrix3f& matrix) {
		std::array<float, 9> arr;
		std::copy(matrix.data(), matrix.data() + 9, arr.begin());
		return arr;
	}

	Eigen::Matrix3f ComputeCov3D(int index)
	{
		// Criar matriz de escala
		Eigen::Matrix3f scaleMatrix = Eigen::Matrix3f::Identity();
		scaleMatrix(0, 0) = scale[index].scale[0];
		scaleMatrix(1, 1) = scale[index].scale[1];
		scaleMatrix(2, 2) = scale[index].scale[2];

		// Criar matriz de rotação a partir dos ângulos de Euler
		Eigen::Matrix3f rotationMatrix;
		rotationMatrix = Eigen::AngleAxisf(rot[index].rot[0], Eigen::Vector3f::UnitX())
					* Eigen::AngleAxisf(rot[index].rot[1], Eigen::Vector3f::UnitY())
					* Eigen::AngleAxisf(rot[index].rot[2], Eigen::Vector3f::UnitZ());

		// Calcular a matriz de covariância
		Eigen::Matrix3f covarianceMatrix = rotationMatrix * scaleMatrix * rotationMatrix.transpose();

		return covarianceMatrix;
	}

	

	std::vector<Pos> pos;
	std::vector<Pos> hsize;
	std::vector<Rot> rot;
	std::vector<Scale> scale;
	std::vector<float> opacity;
	std::vector<SHs<3>> shs;
	std::vector<std::array<float, 6>> cov3d;
	std::vector<std::array<float, 9>> cov3d9;

	
	private:


		bool _fastCulling = true;
		int _device = 0;
		int _sh_degree = 3;
		float _scalingModifier = 1.0f;


		// Load the PLY data (AoS) to the GPU (SoA)
		

		
		
		int* 	depthIndexCuda;
		int* rect_cuda;
		int resolution_x, resolution_y;

		float* view_cuda;
		float* proj_cuda;
		float* cam_pos_cuda;
		float* background_cuda;

		std::vector<char> fallback_bytes;
		size_t allocdGeom = 0, allocdBinning = 0, allocdImg = 0;
		std::function<char* (size_t N)> geomBufferFunc, binningBufferFunc, imgBufferFunc;
		void* geomPtr = nullptr, * binningPtr = nullptr, * imgPtr = nullptr;

		Eigen::Vector3f _boxmin, _boxmax, _scenemin, _scenemax;

		cudaExternalMemory_t cudaExternalMemory;


		// Load the Gaussians from the given file.
		template<int D>
		int loadPly(const char* filename,
			std::vector<Pos>& pos,
			std::vector<SHs<3>>& shs,
			std::vector<float>& opacities,
			std::vector<Scale>& scales,
			std::vector<Rot>& rot,
			Eigen::Vector3f& minn,
			Eigen::Vector3f& maxx)
		{
			std::ifstream infile(filename, std::ios_base::binary);

			if (!infile.good())
				std::cout << "Unable to find model's PLY file, attempted:\n" << filename << std::endl;

			// "Parse" header (it has to be a specific format anyway)
			std::string buff;
			std::getline(infile, buff);
			std::getline(infile, buff);

			std::string dummy;
			std::getline(infile, buff);
			std::stringstream ss(buff);
			int count;
			ss >> dummy >> dummy >> count;

			// Output number of Gaussians contained
			std::cout << "Loading " << count << " Gaussian splats" << std::endl;

			while (std::getline(infile, buff))
				if (buff.compare("end_header") == 0)
					break;

			// Read all Gaussians at once (AoS)
			std::vector<RichPoint<D>> points(count);
			infile.read((char*)points.data(), count * sizeof(RichPoint<D>));

			// Resize our SoA data
			pos.resize(count);
			shs.resize(count);
			scales.resize(count);
			rot.resize(count);
			opacities.resize(count);

			// Gaussians are done training, they won't move anymore. Arrange
			// them according to 3D Morton order. This means better cache
			// behavior for reading Gaussians that end up in the same tile 
			// (close in 3D --> close in 2D).
			minn = Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX);
			maxx = -minn;
			for (int i = 0; i < count; i++)
			{
				maxx = maxx.cwiseMax(points[i].pos);
				minn = minn.cwiseMin(points[i].pos);
			}
			std::vector<std::pair<uint64_t, int>> mapp(count);
			for (int i = 0; i < count; i++)
			{
				Eigen::Vector3f rel = (points[i].pos - minn).array() / (maxx - minn).array();
				Eigen::Vector3f scaled = ((float((1 << 21) - 1)) * rel);
				Eigen::Vector3i xyz = scaled.cast<int>();

				uint64_t code = 0;
				for (int i = 0; i < 21; i++) {
					code |= ((uint64_t(xyz.x() & (1 << i))) << (2 * i + 0));
					code |= ((uint64_t(xyz.y() & (1 << i))) << (2 * i + 1));
					code |= ((uint64_t(xyz.z() & (1 << i))) << (2 * i + 2));
				}

				mapp[i].first = code;
				mapp[i].second = i;
			}
			auto sorter = [](const std::pair < uint64_t, int>& a, const std::pair < uint64_t, int>& b) {
				return a.first < b.first;
			};
			std::sort(mapp.begin(), mapp.end(), sorter);

			// Move data from AoS to SoA
			int SH_N = (D + 1) * (D + 1);
			for (int k = 0; k < count; k++)
			{
				int i = mapp[k].second;
				pos[k] = points[i].pos;

				// Normalize quaternion
				float length2 = 0;
				for (int j = 0; j < 4; j++)
					length2 += points[i].rot.rot[j] * points[i].rot.rot[j];
				float length = sqrt(length2);
				for (int j = 0; j < 4; j++)
					rot[k].rot[j] = points[i].rot.rot[j] / length;

				// Exponentiate scale
				for(int j = 0; j < 3; j++)
					scales[k].scale[j] = exp(points[i].scale.scale[j]);

				// Activate alpha
				opacities[k] = sigmoid(points[i].opacity);

				shs[k].shs[0] = points[i].shs.shs[0];
				shs[k].shs[1] = points[i].shs.shs[1];
				shs[k].shs[2] = points[i].shs.shs[2];
				for (int j = 1; j < SH_N; j++)
				{
					shs[k].shs[j * 3 + 0] = points[i].shs.shs[(j - 1) + 3];
					shs[k].shs[j * 3 + 1] = points[i].shs.shs[(j - 1) + SH_N + 2];
					shs[k].shs[j * 3 + 2] = points[i].shs.shs[(j - 1) + 2 * SH_N + 1];
				}

			}
			cov3d = ComputeCov3D(scales, rot, 1);
			// cov3d9 = ComputeCov3D9(scales, rot, 1);
			// cov3d = ComputeCov3Ds(count);
			this->hsize.resize(count);
			for (int i=0; i < count; i++) {
				auto cov3d_9 = ComputeCov3D(i);
				glm::vec3 posi = glm::vec3(pos[i][0], pos[i][1], pos[i][2]);

				auto hs = calculateBoundingBoxSize(cov3d_9, posi) * 0.35f;
				this->hsize[i] = {hs.x, hs.y, hs.z};
			}

			return count;
		}

		
		std::vector<std::array<float, 6>> ComputeCov3D(const std::vector<Scale>& scales, const std::vector<Rot>& rotations, float scale_modifier)
		{
			std::vector<std::array<float, 6>> cov3ds;
			cov3ds.reserve(scales.size());

			for (int i = 0; i < scales.size(); i++) {
				std::array<float, 6> cov;
				glm::mat3 S = glm::mat3(1.0f);
				S[0][0] = scale_modifier * scales[i].scale[0];
				S[1][1] = scale_modifier * scales[i].scale[1];
				S[2][2] = scale_modifier * scales[i].scale[2];

				float r = rotations[i].rot[0];
				float x = rotations[i].rot[1];
				float y = rotations[i].rot[2];
				float z = rotations[i].rot[3];

				glm::mat3 R = glm::mat3(
					1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
					2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
					2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
				);

				glm::mat3 M = S * R;
				glm::mat3 sigma = transpose(M) * M;
				cov[0] = sigma[0][0];
				cov[1] = sigma[0][1];
				cov[2] = sigma[0][2];
				cov[3] = sigma[1][1];
				cov[4] = sigma[1][2];
				cov[5] = sigma[2][2];

				cov3ds.push_back(cov);
			}
			return cov3ds;
		}

		std::vector<std::array<float, 9>> ComputeCov3D9(const std::vector<Scale>& scales, const std::vector<Rot>& rotations, float scale_modifier)
		{
			std::vector<std::array<float, 9>> cov3ds;
			cov3ds.reserve(scales.size());

			for (size_t i = 0; i < scales.size(); i++) {
				std::array<float, 9> cov;
				
				// Construir a matriz de escala S
				glm::mat3 S(1.0f);
				S[0][0] = scale_modifier * scales[i].scale[0];
				S[1][1] = scale_modifier * scales[i].scale[1];
				S[2][2] = scale_modifier * scales[i].scale[2];

				// Obter os componentes do quaternion (r, x, y, z)
				float r = rotations[i].rot[0];
				float x = rotations[i].rot[1];
				float y = rotations[i].rot[2];
				float z = rotations[i].rot[3];

				// Construir a matriz de rotação R a partir do quaternion
				glm::mat3 R = glm::mat3(
					1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z),       2.f * (x * z + r * y),
					2.f * (x * y + r * z),       1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
					2.f * (x * z - r * y),       2.f * (y * z + r * x),       1.f - 2.f * (x * x + y * y)
				);

				// Calcular M e a matriz de covariância sigma = transpose(M)*M
				glm::mat3 M = S * R;
				glm::mat3 sigma = glm::transpose(M) * M;

				// Inverter sigma para obter sigmaInv (já que a GPU espera a matriz invertida)
				// glm::mat3 sigmaInv = glm::inverse(sigma);

				// Armazenar sigmaInv em ordem row-major (GPU espera float[9] em row-major)
				cov[0] = sigma[0][0];
				cov[1] = sigma[0][1];
				cov[2] = sigma[0][2];

				cov[3] = sigma[1][0];
				cov[4] = sigma[1][1];
				cov[5] = sigma[1][2];

				cov[6] = sigma[2][0];
				cov[7] = sigma[2][1];
				cov[8] = sigma[2][2];

				cov3ds.push_back(cov);
			}
			return cov3ds;
		}

		// Função para calcular o tamanho do cubo a partir da matriz de covariância
		glm::vec3 calculateBoundingBoxSize(const Eigen::Matrix3f covariance, glm::vec3 P) {
			
			// Obter os eigenvalues da matriz de covariância 3D
		   Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
		   // Eigen::Vector3f eigenvalues = solver.eigenvalues().cwiseSqrt() * 2.0f;
		   Eigen::Vector3f eigenvalues = solver.eigenvalues();
		   Eigen::Matrix3f eigenvectors = solver.eigenvectors();

		   // Defina o fator de dispersão (tipicamente 3 para 99.7% de cobertura)
		   float factor = 3.0;
		   Eigen::Vector3f pos(P.x, P.y, P.z);

		   // Calcule a dispersão nas direções principais
		   Eigen::Vector3f dispersion = factor * eigenvalues.cwiseSqrt();

		   // Calcule as extremidades da bounding box
		   Eigen::Vector3f bounding_box_min = pos - eigenvectors * dispersion;
		   Eigen::Vector3f bounding_box_max = pos + eigenvectors * dispersion;

		   // Calcule os tamanhos ao longo das direções X, Y e Z
		   Eigen::Vector3f sizes = (bounding_box_max - bounding_box_min).cwiseAbs();

		   // Criar o tamanho do cubo utilizando os eigenvalues da matriz 3D
		   return glm::vec3(sizes(0), sizes(1), sizes(2));

	   }


};

#endif