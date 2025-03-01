
#include "gaussian.hpp"
#include "octree.cu"
#include "eigen/Eigen/Eigen"
//#include <Eigen/Dense>
// Define the types and sizes that make up the contents of each Gaussian 
// in the trained model.
 

namespace oct {



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




class OctreeGaussian {

	public:

		OctreeGaussian(Gaussian& gaussian ) {

			// Definir o limite da Octree
			Cube boundary{{0, 0, 0}, {38, 30, 38}, -1};
			Octree hostOctree(boundary);
			std::cout <<" Criando a octree" << std::endl;
			// Inserir cubos na Octree no host
			for (int i = 0; i < gaussian.count; ++i) {
				std::cout <<" " << i;
				auto cov3d = gaussian.ComputeCov3D(i);
				float3 pos = make_float3(gaussian.pos[i][0], gaussian.pos[i][1], gaussian.pos[i][2]);

				float3 hsize = calculateBoundingBoxSize(cov3d, pos) * 0.5f;
				hostOctree.insert(Cube({pos, hsize, i}));
			}

			// for (int i = 0; i < 4; ++i) {
			// 	Cube cube(make_float3(1.0f * i, 1.0f * i, 1.0f * i), make_float3(0.1f, 0.1f, 0.1f), 100 + i);
			// 	hostOctree.insert(cube);
			// }
			std::cout <<" Copiando a octree para a GPU" << std::endl;

			float3 origin ={0,0,0};
			float3 direction = {1, 0, 1};
			int count = 0;
			int tempFoundNumbers[MAX_RESULTS];
			sutil::Matrix4x4 m1({1.0f});
			hostOctree.query(origin, direction, m1, &tempFoundNumbers[0], count);
			std::cout << "Encontrados: " << count <<std::endl;

			std::cout << "Tamanho da Arvore: " << hostOctree.getSizeTree(&hostOctree) << "; Nós: " << hostOctree.getLength(&hostOctree)<< "; Depth: " << hostOctree.getDepth(&hostOctree) << std::endl;

			
			deviceOctree = copyOctreeToGPU(&hostOctree);

		}

		~OctreeGaussian() {

			 // Limpar memória
			cudaFree(deviceOctree);
		}

		Octree* deviceOctree = nullptr;

	private: 

		// Função para calcular o tamanho do cubo a partir da matriz de covariância
		float3 calculateBoundingBoxSize(const Eigen::Matrix3f covariance, float3 P) {
			
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
			return make_float3(sizes(0), sizes(1), sizes(2));

		}


};

}