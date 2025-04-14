
#include "gaussian.hpp"
#include "octree.cu"
#include "eigen/Eigen/Eigen"
//#include <Eigen/Dense>
// Define the types and sizes that make up the contents of each Gaussian 
// in the trained model.
 

namespace oct {


class OctreeGaussian {

	public:

		OctreeGaussian(Gaussian& gaussian_0, Gaussian& gaussian_1) {

			// Definir o limite da Octree
			Cube rootCube{{0, -2.f, 0.f}, {18, 12, 18}, -1}; // Quarto
			// Cube rootCube{{0, 4.f, 3.f}, {80, 30, 80}, -1}; //Sponza

			// Cube rootCube{{0, -2.f, 0.f}, {30, 12, 30}, -1}; // Outros
			octree = new Octree(rootCube.center, rootCube.half_size);

			octree->buildOctree();
			std::cout << "Copying level 0 Gaussian: " << std::endl;
			for (int i = 0; i < gaussian_0.count; ++i) {
				glm::vec3 pos = glm::vec3(gaussian_0.pos[i][0], gaussian_0.pos[i][1], gaussian_0.pos[i][2]);
				glm::vec3 hsize = glm::vec3(gaussian_0.hsize[i][0], gaussian_0.hsize[i][1], gaussian_0.hsize[i][2]);
				int count = 0;
				octree->addCubeOnLeafs(Cube({pos, hsize, i}), 0, count, 0);
			}

			std::cout << "Copying level 1 Gaussian: " << std::endl;
			for (int i = 0; i < gaussian_1.count; ++i) {
				glm::vec3 pos = glm::vec3(gaussian_1.pos[i][0], gaussian_1.pos[i][1], gaussian_1.pos[i][2]);
				glm::vec3 hsize = glm::vec3(gaussian_1.hsize[i][0], gaussian_1.hsize[i][1], gaussian_1.hsize[i][2]);
				int count = 0;
				octree->addCubeOnLeafs(Cube({pos, hsize, i}), 0, count, 1);
			}

			octree->flagOctree(0);

			std::cout <<" Copying octree to the GPU" << std::endl;
			deviceOctree = octree->sendToGPU();

		}

		~OctreeGaussian() {

			 // Limpar memória
			 delete octree;
			 
		}


		Octree* octree;

		OctreeNodeD* deviceOctree = nullptr;
		int numNodes = 0;



	private: 

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

}