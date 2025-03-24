#pragma once

#include <iostream>
#include <cuda_runtime.h>
#include <array>
#include <vector>
#include <memory>
#include <complex>
#include <whitted.h>
#include <fstream>
#include <sstream>
#include "minstack.cu"
#include "whitted.h"

# define CUDA_SAFE_CALL_ALWAYS(A) \ 
A; \
cudaDeviceSynchronize(); \
if (cudaPeekAtLastError() != cudaSuccess) \
	std::cerr << cudaGetErrorString(cudaGetLastError()) << std::endl;

#define MAX_RESULTS 20
#define MIN_SIZE .6f
// Estrutura do cubo

// Função para verificar erros CUDA
void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }

}

using namespace minstack;


struct OctreeD {
    int numNodes;
    OctreeNodeD* nodes;
    float3 root_center;
    float3 root_half_size;
    
};

// Estrutura do nó da Octree
struct OctreeNode {
    std::vector<Cube> cubes_0;
    std::vector<Cube> cubes_1;
    int children[8]; // Índices dos filhos
    Cube boundary;
    bool is_leaf;
    int numCubes_0;
    int numCubes_1;
    int branchCubes;
    int level;

    __host__ __device__
    OctreeNode() : numCubes_0(0), numCubes_1(0), branchCubes(0) {
        cubes_0.resize(0);
        cubes_1.resize(0);
        level = 0;
        for (int i = 0; i < 8; ++i) {
            children[i] = -1;
        }
    }
};

// Estrutura da Octree
struct Octree {
    std::vector<OctreeNode*> nodes_host;
    std::vector<OctreeNodeD*> dnodes;
    OctreeD* octreed;
    int numNodes;
    glm::vec3 root_center;
    glm::vec3 root_half_size;

    Octree() {
        numNodes = 0;
    }

    Octree(glm::vec3 center, glm::vec3 half_size) : numNodes(0), root_center(center), root_half_size(half_size) {
        OctreeNode* root = new OctreeNode();
        Cube boundary(center, half_size, -1);
        boundary.index = -1;
        root->boundary = boundary;

        nodes_host.push_back(root);
    }

    ~Octree() {
        for (int i = 0; i< nodes_host.size(); i ++) {
            delete nodes_host[i];
        }
        nodes_host.resize(0);
    }

    __host__ __device__
    bool cubeFitsInNode(OctreeNode* node, const Cube& cube) {
        return (cube.half_size.x <= node->boundary.half_size.x &&
                cube.half_size.y <= node->boundary.half_size.y &&
                cube.half_size.z <= node->boundary.half_size.z);
    }

    __host__ 
    bool cubeIntersectsNode(OctreeNode* node, const Cube& cube) {
        if (abs(node->boundary.center.x - cube.center.x) > (node->boundary.half_size.x + cube.half_size.x)) {
            return false;
        }
        // Verificar separação no eixo Y
        if (abs(node->boundary.center.y - cube.center.y) > (node->boundary.half_size.y + cube.half_size.y)) {
            return false;
        }
        // Verificar separação no eixo Z
        if (abs(node->boundary.center.z - cube.center.z) > (node->boundary.half_size.z + cube.half_size.z)) {
            return false;
        }
        
        // Não há separação em nenhum dos eixos, os cubos se intersectam
        return true;
    }

    __host__ bool createChildren(int nodeIndex) {
        OctreeNode* node = nodes_host.at(nodeIndex);
        glm::vec3 center = node->boundary.center;
        glm::vec3 halfSize = node->boundary.half_size;
        glm::vec3 quarterSize = {halfSize.x / 2.0f, halfSize.y / 2.0f, halfSize.z / 2.0f};
        if (quarterSize.x <= MIN_SIZE || quarterSize.y <= MIN_SIZE || quarterSize.z <= MIN_SIZE) {
            node->is_leaf = true;
            return false;
        }
        int level = node->level + 1;

        for (int i = 0; i < 8; ++i) {
            numNodes += 1;
            glm::vec3 offset = {
                ((i & 1) ? 1.0f : -1.0f) * quarterSize.x,
                ((i & 2) ? 1.0f : -1.0f) * quarterSize.y,
                ((i & 4) ? 1.0f : -1.0f) * quarterSize.z
            };
            Cube childBoundary(center + offset, quarterSize, -1);
            OctreeNode *childNode = new OctreeNode();
            childNode->level = level;
            childNode->cubes_0.resize(0);
            childNode->cubes_1.resize(0);
            childNode->boundary = childBoundary;
            childNode->is_leaf = false;
            nodes_host.push_back(childNode);
            node->children[i] = nodes_host.size() - 1;
            node->is_leaf = false;
        }
        return true;
    }

    void createOctreeNodes(int nodeIndex) {

        if (!createChildren(nodeIndex))
            return;

        OctreeNode* node = nodes_host.at(nodeIndex);
        for (auto& child : node->children) {
            createOctreeNodes(child);
        }
    }

    __host__ void buildOctree() {
        createOctreeNodes(0);

    }

    // __host__ void addCube(const Cube& cube, int nodeIndex, int& count, int level) {
    //     OctreeNode* node = nodes_host.at(nodeIndex);

    //     node = nodes_host.at(nodeIndex);
    //     if (!node->is_leaf && node->children[0] == -1)
    //         createChildren(nodeIndex);

    //     node = nodes_host.at(nodeIndex);

    //     if (node->is_leaf) {
            
    //         node->numCubes++;
    //         node->cubes.push_back(cube);
    //         count = count + 1;
    //         return;

    //     }

    //     std::vector<int> hm; //how many cubes will collides on childrens bounding box
    //     hm.resize(0);
    //     for (int i = 0; i < 8; ++i) {
    //         OctreeNode* childNode = nodes_host.at(node->children[i]);
    //         if (cubeIntersectsNode(childNode, cube) && cubeFitsInNode(childNode, cube))
    //             hm.push_back(i);
    //     }

    //     if (hm.size() == 0) {
    //         node->numCubes++;
    //         node->cubes.push_back(cube);
    //         count = count + 1;
    //         return;
    //     }
    //     else {
    //         for (const int& i : hm) {
    //             addCube(cube, node->children[i], count);
    //         }
    //     }

    //     return;
        
    // }

    
    // Estratégia para adicionar cubos apenas nos nós folhas
    __host__ void addCubeOnLeafs(const Cube& cube, int nodeIndex, int& count, int level) {
        OctreeNode* node = nodes_host.at(nodeIndex);
        // std::cout << "Cube X: " << cube.center.x << " Y: " << cube.center.y << " Z: " << cube.center.z << std::endl;
        // std::cout << "Size X: " << cube.half_size.x << " Y: " << cube.half_size.y << " Z: " << cube.half_size.z << std::endl;
        if (!cubeIntersectsNode(node, cube)) {
            return;
        }

        node = nodes_host.at(nodeIndex);

        if (node->is_leaf) {
            if (level == 0) {
                node->numCubes_0++;
                node->cubes_0.push_back(cube);
            }
            else {
                node->numCubes_1++;
                node->cubes_1.push_back(cube);
            }
            
            count = count + 1;
            return;

        }
        else {
            for (int i = 0; i < 8; ++i) {
                addCubeOnLeafs(cube, node->children[i], count, level);   
            }
            return;
        }
        
    }

    __host__
    int flagOctree(int nodeIndex) {
        if (nodeIndex >=0) {
            OctreeNode* node = nodes_host.at(nodeIndex);

            if (!node->is_leaf) {
                for (int i=0; i<8; i++)
                    node->branchCubes +=flagOctree(node->children[i]);
            }
            node->branchCubes += node->numCubes_0;
            node->branchCubes += node->numCubes_1;
            return node->branchCubes;
        }

    }

    __host__ OctreeNodeD* sendToGPU() {
        
        numNodes = nodes_host.size();
        OctreeNodeD* h_data = new OctreeNodeD[numNodes];


        for (int i = 0; i < numNodes; i++) {

            OctreeNode* node = nodes_host.at(i);
            if (node->branchCubes >0) 
                node->is_leaf = true;

            for (int j=0; j < 8; j++)
                h_data[i].children[j] = node->children[j];

            h_data[i].boundary = node->boundary;
            h_data[i].numCubes_0 = node->numCubes_0;
            h_data[i].numCubes_1 = node->numCubes_1;
            h_data[i].cubes_0 = nullptr;
            h_data[i].cubes_1 = nullptr;
            h_data[i].branchCubes = node->branchCubes;
            // h_data[i].is_leaf = node->is_leaf;
            if (node->numCubes_0 > 0) {
                int *ccubes;
                ccubes = (int *)malloc(node->numCubes_0 * sizeof(int));
                for (int j=0; j < node->numCubes_0; j++)
                    ccubes[j] = node->cubes_0[j].index;
                h_data[i].cubes_0= ccubes;
            }
            if (node->numCubes_1 > 0) {
                int *ccubes;
                ccubes = (int *)malloc(node->numCubes_1 * sizeof(int));
                for (int j=0; j < node->numCubes_1; j++)
                    ccubes[j] = node->cubes_1[j].index;
                h_data[i].cubes_1 = ccubes;
            }
        }

         // Alocar memória na GPU para o array de OctreeNodeD
        OctreeNodeD *d_nodes;
        checkCudaError(cudaMalloc(&d_nodes, numNodes * sizeof(OctreeNodeD)), "Allocating d_nodes");

        // Copiar dados do array principal (exceto os ponteiros cubes) para a GPU
        checkCudaError(cudaMemcpy(d_nodes, h_data, numNodes * sizeof(OctreeNodeD), cudaMemcpyHostToDevice), "Copying h_nodes to d_nodes");

        // Para cada nó, alocar memória na GPU para o array cubes e copiar os dados
        for (int i = 0; i < numNodes; ++i) {
            int *d_cubes_0;
            checkCudaError(cudaMalloc(&d_cubes_0, h_data[i].numCubes_0 * sizeof(int)), "Allocating d_cubes");
            checkCudaError(cudaMemcpy(d_cubes_0, h_data[i].cubes_0, h_data[i].numCubes_0 * sizeof(int), cudaMemcpyHostToDevice), "Copying cubes to d_cubes");

            // Atualizar o ponteiro cubes na GPU
            checkCudaError(cudaMemcpy(&d_nodes[i].cubes_0, &d_cubes_0, sizeof(int*), cudaMemcpyHostToDevice), "Updating d_nodes[i].cubes");
            delete h_data[i].cubes_0;

            int *d_cubes_1;
            checkCudaError(cudaMalloc(&d_cubes_1, h_data[i].numCubes_1 * sizeof(int)), "Allocating d_cubes");
            checkCudaError(cudaMemcpy(d_cubes_1, h_data[i].cubes_1, h_data[i].numCubes_1 * sizeof(int), cudaMemcpyHostToDevice), "Copying cubes to d_cubes");

            // Atualizar o ponteiro cubes na GPU
            checkCudaError(cudaMemcpy(&d_nodes[i].cubes_1, &d_cubes_1, sizeof(int*), cudaMemcpyHostToDevice), "Updating d_nodes[i].cubes");
            delete h_data[i].cubes_1;
        }
        delete h_data;

        return d_nodes;
    }

};


__host__ __device__ bool rayIntersectsCube1(const Cube& cube, const glm::vec3& origin, const glm::vec3& direction) {

    // Inicializar valores de tmin e tmax para cada eixo
    float tmin = (cube.center.x - cube.half_size.x - origin.x) / direction.x;
    float tmax = (cube.center.x + cube.half_size.x - origin.x) / direction.x;

    if (tmin > tmax) {
        float temp = tmin;
        tmin = tmax;
        tmax = temp;
    }

    float tymin = (cube.center.y - cube.half_size.y  - origin.y) / direction.y;
    float tymax = (cube.center.y + cube.half_size.y  - origin.y) / direction.y;

    if (tymin > tymax) {
        float temp = tymin;
        tymin = tymax;
        tymax = temp;
    }

    // Verificar se há interseção no eixo Y
    if ((tmin > tymax) || (tymin > tmax)) {
        return false;
    }

    // Atualizar tmin e tmax para o intervalo válido
    if (tymin > tmin) {
        tmin = tymin;
    }
    if (tymax < tmax) {
        tmax = tymax;
    }

    float tzmin = (cube.center.z - cube.half_size.z - origin.z) / direction.z;
    float tzmax = (cube.center.z + cube.half_size.z - origin.z) / direction.z;

    if (tzmin > tzmax) {
        float temp = tzmin;
        tzmin = tzmax;
        tzmax = temp;
    }

    // Verificar se há interseção no eixo Z
    if ((tmin > tzmax) || (tzmin > tmax)) {
        return false;
    }
    return true;
}

__device__
bool isPointInsideCube(const glm::vec3& point, const Cube& cube) {
    return (point.x >= cube.center.x - cube.half_size.x && point.x <= cube.center.x + cube.half_size.x &&
            point.y >= cube.center.y - cube.half_size.y && point.y <= cube.center.y + cube.half_size.y &&
            point.z >= cube.center.z - cube.half_size.z && point.z <= cube.center.z + cube.half_size.z);
}
__device__
bool cubeIntersectsNode(const Cube&  node, const Cube& cube) {
        if (abs(node.center.x - cube.center.x) > (node.half_size.x + cube.half_size.x)) {
            return false;
        }
        // Verificar separação no eixo Y
        if (abs(node.center.y - cube.center.y) > (node.half_size.y + cube.half_size.y)) {
            return false;
        }
        // Verificar separação no eixo Z
        if (abs(node.center.z - cube.center.z) > (node.half_size.z + cube.half_size.z)) {
            return false;
        }
        
        // Não há separação em nenhum dos eixos, os cubos se intersectam
        return true;
    }

__device__ int searchIntersectingNodes(OctreeNodeD* nodes, const glm::vec3& origin, const glm::vec3& direction, octnode *octStack,const glm::mat4& viewMatrix, const int& level) {
    

    int stack[256]; // Pilha para armazenar os índices dos nós a serem processados
    int stackSize = 0;
    stack[stackSize++] = 0; // Começa pelo nó root
    int count = 0;

    while (stackSize > 0) {
        int nodeIndex = stack[--stackSize];
        OctreeNodeD node = nodes[nodeIndex];

        if (level == 0 && node.numCubes_0 > 0) {
            octnode n;
            n.index = nodeIndex;
            Cube boundary = applyCenterTransformation(node.boundary, viewMatrix);
            n.z = boundary.center.z;
            // n.type = 0;
            insert(n, octStack, count);
        }
        else {
            if (level == 1 && node.numCubes_1 > 0) {
                octnode n;
                n.index = nodeIndex;
                Cube boundary = applyCenterTransformation(node.boundary, viewMatrix);
                n.z = boundary.center.z;
                // n.type = 1;
                insert(n, octStack, count);
            }
        }
        

        // Adiciona os filhos à pilha
        for (int i = 0; i < 8; ++i) {
            int nchild = node.children[i];
            if ( nchild != -1) {
                OctreeNodeD child = nodes[nchild];
                if ((child.branchCubes > 0) && rayIntersectsCube1(child.boundary, origin, direction)) {
            
                    stack[stackSize++] = nchild;
                }
            }
        }
    }

    return count;
}

__device__ int searchIntersectingNodes(OctreeNodeD* nodes, const glm::vec3& origin, const glm::vec3& direction, int *results, int level) {
    

    int stack[256]; // Pilha para armazenar os índices dos nós a serem processados
    int stackSize = 0;
    stack[stackSize++] = 0; // Começa pelo nó root
    int count = 0;

    while (stackSize > 0 && count < MAX_RESULTS) {
        int nodeIndex = stack[--stackSize];
        OctreeNodeD node = nodes[nodeIndex];

        if (level == 0 && node.numCubes_0 > 0) {
            results[count++] = nodeIndex;
        }

        if (level == 1 && node.numCubes_1 > 0) {
            results[count++] = nodeIndex;
        }

        // Adiciona os filhos à pilha
        for (int i = 0; i < 8; ++i) {
            int nchild = node.children[i];
            if ( nchild != -1) {
                OctreeNodeD child = nodes[nchild];
                if ((child.branchCubes > 0) && rayIntersectsCube1(child.boundary, origin, direction)) {
            
                    stack[stackSize++] = nchild;
                }
            }
        }
    }

    return count;
}


__device__ int searchInsideNode(OctreeNodeD* nodes, const glm::vec3& origin, int *results, int level) {

    int stack[256]; // Pilha para armazenar os índices dos nós a serem processados
    int stackSize = 0;
    stack[stackSize++] = 0; // Começa pelo nó root
    int count = 0;

    auto size = glm::vec3(0.2f, 0.2f, 0.2f);
    Cube cube(origin, size, -1);

    while (stackSize > 0 && count < MAX_RESULTS) {
        int nodeIndex = stack[--stackSize];
        OctreeNodeD node = nodes[nodeIndex];

        if (level == 0 && node.numCubes_0 > 0) {
            results[count++] = nodeIndex;
        }

        if (level == 1 && node.numCubes_1 > 0) {
            results[count++] = nodeIndex;
        }

        // Adiciona os filhos à pilha
        for (int i = 0; i < 8; ++i) {
            int nchild = node.children[i];
            if ( nchild != -1) {
                OctreeNodeD child = nodes[nchild];
                if ((child.branchCubes > 0) && cubeIntersectsNode(cube, child.boundary)) {
            
                    stack[stackSize++] = nchild;
                }
            }
        }
    }

    return count;
}