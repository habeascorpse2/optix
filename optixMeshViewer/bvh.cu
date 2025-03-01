#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include "minstack.cu"
#include "whitted.h"

using namespace minstack;


namespace BVH {

    # define CUDA_SAFE_CALL_ALWAYS(A) \ 
    A; \
    cudaDeviceSynchronize(); \
    if (cudaPeekAtLastError() != cudaSuccess) \
        std::cerr << cudaGetErrorString(cudaGetLastError()) << std::endl;

    // Função para verificar erros CUDA
    void checkCudaError(cudaError_t err, const char* msg) {
        if (err != cudaSuccess) {
            std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
            exit(EXIT_FAILURE);
        }

    }
    

    struct BVHNode {
        float3 bboxMin;
        float3 bboxMax;
        int leftChild;
        int rightChild;
        int numCubes;
        bool isLeaf;
        int cubeIndex; // -1 if not a leaf node
        std::vector<Cube> cubes;
    };

    class BVH {
    public:
        std::vector<BVHNode> nodes;
        std::vector<BVHNodeD*> dnodes;
        std::vector<Cube> cubes;
        int numNodes;

        BVH(const std::vector<Cube>& cubes_host) {
            cubes = std::vector<Cube>(cubes_host.begin(), cubes_host.end());
            nodes.resize(2 * cubes.size() - 1); // Maximum number of nodes
            buildBVH();
        }

        void buildBVH() {
            std::vector<BVHNode> nodes_host(nodes.size());
            buildBVHRecursive(nodes_host, 0, cubes.size(), 0);
            // thrust::copy(nodes_host.begin(), nodes_host.end(), nodes.begin());
        }

        __device__ bool intersect(const float3& rayOrigin, const float3& rayDirection, int* hitCubeIndex) const {
            bool hit = false;
            float tmin = 0.0f;
            float tmax = 1e20f;
            int nodeStack[64];
            int stackPtr = 0;
            nodeStack[stackPtr++] = 0;

            while (stackPtr > 0) {
                int nodeIndex = nodeStack[--stackPtr];
                const BVHNode& node = nodes[nodeIndex];

                if (!rayIntersectsAABB(rayOrigin, rayDirection, node.bboxMin, node.bboxMax, tmin, tmax)) {
                    continue;
                }

                if (node.cubeIndex != -1) { 
                    const Cube& cube = cubes[node.cubeIndex];
                    if (rayIntersectsCube(rayOrigin, rayDirection, cube)) {
                        *hitCubeIndex = cube.index;
                        hit = true;
                    }
                } else {
                    if (node.leftChild != -1) nodeStack[stackPtr++] = node.leftChild;
                    if (node.rightChild != -1) nodeStack[stackPtr++] = node.rightChild;
                }
            }

            return hit;
        }


        __host__ BVHNodeD* sendToGPU() {
        
            numNodes = nodes.size();
            BVHNodeD* h_data = new BVHNodeD[numNodes];


            for (int i = 0; i < numNodes; i++) {

                BVHNode* node = &nodes.at(i);

                h_data[i].leftChild = node->leftChild;
                h_data[i].rightChild = node->rightChild;

                h_data[i].bboxMax = node->bboxMax;
                h_data[i].bboxMin = node->bboxMin;
                h_data[i].numCubes = node->numCubes;
                h_data[i].cubes = nullptr;
                h_data[i].isLeaf = node->isLeaf;
                if (node->numCubes > 0) {
                    int *ccubes;
                    ccubes = (int *)malloc(node->numCubes * sizeof(int));
                    for (int j=0; j < node->numCubes; j++)
                        ccubes[j] = node->cubes[j].index;
                    h_data[i].cubes = ccubes;
                }
            }

            // Alocar memória na GPU para o array de OctreeNodeD
            BVHNodeD *d_nodes;
            checkCudaError(cudaMalloc(&d_nodes, numNodes * sizeof(BVHNodeD)), "Allocating d_nodes");

            // Copiar dados do array principal (exceto os ponteiros cubes) para a GPU
            checkCudaError(cudaMemcpy(d_nodes, h_data, numNodes * sizeof(OctreeNodeD), cudaMemcpyHostToDevice), "Copying h_nodes to d_nodes");

            // Para cada nó, alocar memória na GPU para o array cubes e copiar os dados
            for (int i = 0; i < numNodes; ++i) {
                int *d_cubes;
                checkCudaError(cudaMalloc(&d_cubes, h_data[i].numCubes * sizeof(int)), "Allocating d_cubes");
                checkCudaError(cudaMemcpy(d_cubes, h_data[i].cubes, h_data[i].numCubes * sizeof(int), cudaMemcpyHostToDevice), "Copying cubes to d_cubes");

                // Atualizar o ponteiro cubes na GPU
                checkCudaError(cudaMemcpy(&d_nodes[i].cubes, &d_cubes, sizeof(int*), cudaMemcpyHostToDevice), "Updating d_nodes[i].cubes");
                delete h_data[i].cubes;
            }
            delete h_data;

            return d_nodes;
        }

    private:
        int buildBVHRecursive(std::vector<BVHNode>& nodes_host, int start, int end, int nodeIndex) {
            if (start == end - 1) {
                BVHNode& leaf = nodes_host[nodeIndex];
                leaf.bboxMin = cubes[start].center - cubes[start].half_size;
                leaf.bboxMax = cubes[start].center + cubes[start].half_size;
                leaf.cubeIndex = start;
                leaf.leftChild = -1;
                leaf.rightChild = -1;
                return nodeIndex;
            }

            int mid = (start + end) / 2;
            BVHNode& node = nodes_host[nodeIndex];
            node.leftChild = buildBVHRecursive(nodes_host, start, mid, nodeIndex + 1);
            node.rightChild = buildBVHRecursive(nodes_host, mid, end, node.leftChild + 1);

            node.bboxMin = fminf(nodes_host[node.leftChild].bboxMin, nodes_host[node.rightChild].bboxMin);
            node.bboxMax = fminf(nodes_host[node.leftChild].bboxMax, nodes_host[node.rightChild].bboxMax);
            node.cubeIndex = -1;

            return nodeIndex;
        }

        __device__ bool rayIntersectsCube(const float3& rayOrigin, const float3& rayDirection, const Cube& cube) const {
            float3 tMin = (cube.center - cube.half_size - rayOrigin) / rayDirection;
            float3 tMax = (cube.center + cube.half_size - rayOrigin) / rayDirection;
            float3 t1 = fminf(tMin, tMax);
            float3 t2 = fmaxf(tMin, tMax);

            float tNear = fmaxf(fmaxf(t1.x, t1.y), t1.z);
            float tFar = fminf(fminf(t2.x, t2.y), t2.z);

            return tNear <= tFar && tFar > 0.0f;
        }

        __device__ bool rayIntersectsAABB(const float3& rayOrigin, const float3& rayDirection, const float3& bboxMin, const float3& bboxMax, float& tmin, float& tmax) const {
            float3 invDir = 1.0f / rayDirection;
            float3 t0 = (bboxMin - rayOrigin) * invDir;
            float3 t1 = (bboxMax - rayOrigin) * invDir;

            float3 tmin3 = fminf(t0, t1);
            float3 tmax3 = fmaxf(t0, t1);

            tmin = fmaxf(tmin, fmaxf(fmaxf(tmin3.x, tmin3.y), tmin3.z));
            tmax = fminf(tmax, fminf(fminf(tmax3.x, tmax3.y), tmax3.z));

            return tmax > tmin;
        }
    };

}