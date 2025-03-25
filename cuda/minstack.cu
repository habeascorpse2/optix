#pragma once
#include "whitted.h"
#define OCTNODE_SIZE 25

namespace minstack {

    
    struct octnode{
        float z;
        int index;
        // unsigned char type;
    };
    //Não alterar
    __device__ void heapifyUp(octnode* dtree, int &size,int index, octnode value) {

        //Verifica se o que será adicionado é menor que o último
        //Se for menor, ele substituirá o último
        if (size == OCTNODE_SIZE) {
          if (dtree[size - 1].z > value.z) {
            dtree[size - 1] = value;
            return;
          }
          else 
            return;
        
        }
        else {
            dtree[size] = value;
            size++;
        }

        while (index > 0) {
            
            int parent = (index - 1) / 2;
            // if (dtree[index].index == dtree[parent].index)
            //     break;
            if (dtree[index].z < dtree[parent].z) {
                octnode d3 = dtree[parent];
                dtree[parent] = dtree[index];
                dtree[index] = d3;
                index = parent;
            } else {
                break;
            }
        }
    }

    __device__ void heapifyDown(int index, octnode* dtree, int& size) {
        while (true) {
            int leftChild = 2 * index + 1;
            int rightChild = 2 * index + 2;
            int smalest = index;

            if (leftChild < size && dtree[leftChild].z < dtree[smalest].z) {
                smalest = leftChild;
            }

            if (rightChild < size && dtree[rightChild].z < dtree[smalest].z) {
                smalest = rightChild;
            }

            if (smalest != index) {
            octnode d3 = dtree[smalest];
                dtree[smalest] = dtree[index];
                dtree[index] = d3;
                index = smalest;
            } else {
                break;
            }
        }
    }


    // Max Stack GSM: Gaussian Max Stack
    __device__ void insert(octnode value, octnode* dtree, int& size) {
        heapifyUp(dtree,size, size, value);
    }



    __device__ void removeMin(octnode* dtree, int& size) {
        if (size > 0) {
            dtree[0] = dtree[size - 1];
            size--;
            heapifyDown(0, dtree, size);
        }
    }
}


__forceinline__ __device__ void GSM_heapifyUp(whitted::DepthGaussian* dtree, int &size,int index, whitted::DepthGaussian value) {

    //Verifica se o que será adicionado é menor que o último
    //Se for menor, ele substituirá o último
    if (size == whitted::GSM_MAX_SIZE) {
      if (dtree[size - 1].z > value.z) {// || dtree[size -1].index) {
        dtree[size - 1] = value;
        return;
      }
      else 
        return;
      
    }
    else {
      dtree[size] = value;
      size++;
    }

    while (index > 0) {
        
        int parent = (index - 1) / 2;
        if (dtree[index].z < dtree[parent].z) {
            whitted::DepthGaussian d3 = dtree[parent];
            dtree[parent] = dtree[index];
            dtree[index] = d3;
            index = parent;
        } else {
            break;
        }
    }
}

__forceinline__ __device__ void GSM_heapifyDown(int index, whitted::DepthGaussian* dtree, int& size) {
    while (true) {
        int leftChild = 2 * index + 1;
        int rightChild = 2 * index + 2;
        int smalest = index;

        if (leftChild < size && dtree[leftChild].z < dtree[smalest].z) {
            smalest = leftChild;
        }

        if (rightChild < size && dtree[rightChild].z < dtree[smalest].z) {
            smalest = rightChild;
        }

        if (smalest != index) {
          whitted::DepthGaussian d3 = dtree[smalest];
            dtree[smalest] = dtree[index];
            dtree[index] = d3;
            index = smalest;
        } else {
            break;
        }
    }
}


// Max Stack GSM: Gaussian Max Stack
__forceinline__ __device__ void GSM_insert(whitted::DepthGaussian value, whitted::DepthGaussian* dtree, int& size) {
    GSM_heapifyUp(dtree,size, size, value);

}



__forceinline__ __device__ void GSM_removeMin(whitted::DepthGaussian* dtree, int& size) {
    if (size > 0) {
        dtree[0] = dtree[size - 1];
        size--;
        GSM_heapifyDown(0, dtree, size);
    }
}