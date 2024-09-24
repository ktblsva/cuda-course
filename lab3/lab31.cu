#include <cuda.h>
#include <stdio.h>
#include <malloc.h>
#include <math.h>

#define onekk 1000000

#define CUDA_CHECK_RETURN(value) {\
    cudaError_t _m_cudaStat = value;\
    if (_m_cudaStat != cudaSuccess) {\
        fprintf(stderr, "Error %s at line %d in file %s\n",\
        cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);\
        exit(1);\
    }}

__global__ void initializing_vector(int *vector){
    int index = threadIdx.x + blockDim.x * blockIdx.x; 
    vector[index] = index;
}

int main(int argc, char *argv[]){

    int num_of_blocks = atoi(argv[1]); 
    int threads_per_block = atoi(argv[2]);
    int N = num_of_blocks * threads_per_block;

    printf("N = %d\n",N);
    printf("num_of_blocks = %d\n",num_of_blocks);
    printf("threads_per_block = %d\n",threads_per_block);


    int *a;
    int *a_gpu;

    a = (int*)malloc(N * sizeof(int));
    
    CUDA_CHECK_RETURN(cudaMalloc((void**)&a_gpu, N * sizeof(int)));
    
    initializing_vector <<< dim3(num_of_blocks), 
    dim3(threads_per_block) >>> (a_gpu);
    CUDA_CHECK_RETURN(cudaGetLastError());
    CUDA_CHECK_RETURN(cudaMemcpy(a, a_gpu, N * sizeof(int), cudaMemcpyDeviceToHost));

    free(a);
    CUDA_CHECK_RETURN(cudaFree(a_gpu));
}
