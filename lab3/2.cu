#include <cuda.h>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <malloc.h>
#include <stdio.h>

using namespace std;

#define CUDA_CHECK_RETURN(value)                                               \
  {                                                                            \
    cudaError_t _m_cudaStat = value;                                           \
    if (_m_cudaStat != cudaSuccess) {                                          \
      fprintf(stderr, "Error %s at line %d in file %s\n",                      \
              cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);            \
      exit(1);                                                                 \
    }                                                                          \
  }

__global__ void init(int *c, int N) {
int i = threadIdx.x + blockIdx.x * blockDim.x;
  if(i >= N) return;
  c[i] = 0;
}


int main(int argc,char* argv[]) 
{ 
    // char dev;
    // cudaSetDevice(dev); 
    // cudaDeviceProp deviceProp; 
    // cudaGetDeviceProperties(&deviceProp, dev);
    // printf("  Total amount of constant memory:  %lu bytes\n", deviceProp.totalConstMem); 
    // printf("  Total amount of shared memory per block: %lu bytes\n", deviceProp.sharedMemPerBlock);
    // printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock); 
    // printf("  Warp size: %d\n", deviceProp.warpSize); 
    // printf("  Maximum number of threads per multiprocessor:  %d\n", deviceProp.maxThreadsPerMultiProcessor); 
    // printf("  Maximum number of threads per block:  %d\n", deviceProp.maxThreadsPerBlock);
    float elapsedTime;
    int N = 0;
    int *dev_c, *c;
    N = atoi(argv[1]);
    int th = atoi(argv[2]);
    cudaEvent_t start, stop;          
    cudaEventCreate(&start); 
    cudaEventCreate(&stop);
    fprintf(stderr, "%d blocks\n", N / th);
    c = (int*)calloc(N, sizeof(int));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&dev_c, N * sizeof(int)));

    cudaEventRecord(start, 0); 
    init<<<N / th, th>>>(dev_c, N);
    cudaEventRecord(stop, 0); 
    cudaEventSynchronize(stop); 

    CUDA_CHECK_RETURN(cudaGetLastError());
    cudaEventElapsedTime(&elapsedTime, start, stop); 
    CUDA_CHECK_RETURN(cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));
    fprintf(stderr, "%d: %.6f ms\n", N, elapsedTime);
    cudaEventDestroy(start); 
    cudaEventDestroy(stop);  
    free(c);
    cudaFree(dev_c);
    cout << endl;    	
    return 0;
}
