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

void Output(float *a, int N) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++)
      fprintf(stdout, "%g\t", a[j + i * N]);
    fprintf(stdout, "\n");
  }
  fprintf(stdout, "\n\n\n");
}



__global__ void gTrans(float* matrix ,float* matrixT) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int N = blockDim.x*gridDim.x;

  matrixT[j+i*N] = matrix[i+j*N];

}


__global__ void gInit(float* matrix) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int N = blockDim.x*gridDim.x;

  matrix[i+j*N] = (float)(i+j*N);

}

__global__ void gInit1(float* matrix) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int N = blockDim.y*gridDim.y;

  matrix[j+i*N] = (float)(j+i*N);

}



int main(int argc,char* argv[]) {
  float elapsedTime;
  int N = 8192;
  int dimB = 32;
  int dimG = N/dimB;
  float *dev_matrix,*dev_matrixT,*dev_matrix1,*matrix,*matrixT,*matrix1;
  cudaEvent_t start, stop;
  cudaEventCreate(&start); // инициализация
  cudaEventCreate(&stop); 

  matrix = (float *)calloc(N*N, sizeof(float));
  matrix1 = (float *)calloc(N*N, sizeof(float));
  matrixT = (float *)calloc(N*N, sizeof(float));
  CUDA_CHECK_RETURN(cudaMalloc((void **)&dev_matrix, N*N * sizeof(float)));
  CUDA_CHECK_RETURN(cudaMalloc((void **)&dev_matrix1, N*N * sizeof(float)));
  CUDA_CHECK_RETURN(cudaMalloc((void **)&dev_matrixT, N*N * sizeof(float)));

  gInit<<<dim3(dimG,dimG),dim3(dimB,dimB)>>>(dev_matrix);
  cudaDeviceSynchronize();
  CUDA_CHECK_RETURN(cudaMemcpy(matrix, dev_matrix, N * N * sizeof(float), cudaMemcpyDeviceToHost));
  
  gInit1<<<dim3(dimG,dimG),dim3(dimB,dimB)>>>(dev_matrix1);
  cudaDeviceSynchronize();
  CUDA_CHECK_RETURN(cudaMemcpy(matrix1, dev_matrix1, N * N * sizeof(float), cudaMemcpyDeviceToHost));

  cudaEventRecord(start, 0); // привязка (регистрация) события start
  gTrans<<<dim3(dimG,dimG),dim3(dimB,dimB)>>>(dev_matrix,dev_matrixT);
  cudaEventRecord(stop, 0); // привязка события stop
  cudaEventSynchronize(stop); // синхронизация по событию
  CUDA_CHECK_RETURN(cudaMemcpy(matrixT, dev_matrixT, N * N * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK_RETURN(cudaGetLastError());
  cudaEventElapsedTime(&elapsedTime, start,stop);
  printf("gTranspose took %gms\n", elapsedTime);

  cudaEventDestroy(start); // освобождение
  cudaEventDestroy(stop);  // памяти
  free(matrix);
  free(matrixT);
  cudaFree(dev_matrix);
  cudaFree(dev_matrixT);
  return 0;
}
    
