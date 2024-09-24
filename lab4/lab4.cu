#include <cuda.h>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <malloc.h>
#include <stdio.h>

#define SH_DIM 64

using namespace std;

#define CUDA_CHECK_RETURN(value)                                               \
  {                                                                            \
    cudaError_t _m_cudaStat = value;                                           \
    if (_m_cudaStat != cudaSuccess) {                                          \
     fprintf(stderr,"Error %s at line %d in file %s\n",                               \
              cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);            \
      exit(1);                                                                 \
    }                                                                          \
  }

__global__ void gTrans3(float *matrix, float *matrixT) {
  __shared__ float buffer_s[SH_DIM][SH_DIM + 1];
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int N = blockDim.x * gridDim.x;

  buffer_s[threadIdx.y][threadIdx.x] = matrix[i + j * N];
  __syncthreads();

  i = threadIdx.x + blockIdx.y * blockDim.x;
  j = threadIdx.y + blockIdx.x * blockDim.y;
  matrixT[i + j * N] = buffer_s[threadIdx.x][threadIdx.y];
}

__global__ void gTrans2(float *matrix, float *matrixT) {
  __shared__ float buffer_s[SH_DIM][SH_DIM];
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int N = blockDim.x * gridDim.x;

  buffer_s[threadIdx.y][threadIdx.x] = matrix[i + j * N];
  __syncthreads();

  i = threadIdx.x + blockIdx.y * blockDim.x;
  j = threadIdx.y + blockIdx.x * blockDim.y;
  matrixT[i + j * N] = buffer_s[threadIdx.x][threadIdx.y];
}

__global__ void gTrans1(float *matrix, float *matrixT) {
  extern __shared__ float buffer[];
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int N = blockDim.x * gridDim.x;

  buffer[threadIdx.y + threadIdx.x * blockDim.y] = matrix[i + j * N];
  __syncthreads();

  i = threadIdx.x + blockIdx.y * blockDim.x;
  j = threadIdx.y + blockIdx.x * blockDim.y;
  matrixT[i + j * N] = buffer[threadIdx.x + threadIdx.y * blockDim.x];
}

__global__ void gTrans(float *matrix, float *matrixT) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int N = blockDim.x * gridDim.x;
  matrixT[j + i * N] = matrix[i + j * N];
}

__global__ void gInit(float *matrix) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int N = blockDim.x * gridDim.x;
  matrix[i + j * N] = (float)(i + j * N);
}

void Output(float *a, int N) {
  //for (int i = 0; i < N; i++) {
   // for (int j = 0; j < N; j++)
     // fprintf(stdout, "%g\t", a[j + i * N]);
    //fprintf(stdout, "\n");
  //}
  for(int i = 0; i < N; i++)
  {
   fprintf(stdout, "%g\t", a[i]);
  }
  fprintf(stdout, "\n\n\n");
}

int main(int argc, char *argv[]) {
  int N;
  int dimB;
  if (argc == 3) {
    N = atoi(argv[1]);
    dimB = atoi(argv[2]);
  } else {
    N = 1 << 10;
    dimB = 1 << 5;
  }
  float elapsedTime;
  int dimG = N / dimB;
  float *dev_matrix, *dev_matrixT,*dev_matrixT1,*dev_matrixT2,*dev_matrixT3,*matrix, *matrixT;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  matrix = (float *)calloc(N * N,sizeof(float));
  matrixT = (float *)calloc(N * N, sizeof(float));
  CUDA_CHECK_RETURN(cudaMalloc((void **)&dev_matrix, N * N * sizeof(float)));
  CUDA_CHECK_RETURN(cudaMalloc((void **)&dev_matrixT, N * N * sizeof(float)));
  CUDA_CHECK_RETURN(cudaMalloc((void **)&dev_matrixT1, N * N * sizeof(float)));
  CUDA_CHECK_RETURN(cudaMalloc((void **)&dev_matrixT2, N * N * sizeof(float)));
  CUDA_CHECK_RETURN(cudaMalloc((void **)&dev_matrixT3, N * N * sizeof(float)));
  gInit<<<dim3(dimG,dimG),dim3(dimB,dimB) >>>(dev_matrix);
  CUDA_CHECK_RETURN(cudaDeviceSynchronize());
  CUDA_CHECK_RETURN(cudaGetLastError());
  CUDA_CHECK_RETURN(cudaMemcpy(matrix, dev_matrix, N * N * sizeof(float), cudaMemcpyDeviceToHost));
  Output(matrix, 8);	
  		
  cudaEventRecord(start, 0);
  gTrans<<<dim3(dimG,dimG),dim3(dimB,dimB)>>>(dev_matrix, dev_matrixT);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  CUDA_CHECK_RETURN(cudaGetLastError());
  cudaEventElapsedTime(&elapsedTime, start,stop);
  printf("gTranspose0 took %gms\n", elapsedTime);
  CUDA_CHECK_RETURN(cudaMemcpy(matrixT, dev_matrixT, N * N * sizeof(float), cudaMemcpyDeviceToHost));
  Output(matrixT, 8);
  
  CUDA_CHECK_RETURN(cudaMemcpy(dev_matrix, matrix, N * N * sizeof(float), cudaMemcpyHostToDevice));
  cudaEventRecord(start, 0);
  gTrans1<<<dim3(dimG,dimG),dim3(dimB,dimB),dimB*dimB*sizeof(float)>>>(dev_matrix, dev_matrixT1);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  CUDA_CHECK_RETURN(cudaGetLastError());
  cudaEventElapsedTime(&elapsedTime, start,stop);
  printf("gTranspose1 took %gms\n", elapsedTime);
  CUDA_CHECK_RETURN(cudaMemcpy(matrixT, dev_matrixT1, N * N * sizeof(float), cudaMemcpyDeviceToHost));
  Output(matrixT, 8);
  
  CUDA_CHECK_RETURN(cudaMemcpy(dev_matrix, matrix, N * N * sizeof(float), cudaMemcpyHostToDevice));
  cudaEventRecord(start, 0);
  gTrans2<<<dim3(dimG,dimG),dim3(dimB,dimB)>>>(dev_matrix, dev_matrixT2);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  CUDA_CHECK_RETURN(cudaGetLastError());
  cudaEventElapsedTime(&elapsedTime, start,stop);
  printf("gTranspose2 took %gms\n", elapsedTime);
  CUDA_CHECK_RETURN(cudaMemcpy(matrixT, dev_matrixT2, N * N * sizeof(float), cudaMemcpyDeviceToHost));
  Output(matrixT, 8);
  
  CUDA_CHECK_RETURN(cudaMemcpy(dev_matrix, matrix, N * N * sizeof(float), cudaMemcpyHostToDevice));
  cudaEventRecord(start, 0);
  gTrans3<<<dim3(dimG,dimG),dim3(dimB,dimB)>>>(dev_matrix, dev_matrixT3);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start,stop);
  CUDA_CHECK_RETURN(cudaGetLastError());
  printf("gTranspose3 took %gms\n", elapsedTime);
  CUDA_CHECK_RETURN(cudaMemcpy(matrixT, dev_matrixT3, N * N * sizeof(float), cudaMemcpyDeviceToHost));
  Output(matrixT, 8);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  free(matrix);
  free(matrixT);
  cudaFree(dev_matrix);
  cudaFree(dev_matrixT);
  cudaFree(dev_matrixT1);
  cudaFree(dev_matrixT2);
  cudaFree(dev_matrixT3);
  return 0;
}

