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

__global__ void add(int*a, int *b, int *c, int N) {
int i = threadIdx.x + blockIdx.x * blockDim.x;
  if(i >= N) return;
  c[i] = a[i] + b[i];
}

void init(int*a, int *b, int *c, int N)
{
    for(int k = 0; k < N; k++)
    {
      a[k] = k;
      b[k] = k;
      c[k] = 0;
    }
}

int main(int argc,char* argv[]) 
{ 
    float elapsedTime;
    int N = 0;
    int *dev_a, *dev_b, *dev_c, *c, *a, *b;

    if(argc == 3)
    {
         int N = atoi(argv[1]);
         int th = atoi(argv[2]);
         cudaEvent_t start, stop;          
         cudaEventCreate(&start); 
         cudaEventCreate(&stop);

         a = (int*)calloc(N, sizeof(int));
         b = (int*)calloc(N, sizeof(int));
         c = (int*)calloc(N, sizeof(int));
         init(a, b, c, N);
         CUDA_CHECK_RETURN(cudaMalloc((void **)&dev_a, N * sizeof(int)));
         CUDA_CHECK_RETURN(cudaMalloc((void **)&dev_b, N * sizeof(int)));
         CUDA_CHECK_RETURN(cudaMalloc((void **)&dev_c, N * sizeof(int)));

         cudaEventRecord(start, 0); 
         add<<<N / th, th>>>(dev_a, dev_b, dev_c, N);
         cudaEventRecord(stop, 0); 
         cudaEventSynchronize(stop); 

         CUDA_CHECK_RETURN(cudaGetLastError());
         cudaEventElapsedTime(&elapsedTime, start, stop); 
         CUDA_CHECK_RETURN(cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));
         fprintf(stderr, "%d: %.6f ms\n", N, elapsedTime);
         cudaEventDestroy(start); 
         cudaEventDestroy(stop);  
         free(c);
         cudaFree(dev_a);
         cudaFree(dev_b);
         cudaFree(dev_c);
         cout << endl;    	
    }
    else if(argc == 1){
      for (int k = 1 << 0; k <= 1 << 10; k = k << 1) {
        fprintf(stderr, "Threads per block(%i):\n", k);
        for (int j = 10; j <= 23; j++) {
          cudaEvent_t start, stop; 
          cudaEventCreate(&start);
          cudaEventCreate(&stop); 
          N = 1 << j;

          a = (int*)calloc(N, sizeof(int));
          b = (int*)calloc(N, sizeof(int));
          c = (int*)calloc(N, sizeof(int));
          init(a, b, c, N);
          CUDA_CHECK_RETURN(cudaMalloc((void **)&dev_a, N * sizeof(int)));
          CUDA_CHECK_RETURN(cudaMalloc((void **)&dev_b, N * sizeof(int)));
          CUDA_CHECK_RETURN(cudaMalloc((void **)&dev_c, N * sizeof(int)));

          cudaEventRecord(start, 0); 
          add<<<N / k, k>>>(dev_a, dev_b, dev_c, N);
          cudaEventRecord(stop, 0); 
          cudaEventSynchronize(stop); 

          CUDA_CHECK_RETURN(cudaGetLastError());
          cudaEventElapsedTime(&elapsedTime, start, stop);
          CUDA_CHECK_RETURN(cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));
          fprintf(stderr, " %d: %.6f ms\n", N, elapsedTime);
          cudaEventDestroy(start); 
          cudaEventDestroy(stop);  
          free(c);
          cudaFree(dev_a);
          cudaFree(dev_b);
          cudaFree(dev_c);
        }
        cout << endl;
      }
    }
    return 0;
}
