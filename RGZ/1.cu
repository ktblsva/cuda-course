#include "device_launch_parameters.h"
#include <cufft.h>
#include <stdio.h>
#include <cuda.h>
#include <malloc.h>
#include <string.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cctype>
#include <list>
#include <stdlib.h>
#include <ctime>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>


#include <cublas.h>
#include <cublas_v2.h>

#define CUDA_CHECK_RETURN(value)                                               \
  {                                                                            \
    cudaError_t _m_cudaStat = value;                                           \
    if (_m_cudaStat != cudaSuccess) {                                          \
      fprintf(stderr, "Error %s at line %d in file %s\n",                      \
              cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);            \
      exit(1);                                                                 \
    }                                                                          \
  }                                                              \


#define NX 64
#define BATCH 1
#define pi 3.141592
#define SZ (1<<23)
#define ALPHA 2.0f

using namespace std;


__global__ void cusaxpy(int n, float a, float *x, float *y)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) y[i] = a*x[i] + y[i];
}


struct functor {
        const float koef;
        functor(float _koef) : koef(_koef) {}
        __host__ __device__ float operator()(float x, float y) { return koef * x + y; }
};

void saxpy(float _koef, thrust::device_vector<float> &x, thrust::device_vector<float> &y)
{
        functor func(_koef);
        thrust::transform(x.begin(), x.end(), y.begin(), y.begin(), func);

}


int thrustFunc(float* a, int N) {
	cudaEvent_t start, stop;
	float time;
	printf("here!\n");
	thrust::host_vector<float> h1(N);
       	thrust::host_vector<float> h2(N);
        float alpha = ALPHA;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        for(int k = 0; k < N; k++){
                h1[k] = a[k];
                h2[k] = a[k];
                	//printf("%g ", h1[k]);
        }
        //printf("\n");
        thrust::device_vector<float> gpumem1 = h1;
        thrust::device_vector<float> gpumem2 = h2;
        printf("after gpu!\n");
        cudaEventRecord(start, 0);
        saxpy(alpha, gpumem1, gpumem2);
  	cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        printf("after saxpy!\n");
        h1 = gpumem1;
        h2 = gpumem2;
        for(int k = 0; k < N; k++) {
        	if(h2[k] == h1[k]*ALPHA + h1[k]) {
        		continue;
        	} else {
        		printf("Thrust: wrong answer!\n");
        		return -1;
        	}
        	printf("%g\t %g\n", h1[k], h2[k]);
        }
        printf("Thrust: correct answer!\n");
        
        printf("Thrust time: %g ms \n", time);
        return 0;
}

int cublasFunc(float* a, int N) {
	cudaEvent_t start, stop;
	float time;
	cublasHandle_t handle;
        cublasCreate(&handle);
        float *res = new float[N];
        float *dev_x, *dev_y;
        cudaMalloc(&dev_x, N * sizeof(float));
	cudaMalloc(&dev_y, N * sizeof(float));
	cudaEventCreate(&start);
        cudaEventCreate(&stop);
	cublasInit();
	cublasSetVector(N, sizeof(a[0]), a, 1, dev_x, 1);
        cublasSetVector(N, sizeof(a[0]), a, 1, dev_y, 1);
        float alpha = ALPHA;
	cudaEventRecord(start, 0);
	cublasSaxpy(handle, N, &alpha, dev_x, 1, dev_y, 1);
	cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	cublasGetVector(N, sizeof(res[0]), dev_y, 1, res, 1);
	cublasShutdown();
	for(int k = 0; k < N; k++) {
        	if(res[k] == a[k]*ALPHA + a[k]) {
        		continue;
        	} else {
        		printf("cuBLAS: wrong answer!\n");
        		return -1;
        	}
        	printf("%g\t %g\n", a[k], res[k]);
        }
        printf("cuBLAS: correct answer!\n");
        
        printf("cuBLAS time: %g ms \n", time);
	cublasDestroy(handle);
	cudaFree(dev_x);
	cudaFree(dev_y);
	cudaFreeHost(res);
	cudaDeviceReset();
	return 0;
}

int cudaFunc(float* a, int N) {
	cudaEvent_t start, stop;
	float time;
	float alpha = ALPHA;        
        cudaEventCreate(&start); 
        cudaEventCreate(&stop);
	float *x, *y, *d_x, *d_y;
	x = (float*)malloc(N*sizeof(float));
	y = (float*)malloc(N*sizeof(float));
	cudaMalloc(&d_x, N*sizeof(float));
	cudaMalloc(&d_y, N*sizeof(float));
	for (int k = 0; k < N; k++) {
		x[k] = k;
		y[k] = k;
	}
	cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaEventRecord(start,0);
	cusaxpy<<<(N)/256, 256>>>(N, alpha, d_x, d_y);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
	cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);
   
	for(int k = 0; k < N; k++) {
        	if(y[k] == a[k]*ALPHA + a[k]) {
        		continue;
        	} else {
        		printf("cuda C: wrong answer!\n");
        		return -1;
        	}
        	//printf("%g\t %g\n", a[k], y[k]);
        }
        cudaFree(d_x);
	cudaFree(d_y);
	free(x);
	free(y);
        printf("cuda C: correct answer!\n");
        printf("cuda C time: %g ms \n", time);
        return 0;
}


int main() {
        srand(time(nullptr));
	float *a = (float*)calloc(SZ, sizeof(float));
    
	for (int i = 0; i < SZ; i++) {
		a[i] = i;
	}

	ofstream os("N.dat");
	ofstream th("threads.dat");
	ofstream os1("dataThrust.dat");
	ofstream os2("dataCublas.dat");
	ofstream os3("dataCuda.dat");

    	if(!os1.is_open()){
        	cout << "Error" << endl;
        	return -1;
   	}
    	if(!os2.is_open()){
                cout << "Error" << endl;
                return -1;
        }
	if(!os3.is_open()){
                cout << "Error" << endl;
                return -1;
        }

	int i = 256;
       	for (int j = 10; j <= 23; j++) {
		//int N = 1 << j; 
		int N = 2048;
		printf("N = %d\n", N);
		if (thrustFunc(a, N) != 0) {
			return -1;
		}
		if (cublasFunc(a, N) != 0) {
			return -1;
		}
		cudaFunc(a, N);
	
	}

	os1.close();
	os2.close();
	os3.close();
	return 0;
}