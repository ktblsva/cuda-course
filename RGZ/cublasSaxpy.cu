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

#include <cublas.h>
#include <cublas_v2.h>
#define NX 64
#define BATCH 1
#define pi 3.141592
#define SZ (1<<23)
#define ALPHA 2.0f
using namespace std;

int main() {
	float *a = (float*)calloc(SZ, sizeof(float));
	ofstream os1("dataCublas.dat");

	for (int i = 0; i < SZ; i++) {
		a[i] = i;
	}
	for(int i = 10; i <=23; i++) {
		int N = 1 << i;
		printf("N = %d\n", N);
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
	    os1 << time << endl;
		cublasDestroy(handle);
		cudaFree(dev_x);
		cudaFree(dev_y);
		cudaFreeHost(res);
		cudaDeviceReset();
	}
}