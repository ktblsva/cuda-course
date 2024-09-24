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

#define ALPHA 2.0f
#define SZ (1<<23)
using namespace std;

__global__ void cusaxpy(int n, float a, float *x, float *y)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) y[i] = a*x[i] + y[i];
}


int main() {
	float *a = (float*)calloc(SZ, sizeof(float));
	ofstream os1("dataCuda.dat");

	for (int i = 0; i < SZ; i++) {
		a[i] = i;
	}
	for(int i = 10; i <= 23; i++) {
		int N = 1 << i;
		printf("N = %d\n", N);
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
	    os1 << time << endl;
    }
}