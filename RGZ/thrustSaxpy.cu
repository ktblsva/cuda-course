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
#define ALPHA 2.0f
#define SZ (1<<23)

using namespace std;

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

int main(void)
{
	float *a = (float*)calloc(SZ, sizeof(float));
    ofstream os("N.dat");
	ofstream os1("dataThrust.dat");

	for (int i = 0; i < SZ; i++) {
		a[i] = i;
	}
	for(int i = 10; i <= 23; i++) {
		int N = 1 << i;
		printf("N = %d\n", N);
		os << N << endl;
		cudaEvent_t start, stop;
		float time;
		thrust::host_vector<float> h1(N);
	    thrust::host_vector<float> h2(N);
	    float alpha = ALPHA;
	    cudaEventCreate(&start);
	    cudaEventCreate(&stop);
	    for(int k = 0; k < N; k++){
	            h1[k] = a[k];
	            h2[k] = a[k];
	    }
	     
	    thrust::device_vector<float> gpumem1 = h1;
	    thrust::device_vector<float> gpumem2 = h2;
	    cudaEventRecord(start, 0);
	    saxpy(alpha, gpumem1, gpumem2);
	  	cudaEventRecord(stop, 0);
	    cudaEventSynchronize(stop);
	    cudaEventElapsedTime(&time, start, stop);
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
	    os1 << time << endl;
        
    }
}
