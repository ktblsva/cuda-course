#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <stdio.h>
#include <fstream>
#include <string.h>

using namespace std;

#define step 256
#define N 1024
#define Nt 20
#define ALPHA 0.2
#define T 2
struct functor
{
    const float koef;
    functor(float _koef) : koef(_koef){}
    __host__ __device__ float operator()(float x, float y)
    {
        return x + koef * (y - x);
    }
};


__global__ void kernel(float *f, float *res)
{
    int cur = threadIdx.x + blockDim.x * blockIdx.x;
    int prev = cur - 1;
    if(prev == -1)
    {
        res[cur] = f[cur];
    }else
    {
        res[cur] = f[cur] + ALPHA * T * (f[prev] - f[cur]);
    }
}

int data(int curr)
{
	// if(curr < 4 || curr >= 8 && curr < 12) return 0;
	// else return 1;
	return (curr / step) % 2 == 1;
}

int main()
{
	float funct[N * Nt];
	float funcData[N * Nt];
	float *temp;

	cudaEvent_t start, stop;
	float time;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	for(int i = 0; i < N; i++) {
       for(int j = 0; j < Nt; j++) {
                funcData[i + j * Nt] = 0;
        }
    }
     for(int i = 0; i < N; i++) {
        funcData[i + 0 * Nt] = data(i);
        printf("%0.2f ", funcData[i + 0 * Nt]);
    }
	printf("\n\n");
	cudaMalloc((void **)&temp, sizeof(float) * N * Nt);
	cudaMemcpy(temp, funcData, sizeof(float) * N * Nt, cudaMemcpyHostToDevice);

	cudaEventSynchronize(start);
	cudaEventRecord(start, 0);
	for (int i = 0; i < Nt; i++) {
		kernel <<< 1, N >>> (temp + (i * N), temp + ((i + 1)* N));
		cudaDeviceSynchronize();
	}
	cudaDeviceSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cudaMemcpy(funct, temp, N * Nt * sizeof(float), cudaMemcpyDeviceToHost);
	printf("cuda: %f ms\n", time);
	ofstream of("x.txt");

	for(int i = 0; i < Nt; i++) {
		printf("f[%d] ", i);
	}
	for(int i = 0; i < N; i++) {
		of << i << endl;
	}
	printf("\n");
 	for(int i = 0; i < N; i++) {
       for(int j = 0; j < Nt; j++) {
            printf("%0.2f ", funct[i + j * Nt]);
        }
        printf("\n");
    }
    for(int i = 0; i < Nt; i++)
    {
    	string num("y" + to_string(i) + ".txt");
    	ofstream y(num);
    	for(int j = 0; j < N; j++)
    	{
    		y << funct [j + i * N]<< endl;
    	}
    }

	thrust::host_vector<float> vect(N * 10);

	for (int i = 0; i < N; i++)
	{
		vect[i] = funcData[i];
	}
	thrust::device_vector<float> x(N * 10);
	thrust::copy(vect.begin(), vect.end(),x.begin());
	cudaEventSynchronize(start);
	cudaEventRecord(start, 0);
	functor func(ALPHA * T);
	for(int j = 0; j < Nt; j++){
        thrust::transform(x.begin()+(j*N)+1, x.begin()+((j+1)*N), x.begin()+(j*N), x.begin() +((j+1)*N)+1, func);
    }
    thrust::copy(x.begin(),x.end(),vect.begin());
	cudaDeviceSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	std::cout << std::endl;
	printf("thrust: %f ms\n", time);
	for(int i = 0; i < Nt; i++)
	{
		printf("f[%d] ", i);
	}
	printf("\n");
 	for(int i = 0; i < N; i++){
       for(int j = 0; j < Nt; j++){
                printf("%0.2f ", vect[i + j * Nt]);
        }
        printf("\n");
    }
    system("python3 1.py");
    return 0;
}