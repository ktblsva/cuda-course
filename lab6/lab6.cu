#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_CHECK_RETURN(value) {\
cudaError_t _m_cudaStat = value;\
if (_m_cudaStat != cudaSuccess) {\
	fprintf(stderr, "Error %s at line %d in file %s\n",\
	cudaGetErrorString(_m_cudaStat),__LINE__,__FILE__);\
	exit(1);\
}}


void allocTest(int size, bool hostToDevice, float &elapsed_time)
{
	cudaEvent_t start, stop;
	int *a, *dev_a;

	CUDA_CHECK_RETURN(cudaEventCreate(&start));
	CUDA_CHECK_RETURN(cudaEventCreate(&stop));

	CUDA_CHECK_RETURN(cudaHostAlloc((void**)&a, size * sizeof(*a), cudaHostAllocDefault));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_a, size * sizeof(*dev_a)));

	for (int i = 0; i < size; i++)
    	a[i] = i;

	CUDA_CHECK_RETURN(cudaEventRecord(start, 0));
	if (hostToDevice == true) {
		CUDA_CHECK_RETURN(cudaMemcpy(dev_a, a, size * sizeof(*a), cudaMemcpyHostToDevice));
	}
	else {
		CUDA_CHECK_RETURN(cudaMemcpy(a, dev_a, size * sizeof(*dev_a), cudaMemcpyDeviceToHost));
	}

	CUDA_CHECK_RETURN(cudaEventRecord(stop, 0));
	CUDA_CHECK_RETURN(cudaEventSynchronize(stop));
	CUDA_CHECK_RETURN(cudaEventElapsedTime(&elapsed_time, start, stop));

	CUDA_CHECK_RETURN(cudaFreeHost(a));
	CUDA_CHECK_RETURN(cudaFree(dev_a));
	CUDA_CHECK_RETURN(cudaEventDestroy(start));
	CUDA_CHECK_RETURN(cudaEventDestroy(stop));
}

void mallocTest(int size, bool hostToDevice, float& elapsed_time)
{
	cudaEvent_t start, stop;
	int *a, *dev_a;

	CUDA_CHECK_RETURN(cudaEventCreate(&start));
	CUDA_CHECK_RETURN(cudaEventCreate(&stop));

	a = (int*)malloc(size * sizeof(int));
	for (int i = 0; i < size; i++)
      	a[i] = i;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_a, size * sizeof(*dev_a)));

	CUDA_CHECK_RETURN(cudaEventRecord(start, 0));
	if (hostToDevice == true) {
		CUDA_CHECK_RETURN(cudaMemcpy(dev_a, a, size * sizeof(*a), cudaMemcpyHostToDevice));
	}
	else {
		CUDA_CHECK_RETURN(cudaMemcpy(a, dev_a, size * sizeof(*dev_a), cudaMemcpyDeviceToHost));
	}

	CUDA_CHECK_RETURN(cudaEventRecord(stop, 0));
	CUDA_CHECK_RETURN(cudaEventSynchronize(stop));
	CUDA_CHECK_RETURN(cudaEventElapsedTime(&elapsed_time, start, stop));

	free(a);
	CUDA_CHECK_RETURN(cudaFree(dev_a));
	CUDA_CHECK_RETURN(cudaEventDestroy(start));
	CUDA_CHECK_RETURN(cudaEventDestroy(stop));

}

int main(int argc, char const *argv[])
{
	cudaDeviceProp deviceProp;
  	cudaGetDeviceProperties(&deviceProp, 0);
  	printf("\nDevice:\t%s\n\n", deviceProp.name);
  	int size = 0;
  	if (argc == 2) {
    	size = 1 << atoi(argv[1]);
  	} else {
    	size = 1 << 13;
  	}
 	size *= size;
	float elapsed_time;
	printf("Size = %d\n\n", size);
	mallocTest(size, true, elapsed_time);
	printf("cudaMalloc [host to device]: %.6f ms\n", elapsed_time);

	mallocTest(size, false, elapsed_time);
	printf("cudaMalloc [device to host]: %.6f ms\n", elapsed_time);
	

	allocTest(size, true, elapsed_time);
	printf("cudaHostAlloc [host to device]: %.6f ms\n", elapsed_time);
	
	allocTest(size, false, elapsed_time);
	printf("cudaHostAlloc [device to host]: %.6f ms\n", elapsed_time);

	return 0;
}