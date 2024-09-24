#include <bits/stdc++.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <malloc.h>
#include <time.h>

__global__ void addVec(int* a, int* b, int* c, int N){
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i >= N) return;
	c[i] = a[i] + b[i];
}

void addVectors(int N, int j)
{
	int *a, *b, *c, *d, *cuA, *cuB, *cuC;
	//timespec start, end;
	a = (int*)calloc(N, sizeof(int));
	b = (int*)calloc(N, sizeof(int));
	c = (int*)calloc(N, sizeof(int));
	d = (int*)calloc(N, sizeof(int));
	cudaMalloc((void**)&cuA, N*sizeof(int));
	cudaMalloc((void**)&cuB, N*sizeof(int));
	cudaMalloc((void**)&cuC, N*sizeof(int));

	for(int k = 0; k < N; k++)
	{
		a[k] = k;
		b[k] = k;
		c[k] = 0;
		d[k] = k + k;
	}

	cudaMemcpy(cuA, a, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(cuB, b, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(cuC, c, N * sizeof(int), cudaMemcpyHostToDevice);
	//clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	auto start = std::chrono::high_resolution_clock::now();
	addVec << <N / j, j >> > (cuA, cuB, cuC, N);
	cudaDeviceSynchronize();
	auto elapsed = std::chrono::high_resolution_clock::now() - start;
	//clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
	//double time = (double)(end.tv_nsec-start.tv_nsec);
	//printf("%0.9f\n", time);
	cudaMemcpy(c, cuC, N * sizeof(int), cudaMemcpyDeviceToHost);

			/*for(int k = 0; k < N; k++)
			{
				if(c[k] != d[k])
				{
					printf("Wrong!\n");
					free(a);
					free(b);
					free(c);
					cudaFree(cuA);
					cudaFree(cuB);
					cudaFree(cuC);
					return;
				}
			}
			*/
	std::cout << N << " size  \t\t"<< N/j << " blocks \t\t" << std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count() << " ns" <<std::endl;
	free(a);
	free(b);
	free(c);
	cudaFree(cuA);
	cudaFree(cuB);
	cudaFree(cuC);
}

void changeThreads()
{
  	int N = 0;
  	for(int j = 1<<0; j <= 1<<10; j<<=1)
	{
		std::cout << "Threads per block " << j << std::endl;
		for(int i = 10; i <= 23; i++)
		{
			N = 1 << i;
			addVectors(N, j);
		}
	}
}

int main(int argc, char* argv[])
{
  	if (argc == 1)
  	{
  		changeThreads();
  	}
  	else if(argc == 3)
  	{
  		int N = std::stoi(argv[1]);
  		int threads = std::stoi(argv[2]);
  		addVectors(N, threads);
  	}
	
	return 0;
}