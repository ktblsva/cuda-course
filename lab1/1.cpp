#include <bits/stdc++.h>
#include <stdio.h>
#include <malloc.h>
#include <time.h>

void addVec(int* a, int* b, int* c, int N){
	for(int i = 0; i < N; i++)
	{
		c[i] = a[i] + b[i];
	}
}

void addVectors(int N, int j)
{
	int *a, *b, *c, *d, *cuA, *cuB, *cuC;
	//timespec start, end;
	a = (int*)calloc(N, sizeof(int));
	b = (int*)calloc(N, sizeof(int));
	c = (int*)calloc(N, sizeof(int));
	d = (int*)calloc(N, sizeof(int));

	for(int k = 0; k < N; k++)
	{
		a[k] = k;
		b[k] = k;
		c[k] = 0;
		d[k] = k + k;
	}

	auto start = std::chrono::high_resolution_clock::now();
	addVec(a, b, c, N);
	auto elapsed = std::chrono::high_resolution_clock::now() - start;

	std::cout << N <<" \t"<< std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count() <<std::endl;
	free(a);
	free(b);
	free(c);
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