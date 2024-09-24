#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cufft.h>
#include <stdio.h>
#include <malloc.h>
#include <string.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <ctime>

#include <cublas.h>
#include <cublas_v2.h>


#define NX 64
#define BATCH 1
#define pi 3.141592
#define SZ (1<<25)
#define ALPHA 3.0f

using namespace std;


int main()
{
	string line;
	string buffer;
	cufftHandle plan;
	cufftComplex *cpu_data;
	cufftComplex *gpu_data;
	vector<string> fromFile;
	vector<float> data;
	vector<float> period;
	vector<float> power;
	ifstream in;
	cufftComplex *data_h;
	int j=0;
	for(int i = 1938; i <= 1991; i++)
	{
		string path = string("data/") + to_string(i) + string(".dat");
		in.open(path);
		if(!in.is_open())
		{
			cout << "Can't open file" << endl;
			return -1;
		}
		while(getline(in, line))
		{
			buffer = "";
			line += " ";
			for(int k = 0; k < line.size(); k++)
			{
				if(line[k] != ' ')
				{
					buffer += line[k];
				} else 
				{
					if (buffer != "") fromFile.push_back(buffer);
					buffer = "";
				}
			}
			if (fromFile.size() != 0)
			{
				if(fromFile[2] == "999")
				{
					fromFile[2] = to_string(data.back());
				}
				data.push_back(stoi(fromFile[2]));
				fromFile.clear();
			} 
			j++;
		}

		in.close();
	}
	int N = data.size();

	cudaMalloc((void**)&gpu_data, sizeof(cufftComplex) *  N * BATCH);
	data_h = (cufftComplex*)calloc(N, sizeof(cufftComplex));
	cpu_data = new cufftComplex[N * BATCH];
	for (int i = 0; i < N * BATCH; i++)
	{
		cpu_data[i].x = data[i];
		cpu_data[i].y = 0.0f;
	}
	cudaMemcpy(gpu_data, cpu_data, sizeof(cufftComplex) *  N * BATCH, cudaMemcpyHostToDevice);

	if (cufftPlan1d(&plan, N * BATCH, CUFFT_C2C, BATCH) != CUFFT_SUCCESS)
	{
		cerr << "ERROR cufftPlan1d" << endl;
		return -1;
	}
	if (cufftExecC2C(plan, gpu_data, gpu_data, CUFFT_FORWARD) != CUFFT_SUCCESS)
	{
		cerr << "ERROR cufftPlan1d" << endl;
		return -1;
	}
	if (cudaDeviceSynchronize() != cudaSuccess)
	{
		cerr << "ERROR cufftPlan1d" << endl;
		return -1;
	}

	cudaMemcpy(data_h,gpu_data, N * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
	ofstream of("x.txt");
	ofstream of1("y.txt");
	power.resize(N/2 + 1);
	for(int i = 1; i <= N/2; i++)
	{
		power[i] = sqrt(data_h[i].x * data_h[i].x + data_h[i].y * data_h[i].y);
		of << i << endl;
		of1 << power[i] << endl;
	}


	float max_freq = 0.5;
	period.resize(N / 2 + 1);
	ofstream of4("y1.txt");
	for(int i = 1; i <= N / 2; i++)
	{
		period[i] = 1 / (float(i) / float(N / 2) * max_freq);
		of4 << period[i] << endl;
	}

	int maxind = 1;
	for (int i = 1; i <= N/2; i++)
	{
		if(power[i] > power[maxind]) maxind = i;
	}

	cout << "Calculated periodicity: " << period[maxind]/365 << " years." << endl;

	cufftDestroy(plan);
	cudaFree(gpu_data);
	free(data_h);
	free(cpu_data);
}