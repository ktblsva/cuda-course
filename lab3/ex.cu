#include "stdio.h"

__global__ void MyKernel(int *array, int arrayCount) 
{ 
  int idx = threadIdx.x + blockIdx.x * blockDim.x; 
  if (idx < arrayCount) 
  { 
    array[idx] *= array[idx]; 
  } 
} 

void launchMyKernel(int *array, int arrayCount) 
{ 
  int blockSize;   // The launch configurator returned block size 
  int minGridSize; // The minimum grid size needed to achieve the 
                   // maximum occupancy for a full device launch 
  int gridSize;    // The actual grid size needed, based on input size 

  cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, 
                                      MyKernel, 0, 0); 
  // Round up according to array size 
  gridSize = (arrayCount + blockSize - 1) / blockSize; 

  MyKernel<<< gridSize, blockSize >>>(array, arrayCount); 

  cudaDeviceSynchronize(); 

  // calculate theoretical occupancy
  int maxActiveBlocks;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor( &maxActiveBlocks, 
                                                 MyKernel, blockSize, 
                                                 0);

  int device;
  cudaDeviceProp props;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&props, device);

  float occupancy = (maxActiveBlocks * blockSize / props.warpSize) / 
                    (float)(props.maxThreadsPerMultiProcessor / 
                            props.warpSize);

  printf("Max active block size: %d.\n", 
         maxActiveBlocks);
  printf("Launched blocks of size %d. Theoretical occupancy: %f\n", 
         blockSize, occupancy);
}

int main(int argc,char* argv[])
{
    int N = 0;
    int *c;
    N = atoi(argv[1]);
    c = (int*)calloc(N, sizeof(int));
    for(int k = 0; k < N; k++)
    {
      c[k] = k;
    }
    launchMyKernel(c, N);
    return 0;
}