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
#include <algorithm>
#include <cctype>
#include <list>
#include <stdlib.h>
#include <ctime>

#include <cuComplex.h>

using namespace std;

#define IMG_WIDTH 1536
#define IMG_HEIGHT 1024

__device__ int mandel(float x, float y, unsigned int max_iters){
    cuFloatComplex c = make_cuFloatComplex(x, y);
    cuFloatComplex z = make_cuFloatComplex(0.0f, 0.0f); 

    for(int i=0;i<max_iters;i++){
        z = cuCaddf(cuCmulf(z,z),c);
        if (z.x*z.x + z.y*z.y >= 4){
            return i;
        }
    }
    return max_iters;
}

__global__ void create_fractal(float min_x, float max_x, float min_y, float max_y, unsigned char* image, unsigned int iters){
    int height = IMG_HEIGHT;
    int width = IMG_WIDTH;

    float pixel_size_x = (max_x - min_x)/ width;
    float pixel_size_y = (max_y - min_y)/ height;

    int startX = threadIdx.x+blockDim.x*blockIdx.x;
    int startY = threadIdx.y+blockDim.y*blockIdx.y;

    int gridX = gridDim.x * blockDim.x; 
    int gridY = gridDim.y * blockDim.y; 

    for(int x = startX; x<width; x+=gridX){
        float real = min_x + x * pixel_size_x;
        for(int y = startY;y<height; y+=gridY){
            float imag = min_y + y * pixel_size_y;
            image[x+y*width] = mandel(real,imag,iters);
        }
    }

} 

int main(){
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float dt;

    unsigned char* image = (unsigned char*)calloc(IMG_WIDTH*IMG_HEIGHT,sizeof(unsigned char));
    unsigned char* d_image;

    cudaMalloc((void**)&d_image, sizeof(unsigned char) * IMG_WIDTH * IMG_HEIGHT);

    dim3 blockdim(32,8);
    dim3 griddim(32,16);

    cudaMemcpy(d_image,image,sizeof(unsigned char) * IMG_WIDTH * IMG_HEIGHT,cudaMemcpyHostToDevice);

    cudaEventRecord(start, 0);

    create_fractal<<<griddim, blockdim>>>(-2.0, 1.0, -1.0, 1.0, d_image, 20);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&dt, start, stop);

    cudaMemcpy(image,d_image,sizeof(unsigned char) * IMG_WIDTH * IMG_HEIGHT,cudaMemcpyDeviceToHost);

    cout << "Mandelbrot created in " << (dt/1000) << " s \n";    
    return 0;
}
