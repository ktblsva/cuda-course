import numpy as np
from pylab import imshow, show
from timeit import default_timer as timer

import pycuda.autoinit
from pycuda import driver, compiler, gpuarray, tools  

import os

kernel_source = """

	#include <cuComplex.h>

	__device__ unsigned char mandel(float x, float y, unsigned int max_iters){
	    cuFloatComplex c = make_cuFloatComplex(x, y);
	    cuFloatComplex z = make_cuFloatComplex(0.0f, 0.0f); 

	    for(unsigned int i=0;i<max_iters;i++){
	        z = cuCaddf(cuCmulf(z,z),c);
	        if (z.x*z.x + z.y*z.y >= 4){
	            return i;
	        }
	    }
	    return max_iters;
	}

	__global__ void create_fractal(float min_x, float max_x, float min_y, float max_y, unsigned char* image, unsigned int iters){
	    int height = %(IMG_HEIGHT)s;
	    int width = %(IMG_WIDTH)s;

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

	} """



IMG_WIDTH = 1536
IMG_HEIGHT = 1024

image = np.zeros((IMG_HEIGHT,IMG_WIDTH), dtype = np.uint8)

blockdim = (32,8,1)
griddim = (32,16,1)

d_image = gpuarray.to_gpu(image)

kernel = kernel_source % {
        'IMG_HEIGHT': IMG_HEIGHT,
        'IMG_WIDTH': IMG_WIDTH,
        }

mod = compiler.SourceModule(kernel)

create_fractal = mod.get_function("create_fractal")

start = timer()
create_fractal(np.float32(-2.0), np.float32(1.0), np.float32(-1.0), np.float32(1.0), d_image, np.uint32(20), grid = griddim, block = blockdim)

dt = timer() - start

image = d_image.get()


print ("Mandelbrot created in %f s" % dt)

imshow(image)
show()

