#include <stdio.h>

int main ( int argc, char *  argv [] )
{
    int            deviceCount;
    cudaDeviceProp devProp;

    cudaGetDeviceCount ( &deviceCount );

    printf ( "Found %d devices\n", deviceCount );

    for ( int device = 0; device < deviceCount; device++ )
    {
        cudaGetDeviceProperties ( &devProp, device );

        printf ( "Device %d\n", device );
        printf ( "Compute capability     : %d.%d\n", devProp.major, devProp.minor );
        printf ( "Name                   : %s\n", devProp.name );
        printf ( "Total Global Memory    : %d\n", devProp.totalGlobalMem );
        printf ( "Shared memory per block: %d\n", devProp.sharedMemPerBlock );
        printf ( "Registers per block    : %d\n", devProp.regsPerBlock );
        printf ( "Warp size              : %d\n", devProp.warpSize );
        printf ( "Max threads per block  : %d\n", devProp.maxThreadsPerBlock );
        printf ( "Total constant memory  : %d\n", devProp.totalConstMem );
        printf("Max Thread Dimensions    : %i x %i x %i\n", devProp.maxThreadsDim[0], devProp.maxThreadsDim[1], devProp.maxThreadsDim[2]);
        printf("Max Block Dimensions     :   %i x %i x %i\n", devProp.maxGridSize[0], devProp.maxGridSize[1], devProp.maxGridSize[2]);  
    }

    return 0;
}