#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#define M_PI 3.14159265358979323846
#define COEF 48
#define VERTCOUNT COEF *COEF * 2
#define RADIUS 160.0f
#define FGSIZE 320
#define FGSHIFT FGSIZE / 2
#define IMIN(A, B) (A < B ? A : B)
#define THREADSPERBLOCK 256
#define BLOCKSPERGRID                                                          \
  IMIN(32, (VERTCOUNT + THREADSPERBLOCK - 1) / THREADSPERBLOCK)

typedef float (*ptr_f)(float, float, float);

struct Vertex {
  float x, y, z;
};

__constant__ Vertex vert[VERTCOUNT];

texture<float, 3, cudaReadModeElementType> df_tex;
cudaArray *df_Array = 0;

// Функция в разложении по ортонормированному базису в Гильбертовом пространстве
float func(float x, float y, float z) {
  return (0.5 * sqrtf(15.0 / M_PI)) * (0.5 * sqrtf(15.0 / M_PI)) * z * z * y *
         y * sqrtf(1.0f - z * z / RADIUS / RADIUS) / RADIUS / RADIUS / RADIUS /
         RADIUS;
}

// Проверка суммы по функции от декартовых координат вершины
float check(Vertex *v, ptr_f f) {
  float sum = 0.0f;

  for (int i = 0; i < VERTCOUNT; i++) {
    sum += f(v[i].x, v[i].y, v[i].z);
  }
  return sum;
}

void calc_f(float *arr_f, int x_size, int y_size, int z_size, ptr_f f) {
  for (int x = 0; x < x_size; x++)
    for (int y = 0; y < y_size; y++)
      for (int z = 0; z < z_size; z++)
        arr_f[z_size * (x * y_size + y) + z] =
            f(x - FGSHIFT, y - FGSHIFT, z - FGSHIFT);
}

void init_vertices(Vertex *vertex_dev) {
  Vertex *temp_vert = (Vertex *)malloc(sizeof(Vertex) * VERTCOUNT);
  int i = 0;
  for (int iphi = 0; iphi < 2 * COEF; iphi++) {
    for (int ipsi = 0; ipsi < COEF; ipsi++, i++) {
      float phi = iphi * M_PI / COEF;
      float psi = ipsi * M_PI / COEF;
      temp_vert[i].x = RADIUS * sinf(psi) * cosf(phi);
      temp_vert[i].y = RADIUS * sinf(psi) * sinf(phi);
      temp_vert[i].z = RADIUS * cosf(psi);
    }
  }
  printf("Проверка суммы = %f\n",
         check(temp_vert, &func) * M_PI * M_PI / COEF / COEF);
  // Функция для копирования данных с host'а в текстурную память
  cudaMemcpyToSymbol(vert, temp_vert, sizeof(Vertex) * VERTCOUNT, 0,
                     cudaMemcpyHostToDevice);
  cudaMemcpy(vertex_dev, temp_vert, sizeof(Vertex) * VERTCOUNT,
             cudaMemcpyHostToDevice);
  free(temp_vert);
}

void init_texture(float *df_h) {
  const cudaExtent volumeSize = make_cudaExtent(FGSIZE, FGSIZE, FGSIZE);
  // Формат дескриптора канала // float = 2
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

  cudaMalloc3DArray(&df_Array, &channelDesc, volumeSize);
  cudaMemcpy3DParms cpyParams = {0};
  // Адрес исходной памяти
  cpyParams.srcPtr =
      make_cudaPitchedPtr((void *)df_h, volumeSize.width * sizeof(float),
                          volumeSize.width, volumeSize.height);
  // df_h - Указатель на выделенную память
  // volumeSize.width * sizeof(float) - шаг выделенной памяти в байтах
  // volumeSize.width - логическая ширина(высота) размещения в элементах
  // Адрес целевой памяти
  cpyParams.dstArray = df_Array;
  // Запрошенный размер экземпляра памяти
  cpyParams.extent = volumeSize;
  // Тип копирования
  cpyParams.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&cpyParams);

  df_tex.normalized =
      false; // Указывает, нормализовано ли чтение текстуры или нет
  df_tex.filterMode =
      cudaFilterModeLinear; // cudaFilterModePoint | cudaFilterModeLinear
  // Режим текстурной адресации для 3-х измерений
  df_tex.addressMode[0] = cudaAddressModeClamp; // Clamp to edge address mode
  df_tex.addressMode[1] = cudaAddressModeClamp;
  df_tex.addressMode[2] = cudaAddressModeClamp;
  // Привязывает массив к текстуре
  cudaBindTextureToArray(df_tex, df_Array, channelDesc);
}

void release_texture() {
  cudaUnbindTexture(df_tex);
  cudaFreeArray(df_Array);
}

__global__ void kernel(float *a) {
  // Использование разделяемой памяти для кеширования фильтрованных значений
  // функции
  __shared__ float cache[THREADSPERBLOCK];
  // Индекс потока
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int cacheIndex = threadIdx.x;
  // Получаем координаты вершин в которых нужно посчитать значение функции
  float x, y, z;
  x = vert[tid].x + FGSHIFT + 0.5f;
  y = vert[tid].y + FGSHIFT + 0.5f;
  z = vert[tid].z + FGSHIFT + 0.5f;

  cache[cacheIndex] = tex3D(df_tex, z, y, x);

  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (cacheIndex < s)
      cache[cacheIndex] += cache[cacheIndex + s];
    __syncthreads();
  }
  if (cacheIndex == 0)
    a[blockIdx.x] = cache[0];
}

__device__ float getDistance(Vertex v0, Vertex v) {
  return sqrtf((v0.x - v.x) * (v0.x - v.x) + (v0.y - v.y) * (v0.y - v.y) +
               (v0.z - v.z) * (v0.z - v.z));
}

__device__ float interpolate1D(float a, float b, float t) {
  return a * (1 - t) + b * t;
}

__device__ float interpolate2D(float v1, float v2, float v3, float v4, float x,
                               float y) {

  float s = interpolate1D(v1, v2, x);
  float t = interpolate1D(v3, v4, x);

  return interpolate1D(s, t, y);
}

__device__ float interpolate3D(float *arr_f, Vertex v) {

  int gx = int(v.x);
  int gy = int(v.y);
  int gz = int(v.z);
  float tx = v.x - (float)gx;
  float ty = v.z - (float)gz;
  float tz = v.z - (float)gz;

  if (gx + 1 >= FGSIZE || gy + 1 >= FGSIZE || gz + 1 >= FGSIZE)
    return 0.0f;

  float c000 = arr_f[FGSIZE * (gx * FGSIZE + gy) + gz];
  float c001 = arr_f[FGSIZE * ((gx + 1) * FGSIZE + gy) + gz];
  float c010 = arr_f[FGSIZE * (gx * FGSIZE + (gy + 1)) + gz];
  float c011 = arr_f[FGSIZE * ((gx + 1) * FGSIZE + (gy + 1)) + gz];

  float c100 = arr_f[FGSIZE * (gx * FGSIZE + gy) + (gz + 1)];
  float c101 = arr_f[FGSIZE * ((gx + 1) * FGSIZE + gy) + (gz + 1)];
  float c110 = arr_f[FGSIZE * (gx * FGSIZE + (gy + 1)) + (gz + 1)];
  float c111 = arr_f[FGSIZE * ((gx + 1) * FGSIZE + (gy + 1)) + (gz + 1)];

  float e = interpolate2D(c000, c001, c010, c011, tx, ty);
  float f = interpolate2D(c100, c101, c110, c111, tx, ty);

  return interpolate1D(e, f, tz);
}

__device__ float interpolateStep(float *arr_f, Vertex v) {

  int gx = int(v.x);
  int gy = int(v.y);
  int gz = int(v.z);

  if (gx + 1 >= FGSIZE || gy + 1 >= FGSIZE || gz + 1 >= FGSIZE)
    return 0.0f;

  float fgx = float(gx);
  float fgy = float(gy);
  float fgz = float(gz);

  Vertex coords[8] = {{fgx, fgy, fgz},         {fgx + 1, fgy, fgz},
                      {fgx, fgy, fgz + 1},     {fgx + 1, fgy, fgz + 1},
                      {fgx, fgy + 1, fgz},     {fgx + 1, fgy + 1, fgz},
                      {fgx, fgy + 1, fgz + 1}, {fgx + 1, fgy + 1, fgz + 1}};

  float value = arr_f[FGSIZE * (gx * FGSIZE + gy) + gz];
  Vertex vrt;
  vrt.x = coords[0].x;
  vrt.y = coords[0].y;
  vrt.z = coords[0].z;
  float distance = getDistance(vrt, v);
  float tmp = 0;
  for (int i = 1; i < 8; i++) {
    Vertex vrt1;
    vrt1.x = coords[i].x;
    vrt1.y = coords[i].y;
    vrt1.z = coords[i].z;
    tmp = getDistance(vrt1, v);
    if (tmp < distance) {
      distance = tmp;
      value = arr_f[FGSIZE * (int(coords[i].x) * FGSIZE + int(coords[i].y)) +
                    int(coords[i].z)];
    }
  }

  return value;
}

__global__ void Trilinear(float *a, float *arr, Vertex *v) {
  __shared__ float cache[THREADSPERBLOCK];
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int cacheIndex = threadIdx.x;
  float x = v[tid].x + FGSHIFT;
  float y = v[tid].y + FGSHIFT;
  float z = v[tid].z + FGSHIFT;
  Vertex vrt;
  vrt.x = x;
  vrt.y = y;
  vrt.z = z;
  cache[cacheIndex] = interpolate3D(arr, vrt);
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (cacheIndex < s)
      cache[cacheIndex] += cache[cacheIndex + s];
    __syncthreads();
  }
  if (cacheIndex == 0)
    a[blockIdx.x] = cache[0];
}

__global__ void Stepped(float *a, float *arr, Vertex *v) {
  __shared__ float cache[THREADSPERBLOCK];
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int cacheIndex = threadIdx.x;
  float x = v[tid].x + FGSHIFT;
  float y = v[tid].y + FGSHIFT;
  float z = v[tid].z + FGSHIFT;
  Vertex vrt;
  vrt.x = x;
  vrt.y = y;
  vrt.z = z;
  cache[cacheIndex] = interpolateStep(arr, vrt);
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (cacheIndex < s)
      cache[cacheIndex] += cache[cacheIndex + s];
    __syncthreads();
  }
  if (cacheIndex == 0)
    a[blockIdx.x] = cache[0];
}

int main(int argc, char *argv[]) {
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  printf("\nDevice:\t%s\n\n", deviceProp.name);

  Vertex *vert_dev;
  float elapsedTime;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float *arr = (float *)malloc(sizeof(float) * FGSIZE * FGSIZE * FGSIZE);
  float *sum = (float *)malloc(sizeof(float) * BLOCKSPERGRID);
  float *sum_dev, *arr_dev;

  cudaMalloc((void **)&sum_dev, sizeof(float) * BLOCKSPERGRID);
  cudaMalloc((void **)&arr_dev, sizeof(float) * FGSIZE * FGSIZE * FGSIZE);
  cudaMalloc((void **)&vert_dev, sizeof(Vertex) * VERTCOUNT);

  init_vertices(vert_dev);
  calc_f(arr, FGSIZE, FGSIZE, FGSIZE, &func);
  init_texture(arr);

  cudaMemcpy(arr_dev, arr, sizeof(float) * FGSIZE * FGSIZE * FGSIZE,
             cudaMemcpyHostToDevice);
  /* Texture Kernel */
  cudaEventRecord(start, 0);
  kernel<<<BLOCKSPERGRID, THREADSPERBLOCK>>>(sum_dev);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  cudaMemcpy(sum, sum_dev, sizeof(float) * BLOCKSPERGRID,
             cudaMemcpyDeviceToHost);

  float s = 0.0f;
  for (int i = 0; i < BLOCKSPERGRID; i++) {
    s += sum[i];
  }
  cudaEventElapsedTime(&elapsedTime, start, stop);
  fprintf(stderr, "TextureSum = %f\n",s * M_PI * M_PI / COEF / COEF);
  fprintf(stderr, "Time = %f\n\n", elapsedTime);
  /* Trilinear Kernel */
  cudaEventRecord(start, 0);
  Trilinear<<<BLOCKSPERGRID, THREADSPERBLOCK>>>(sum_dev, arr_dev, vert_dev);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  cudaMemcpy(sum, sum_dev, sizeof(float) * BLOCKSPERGRID,
             cudaMemcpyDeviceToHost);

  s = 0.0f;
  for (int i = 0; i < BLOCKSPERGRID; i++) {
    s += sum[i];
  }
  cudaEventElapsedTime(&elapsedTime, start, stop);
  fprintf(stderr, "TrilinearSum = %f\n", s * M_PI * M_PI / COEF / COEF);
  fprintf(stderr, "Time = %f\n\n", elapsedTime);
  /* Stepped Kernel */
  cudaEventRecord(start, 0);
  Stepped<<<BLOCKSPERGRID, THREADSPERBLOCK>>>(sum_dev, arr_dev, vert_dev);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  cudaMemcpy(sum, sum_dev, sizeof(float) * BLOCKSPERGRID,
             cudaMemcpyDeviceToHost);

  s = 0.0f;
  for (int i = 0; i < BLOCKSPERGRID; i++) {
    s += sum[i];
  }
  cudaEventElapsedTime(&elapsedTime, start, stop);
  fprintf(stderr, "SteppedSum = %f\n", s * M_PI * M_PI / COEF / COEF);
  fprintf(stderr, "Time = %f\n\n", elapsedTime);
  cudaFree(sum);
  release_texture();
  free(arr);
  return 0;
}