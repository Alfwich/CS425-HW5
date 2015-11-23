#include <iostream>
#include <ctime>
#include <cstdlib>
#include <stdio.h>
#include <math.h>

#include "support_code.hpp"

const int N = 100000;
const int SPLIT_SIZE = 1024;

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

__global__ void calculateFeatureSquareDifference(int *a, int *b, int *c, int N) {
 int tid = blockIdx.x; // handle the data at this index
 printf("%d\n", tid);
 if (tid < N) {
   int diff = a[tid] - b[tid];
   c[tid] = diff * diff;
 }
}

__global__ void reduce(int *a, int *b, int *c, int N) {
  extern __shared__ int sdata[];

  // each thread loads one element from global to shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  sdata[tid] = 0;
  if( i < N ) {
    int before = sdata[tid] = c[i];
    __syncthreads();
    //printf("Before Value: %d, Value in sdata - %d\n", before, sdata[tid]);
    //printf("tid = %d, i = %d, blockDim = (%d, %d), blockIdx = (%d, %d), threadIdx = (%d,%d), gridDim = (%d, %d)\n", tid, i, blockDim.x, blockDim.y, blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, gridDim.x, gridDim.y);
    //printf("1 - %d\n", sdata[0]);
    //printf("%d\n", tid);
    // do reduction in shared mem
    for(unsigned int s=1; s < blockDim.x; s *= 2) {
      if (tid % (2*s) == 0) {
        sdata[tid] += sdata[tid + s];
      }
      __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) {
      c[blockIdx.x] = sdata[0];
    }
    __syncthreads();

    if(tid == 0 && blockIdx.x == 0) {
      int localSum = 0;
      for(int i = 0; i < gridDim.x; ++i) {
        //printf("Adding location %d localSum\n", i);
        //__syncthreads();
        localSum += c[i];
      }

      c[0] = localSum;
    }
  }
}

int main(int argc, char** argv) {
  //std::srand(time(0));
  std::srand(0);

  int *dev_a, *dev_b, *dev_c;
  HANDLE_ERROR(cudaMalloc( (void**)&dev_a, N * sizeof(int) ));
  HANDLE_ERROR(cudaMalloc( (void**)&dev_b, N * sizeof(int) ));
  HANDLE_ERROR(cudaMalloc( (void**)&dev_c, N * sizeof(int) ));

  int *A = (int*)malloc(sizeof(int)*N);
  int *B = (int*)malloc(sizeof(int)*N);
  int *C = (int*)malloc(sizeof(int)*N);

  InitArray(A, N);
  InitArray(B, N);

  HANDLE_ERROR(cudaMemcpy(dev_a, A, N * sizeof(int), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_b, B, N * sizeof(int), cudaMemcpyHostToDevice));

  calculateFeatureSquareDifference<<<N,1>>>(dev_a, dev_b, dev_c, N);

  HANDLE_ERROR(cudaMemcpy(C, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));

  double gpuDistanceTotal = 0.0;
  for(int i = 0; i < N; ++i) {
    gpuDistanceTotal += C[i];
    //std::cout << C[i] << " ";
  }
  //std::cout << std::endl;

  double gpuDistance = sqrt(gpuDistanceTotal);
  double actualDistance = ComputeDistance(A, B, N);

  int numBlocks = N/SPLIT_SIZE,
      blockSize = SPLIT_SIZE,
      memorySize = sizeof(int)*SPLIT_SIZE;
  if(N % SPLIT_SIZE != 0) {
    numBlocks += 1;
  }
  reduce<<<numBlocks,blockSize,memorySize>>>(dev_a, dev_b, dev_c, N);
  HANDLE_ERROR(cudaMemcpy(C, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));
  /*
  for(int i = 0; i < N; ++i) {
    gpuDistanceTotal += C[i];
    std::cout << C[i] << " ";
  }
  std::cout << std::endl;
  */

  double deviceDistance = sqrt(C[0]); // Some way to get the reduction distance
  std::cout << "Host result: " << actualDistance << std::endl;
  std::cout << "CUDA result with host reduction: " << gpuDistance << std::endl;
  std::cout << "CUDA result with device reduction: " << deviceDistance << std::endl;

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);
  free(A);
  free(B);
  free(C);

  return 0;
}
