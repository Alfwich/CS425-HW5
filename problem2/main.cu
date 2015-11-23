#include <iostream>
#include <ctime>
#include <cstdlib>
#include <stdio.h>
#include <math.h>

#include "support_code.hpp"

const int N = 100000000;
const int SPLIT_SIZE = 1024;
const int ITERATION_SIZE = 3000;

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
 //printf("%d\n", tid);
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
  unsigned long cpuDistanceTotal = 0L,
                gpuHostDistanceTotal = 0L,
                gpuDeviceDistanceTotal = 0L;

  int *dev_a, *dev_b, *dev_c;
  HANDLE_ERROR(cudaMalloc( (void**)&dev_a, ITERATION_SIZE * sizeof(int) ));
  HANDLE_ERROR(cudaMalloc( (void**)&dev_b, ITERATION_SIZE * sizeof(int) ));
  HANDLE_ERROR(cudaMalloc( (void**)&dev_c, ITERATION_SIZE * sizeof(int) ));
  int *A = (int*)malloc(sizeof(int)*ITERATION_SIZE);
  int *B = (int*)malloc(sizeof(int)*ITERATION_SIZE);
  int *C = (int*)malloc(sizeof(int)*ITERATION_SIZE);
  for(int i = 0; i < N ;i += ITERATION_SIZE) {
    int computationBlockSize = ITERATION_SIZE;
    if( i + ITERATION_SIZE >= N ) {
      computationBlockSize = ITERATION_SIZE - (i+ITERATION_SIZE - N);
    }
    if(computationBlockSize <= 0) {
      break;
    }
    //std::cout << "Computation block size: " << computationBlockSize << ", Running Totals: " << cpuDistanceTotal << ", " << gpuHostDistanceTotal << ", " << gpuDeviceDistanceTotal << std::endl;


    InitArray(A, computationBlockSize);
    InitArray(B, computationBlockSize);

    HANDLE_ERROR(cudaMemcpy(dev_a, A, computationBlockSize * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, B, computationBlockSize * sizeof(int), cudaMemcpyHostToDevice));

    calculateFeatureSquareDifference<<<computationBlockSize,1>>>(dev_a, dev_b, dev_c, computationBlockSize);

    HANDLE_ERROR(cudaMemcpy(C, dev_c, computationBlockSize * sizeof(int), cudaMemcpyDeviceToHost));

    double gpuDistanceTotal = 0.0;
    for(int i = 0; i < computationBlockSize; ++i) {
      gpuDistanceTotal += C[i];
      //std::cout << C[i] << " ";
    }
    //std::cout << std::endl;

    gpuHostDistanceTotal += gpuDistanceTotal;
    cpuDistanceTotal += ComputeDistanceRaw(A, B, computationBlockSize);

    int numBlocks = computationBlockSize/SPLIT_SIZE,
        blockSize = SPLIT_SIZE,
        memorySize = sizeof(int)*SPLIT_SIZE;
    if(computationBlockSize % SPLIT_SIZE != 0) {
      numBlocks += 1;
    }
    reduce<<<numBlocks,blockSize,memorySize>>>(dev_a, dev_b, dev_c, computationBlockSize);
    HANDLE_ERROR(cudaMemcpy(C, dev_c, computationBlockSize * sizeof(int), cudaMemcpyDeviceToHost));
    /*
    for(int i = 0; i < N; ++i) {
      gpuDistanceTotal += C[i];
      std::cout << C[i] << " ";
    }
    std::cout << std::endl;
    */

    gpuDeviceDistanceTotal += C[0];

  }

  std::cout << "Host result: " << sqrt(cpuDistanceTotal) << std::endl;
  std::cout << "CUDA result with host reduction: " << sqrt(gpuHostDistanceTotal) << std::endl;
  std::cout << "CUDA result with device reduction: " << sqrt(gpuDeviceDistanceTotal) << std::endl;

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);
  free(A);
  free(B);
  free(C);

  return 0;
}
