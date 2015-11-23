// CS425: HW5 CUDA
// Cameron Hall, Arthur Wuterich
// 11/22/2015

#include <iostream>
#include <ctime>
#include <cstdlib>
#include <stdio.h>
#include <math.h>
#include "support_code.hpp"

// Problem size
const int N = 1000000;

// Size to split within the reduce CUDA kernel
const int SPLIT_SIZE = 1024;

// Number of values to compute per iteration. This allows the logic for the
// GPU calculations to work on any hardware specification
const int ITERATION_SIZE = 3000;

// HandleError helper function from the CUDA book
// http://stackoverflow.com/questions/13245258/handle-error-not-found-error-in-cuda
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

// Calculates square difference between two N dimensional points
__global__ void calculateFeatureSquareDifference(int *a, int *b, int *c, int N) {
 int tid = blockIdx.x; // handle the data at this index
 if (tid < N) {
   int diff = a[tid] - b[tid];
   c[tid] = diff * diff;
 }
}

// Performs a reduction on the GPU as per the algorithms provided by Mark Harris
// https://docs.nvidia.com/cuda/samples/6_Advanced/reduction/doc/reduction.pdf
__global__ void reduce(int *a, int *b, int *c, int N) {
  extern __shared__ int sdata[];

  // each thread loads one element from global to shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  sdata[tid] = 0;

  // Only complete the reduction if within the problem size
  if( i < N ) {
    // Calculate this threads squared difference
    int diff = a[i] - b[i];
    sdata[tid] = diff*diff;
    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s=1; s < blockDim.x; s *= 2) {
      if (tid % (2*s) == 0) {
        sdata[tid] += sdata[tid + s];
      }
      __syncthreads();
    }

    // write result for this block to c totals array
    if (tid == 0) {
      c[blockIdx.x] = sdata[0];
    }
    __syncthreads();

    // Have the first block and first thread do the final reduction
    if(tid == 0 && blockIdx.x == 0) {
      int localSum = 0;
      for(int i = 0; i < gridDim.x; ++i) {
        localSum += c[i];
      }
      // Place the final reduction into c[0]
      c[0] = localSum;
    }
  }
}

int main(int argc, char** argv) {
  std::srand(time(0));
  // Totals for all of the iterations
  unsigned long cpuDistanceTotal = 0L,
                gpuHostDistanceTotal = 0L,
                gpuDeviceDistanceTotal = 0L;

  // Allocate memory on both the device and the host to the size of ITERATION_SIZE
  // as we will never expect the hardware to perform operations outside this limit
  int *A = (int*)malloc(sizeof(int)*ITERATION_SIZE);
  int *B = (int*)malloc(sizeof(int)*ITERATION_SIZE);
  int *C = (int*)malloc(sizeof(int)*ITERATION_SIZE);
  int *dev_a, *dev_b, *dev_c;
  HANDLE_ERROR(cudaMalloc( (void**)&dev_a, ITERATION_SIZE * sizeof(int) ));
  HANDLE_ERROR(cudaMalloc( (void**)&dev_b, ITERATION_SIZE * sizeof(int) ));
  HANDLE_ERROR(cudaMalloc( (void**)&dev_c, ITERATION_SIZE * sizeof(int) ));
  for(int i = 0; i < N ; i += ITERATION_SIZE) {
    // Compute the next block of features to compare. If we are over the N on the next
    // iteration then compute the difference to bring us to the total N computations
    int computationBlockSize = (i + ITERATION_SIZE < N) ? ITERATION_SIZE : (ITERATION_SIZE - (i+ITERATION_SIZE - N));

    // Get the random values for the points
    InitArray(A, computationBlockSize);
    InitArray(B, computationBlockSize);

    // Send these values to the GPU
    HANDLE_ERROR(cudaMemcpy(dev_a, A, computationBlockSize * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, B, computationBlockSize * sizeof(int), cudaMemcpyHostToDevice));

    // Perform the Square Difference calculation on the GPU
    calculateFeatureSquareDifference<<<computationBlockSize,1>>>(dev_a, dev_b, dev_c, computationBlockSize);

    // Get the results of the Square Difference calculations into the C array
    HANDLE_ERROR(cudaMemcpy(C, dev_c, computationBlockSize * sizeof(int), cudaMemcpyDeviceToHost));

    // Calculate the C array total distance
    double gpuDistanceTotal = 0.0;
    for(int i = 0; i < computationBlockSize; ++i) {
      gpuDistanceTotal += C[i];
    }

    // Determine the offsets for the GPU reduction
    int numBlocks = (computationBlockSize/SPLIT_SIZE)+1,
        blockSize = SPLIT_SIZE,
        memorySize = sizeof(int)*SPLIT_SIZE;

    // Perform the GPU reduction
    reduce<<<numBlocks,blockSize,memorySize>>>(dev_a, dev_b, dev_c, computationBlockSize);

    // Get the result from C[0] from the GPU reduction
    HANDLE_ERROR(cudaMemcpy(C, dev_c, computationBlockSize * sizeof(int), cudaMemcpyDeviceToHost));


    // Add the distances to the running totals
    gpuHostDistanceTotal += gpuDistanceTotal;
    cpuDistanceTotal += ComputeDistanceRaw(A, B, computationBlockSize);
    gpuDeviceDistanceTotal += C[0];
  }

  // Output the totals
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
