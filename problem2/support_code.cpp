#include "support_code.hpp"
#include <cstdlib>

void InitArray(int A[], int N) {
  for(int i = 0; i < N; ++i) {
    A[i] = rand() % 100;
  }
}

int ComputeDistanceRaw(int A[], int B[], int N) {
  int total = 0;

  for(int i = 0; i < N; ++i) {
    int diff = A[i] - B[i];
    total += (diff*diff);
  }

  return total;
}

double ComputeDistance(int A[], int B[], int N) {
  return sqrt(ComputeDistanceRaw(A,B,N));
}
