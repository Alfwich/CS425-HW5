#include "support_code.hpp"

void InitArray(int A[], int N) {
  for(int i = 0; i < N; ++i) {
    A[i] = rand() % 100;
  }
}

double ComputeDistance(int A[], int B[], int N) {
  int total = 0;

  for(int i = 0; i < N; ++i) {
    int diff = A[i] - B[i];
    total += (diff*diff);
  }

  return sqrt(total);
}
