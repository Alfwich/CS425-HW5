#include "support_code.hpp"

// Fills the provided array with random values between [0,99]
// Precondition: srand has been called
void InitArray(int A[], int N) {
  for(int i = 0; i < N; ++i) {
    A[i] = rand() % 100;
  }
}

// Returns the non sqrt'd value for the distance between two points A and B
int ComputeDistanceRaw(int A[], int B[], int N) {
  int total = 0;

  for(int i = 0; i < N; ++i) {
    int diff = A[i] - B[i];
    total += (diff*diff);
  }

  return total;
}

// Returns the sqrt'd distance between two points A and B with N features
double ComputeDistance(int A[], int B[], int N) {
  return sqrt(ComputeDistanceRaw(A,B,N));
}
