#include "handle_error.h"
#include <stdio.h>

// HandleError helper function from the CUDA book
// http://stackoverflow.com/questions/13245258/handle-error-not-found-error-in-cuda
void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
