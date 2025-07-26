// hello_cuda.cu

#include <stdio.h>
#include "hello_cuda.cuh" // Include the header for the kernel declaration

__global__ void helloCUDA() {
    // Print from the GPU. threadIdx.x and blockDim.x are CUDA built-in variables.
    printf("Hello from GPU! Thread %d/%d\n", threadIdx.x, blockDim.x);
}
