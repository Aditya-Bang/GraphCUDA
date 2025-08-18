#include <cstdio>
#include "hello_cuda.cuh"
#include <cuda_runtime.h>

int main() {
    // Launch kernel: 1 block, 1 thread
    helloCUDA<<<1, 1>>>();

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    printf("Hello from CPU!\n");

    return 0;
}
