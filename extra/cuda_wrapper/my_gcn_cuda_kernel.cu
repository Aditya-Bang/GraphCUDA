#include <cuda_runtime.h>
#include <iostream>

__global__ void my_test_kernel(float* x, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < rows * cols) {
        printf("x[%d] = %f\n", idx, x[idx]);
    }
}

void my_test_kernel_launcher(float* x, int rows, int cols) {
    int threads = 256;
    int blocks = (rows * cols + threads - 1) / threads;

    my_test_kernel<<<blocks, threads>>>(x, rows, cols);
    cudaDeviceSynchronize();
}
