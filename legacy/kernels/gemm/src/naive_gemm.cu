#include <cuda_runtime.h>
#include "utils.h"
#include "naive_gemm.h"

__global__ void naive_gemm_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int K,
    int N
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}

void launch_naive_gemm(
    const float* A,
    const float* B,
    float* C,
    int M,
    int K,
    int N
) {
    dim3 block(16, 16);
    dim3 grid(
        CEIL_DIV(N, block.x),
        CEIL_DIV(M, block.y)
    );

    naive_gemm_kernel<<<grid, block>>>(A, B, C, M, K, N);
}
