#include <cuda_runtime.h>
#include "utils.h"
#include "tiled_gemm.h"

// need block of same length and width, otherwise if bk > bn or bm, can't load shared memory properly
// let x be N direction (col), y be M direction (row)

#define BLOCK_SIZE 16


__global__ void tiled_gemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {
    const int cRow = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    const int cCol = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    __shared__ float As[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE * BLOCK_SIZE];

    float cValue = 0.0f;
    for (int blockIdxK = 0; blockIdxK < CEIL_DIV(K, BLOCK_SIZE); blockIdxK++) {
        if (cRow < M && (blockIdxK * BLOCK_SIZE + threadIdx.x) < K)
            As[threadIdx.y * BLOCK_SIZE + threadIdx.x] = A[cRow * K + blockIdxK * BLOCK_SIZE + threadIdx.x];
        else
            As[threadIdx.y * BLOCK_SIZE + threadIdx.x] = 0.0f;
        
        if (cCol < N && (blockIdxK * BLOCK_SIZE + threadIdx.y) < K)
            Bs[threadIdx.y * BLOCK_SIZE + threadIdx.x] = B[(blockIdxK * BLOCK_SIZE + threadIdx.y) * N + cCol];
        else
            Bs[threadIdx.y * BLOCK_SIZE + threadIdx.x] = 0.0f;
        
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; k++) {
            cValue += As[threadIdx.y * BLOCK_SIZE + k] * Bs[k * BLOCK_SIZE + threadIdx.x];
        }

        __syncthreads();
    }

    if (cRow < M && cCol < N) {
        C[cRow * N + cCol] = cValue;
    }
}

void launch_tiled_gemm(
    const float* A,
    const float* B,
    float* C,
    int M,
    int K,
    int N
) {
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(
        CEIL_DIV(N, BLOCK_SIZE),
        CEIL_DIV(M, BLOCK_SIZE)
    );

    tiled_gemm_kernel<<<grid, block>>>(A, B, C, M, K, N);
}
