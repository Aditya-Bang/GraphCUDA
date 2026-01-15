#include <cuda_runtime.h>
#include "tiled_gemm.h"

#define BM 128
#define BN 128
#define BK 8

#define TM 8
#define TN 8

__global__ void tiled_gemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;

    int row = blockRow * BM + threadRow * TM;
    int col = blockCol * BN + threadCol * TN;

    float acc[TM][TN] = {0};

    for (int k0 = 0; k0 < K; k0 += BK) {

        // Load A tile
        for (int i = 0; i < TM; ++i) {
            int r = blockRow * BM + threadRow * TM + i;
            int c = k0 + threadCol;
            if (r < M && c < K) {
                As[threadRow * TM + i][threadCol] = A[r * K + c];
            } else {
                As[threadRow * TM + i][threadCol] = 0.0f;
            }
        }

        // Load B tile
        for (int j = 0; j < TN; ++j) {
            int r = k0 + threadRow;
            int c = blockCol * BN + threadCol * TN + j;
            if (r < K && c < N) {
                Bs[threadRow][threadCol * TN + j] = B[r * N + c];
            } else {
                Bs[threadRow][threadCol * TN + j] = 0.0f;
            }
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                float a = As[threadRow * TM + i][k];
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += a * Bs[k][threadCol * TN + j];
                }
            }
        }

        __syncthreads();
    }

    // Write back
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int r = row + i;
        if (r < M) {
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                int c = col + j;
                if (c < N) {
                    C[r * N + c] = acc[i][j];
                }
            }
        }
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
    dim3 block(16, 16);
    dim3 grid(
        (N + BN - 1) / BN,
        (M + BM - 1) / BM
    );

    tiled_gemm_kernel<<<grid, block>>>(A, B, C, M, K, N);
}
