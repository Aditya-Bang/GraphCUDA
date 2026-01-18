#include <cuda_runtime.h>
#include "utils.h"
#include "tiled_gemm.h"

// need block of same length and width, otherwise if bk > bn or bm, can't load shared memory properly
// let x be N direction (iterating over cols in C), y be M direction (iterating over rows in C)

// each block actually computes a (BLOCK_SIZE)**2 tile of C
#define BLOCK_SIZE 64 // (BLOCK_SIZE/THREAD_TILE_SIZE)**2 threads per block
#define THREAD_TILE_SIZE 8 // each thread computes (THREAD_TILE_SIZE)**2 tile, must divide BLOCK_SIZE, must be divisible by 4 for SIMD


__global__ void tiled_gemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {

    // in shared memory
    __shared__ float As[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE * BLOCK_SIZE];

    // in register memory
    float regA[THREAD_TILE_SIZE];
    float regB[THREAD_TILE_SIZE];
    float cTileValues[THREAD_TILE_SIZE * THREAD_TILE_SIZE] = {0.0f};
    
    for (int blockIdxK = 0; blockIdxK < CEIL_DIV(K, BLOCK_SIZE); blockIdxK++) {
        for (int threadTileY = 0; threadTileY < THREAD_TILE_SIZE; threadTileY++) {
            for (int threadTileX = 0; threadTileX < THREAD_TILE_SIZE; threadTileX++) {
                int aRow = blockIdx.y * BLOCK_SIZE + threadIdx.y * THREAD_TILE_SIZE + threadTileY;
                int aCol = blockIdxK * BLOCK_SIZE + threadIdx.x * THREAD_TILE_SIZE + threadTileX;
                int aRowShared = threadIdx.y * THREAD_TILE_SIZE + threadTileY;
                int aColShared = threadIdx.x * THREAD_TILE_SIZE + threadTileX;

                if (aRow < M && aCol < K) {
                    As[RM_INDEX(aRowShared, aColShared, BLOCK_SIZE)] = A[RM_INDEX(aRow, aCol, K)];
                } else {
                    As[RM_INDEX(aRowShared, aColShared, BLOCK_SIZE)] = 0.0f;
                }

                int bRow = blockIdxK * BLOCK_SIZE + threadIdx.y * THREAD_TILE_SIZE + threadTileY;
                int bCol = blockIdx.x * BLOCK_SIZE + threadIdx.x * THREAD_TILE_SIZE + threadTileX;
                // same as aRowShared and aColShared
                int bRowShared = threadIdx.y * THREAD_TILE_SIZE + threadTileY;
                int bColShared = threadIdx.x * THREAD_TILE_SIZE + threadTileX;

                if (bRow < K && bCol < N) {
                    Bs[RM_INDEX(bRowShared, bColShared, BLOCK_SIZE)] = B[RM_INDEX(bRow, bCol, N)];
                } else {
                    Bs[RM_INDEX(bRowShared, bColShared, BLOCK_SIZE)] = 0.0f;
                }
            }
        }
        
        __syncthreads();

        // calculate c values in tile
        for (int threadIdxK = 0; threadIdxK < BLOCK_SIZE; threadIdxK++) {
            for (int threadTileY = 0; threadTileY < THREAD_TILE_SIZE; threadTileY++) {
                int aRowShared = threadIdx.y * THREAD_TILE_SIZE + threadTileY;
                int aColShared = threadIdxK;
                regA[threadTileY] = As[RM_INDEX(aRowShared, aColShared, BLOCK_SIZE)];
            }

            for (int threadTileX = 0; threadTileX < THREAD_TILE_SIZE; threadTileX++) {
                int bRowShared = threadIdxK;
                int bColShared = threadIdx.x * THREAD_TILE_SIZE + threadTileX;
                regB[threadTileX] = Bs[RM_INDEX(bRowShared, bColShared, BLOCK_SIZE)];
            }

            for (int threadTileY = 0; threadTileY < THREAD_TILE_SIZE; threadTileY++) {
                for (int threadTileX = 0; threadTileX < THREAD_TILE_SIZE; threadTileX++) {
                    int cRowReg = threadTileY;
                    int cColReg = threadTileX;
                    cTileValues[RM_INDEX(cRowReg, cColReg, THREAD_TILE_SIZE)] += regA[threadTileY] * regB[threadTileX];
                }
            }
        }

        // for (int threadTileY = 0; threadTileY < THREAD_TILE_SIZE; threadTileY++) {
        //     for (int threadTileX = 0; threadTileX < THREAD_TILE_SIZE; threadTileX++) {
        //         for (int k = 0; k < BLOCK_SIZE; k++) {
        //             int aRowShared = threadIdx.y * THREAD_TILE_SIZE + threadTileY;
        //             int aColShared = k;
        //             int bRowShared = k;
        //             int bColShared = threadIdx.x * THREAD_TILE_SIZE + threadTileX;
        //             int cRowReg = threadTileY;
        //             int cColReg = threadTileX;
        //             cTileValues[RM_INDEX(cRowReg, cColReg, THREAD_TILE_SIZE)] += As[RM_INDEX(aRowShared, aColShared, BLOCK_SIZE)] * Bs[RM_INDEX(bRowShared, bColShared, BLOCK_SIZE)];
        //         }
        //     }
        // }

        __syncthreads();
    }

    // write back to memory
    for (int cRowReg = 0; cRowReg < THREAD_TILE_SIZE; cRowReg++) {
        for (int cColReg = 0; cColReg < THREAD_TILE_SIZE; cColReg++) {
            int cRow = blockIdx.y * BLOCK_SIZE + threadIdx.y * THREAD_TILE_SIZE + cRowReg;
            int cCol = blockIdx.x * BLOCK_SIZE + threadIdx.x * THREAD_TILE_SIZE + cColReg;
            if (cRow < M && cCol < N) {
                C[RM_INDEX(cRow, cCol, N)] = cTileValues[RM_INDEX(cRowReg, cColReg, THREAD_TILE_SIZE)];
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
    dim3 block(BLOCK_SIZE/THREAD_TILE_SIZE, BLOCK_SIZE/THREAD_TILE_SIZE);
    dim3 grid(
        CEIL_DIV(N, BLOCK_SIZE),
        CEIL_DIV(M, BLOCK_SIZE)
    );

    tiled_gemm_kernel<<<grid, block>>>(A, B, C, M, K, N);
}
