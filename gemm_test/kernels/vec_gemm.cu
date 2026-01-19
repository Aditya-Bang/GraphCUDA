#include <cuda_runtime.h>
#include "utils.h"
#include "vec_gemm.h"

// need block of same length and width, otherwise if bk > bn or bm, can't load shared memory properly
// let x be N direction (iterating over cols in C), y be M direction (iterating over rows in C)

// each block actually computes a (BLOCK_SIZE)**2 tile of C
#define BLOCK_SIZE 64 // (BLOCK_SIZE/THREAD_TILE_SIZE)**2 threads per block
#define THREAD_TILE_SIZE 8 // each thread computes (THREAD_TILE_SIZE)**2 tile, must divide BLOCK_SIZE, must be divisible by 4 for SIMD


__global__ void vec_gemm_kernel(
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
        #pragma unroll
        for (int threadTileY = 0; threadTileY < THREAD_TILE_SIZE; threadTileY++) {
            #pragma unroll
            for (int threadTileX = 0; threadTileX < THREAD_TILE_SIZE; threadTileX += 4) { // Assume K % 4 == 0 and N % 4 == 0
                int aRow = blockIdx.y * BLOCK_SIZE + threadIdx.y * THREAD_TILE_SIZE + threadTileY;
                int aCol = blockIdxK * BLOCK_SIZE + threadIdx.x * THREAD_TILE_SIZE + threadTileX;
                int aRowShared = threadIdx.y * THREAD_TILE_SIZE + threadTileY;
                int aColShared = threadIdx.x * THREAD_TILE_SIZE + threadTileX;

                // transpose As for coalesced access
                if (aRow < M && aCol + 3 < K) {
                    // As[RM_INDEX(aColShared, aRowShared, BLOCK_SIZE)] = A[RM_INDEX(aRow, aCol, K)];

                    const float4 a4 = *reinterpret_cast<const float4*>(
                        &A[RM_INDEX(aRow, aCol, K)]
                    );

                    As[RM_INDEX(aColShared, aRowShared, BLOCK_SIZE)] = a4.x;
                    As[RM_INDEX(aColShared + 1, aRowShared, BLOCK_SIZE)] = a4.y;
                    As[RM_INDEX(aColShared + 2, aRowShared, BLOCK_SIZE)] = a4.z;
                    As[RM_INDEX(aColShared + 3, aRowShared, BLOCK_SIZE)] = a4.w;
                } else {
                    As[RM_INDEX(aColShared, aRowShared, BLOCK_SIZE)] = 0.0f;
                    As[RM_INDEX(aColShared + 1, aRowShared, BLOCK_SIZE)] = 0.0f;
                    As[RM_INDEX(aColShared + 2, aRowShared, BLOCK_SIZE)] = 0.0f;
                    As[RM_INDEX(aColShared + 3, aRowShared, BLOCK_SIZE)] = 0.0f;
                }

                int bRow = blockIdxK * BLOCK_SIZE + threadIdx.y * THREAD_TILE_SIZE + threadTileY;
                int bCol = blockIdx.x * BLOCK_SIZE + threadIdx.x * THREAD_TILE_SIZE + threadTileX;
                // same as aRowShared and aColShared
                int bRowShared = threadIdx.y * THREAD_TILE_SIZE + threadTileY;
                int bColShared = threadIdx.x * THREAD_TILE_SIZE + threadTileX;

                if (bRow < K && bCol + 3 < N) {
                    // Bs[RM_INDEX(bRowShared, bColShared, BLOCK_SIZE)] = B[RM_INDEX(bRow, bCol, N)];

                    reinterpret_cast<float4*>(
                        &Bs[RM_INDEX(bRowShared, bColShared, BLOCK_SIZE)]
                    )[0] = *reinterpret_cast<const float4*>(
                        &B[RM_INDEX(bRow, bCol, N)]
                    );
                } else {
                    Bs[RM_INDEX(bRowShared, bColShared, BLOCK_SIZE)] = 0.0f;
                    Bs[RM_INDEX(bRowShared, bColShared + 1, BLOCK_SIZE)] = 0.0f;
                    Bs[RM_INDEX(bRowShared, bColShared + 2, BLOCK_SIZE)] = 0.0f;
                    Bs[RM_INDEX(bRowShared, bColShared + 3, BLOCK_SIZE)] = 0.0f;
                }
            }
        }
        
        __syncthreads();

        // calculate c values in tile
        // no bounds checking needed because BLOCK_SIZE multiple of THREAD_TILE_SIZE and THREAD_TILE_SIZE is a multiple of 4
        for (int threadIdxK = 0; threadIdxK < BLOCK_SIZE; threadIdxK++) {
            #pragma unroll
            for (int threadTileY = 0; threadTileY < THREAD_TILE_SIZE; threadTileY += 4) {
                int aRowShared = threadIdx.y * THREAD_TILE_SIZE + threadTileY;
                int aColShared = threadIdxK;

                const float4 a4 = *reinterpret_cast<const float4*>(
                    &As[RM_INDEX(aColShared, aRowShared, BLOCK_SIZE)]
                );

                regA[threadTileY] = a4.x;
                regA[threadTileY + 1] = a4.y;
                regA[threadTileY + 2] = a4.z;
                regA[threadTileY + 3] = a4.w;
            }

            #pragma unroll
            for (int threadTileX = 0; threadTileX < THREAD_TILE_SIZE; threadTileX += 4) {
                int bRowShared = threadIdxK;
                int bColShared = threadIdx.x * THREAD_TILE_SIZE + threadTileX;

                const float4 b4 = *reinterpret_cast<const float4*>(
                    &Bs[RM_INDEX(bRowShared, bColShared, BLOCK_SIZE)]
                );

                regB[threadTileX] = b4.x;
                regB[threadTileX + 1] = b4.y;
                regB[threadTileX + 2] = b4.z;
                regB[threadTileX + 3] = b4.w;
            }

            #pragma unroll
            for (int threadTileY = 0; threadTileY < THREAD_TILE_SIZE; threadTileY++) {
                #pragma unroll
                for (int threadTileX = 0; threadTileX < THREAD_TILE_SIZE; threadTileX++) {
                    int cRowReg = threadTileY;
                    int cColReg = threadTileX;
                    cTileValues[RM_INDEX(cRowReg, cColReg, THREAD_TILE_SIZE)] += regA[threadTileY] * regB[threadTileX];
                }
            }
        }

        __syncthreads();
    }

    // write back to memory
    #pragma unroll
    for (int cRowReg = 0; cRowReg < THREAD_TILE_SIZE; cRowReg++) {
        #pragma unroll
        for (int cColReg = 0; cColReg < THREAD_TILE_SIZE; cColReg++) {
            int cRow = blockIdx.y * BLOCK_SIZE + threadIdx.y * THREAD_TILE_SIZE + cRowReg;
            int cCol = blockIdx.x * BLOCK_SIZE + threadIdx.x * THREAD_TILE_SIZE + cColReg;
            if (cRow < M && cCol < N) {
                C[RM_INDEX(cRow, cCol, N)] = cTileValues[RM_INDEX(cRowReg, cColReg, THREAD_TILE_SIZE)];
            }
        }
    }
}

void launch_vec_gemm(
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

    vec_gemm_kernel<<<grid, block>>>(A, B, C, M, K, N);
}
