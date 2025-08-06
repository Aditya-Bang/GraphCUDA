#include "matmul.cuh"

// Naive CUDA kernel for matrix multiplication
__global__ void matmul_kernel(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // Row in A and C
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // Column in B and C

    if (row < M && col < N) {
        float value = 0.0f;
        for (int i = 0; i < K; ++i) {
            value += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = value;
    }
}

#define TILE_SIZE 32

// Tiled matrix multiplication kernel using shared memory
__global__ void matmul_kernel_optimized(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int tile_idx = 0; tile_idx < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile_idx) {
        if (row < M && tile_idx * TILE_SIZE + threadIdx.x < K)
            tile_A[threadIdx.y][threadIdx.x] = A[row * K + tile_idx * TILE_SIZE + threadIdx.x];
        else
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && tile_idx * TILE_SIZE + threadIdx.y < K)
            tile_B[threadIdx.y][threadIdx.x] = B[(tile_idx * TILE_SIZE + threadIdx.y) * N + col];
        else
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; ++i)
            sum += tile_A[threadIdx.y][i] * tile_B[i][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

// PyTorch wrapper
torch::Tensor matmul(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Input tensors must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "A.cols must match B.rows");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    // auto A_contig = A.contiguous();
    // auto B_contig = B.contiguous();

    auto C = torch::zeros({M, N}, A.options());

    float* A_ptr = A.data_ptr<float>();
    float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);

    // matmul_kernel<<<blocks, threads>>>(A_ptr, B_ptr, C_ptr, M, N, K);
    matmul_kernel_optimized<<<blocks, threads>>>(A_ptr, B_ptr, C_ptr, M, N, K);
    // cudaDeviceSynchronize();  // Optional: helpful for debugging

    return C;
}
