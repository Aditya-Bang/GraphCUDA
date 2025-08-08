#include "matmul.cuh"


// Naive CUDA kernel for matrix multiplication
__global__ void matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int M,
    const int N,
    const int K
) {
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


// PyTorch wrapper
torch::Tensor matmul1(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "A.cols must match B.rows");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    A = A.contiguous();
    B = B.contiguous();

    const int M = static_cast<int>(A.size(0));
    const int K = static_cast<int>(A.size(1));
    const int N = static_cast<int>(B.size(1));

    auto C = torch::zeros({M, N}, A.options());

    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    // KERNEL 1 - Global Memory Coalescing
    constexpr int BS = 32;
    dim3 grid(CEIL_DIV(N, BS), CEIL_DIV(M, BS)); // swapped N, M here
    dim3 block(BS, BS);

    matmul_kernel<<<grid, block>>>(A_ptr, B_ptr, C_ptr, M, N, K);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::string msg = std::string("Global Memory Coalescing kernel launch failed: ") + cudaGetErrorString(err);
        TORCH_CHECK(false, msg);
    }

    // Optionally synchronize here while debugging:
    // cudaDeviceSynchronize();

    return C;
}
