#include "matmul.cuh"

// Placeholder CUDA kernel (optional, not implemented)
__global__ void matmul_kernel(float* A, float* B, float* C, int M, int N, int K) {
    // TODO: implement CUDA matrix multiplication kernel
}

// Wrapper calling torch::matmul for now
torch::Tensor matmul(torch::Tensor A, torch::Tensor B) {
    return torch::matmul(A, B);
}
