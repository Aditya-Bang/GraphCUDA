#include "matmul.cuh"

template <int BLOCKSIZE>
__global__ void sgemm_shared_mem_block(const int M, const int N, const int K,
                                        const float alpha,
                                        const float* __restrict__ A,
                                        const float* __restrict__ B,
                                        const float beta,
                                        float* __restrict__ C) {
    // block coordinates (block row, block col)
    const unsigned int cRow = blockIdx.x;
    const unsigned int cCol = blockIdx.y;

    // thread coordinates inside block
    const unsigned int threadRow = threadIdx.y; // 0..BLOCKSIZE-1
    const unsigned int threadCol = threadIdx.x; // 0..BLOCKSIZE-1

    // global indices for this thread's element in the C block
    const int globalRow = cRow * BLOCKSIZE + threadRow;
    const int globalCol = cCol * BLOCKSIZE + threadCol;

    // shared memory for A and B blocks
    __shared__ float As[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

    float tmp = 0.0f;

    // iterate over K in BLOCKSIZE chunks
    for (int bk = 0; bk < K; bk += BLOCKSIZE) {
        // Load A block: rows (cRow*BLOCKSIZE + threadRow), cols (bk + threadCol)
        const int a_row = cRow * BLOCKSIZE + threadRow;
        const int a_col = bk + threadCol;
        if (a_row < M && a_col < K) {
            As[threadRow * BLOCKSIZE + threadCol] = A[a_row * K + a_col];
        } else {
            As[threadRow * BLOCKSIZE + threadCol] = 0.0f;
        }

        // Load B block: rows (bk + threadRow), cols (cCol*BLOCKSIZE + threadCol)
        const int b_row = bk + threadRow;
        const int b_col = cCol * BLOCKSIZE + threadCol;
        if (b_row < K && b_col < N) {
            Bs[threadRow * BLOCKSIZE + threadCol] = B[b_row * N + b_col];
        } else {
            Bs[threadRow * BLOCKSIZE + threadCol] = 0.0f;
        }

        __syncthreads();

        // dot product across the BLOCKSIZE dimension
        for (int d = 0; d < BLOCKSIZE; ++d) {
            tmp += As[threadRow * BLOCKSIZE + d] * Bs[d * BLOCKSIZE + threadCol];
        }

        __syncthreads();
    }

    // Write back to C if inside bounds
    if (globalRow < M && globalCol < N) {
        C[globalRow * N + globalCol] = alpha * tmp + beta * C[globalRow * N + globalCol];
    }
}

// PyTorch wrapper
torch::Tensor matmul2(torch::Tensor A, torch::Tensor B) {
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

    // KERNEL 1 - Shared Memory Cache-Blocking
    constexpr int BS = 32;
    dim3 grid(CEIL_DIV(M, BS), CEIL_DIV(N, BS));
    dim3 block(BS, BS);

    sgemm_shared_mem_block<BS><<<grid, block>>>(
        M, N, K, 1.0f, A_ptr, B_ptr, 0.0f, C_ptr);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::string msg = std::string("sgemm kernel launch failed: ") + cudaGetErrorString(err);
        TORCH_CHECK(false, msg);
    }

    // Optionally synchronize here while debugging:
    // cudaDeviceSynchronize();

    return C;
}
