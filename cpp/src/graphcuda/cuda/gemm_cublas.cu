#include "gemm_cublas.cuh"

torch::Tensor gemm_cublas(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "A.cols must match B.rows");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Only float32 is supported here");

    // Ensure contiguous layout
    // Check and ensure contiguous layout only if needed
    if (!A.is_contiguous()) {
        A = A.contiguous();
    }
    if (!B.is_contiguous()) {
        B = B.contiguous();
    }

    const int M = static_cast<int>(A.size(0)); // rows of A, C
    const int K = static_cast<int>(A.size(1)); // cols of A, rows of B
    const int N = static_cast<int>(B.size(1)); // cols of B, C

    // Output tensor
    auto C = torch::empty({M, N}, A.options());

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // cuBLAS is column-major by default.
    // PyTorch is row-major, so we swap A and B in GEMM and transpose the operation:
    // C_rowmajor = A_rowmajor * B_rowmajor
    //   = (B_colmajor^T * A_colmajor^T)^T in column-major
    // The simplest: call cublasSgemm with swapped args and trans flags.

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cublasSetStream(handle, at::cuda::getCurrentCUDAStream());

    cublasStatus_t stat = cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,   // op(A), op(B) in column-major
        N,                          // m (columns of C in column-major)
        M,                          // n (rows of C in column-major)
        K,                          // k
        &alpha,
        B.data_ptr<float>(), N,     // B_colmajor, ldb
        A.data_ptr<float>(), K,     // A_colmajor, lda
        &beta,
        C.data_ptr<float>(), N      // C_colmajor, ldc
    );

    if (stat != CUBLAS_STATUS_SUCCESS) {
        TORCH_CHECK(false, "cublasSgemm failed with code ", stat);
    }

    return C;
}
