#include "spmm_cusparse.cuh"

torch::Tensor spmm_cusparse(torch::Tensor A, torch::Tensor B) {
    // ---- Checks ----
    TORCH_CHECK(A.is_sparse(), "A must be a sparse COO tensor");
    TORCH_CHECK(A.layout() == c10::kSparse, "A must be sparse COO");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "A.cols must match B.rows");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A and B must be CUDA tensors");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "A values must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "B must be float32");

    // Ensure contiguous row-major for B
    if (!B.is_contiguous()) B = B.contiguous();

    // Coalesce A (required for predictable COO)
    if (!A.is_coalesced()) A = A.coalesce();

    const int64_t M = A.size(0);
    const int64_t K = A.size(1);
    const int64_t N = B.size(1);
    const int64_t nnz = A._nnz();

    // Early-exit: if A has no nnz, return zeros
    auto C = torch::empty({M, N}, B.options());
    if (nnz == 0) {
        C.zero_();
        return C;
    }

    // Extract COO indices/values
    // indices: [2, nnz] int64 (rows, cols), values: [nnz] float32
    torch::Tensor indices = A._indices().contiguous();
    torch::Tensor values  = A._values().contiguous();

    TORCH_CHECK(indices.dtype() == torch::kLong, "A indices must be int64");
    TORCH_CHECK(values.dtype()  == torch::kFloat32, "A values must be float32");

    const int64_t* rows64 = indices[0].data_ptr<int64_t>();
    const int64_t* cols64 = indices[1].data_ptr<int64_t>();
    const float*   vals   = values.data_ptr<float>();
    const float*   Bptr   = B.data_ptr<float>();
    float*         Cptr   = C.data_ptr<float>();

    // cuSPARSE handle + stream
    cusparseHandle_t handle = at::cuda::getCurrentCUDASparseHandle();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cusparseSetStream(handle, stream);

    // Create descriptors
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;

    // A is COO (M x K) with int64 indices, base 0
    cusparseStatus_t stat;
    stat = cusparseCreateCoo(
        &matA,
        M, K, nnz,
        (void*)rows64,
        (void*)cols64,
        (void*)vals,
        CUSPARSE_INDEX_64I,         // index type (row+col)
        CUSPARSE_INDEX_BASE_ZERO,   // base
        CUDA_R_32F                  // values
    );
    TORCH_CHECK(stat == CUSPARSE_STATUS_SUCCESS, "cusparseCreateCoo failed: ", (int)stat);

    // B is dense (K x N), row-major
    // leading dim ldb = N (row-major descriptor)
    stat = cusparseCreateDnMat(&matB, K, N, N, (void*)Bptr, CUDA_R_32F, CUSPARSE_ORDER_ROW);
    TORCH_CHECK(stat == CUSPARSE_STATUS_SUCCESS, "cusparseCreateDnMat(B) failed: ", (int)stat);

    // C is dense (M x N), row-major, ldc = N
    stat = cusparseCreateDnMat(&matC, M, N, N, (void*)Cptr, CUDA_R_32F, CUSPARSE_ORDER_ROW);
    TORCH_CHECK(stat == CUSPARSE_STATUS_SUCCESS, "cusparseCreateDnMat(C) failed: ", (int)stat);

    // Compute C = alpha * A * B + beta * C
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // Workspace size
    size_t bufferSize = 0;
    void*  dBuffer    = nullptr;

    stat = cusparseSpMM_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,   // op(A)
        CUSPARSE_OPERATION_NON_TRANSPOSE,   // op(B)
        &alpha,
        matA,
        matB,
        &beta,
        matC,
        CUDA_R_32F,
        CUSPARSE_SPMM_ALG_DEFAULT,
        &bufferSize);
    TORCH_CHECK(stat == CUSPARSE_STATUS_SUCCESS, "cusparseSpMM_bufferSize failed: ", (int)stat);

    if (bufferSize > 0) {
        cudaError_t cerr = cudaMalloc(&dBuffer, bufferSize);
        TORCH_CHECK(cerr == cudaSuccess, "cudaMalloc workspace failed");
    }

    stat = cusparseSpMM(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        matA,
        matB,
        &beta,
        matC,
        CUDA_R_32F,
        CUSPARSE_SPMM_ALG_DEFAULT,
        dBuffer);
    TORCH_CHECK(stat == CUSPARSE_STATUS_SUCCESS, "cusparseSpMM failed: ", (int)stat);

    if (dBuffer) cudaFree(dBuffer);

    cusparseDestroySpMat(matA);
    cusparseDestroyDnMat(matB);
    cusparseDestroyDnMat(matC);

    return C;
}
