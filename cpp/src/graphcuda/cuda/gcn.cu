#include "gcn.cuh"
#include "gemm_cublas.cuh"
#include "spmm_cusparse.cuh"

std::tuple<torch::Tensor, torch::Tensor> gcn_conv_forward(
    torch::Tensor X,
    torch::Tensor adjm,   // sparse COO
    torch::Tensor weights,
    bool apply_relu) 
{
    // Sparse-dense multiplication: adjm * X
    torch::Tensor AX = spmm_cusparse(adjm, X);

    // Dense-dense multiplication: (adjm*X) * weights
    torch::Tensor y = gemm_cublas(AX, weights);

    torch::Tensor mask;
    if (apply_relu) {
        mask = y.gt(0).to(torch::kFloat32);
        return std::make_tuple(y * mask, mask);
    } else {
        mask = torch::empty({});
        return std::make_tuple(y, mask);
    }
}

std::tuple<torch::Tensor, torch::Tensor> gcn_conv_backward(
    torch::Tensor Y_grad,
    torch::Tensor X_cached,
    torch::Tensor adjm,           // sparse COO
    const torch::Tensor& weights,
    bool apply_relu,
    torch::Tensor mask_cached) 
{
    if (apply_relu) {
        Y_grad = Y_grad * mask_cached;
    }

    // Sparse-dense multiplication: adjm * X_cached
    torch::Tensor A_X = spmm_cusparse(adjm, X_cached);

    // Dense-dense multiplication: (A*X)^T * Y_grad -> dW
    torch::Tensor dW = gemm_cublas(A_X.transpose(0, 1), Y_grad);

    // Dense-dense multiplication: Y_grad * W^T
    torch::Tensor YWt = gemm_cublas(Y_grad, weights.transpose(0, 1));

    // Sparse-dense multiplication: adjm * (Y_grad * W^T) -> dX
    torch::Tensor dX = spmm_cusparse(adjm, YWt);

    return std::make_tuple(dX, dW);
}
