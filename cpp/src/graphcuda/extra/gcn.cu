#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublas_v2.h>

#define CUBLAS_CHECK(expr) do {                         \
    cublasStatus_t status = (expr);                     \
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS,        \
                "cuBLAS error: ", status);              \
} while (0)



std::tuple<torch::Tensor, torch::Tensor> gcn_conv_forward(
    torch::Tensor X,
    torch::Tensor adjm,
    torch::Tensor weights,
    bool apply_relu) {
    
    TORCH_CHECK(X.is_cuda(), "X must be a CUDA tensor");
    TORCH_CHECK(adjm.is_cuda(), "adjm must be a CUDA tensor");
    TORCH_CHECK(weights.is_cuda(), "weights must be a CUDA tensor");

    auto y = at::empty({adjm.size(0), weights.size(1)}, X.options());

    auto handle = at::cuda::getCurrentCUDABlasHandle();
    auto stream = at::cuda::getCurrentCUDAStream();

    cublasSetStream(handle, stream);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // temp = adjm * X
    auto temp = at::empty({adjm.size(0), X.size(1)}, X.options());

    // cuBLAS expects column-major, so we compute: temp^T = X^T * adjm^T
    CUBLAS_CHECK(cublasSgemm(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_T,
        X.size(1), adjm.size(0), X.size(0),
        &alpha,
        X.data_ptr<float>(), X.size(1),
        adjm.data_ptr<float>(), adjm.size(1),
        &beta,
        temp.data_ptr<float>(), temp.size(1)
    ));

    // y = temp * weights
    CUBLAS_CHECK(cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        weights.size(1), temp.size(0), temp.size(1),
        &alpha,
        weights.data_ptr<float>(), weights.size(1),
        temp.data_ptr<float>(), temp.size(1),
        &beta,
        y.data_ptr<float>(), y.size(1)
    ));

    torch::Tensor mask;
    if (apply_relu) {
        mask = y > 0;
        y = y * mask.to(y.scalar_type());
    } else {
        mask = torch::empty({}, X.options());
    }

    return std::make_tuple(y, mask);
}


std::tuple<torch::Tensor, torch::Tensor> gcn_conv_backward(
    torch::Tensor Y_grad,
    torch::Tensor X_cached,
    torch::Tensor adjm,
    const torch::Tensor& weights,
    bool apply_relu,
    torch::Tensor mask_cached) {

    TORCH_CHECK(Y_grad.is_cuda(), "Y_grad must be a CUDA tensor");
    TORCH_CHECK(X_cached.is_cuda(), "X_cached must be a CUDA tensor");
    TORCH_CHECK(adjm.is_cuda(), "adjm must be a CUDA tensor");
    TORCH_CHECK(weights.is_cuda(), "weights must be a CUDA tensor");

    if (apply_relu) {
        Y_grad = Y_grad * mask_cached.to(Y_grad.scalar_type());
    }

    auto dW = at::empty({X_cached.size(1), weights.size(1)}, X_cached.options());
    auto dX = at::empty_like(X_cached);

    auto handle = at::cuda::getCurrentCUDABlasHandle();
    auto stream = at::cuda::getCurrentCUDAStream();
    cublasSetStream(handle, stream);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // A_X = adjm * X_cached
    auto A_X = at::empty({adjm.size(0), X_cached.size(1)}, X_cached.options());
    CUBLAS_CHECK(cublasSgemm(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_T,
        X_cached.size(1), adjm.size(0), X_cached.size(0),
        &alpha,
        X_cached.data_ptr<float>(), X_cached.size(1),
        adjm.data_ptr<float>(), adjm.size(1),
        &beta,
        A_X.data_ptr<float>(), A_X.size(1)
    ));

    // dW = A_X^T * Y_grad
    CUBLAS_CHECK(cublasSgemm(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        A_X.size(1), Y_grad.size(1), A_X.size(0),
        &alpha,
        A_X.data_ptr<float>(), A_X.size(1),
        Y_grad.data_ptr<float>(), Y_grad.size(1),
        &beta,
        dW.data_ptr<float>(), dW.size(1)
    ));

    // dX = adjm * (Y_grad * W^T)
    auto YW = at::empty({Y_grad.size(0), weights.size(0)}, Y_grad.options());
    CUBLAS_CHECK(cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        weights.size(0), Y_grad.size(0), weights.size(1),
        &alpha,
        weights.data_ptr<float>(), weights.size(1),
        Y_grad.data_ptr<float>(), Y_grad.size(1),
        &beta,
        YW.data_ptr<float>(), YW.size(0)
    ));

    CUBLAS_CHECK(cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        YW.size(1), adjm.size(0), YW.size(0),
        &alpha,
        YW.data_ptr<float>(), YW.size(1),
        adjm.data_ptr<float>(), adjm.size(1),
        &beta,
        dX.data_ptr<float>(), dX.size(1)
    ));

    return std::make_tuple(dX, dW);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gcn_conv_forward", &gcn_conv_forward, "GCN Convolution Forward (cuBLAS)");
    m.def("gcn_conv_backward", &gcn_conv_backward, "GCN Convolution Backward (cuBLAS)");
}
