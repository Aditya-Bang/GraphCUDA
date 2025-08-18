#include <torch/extension.h>
#include <torch/script.h>


torch::Tensor gcn_forward_cuda_kernel(
    torch::Tensor X,
    torch::Tensor adjm,
    torch::Tensor weights,
    bool apply_relu,
    torch::Tensor& mask_out);

std::tuple<torch::Tensor, torch::Tensor> gcn_backward_cuda_kernel(
    torch::Tensor Y_grad,
    torch::Tensor X_cached,
    torch::Tensor adjm,
    const torch::Tensor& weights,
    bool apply_relu,
    torch::Tensor mask_cached);


std::tuple<torch::Tensor, torch::Tensor> gcn_conv_forward(
    torch::Tensor X,
    torch::Tensor adjm,
    torch::Tensor weights,
    bool apply_relu) {

    // Ensure all inputs are on the same device
    TORCH_CHECK(X.device() == adjm.device() && X.device() == weights.device(),
                "All input tensors must be on the same device for GCNConv.");

    // Check if all input tensors are on a CUDA device.
    if (X.is_cuda()) {
        torch::Tensor mask_out; // Declare mask_out here to be filled by CUDA kernel
        torch::Tensor Y = gcn_forward_cuda_kernel(X, adjm, weights, apply_relu, mask_out);
        return std::make_tuple(Y, mask_out);
    } else {
        // Fallback to CPU implementation if any tensor is not on CUDA.
        torch::Tensor y = adjm.mm(X).mm(weights);
        torch::Tensor mask;
        if (apply_relu) {
            mask = y.gt(0).to(torch::kFloat32);
            return std::make_tuple(y * mask, mask);
        } else {
            mask = torch::empty({}, y.options()); // Empty tensor on CPU
            return std::make_tuple(y, mask);
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor> gcn_conv_backward(
    torch::Tensor Y_grad,
    torch::Tensor X_cached,
    torch::Tensor adjm,
    const torch::Tensor& weights,
    bool apply_relu,
    torch::Tensor mask_cached) {

    // Ensure all inputs are on the same device
    TORCH_CHECK(Y_grad.device() == X_cached.device() && Y_grad.device() == adjm.device() && Y_grad.device() == weights.device(),
                "All gradient tensors must be on the same device for GCNConv backward.");
    if (!mask_cached.is_empty()) {
        TORCH_CHECK(Y_grad.device() == mask_cached.device(), "Mask tensor must be on the same device as gradients.");
    }


    // Check if all relevant tensors are on a CUDA device.
    if (Y_grad.is_cuda()) {
        // Dispatch to the CUDA kernel for backward computation
        return gcn_backward_cuda_kernel(Y_grad, X_cached, adjm, weights, apply_relu, mask_cached);
    } else {
        // Fallback to CPU implementation
        if (apply_relu) {
            Y_grad = Y_grad * mask_cached;
        }

        torch::Tensor A_X = adjm.mm(X_cached);
        torch::Tensor dW = A_X.transpose(0, 1).mm(Y_grad);
        torch::Tensor dX = adjm.mm(Y_grad.mm(weights.transpose(0, 1)));

        return std::make_tuple(dX, dW);
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gcn_conv_forward", &gcn_conv_forward, "GCN Convolution Forward Pass (CUDA/CPU Dispatch)");
    m.def("gcn_conv_backward", &gcn_conv_backward, "GCN Convolution Backward Pass (CUDA/CPU Dispatch)");
}
