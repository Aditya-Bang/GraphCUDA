#include <torch/extension.h>
#include <torch/script.h>

std::tuple<torch::Tensor, torch::Tensor> gcn_conv_forward(
    torch::Tensor X,
    torch::Tensor adjm,
    torch::Tensor weights,
    bool apply_relu) {

    torch::Tensor y = adjm.mm(X).mm(weights);

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
    torch::Tensor adjm,
    const torch::Tensor& weights,
    bool apply_relu,
    torch::Tensor mask_cached) {

    if (apply_relu) {
        Y_grad = Y_grad * mask_cached;
    }

    torch::Tensor A_X = adjm.mm(X_cached);

    torch::Tensor dW = A_X.transpose(0, 1).mm(Y_grad);

    torch::Tensor dX = adjm.mm(Y_grad.mm(weights.transpose(0, 1)));

    return std::make_tuple(dX, dW);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gcn_conv_forward", &gcn_conv_forward, "GCN Convolution Forward Pass");
    m.def("gcn_conv_backward", &gcn_conv_backward, "GCN Convolution Backward Pass");
}
