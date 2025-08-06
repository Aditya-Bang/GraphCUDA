#pragma once

#include <torch/extension.h>
#include <tuple>

std::tuple<torch::Tensor, torch::Tensor> gcn_conv_forward(
    torch::Tensor X,
    torch::Tensor adjm,
    torch::Tensor weights,
    bool apply_relu);

std::tuple<torch::Tensor, torch::Tensor> gcn_conv_backward(
    torch::Tensor Y_grad,
    torch::Tensor X_cached,
    torch::Tensor adjm,
    const torch::Tensor& weights,
    bool apply_relu,
    torch::Tensor mask_cached);
