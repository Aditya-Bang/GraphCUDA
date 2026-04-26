#pragma once

#include <torch/extension.h>
#include <vector>

namespace graphcuda {
namespace ops {

std::vector<torch::Tensor> fused_spmm_gemm_relu_sm80_forward_cuda(
    torch::Tensor bsr_values_rm,
    torch::Tensor bsr_crow_i32,
    torch::Tensor bsr_col_i32,
    int64_t M,
    int64_t K1,
    torch::Tensor X,
    torch::Tensor W,
    torch::Tensor bias,
    bool apply_relu);

}  // namespace ops
}  // namespace graphcuda
