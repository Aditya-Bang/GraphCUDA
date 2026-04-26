#include <torch/extension.h>

#include "fused_spmm_gemm_relu_sm80_kernel.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward",
        &graphcuda::ops::fused_spmm_gemm_relu_sm80_forward_cuda,
        "Fused SPMM-GEMM-(Bias)-ReLU forward for custom BSR-RM layout on SM80"
    );
}
