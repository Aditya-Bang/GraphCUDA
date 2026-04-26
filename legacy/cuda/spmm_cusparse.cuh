#pragma once

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cusparse.h>


torch::Tensor spmm_cusparse(torch::Tensor A, torch::Tensor B);
