#pragma once

#include <torch/extension.h>
#include <cublas_v2.h>
#include <ATen/cuda/CUDAContext.h>

torch::Tensor gemm_cublas(torch::Tensor A, torch::Tensor B);
