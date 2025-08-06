#pragma once

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>


// Host wrapper function to call CUDA kernel or fallback to torch::matmul
torch::Tensor matmul(torch::Tensor A, torch::Tensor B);
