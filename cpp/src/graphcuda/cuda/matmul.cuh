#pragma once

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
// #include <cmath>
// #include <algorithm>
// #include <cstdio>
// #include <cstdlib>
// #include <cublas_v2.h>


typedef unsigned int uint;

// Host wrapper function to call CUDA kernel or fallback to torch::matmul
torch::Tensor matmul(torch::Tensor A, torch::Tensor B);
