#pragma once

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <iostream>
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <ATen/cuda/CUDAContext.h>

torch::Tensor gemm_cublas(torch::Tensor A, torch::Tensor B);
