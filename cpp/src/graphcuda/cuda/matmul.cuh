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

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

torch::Tensor matmul1(torch::Tensor A, torch::Tensor B);
torch::Tensor matmul2(torch::Tensor A, torch::Tensor B);
torch::Tensor matmul3(torch::Tensor A, torch::Tensor B);
