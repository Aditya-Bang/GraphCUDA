#pragma once

#include <cstdio>
#include <stdexcept>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

// CUDA API error checking
#define CUDA_CHECK(err)                                                    \
    do {                                                                   \
        cudaError_t err_ = (err);                                          \
        if (err_ != cudaSuccess) {                                         \
            std::printf("CUDA error %d at %s:%d\n",                        \
                        err_, __FILE__, __LINE__);                         \
            throw std::runtime_error("CUDA error");                        \
        }                                                                  \
    } while (0)

// cuBLAS API error checking
#define CUBLAS_CHECK(err)                                                  \
    do {                                                                   \
        cublasStatus_t err_ = (err);                                       \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                               \
            std::printf("cuBLAS error %d at %s:%d\n",                      \
                        err_, __FILE__, __LINE__);                         \
            throw std::runtime_error("cuBLAS error");                      \
        }                                                                  \
    } while (0)
