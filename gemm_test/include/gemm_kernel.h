#pragma once

#include <string>

using GemmFn = void (*)(
    const float* A,
    const float* B,
    float* C,
    int M, int K, int N
);

struct GemmKernel {
    std::string name;
    GemmFn launch;
};
