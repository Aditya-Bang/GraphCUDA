#pragma once

void launch_tiled_gemm(
    const float* A,
    const float* B,
    float* C,
    int M,
    int K,
    int N
);
