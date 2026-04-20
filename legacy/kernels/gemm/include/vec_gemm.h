#pragma once

// vectorized mem access tiled GEMM
void launch_vec_gemm(
    const float* A,
    const float* B,
    float* C,
    int M,
    int K,
    int N
);
