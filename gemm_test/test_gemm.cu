#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <iostream>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "utils.h"
#include "naive_gemm.h"

constexpr int WARMUP_ITERS = 5;
constexpr int TIMED_ITERS  = 20;

void init_matrix(std::vector<float>& mat) {
    for (auto& x : mat) {
        x = static_cast<float>(rand()) / RAND_MAX;
    }
}

bool compare_results(
    const std::vector<float>& ref,
    const std::vector<float>& test,
    float tol = 1e-3f
) {
    for (size_t i = 0; i < ref.size(); ++i) {
        float diff = std::fabs(ref[i] - test[i]);
        if (diff > tol) {
            std::printf(
                "Mismatch at %zu: ref=%f test=%f diff=%f\n",
                i, ref[i], test[i], diff
            );
            return false;
        }
    }
    return true;
}

float time_cublas_gemm(
    cublasHandle_t handle,
    const float* dA,
    const float* dB,
    float* dC,
    int M,
    int K,
    int N
) {
    float alpha = 1.0f;
    float beta  = 0.0f;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        CUBLAS_CHECK(cublasSgemm(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            N, M, K,
            &alpha,
            dB, N,
            dA, K,
            &beta,
            dC, N
        ));
    }

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < TIMED_ITERS; ++i) {
        CUBLAS_CHECK(cublasSgemm(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            N, M, K,
            &alpha,
            dB, N,
            dA, K,
            &beta,
            dC, N
        ));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms / TIMED_ITERS;
}

float time_naive_gemm(
    const float* dA,
    const float* dB,
    float* dC,
    int M,
    int K,
    int N
) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        launch_naive_gemm(dA, dB, dC, M, K, N);
    }

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < TIMED_ITERS; ++i) {
        launch_naive_gemm(dA, dB, dC, M, K, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms / TIMED_ITERS;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " M K N\n";
        return 1;
    }

    int M = std::atoi(argv[1]);
    int K = std::atoi(argv[2]);
    int N = std::atoi(argv[3]);

    std::printf("GEMM test: M=%d K=%d N=%d\n", M, K, N);

    size_t sizeA = M * K;
    size_t sizeB = K * N;
    size_t sizeC = M * N;

    std::vector<float> hA(sizeA), hB(sizeB);
    std::vector<float> hC_cublas(sizeC), hC_custom(sizeC);

    init_matrix(hA);
    init_matrix(hB);

    float *dA, *dB, *dC1, *dC2;
    CUDA_CHECK(cudaMalloc(&dA, sizeA * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, sizeB * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC1, sizeC * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC2, sizeC * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(dA, hA.data(), sizeA * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), sizeB * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    float cublas_ms = time_cublas_gemm(handle, dA, dB, dC1, M, K, N);
    float naive_ms  = time_naive_gemm(dA, dB, dC2, M, K, N);

    CUDA_CHECK(cudaMemcpy(hC_cublas.data(), dC1, sizeC * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hC_custom.data(), dC2, sizeC * sizeof(float), cudaMemcpyDeviceToHost));

    bool ok = compare_results(hC_cublas, hC_custom);

    double flops = 2.0 * M * N * K;
    double cublas_gflops = flops / (cublas_ms * 1e6);
    double naive_gflops  = flops / (naive_ms * 1e6);

    std::printf("\nResults:\n");
    std::printf("cuBLAS: %.3f ms (%.2f GFLOP/s)\n", cublas_ms, cublas_gflops);
    std::printf("Naive : %.3f ms (%.2f GFLOP/s)\n", naive_ms, naive_gflops);
    std::printf("Check : %s\n", ok ? "PASSED" : "FAILED");

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC1));
    CUDA_CHECK(cudaFree(dC2));

    return ok ? 0 : 1;
}
