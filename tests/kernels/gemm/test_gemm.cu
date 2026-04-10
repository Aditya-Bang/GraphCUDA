#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <iostream>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "utils.h"
#include "gemm_kernel.h"

#include "naive_gemm.h"
#include "shared_mem_gemm.h"
#include "tiled_gemm.h"
#include "vec_gemm.h"

constexpr int WARMUP_ITERS = 20;
constexpr int TIMED_ITERS  = 100;

std::vector<GemmKernel> kernels = {
    {"naive", launch_naive_gemm},
    {"shared_mem", launch_shared_mem_gemm},
    {"tiled", launch_tiled_gemm},
    {"vectorized", launch_vec_gemm}
};

void init_matrix(std::vector<float>& mat) {
    for (auto& x : mat) {
        int r = (rand() % 11) - 5;   // integers in [-5, 5]
        x = static_cast<float>(r);
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

float time_custom_gemm(
    GemmFn launch,
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
        launch(dA, dB, dC, M, K, N);
    }

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < TIMED_ITERS; ++i) {
        launch(dA, dB, dC, M, K, N);
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
    std::vector<float> hC_ref(sizeC), hC_test(sizeC);

    init_matrix(hA);
    init_matrix(hB);

    float *dA, *dB, *dC_ref, *dC_test;
    CUDA_CHECK(cudaMalloc(&dA, sizeA * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, sizeB * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC_ref, sizeC * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC_test, sizeC * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(dA, hA.data(), sizeA * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), sizeB * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Reference
    float cublas_ms = time_cublas_gemm(handle, dA, dB, dC_ref, M, K, N);
    CUDA_CHECK(cudaMemcpy(hC_ref.data(), dC_ref, sizeC * sizeof(float), cudaMemcpyDeviceToHost));

    double flops = 2.0 * M * N * K;
    double cublas_gflops = flops / (cublas_ms * 1e6);

    std::printf("\nResults:\n");
    std::printf(
        "%-10s: %.3f ms (%.2f GFLOP/s)\n",
        "cuBLAS", cublas_ms, cublas_gflops
    );

    // Custom kernels
    for (const auto& k : kernels) {
        CUDA_CHECK(cudaMemset(dC_test, 0, sizeC * sizeof(float)));

        float ms = time_custom_gemm(
            k.launch, dA, dB, dC_test, M, K, N
        );

        CUDA_CHECK(cudaMemcpy(
            hC_test.data(), dC_test,
            sizeC * sizeof(float),
            cudaMemcpyDeviceToHost
        ));

        bool ok = compare_results(hC_ref, hC_test);

        double gflops = flops / (ms * 1e6);

        std::printf(
            "%-10s: %.3f ms (%.2f GFLOP/s) [%s]\n",
            k.name.c_str(), ms, gflops,
            ok ? "OK" : "FAIL"
        );
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC_ref));
    CUDA_CHECK(cudaFree(dC_test));

    return 0;
}
