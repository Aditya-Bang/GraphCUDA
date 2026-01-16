cmake_minimum_required(VERSION 3.18)

project(gemm_test LANGUAGES CXX CUDA)

# C++ / CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Ensure separable compilation for CUDA
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Optional: target your GPU (T1000 = Turing, sm_75)
set(CMAKE_CUDA_ARCHITECTURES 75)

add_executable(test_gemm
    test_gemm.cu
    kernels/naive_gemm.cu
    kernels/shared_mem_gemm.cu
    kernels/tiled_gemm.cu
)

target_include_directories(test_gemm PRIVATE include)

target_link_libraries(test_gemm PRIVATE cublas)
