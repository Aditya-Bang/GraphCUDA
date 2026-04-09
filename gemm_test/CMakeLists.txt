cmake_minimum_required(VERSION 3.25)

# Default: compile for the build machine's GPU (needs CUDA + visible GPU at configure time).
# Override: cmake -DCMAKE_CUDA_ARCHITECTURES=80 ..
set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA GPU architectures")

project(gemm_test LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

# C++ / CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Ensure separable compilation for CUDA
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

add_executable(test_gemm
    test_gemm.cu
    kernels/naive_gemm.cu
    kernels/shared_mem_gemm.cu
    kernels/tiled_gemm.cu
    kernels/vec_gemm.cu
)

target_include_directories(test_gemm PRIVATE include)

target_link_libraries(test_gemm PRIVATE CUDA::cublas)
