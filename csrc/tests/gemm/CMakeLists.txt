cmake_minimum_required(VERSION 3.25)

# Default: compile for the build machine's GPU (needs CUDA + visible GPU at configure time).
# Override: cmake -DCMAKE_CUDA_ARCHITECTURES=80 ..
set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA GPU architectures")
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

# GraphCUDA repo root (this file lives in csrc/tests/gemm/)
get_filename_component(PROJECT_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
set(GEMM_KERNEL_SRC_DIR "${PROJECT_ROOT}/csrc/gemm/src")
set(GEMM_INCLUDE_DIR "${PROJECT_ROOT}/csrc/gemm/include")

project(gemm_test LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

# C++ / CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Ensure separable compilation for CUDA
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

add_executable(test_gemm
    "${CMAKE_CURRENT_LIST_DIR}/test_gemm.cu"
    "${GEMM_KERNEL_SRC_DIR}/naive_gemm.cu"
    "${GEMM_KERNEL_SRC_DIR}/shared_mem_gemm.cu"
    "${GEMM_KERNEL_SRC_DIR}/tiled_gemm.cu"
    "${GEMM_KERNEL_SRC_DIR}/vec_gemm.cu"
)

target_include_directories(test_gemm PRIVATE "${GEMM_INCLUDE_DIR}")

target_link_libraries(test_gemm PRIVATE CUDA::cublas)
