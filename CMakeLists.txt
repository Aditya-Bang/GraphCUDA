cmake_minimum_required(VERSION 3.24)

project(GraphCUDA LANGUAGES CXX CUDA)

option(GRAPHCUDA_BUILD_FUSED_SPMM_GEMM_RELU "Build the fused SpMM-GEMM-ReLU CUDA extension" ON)
set(GRAPHCUDA_CUDA_ARCHITECTURES "80" CACHE STRING "CUDA architectures for GraphCUDA kernels")

if(GRAPHCUDA_BUILD_FUSED_SPMM_GEMM_RELU)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)

    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path, end='')"
        OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
        RESULT_VARIABLE TORCH_CMAKE_PREFIX_RESULT
    )
    if(NOT TORCH_CMAKE_PREFIX_RESULT EQUAL 0)
        message(FATAL_ERROR "Unable to query torch.utils.cmake_prefix_path from ${Python3_EXECUTABLE}")
    endif()

    list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH}")
    find_package(Torch REQUIRED)
    find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib" REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${TORCH_CXX_FLAGS}")

    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '.so', end='')"
        OUTPUT_VARIABLE PYTHON_EXTENSION_SUFFIX
        RESULT_VARIABLE PYTHON_EXTENSION_SUFFIX_RESULT
    )
    if(NOT PYTHON_EXTENSION_SUFFIX_RESULT EQUAL 0)
        message(FATAL_ERROR "Unable to query Python extension suffix from ${Python3_EXECUTABLE}")
    endif()

    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_config_var('INCLUDEPY') or '', end='')"
        OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
        RESULT_VARIABLE PYTHON_INCLUDE_DIR_RESULT
    )
    if(NOT PYTHON_INCLUDE_DIR_RESULT EQUAL 0 OR NOT EXISTS "${PYTHON_INCLUDE_DIR}/Python.h")
        message(FATAL_ERROR "Unable to find Python.h for ${Python3_EXECUTABLE}; install Python development headers or pass a Python with headers")
    endif()

    add_library(_fused_spmm_gemm_relu_sm80_cuda MODULE
        csrc/bindings.cpp
        csrc/fused_spmm_gemm_relu/fused_spmm_gemm_relu_sm80_kernel.cu
    )

    target_include_directories(_fused_spmm_gemm_relu_sm80_cuda PRIVATE
        csrc/fused_spmm_gemm_relu
        ${PYTHON_INCLUDE_DIR}
        ${TORCH_INCLUDE_DIRS}
    )

    target_link_libraries(_fused_spmm_gemm_relu_sm80_cuda PRIVATE
        ${TORCH_LIBRARIES}
        ${TORCH_PYTHON_LIBRARY}
    )

    target_compile_features(_fused_spmm_gemm_relu_sm80_cuda PRIVATE cxx_std_17)
    target_compile_definitions(_fused_spmm_gemm_relu_sm80_cuda PRIVATE
        TORCH_EXTENSION_NAME=_fused_spmm_gemm_relu_sm80_cuda
    )
    target_compile_options(_fused_spmm_gemm_relu_sm80_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    )

    set_target_properties(_fused_spmm_gemm_relu_sm80_cuda PROPERTIES
        PREFIX ""
        SUFFIX "${PYTHON_EXTENSION_SUFFIX}"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/src/graphcuda/ops/_fused_spmm_gemm_relu/fwd"
        CUDA_ARCHITECTURES "${GRAPHCUDA_CUDA_ARCHITECTURES}"
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
    )
endif()
