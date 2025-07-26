cmake_minimum_required(VERSION 3.18 FATAL_ERROR) # CUDA support since 3.8+; 3.18 recommended for good CUDA features

project(HelloCUDA LANGUAGES CXX CUDA)

# Specify your GPU architecture (compute capability 7.5)
set(CMAKE_CUDA_ARCHITECTURES 75)

# Enable CUDA as a first-class language
enable_language(CUDA)

# Add the executable and specify source files
add_executable(hello_cuda main.cu hello_cuda.cu)

# Set the CUDA standard (optional)
set_target_properties(hello_cuda PROPERTIES
    CUDA_STANDARD 14
    CUDA_STANDARD_REQUIRED ON
)

# Make sure to compile .cu files with NVCC
set_target_properties(hello_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Link CUDA runtime (usually automatic)
target_link_libraries(hello_cuda PRIVATE cuda cudart)
