# Testing Harness for Custom GEMM

## Running the Tests

1. Install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (includes `nvcc`, CUDA headers, and cuBLAS). Ensure `nvcc` is on your `PATH`.

2. Configure and build (single-configuration generators put the binary in `build/`):

On Linux:

```sh
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
./test_gemm M K N
```

On Windows, run in the **x64 Native Tools Command Prompt for VS 2022**:

```cmd
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
.\Release\test_gemm.exe M K N
```

The build defaults to **`CMAKE_CUDA_ARCHITECTURES=native`**, so you need a CUDA-capable GPU visible when you run `cmake`. For cross-compilation, or for headless machines without a GPU at configure time, set an explicit architecture—for example:

```sh
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80
```

### Expected Output:

```
GEMM test: M=1000 K=1000 N=1000

Results:
cuBLAS    : 1.724 ms (1160.37 GFLOP/s)
naive     : 8.943 ms (223.65 GFLOP/s) [OK]
shared_mem: 7.763 ms (257.64 GFLOP/s) [OK]
tiled     : 5.998 ms (333.47 GFLOP/s) [OK]
vectorized: 4.512 ms (443.26 GFLOP/s) [OK]
```

## Adding a GEMM Kernel

1. Create a new `.h` and `.cu` file with a GEMM kernel and launcher following the `GemmFn` interface in `gemm_kernel.h`.
2. Include your new kernel in `test_gemm.cu` and add a `GemmKernel` to `std::vector<GemmKernel> kernels`.
