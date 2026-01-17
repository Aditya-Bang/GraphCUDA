# Testing Harness for Custom GEMM

## Running the Tests (Windows)

1. Open `x64 Native Tools Command Prompt for VS 2022`

2. Run the following with your choice of `M`, `K`, `N`

```
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
./Release/test_gemm.exe M K N
```

### Expected Output:

```
GEMM test: M=1000 K=1000 N=1000

Results:
cuBLAS    : 1.724 ms (1160.37 GFLOP/s)
naive     : 8.943 ms (223.65 GFLOP/s) [OK]
shared_mem: 7.763 ms (257.64 GFLOP/s) [OK]
tiled     : 5.998 ms (333.47 GFLOP/s) [OK]
```

## Adding a GEMM Kernel

1. Create a new `.h` and `.cu` file with GEMM kernel and GEMM launcher following `GemmFn` interface in `gemm_kernel.h`.
2. Include your new kernel in `test_gemm.cu` and add a `GemmKernel` to `std::vector<GemmKernel> kernels`.
