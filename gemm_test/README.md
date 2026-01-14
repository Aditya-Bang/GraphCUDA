# Testing Harness for Custom GEMM

Testing (Windows)
1. Open `x64 Native Tools Command Prompt for VS 2022`

2. Run the following with your choice of `M`, `K`, `N`

```
cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
./Release/test_gemm.exe M K N
```

## Expected Output:

```
GEMM test: M=1000 K=1000 N=1000

Results:
cuBLAS: 1.724 ms (1160.02 GFLOP/s)
Naive : 11.580 ms (172.71 GFLOP/s)
Check : PASSED
```