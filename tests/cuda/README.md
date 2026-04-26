# Running this Test

```
cmake -S . -B /tmp/graphcuda-cmake-build -DPython3_EXECUTABLE=$PWD/.venv/bin/python
cmake --build /tmp/graphcuda-cmake-build -j 4
ython tests/cuda/validate_fused_spmm_gemm_relu_fwd_cuda.py --M 1024 --K1 1024 --K2 1024 --N 16 --adj-density 0.005 --bias --apply-relu
```
