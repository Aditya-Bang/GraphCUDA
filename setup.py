from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent
CSRC_DIR = ROOT / "csrc"
FUSED_SPMM_GEMM_RELU_DIR = CSRC_DIR / "fused_spmm_gemm_relu"


setup(
    ext_modules=[
        CUDAExtension(
            name="graphcuda.ops._fused_spmm_gemm_relu.fwd._fused_spmm_gemm_relu_sm80_cuda",
            sources=[
                str(CSRC_DIR / "bindings.cpp"),
                str(FUSED_SPMM_GEMM_RELU_DIR / "fused_spmm_gemm_relu_sm80_kernel.cu"),
            ],
            include_dirs=[
                str(FUSED_SPMM_GEMM_RELU_DIR),
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--expt-relaxed-constexpr",
                    "-lineinfo",
                    "-gencode=arch=compute_80,code=sm_80",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
