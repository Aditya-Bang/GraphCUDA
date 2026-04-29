from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT_DIR = Path(__file__).parent

setup(
    ext_modules=[
        CUDAExtension(
            name="graphcuda.ops._fused_spmm_gemm_relu.fwd._fused_spmm_gemm_relu_sm80_cuda",
            sources=[
                "csrc/bindings.cpp",
                "csrc/fused_spmm_gemm_relu/fused_spmm_gemm_relu_sm80_kernel.cu",
            ],
            include_dirs=[
                str(ROOT_DIR / "csrc" / "fused_spmm_gemm_relu"),
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
