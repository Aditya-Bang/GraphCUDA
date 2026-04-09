import os
from glob import glob

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_cuda_files(cuda_dir: str = os.path.join("src", "graphcuda", "cuda")) -> list[str]:
    cuda_dir = os.path.normpath(cuda_dir)
    pattern = os.path.join(cuda_dir, "**", "*.cu")
    cu_files = glob(pattern, recursive=True)
    if not cu_files:
        raise RuntimeError(f"No .cu files found in {cuda_dir} (pattern: {pattern})")
    return [os.path.normpath(p) for p in cu_files]


setup(
    ext_modules=[
        CUDAExtension(
            name="graphcuda._graphcuda",
            sources=get_cuda_files(),
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
            libraries=["cublas", "cusparse"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
