from setuptools import setup, find_packages, Extension
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension
import os
from glob import glob
from typing import List

def get_cuda_files(cuda_dir: str = os.path.join("src", "graphcuda", "cuda")) -> List[str]:
    """
    Return a list of all .cu files under `cuda_dir` (recursively).
    Raises RuntimeError if no .cu files are found.
    """
    cuda_dir = os.path.normpath(cuda_dir)
    pattern = os.path.join(cuda_dir, "**", "*.cu")
    cu_files = glob(pattern, recursive=True)
    if not cu_files:
        raise RuntimeError(f"No .cu files found in {cuda_dir} (looked for pattern: {pattern})")
    cu_files = [os.path.normpath(p) for p in cu_files]
    return cu_files

setup(name='graphcuda',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    ext_modules=[CUDAExtension(
        name='graphcuda._graphcuda',
        sources=get_cuda_files(),
        extra_compile_args={
            'cxx': [],
            'nvcc': [
                '-O3',
                '--use_fast_math',
                '-gencode=arch=compute_75,code=sm_75',
            ]
        },
        libraries=['cublas', 'cusparse'],
    )],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
)
