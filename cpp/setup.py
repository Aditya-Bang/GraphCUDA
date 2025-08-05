from setuptools import setup, find_packages, Extension
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension

setup(name='graphcuda',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    ext_modules=[CUDAExtension(
        name='graphcuda._graphcuda',
        sources=['src/graphcuda/cuda/gcn.cu'],
        extra_compile_args={
            'cxx': [],
            'nvcc': [
                '-O3',
                '--use_fast_math',
                '-gencode=arch=compute_75,code=sm_75',
            ]
        },
        libraries=['cublas'],
    )],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
)
