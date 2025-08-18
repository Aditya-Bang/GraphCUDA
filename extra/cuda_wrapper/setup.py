from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='my_gcn_cuda',
    ext_modules=[
        CUDAExtension(
            name='my_gcn_cuda',
            sources=['my_gcn_cuda.cpp', 'my_gcn_cuda_kernel.cu'],
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
