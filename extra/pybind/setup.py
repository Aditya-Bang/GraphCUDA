# setup.py
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='my_op',
    ext_modules=[CppExtension('my_op', ['my_op.cpp'])],
    cmdclass={'build_ext': BuildExtension}
)
