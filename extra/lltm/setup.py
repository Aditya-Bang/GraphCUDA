from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='lltm_cpp',
      ext_modules=[cpp_extension.CppExtension(
          'lltm_cpp',
          ['src/lltm.cpp'],
          extra_compile_args=['-O3']
      )],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
