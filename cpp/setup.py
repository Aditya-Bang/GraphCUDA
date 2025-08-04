from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='graphcuda',
      version='0.1.0',
      packages=['graphcuda'],
      package_dir={'': 'src'},
      ext_modules=[cpp_extension.CppExtension(
          name='graphcuda._graphcuda',
          sources=['src/graphcuda/cuda/gcn.cpp']
      )],
      cmdclass={'build_ext': cpp_extension.BuildExtension},
      zip_safe=False,
)
