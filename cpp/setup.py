from setuptools import setup, find_packages, Extension
from torch.utils import cpp_extension

setup(name='graphcuda',
      version='0.1.0',
      packages=find_packages(where='src'),
      package_dir={'': 'src'},
      ext_modules=[cpp_extension.CppExtension(
          name='graphcuda._graphcuda',
          sources=['src/graphcuda/cpp/gcn.cpp']
      )],
      cmdclass={'build_ext': cpp_extension.BuildExtension},
      zip_safe=False,
)
