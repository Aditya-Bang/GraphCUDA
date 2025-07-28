# Pybind Testing

## 1. Basic example of pybind11

```cpp
// example.cpp

#include <pybind11/pybind11.h>

namespace py = pybind11;

int add(int i, int j) {
    return i + j;
}

PYBIND11_MODULE(example, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("add", &add, "A function which adds two numbers",
      py::arg("i") = 1, py::arg("j") = 2);
}
```

Can build this with ```c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) example.cpp -o example$(python3 -m pybind11 --extension-suffix)```. Note that we must have a venv with ```pybind11``` installed. Additionally, for our python code to find this package, it's ```.so``` file must be in

```python
import sys
print(sys.path)
```

our system path for python or in the same working/current directory as our python script.


## 2. Using setuptools

```cpp
// example.cpp

#include <pybind11/pybind11.h>

namespace py = pybind11;

int add(int i, int j) {
    return i + j;
}

PYBIND11_MODULE(example, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("add", &add, "A function which adds two numbers",
      py::arg("i") = 1, py::arg("j") = 2);
}
```

```python
# setup.py

# setup.py
from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension("example", ["example.cpp"])
]

setup(
    name="example",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
```

Running ```python setup.py build_ext --inplace```, builds the module ```.so``` file in the same repo, requires a ```build``` folder though.

## 3. Using scikit-build-core and CMAKE (NOT WORKING)

```cmake
cmake_minimum_required(VERSION 3.14)
project(example LANGUAGES CXX)

find_package(pybind11 REQUIRED)

pybind11_add_module(example example.cpp)
```

```pyproject.toml
[project]
name = "pybind"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "pybind11>=3.0.0",
    "setuptools>=80.9.0",
]

[build-system]
requires = ["scikit-build-core[pyproject]>=0.4.0", "pybind11"]
build-backend = "scikit_build_core.build"

[tool.scikit-build]
build-dir = "build"
cmake.minimum-version = "3.14"
cmake.verbose = true
```

with ```example.cpp``` same as before, running ```uv pip install .``` builds the python package in ```/build```, but creates a package with

```
[project]
name = "pybind"
```

as in the ```pyproject.toml``` file, but that ```.so``` is not actually in the virtual env, but rather still in the ```/build``` folder.

## 4. Pytorch with setup.py

```cpp
// my_op.cpp

#include <torch/extension.h>

torch::Tensor add_forward(torch::Tensor a, torch::Tensor b) {
    return a + b;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &add_forward);
}
```

```python
# setup.py

from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='my_op',
    ext_modules=[CppExtension('my_op', ['my_op.cpp'])],
    cmdclass={'build_ext': BuildExtension}
)
```

Run ```python setup.py build_ext --inplace``` in python virtual env with necessary packages.

## 5. Pytorch with setup.py and pyproject.toml

```cpp
// my_op.cpp

#include <torch/extension.h>

torch::Tensor add_forward(torch::Tensor a, torch::Tensor b) {
    return a + b;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &add_forward);
}
```

```python
# setup.py
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='my_op',
    ext_modules=[CppExtension('my_op', ['my_op.cpp'])],
    cmdclass={'build_ext': BuildExtension}
)
```

Add this to your pyproject.toml
```
[build-system]
requires = ["setuptools", "wheel", "torch"]
build-backend = "setuptools.build_meta"
```

***IMPORTANT***: This can only run with ```python main.py```, not ```uv run main.py``` because uv creates a clean environment for each run.
