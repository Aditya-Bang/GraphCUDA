# src/graphcuda/__init__.py

# This line is crucial. It ensures that PyTorch is loaded and its
# runtime libraries are initialized before your C++ extension is imported.
# Your C++ extension, built with torch.utils.cpp_extension, depends on PyTorch.
import torch

# Import your compiled C++ extension.
# Based on your setup.py (name='graphcuda', ext_modules=[cpp_extension.CppExtension(name='graphcuda._graphcuda', ...)]),
# the compiled C++ module will be named '_graphcuda' and reside within the 'graphcuda' package.
# The '.' indicates a relative import within the current package.
try:
    from . import _graphcuda
except ImportError as e:
    # This block provides a more informative error message if the C++ extension
    # fails to import, which can happen if PyTorch isn't installed, or if
    # there was an issue during the compilation of your C++ code.
    raise ImportError(
        f"Could not import the native C++ extension '_graphcuda'. "
        f"This typically means PyTorch is not installed, or the extension "
        f"failed to compile correctly. Please ensure PyTorch is installed "
        f"and try rebuilding your package. Original error: {e}"
    ) from e

# Re-export the 'add_forward' function from your C++ extension.
# This allows users to call `graphcuda.add_forward(...)` directly
# instead of `graphcuda._graphcuda.add_forward(...)`.
try:
    add_forward = _graphcuda.add_forward
except AttributeError as e:
    # This handles cases where the C++ extension loaded, but the expected
    # function 'add_forward' was not found within it. This could indicate
    # a mismatch between the C++ PYBIND11_MODULE definition and what's
    # expected in Python.
    raise AttributeError(
        f"The 'add_forward' function was not found in the native C++ extension '_graphcuda'. "
        f"Please check your C++ code (my_op.cpp) and its PYBIND11_MODULE definition. "
        f"Original error: {e}"
    ) from e

# Define __all__ to control what gets imported when a user does 'from graphcuda import *'.
# It's good practice to explicitly list the public API of your package.
__all__ = [
    'add_forward',
]

# You can also add a simple check or print statement for debugging purposes
# or to confirm the package loaded correctly (optional).
# print("graphcuda package loaded successfully with PyTorch support.")
