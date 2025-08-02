import torch

try:
    from . import _graphcuda as cpp_ext
except ImportError as e:
    raise ImportError(
        f"Could not import the native C++ extension '_graphcuda'. "
        f"This typically means PyTorch is not installed, or the extension "
        f"failed to compile correctly. Please ensure PyTorch is installed "
        f"and try rebuilding your package. Original error: {e}"
    ) from e

# Expose the C++ classes to the Python namespace
GCN = cpp_ext.GCN
GCNConv = cpp_ext.GCNConv

__all__ = [
    'GCN',
    'GCNConv',
]
