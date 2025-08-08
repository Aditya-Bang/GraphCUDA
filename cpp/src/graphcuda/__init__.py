import os
import sys

try:
    import torch
    if sys.platform.startswith('win'):
        torch_lib_path = os.path.join(os.path.dirname(os.path.dirname(torch.__file__)), 'torch', 'lib')
        os.add_dll_directory(torch_lib_path)
except ImportError:
    print("Torch not found. Please ensure it's installed in your environment.")
    sys.exit(1)

try:
    from . import _graphcuda as cpp_ext
except ImportError as e:
    raise ImportError(
        "Could not import the native C++ extension '_graphcuda'. Original error: " + str(e)
    ) from e

from .python.gcn import GCN
matmul1 = cpp_ext.matmul1
matmul2 = cpp_ext.matmul2

__all__ = ["GCN", "matmul1", "matmul2"]
