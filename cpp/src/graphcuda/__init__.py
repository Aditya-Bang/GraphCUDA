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


from .python.gcn import GCN

__all__ = ["GCN"]
