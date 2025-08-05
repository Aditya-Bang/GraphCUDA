Update it so that graphcuda init.py handles adding dlls for windows, automatically get all functions setup from pybind

```python
import torch
from . import _graphcuda

# Re-export everything in _graphcuda that's not a special attribute
__all__ = [name for name in dir(_graphcuda) if not name.startswith("_")]

globals().update({name: getattr(_graphcuda, name) for name in __all__})
```

TODO: New structure

.venv/
build/
tests/
    gcn/
        graphcuda_impl.py
        pytorch_impl.py
src/graphcuda/
    __init__.py
    __init__.pyi
    python/
        gcn.py
    cpp/
        gcn.cpp/cu
setup.py
pyproject.toml
README.md
.gitignore


make this work for gcn classes maybe?

Linux:
```bash
uv venv
source .venv/bin/activate
uv pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
uv pip install .
```

Windows cmd (requires cl so for example use x64 Native Command prompt for VS 2022):
```bash
uv venv
.venv\Scripts\activate
set DISTUTILS_USE_SDK=1 # if using x64 Native Tools Command Prompt for VS 2022
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4 # if using cuda 12.4
uv pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
# install all dependancies before this manual or use requirements.txt, then use uv pip install --no-build-isolation .
uv pip install .
```
