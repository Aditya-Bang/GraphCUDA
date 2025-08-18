# GraphCUDA

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
        pygeometric_impl.py
    matmul/
        graphcuda_impl.py
src/graphcuda/
    __init__.py
    python/
        __init__.py
        gcn.py
    cpp/
    cuda/
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
uv pip install -r requirements.txt
uv pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
uv pip install --no-build-isolation .
```

```bash
python tests/gcn/pytorch_impl.py
```

```bash
Model is on device: cuda:0
Data.x is on device: cuda:0
Data.edge_index is on device: cuda:0
Epoch 000 | Train Acc: 0.0714 | Val Acc: 0.1020 | Test Acc: 0.1030
Epoch 001 | Time: 0.1000s | Loss: 1.9518 | Train Acc: 0.2714 | Val Acc: 0.2300 | Test Acc: 0.2340
Epoch 002 | Time: 0.0091s | Loss: 1.9351 | Train Acc: 0.3929 | Val Acc: 0.2960 | Test Acc: 0.3220
Epoch 003 | Time: 0.0085s | Loss: 1.9200 | Train Acc: 0.4786 | Val Acc: 0.3640 | Test Acc: 0.3940
Epoch 004 | Time: 0.0081s | Loss: 1.9029 | Train Acc: 0.5643 | Val Acc: 0.4260 | Test Acc: 0.4490
Epoch 005 | Time: 0.0091s | Loss: 1.8822 | Train Acc: 0.6143 | Val Acc: 0.4780 | Test Acc: 0.4840
Epoch 006 | Time: 0.0102s | Loss: 1.8576 | Train Acc: 0.6429 | Val Acc: 0.4920 | Test Acc: 0.5130
Epoch 007 | Time: 0.0093s | Loss: 1.8290 | Train Acc: 0.6714 | Val Acc: 0.5080 | Test Acc: 0.5450
Epoch 008 | Time: 0.0091s | Loss: 1.7961 | Train Acc: 0.6857 | Val Acc: 0.5380 | Test Acc: 0.5670
Epoch 009 | Time: 0.0080s | Loss: 1.7591 | Train Acc: 0.6929 | Val Acc: 0.5740 | Test Acc: 0.5860
Epoch 010 | Time: 0.0081s | Loss: 1.7171 | Train Acc: 0.7000 | Val Acc: 0.5900 | Test Acc: 0.6060
Epoch 011 | Time: 0.0084s | Loss: 1.6698 | Train Acc: 0.7143 | Val Acc: 0.6040 | Test Acc: 0.6220
Epoch 012 | Time: 0.0092s | Loss: 1.6168 | Train Acc: 0.7143 | Val Acc: 0.6140 | Test Acc: 0.6320
Epoch 013 | Time: 0.0080s | Loss: 1.5579 | Train Acc: 0.7571 | Val Acc: 0.6280 | Test Acc: 0.6610
Epoch 014 | Time: 0.0091s | Loss: 1.4927 | Train Acc: 0.7929 | Val Acc: 0.6640 | Test Acc: 0.6870
Epoch 015 | Time: 0.0082s | Loss: 1.4216 | Train Acc: 0.8643 | Val Acc: 0.7020 | Test Acc: 0.7190
Epoch 016 | Time: 0.0090s | Loss: 1.3452 | Train Acc: 0.8929 | Val Acc: 0.7160 | Test Acc: 0.7530
Epoch 017 | Time: 0.0082s | Loss: 1.2645 | Train Acc: 0.9000 | Val Acc: 0.7280 | Test Acc: 0.7710
Epoch 018 | Time: 0.0092s | Loss: 1.1813 | Train Acc: 0.9214 | Val Acc: 0.7420 | Test Acc: 0.7740
Epoch 019 | Time: 0.0091s | Loss: 1.0973 | Train Acc: 0.9214 | Val Acc: 0.7480 | Test Acc: 0.7850
Epoch 020 | Time: 0.0090s | Loss: 1.0144 | Train Acc: 0.9214 | Val Acc: 0.7480 | Test Acc: 0.7920

Total training + testing time for 20 epochs: 0.2671 seconds
Average time per epoch (train + test): 0.013353 seconds
```

```bash
python tests/gcn/graphcuda_impl.py
```

```bash
Model is on device: cuda:0
Data.x is on device: cuda:0
Data.edge_index is on device: cuda:0
Epoch 000 | Train Acc: 0.1357 | Val Acc: 0.1360 | Test Acc: 0.1400
Epoch 001 | Time: 0.1608s | Loss: 1.9476 | Train Acc: 0.2857 | Val Acc: 0.2020 | Test Acc: 0.2340
Epoch 002 | Time: 0.0224s | Loss: 1.9406 | Train Acc: 0.4714 | Val Acc: 0.3020 | Test Acc: 0.3440
Epoch 003 | Time: 0.0233s | Loss: 1.9308 | Train Acc: 0.5929 | Val Acc: 0.4040 | Test Acc: 0.4510
Epoch 004 | Time: 0.0233s | Loss: 1.9202 | Train Acc: 0.6000 | Val Acc: 0.4720 | Test Acc: 0.4540
Epoch 005 | Time: 0.0223s | Loss: 1.9025 | Train Acc: 0.6286 | Val Acc: 0.4860 | Test Acc: 0.4740
Epoch 006 | Time: 0.0223s | Loss: 1.8835 | Train Acc: 0.6857 | Val Acc: 0.5420 | Test Acc: 0.5220
Epoch 007 | Time: 0.0174s | Loss: 1.8578 | Train Acc: 0.7714 | Val Acc: 0.5840 | Test Acc: 0.5770
Epoch 008 | Time: 0.0132s | Loss: 1.8282 | Train Acc: 0.7714 | Val Acc: 0.5820 | Test Acc: 0.5840
Epoch 009 | Time: 0.0124s | Loss: 1.8027 | Train Acc: 0.7929 | Val Acc: 0.6100 | Test Acc: 0.6090
Epoch 010 | Time: 0.0142s | Loss: 1.7783 | Train Acc: 0.8071 | Val Acc: 0.6120 | Test Acc: 0.6190
Epoch 011 | Time: 0.0122s | Loss: 1.7234 | Train Acc: 0.8143 | Val Acc: 0.6220 | Test Acc: 0.6230
Epoch 012 | Time: 0.0122s | Loss: 1.6921 | Train Acc: 0.8429 | Val Acc: 0.6480 | Test Acc: 0.6510
Epoch 013 | Time: 0.0126s | Loss: 1.6471 | Train Acc: 0.8500 | Val Acc: 0.6460 | Test Acc: 0.6450
Epoch 014 | Time: 0.0131s | Loss: 1.5896 | Train Acc: 0.8643 | Val Acc: 0.6820 | Test Acc: 0.6620
Epoch 015 | Time: 0.0131s | Loss: 1.5400 | Train Acc: 0.8714 | Val Acc: 0.6740 | Test Acc: 0.6710
Epoch 016 | Time: 0.0132s | Loss: 1.4690 | Train Acc: 0.8929 | Val Acc: 0.6860 | Test Acc: 0.6800
Epoch 017 | Time: 0.0128s | Loss: 1.3789 | Train Acc: 0.9071 | Val Acc: 0.7020 | Test Acc: 0.7080
Epoch 018 | Time: 0.0124s | Loss: 1.3449 | Train Acc: 0.8929 | Val Acc: 0.7060 | Test Acc: 0.7190
Epoch 019 | Time: 0.0142s | Loss: 1.2405 | Train Acc: 0.9143 | Val Acc: 0.7160 | Test Acc: 0.7360
Epoch 020 | Time: 0.0124s | Loss: 1.1817 | Train Acc: 0.9000 | Val Acc: 0.7200 | Test Acc: 0.7420

Total training + testing time for 20 epochs: 0.4597 seconds
Average time per epoch (train + test): 0.022984 seconds
```


Todos:

1. custom adjm sparse cuda memory implementation with custom class

1. get initial cuda kernel up and running
2. move cpp code to /, and move all other code to old/ folder, or better name
3. look at maxk gnn fast cuda implementation
4. look at gnn advisor
5. create wavelet gnn
6. run tests against pytorch-native implementation (not pygeometric), both sparse and non sparse mm
7. look at message passing class in pygeometric, and some other gnn's implemented in same folder
8. look for more gnns to optimize and implement
9. env var GRAPHCUDA_USE_CUDA, and cuda available torch function in setup.py


1. check if dlls work on windows for cuda (dependancy inspector + check torch/lib) (setuptools imports need torch+cu12.4?)
2. add support for adding dll directory in init.py based on os
3. figure out how to expose classes in pybind11 + testing, and import all classes/funcs in one command with ns
4. cudaextension setup.py
5. Test running a program with cublas, and checking if it works with cuda extension.
6. actually make optimized gcn + relu (figure out architecture, maybe only gcn layer classes and training loop in python/how to merge with pytorch)
7. proper documentation check all readmes

