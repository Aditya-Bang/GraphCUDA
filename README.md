# GraphCUDA

GraphCUDA is a high-performance Graph Neural Network (GNN) library that leverages custom CUDA kernels and PyTorch C++/CUDA extensions for fast graph convolution and matrix operations. It is designed for research and benchmarking of GNNs on both sparse and dense graphs, with a focus on extensibility and speed.

## Features

- Custom CUDA kernels for GCN layers and matrix multiplication
- PyTorch extension with pybind11 for seamless Python integration
- Easy-to-use Python API, compatible with PyTorch tensors
- Example implementations and benchmarks against PyTorch/torch-geometric
- Cross-platform: Windows (with MSVC + CUDA) and Linux

## Installation

### Prerequisites

- Python 3.12+
- CUDA Toolkit (tested with CUDA 12.4)
- PyTorch (with CUDA support)
- C++17 compiler ([cl](https://visualstudio.microsoft.com/downloads/?q=build+tools) on Windows, GCC/Clang on Linux)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (recommended for fast virtualenv and pip)

### Linux

```bash
uv venv
source .venv/bin/activate
uv pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
uv pip install .
```

### Windows (x64 Native Tools Command Prompt for VS 2022)

```cmd
uv venv
.venv\Scripts\activate
set DISTUTILS_USE_SDK=1
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4
uv pip install -r requirements.txt
uv pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
uv pip install --no-build-isolation .
```

## Usage

After installation, you can import and use the CUDA-accelerated GCN layers and matrix multiplication functions directly from Python:

```python
import torch
from graphcuda import GCNConv

# Example: create a GCN model
class GCN(torch.nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_features, hidden_features, apply_relu=True)
        self.conv2 = GCNConv(hidden_features, out_features, apply_relu=False)

    def forward(self, x, adj):
        x = self.conv1(x, adj)
        x = self.conv2(x, adj)
        return torch.log_softmax(x, dim=1)

model = GCN(in_features, hidden_features, out_features)
output = model(x, adj)
```

See `tests/gcn/graphcuda_impl.py` and `tests/gcn/pytorch_impl.py` for full training and benchmarking scripts.

## Project Structure

```
src/graphcuda/
    __init__.py          # Handles DLLs on Windows, imports all CUDA/C++ functions
    python/              # Python GCNConv and utility code
    cpp/                 # C++ GCNConv layer implementations
    cuda/                # CUDA kernels and pybind11 bindings
        pybind.cu        # Exposes CUDA functions to Python
tests/
    gcn/                 # GCN training/benchmark scripts
    matmul/              # Matrix multiplication tests
setup.py                 # Script for building custom CUDA modules
pyproject.toml
requirements.txt
```

## Development

- All CUDA kernels are in `src/graphcuda/cuda/`
- Pybind11 bindings are in `pybind.cu`
- Python API is auto-populated from the extension module
- On Windows, DLL search paths are set automatically in `__init__.py`
- To add new CUDA functions, expose them in `pybind.cu` and rebuild

### Building from Source

If you change CUDA/C++ code, reinstall the package:

```bash
uv pip install --no-build-isolation .
```

### Running Tests

```bash
python tests/gcn/pytorch_impl.py
python tests/gcn/graphcuda_impl.py
```


Sample Output (PyTorch):
```
Model is on device: cuda:0
Data.x is on device: cuda:0
Data.edge_index is on device: cuda:0
Epoch 000 | Train Acc: 0.1214 | Val Acc: 0.1200 | Test Acc: 0.1300
Epoch 001 | Time: 0.1416s | Loss: 1.9478 | Train Acc: 0.2643 | Val Acc: 0.2080 | Test Acc: 0.2190
...
Epoch 020 | Time: 0.0137s | Loss: 1.0150 | Train Acc: 0.9071 | Val Acc: 0.7540 | Test Acc: 0.7840

Total training + testing time for 20 epochs: 0.4058 seconds
Average time per epoch (train + test): 0.020290 seconds
```

Sample Output (GraphCuda):
```
Model is on device: cuda:0
Data.x is on device: cuda:0
Data.edge_index is on device: cuda:0
Epoch 000 | Train Acc: 0.1429 | Val Acc: 0.1460 | Test Acc: 0.1470
Epoch 001 | Time: 0.0922s | Loss: 1.9448 | Train Acc: 0.3500 | Val Acc: 0.2440 | Test Acc: 0.2570
...
Epoch 020 | Time: 0.0091s | Loss: 0.9753 | Train Acc: 0.9214 | Val Acc: 0.7600 | Test Acc: 0.7960

Total training + testing time for 20 epochs: 0.2559 seconds
Average time per epoch (train + test): 0.012794 seconds
```

Benchmark Summary
- PyTorch baseline: ~0.0203s per epoch
- GraphCUDA optimized: ~0.0128s per epoch
- Speedup: **~1.6×** faster on the tested setup

## TODO

- Custom sparse adjacency CUDA memory implementation
- More optimized GCN and message-passing kernels
- Wavelet GNN and other advanced architectures
- Automated tests and CI
- Improved documentation

## License

MIT License

---

For questions or contributions, please open an issue or pull request!

