# GraphCuda Python Implementations

How to run:
```bash
uv venv
source .venv/bin/activate # source .venv/Scripts/activate on windows
uv pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
uv sync --active
uv run --active app/numpy_gcn.py
uv run --active app/numpy_gcn_v2.py
uv run --active app/pytorch_gcn.py
uv run --active app/pygeometric_gcn.py
```
