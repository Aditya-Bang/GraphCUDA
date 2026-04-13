import torch
from graphcuda.bsr_rm import create_bsr_values_rm

def test_same_matrix_bsr_print():
    """Same 4x4 matrix as BSR(2,2); print crow, col, values (run: pytest -s)."""
    dense = torch.tensor(
        [
            [0, 0, 0, 3, 0],
            [1, 0, 0, 2, 0],
            [0, 4, 0, 0, 213],
            [5, 0, 0, 6, 32],
        ],
        dtype=torch.float32,
    )
    bsr = dense.to_sparse_bsr(blocksize=(2, 1))
    bsr.values_rm = create_bsr_values_rm(bsr)
    print(bsr.crow_indices())
    print(bsr.col_indices())
    print(bsr.values_rm)

