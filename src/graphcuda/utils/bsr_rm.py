import torch

def create_bsr_values_rm(bsr: torch.Tensor) -> torch.Tensor:
    """
    Returns a 1D tensor containing the BSR payload reordered so that each row-block is flattened in row-major order.

    Assumes:
      - 2D BSR tensor
      - blocksize == (block_rows, 1)
    """
    if bsr.layout != torch.sparse_bsr:
        raise TypeError("expected a torch.sparse_bsr tensor")
    if bsr.dim() != 2:
        raise ValueError("expected a 2D BSR tensor")

    crow = bsr.crow_indices()
    vals = bsr.values()
    block_rows, block_cols = vals.shape[-2:]
    if block_cols != 1:
        raise ValueError(f"expected blocksize (b, 1), got ({block_rows}, {block_cols})")

    values_rm = []
    for rb in range(crow.numel() - 1):
        start = int(crow[rb])
        end = int(crow[rb + 1])
        chunk = vals[start:end, :, 0]
        values_rm.append(chunk.transpose(0, 1).contiguous().reshape(-1))
    
    return torch.cat(values_rm, dim=0).detach()

def choose_optimal_block_row_size(dense_matrix: torch.Tensor) -> int:
    """
    TODO: Implement a heuristic to choose the optimal block row size for the given dense matrix.
    Must be a multiple of 16.
    """
    return 16


def to_sparse_bsr_rm(dense_matrix: torch.Tensor) -> torch.Tensor:
    BLOCK_ROW_SIZE = choose_optimal_block_row_size(dense_matrix)
    assert BLOCK_ROW_SIZE % 16 == 0
    assert dense_matrix.dim() == 2
    
    m, n = dense_matrix.shape
    
    m_ceil = (m + BLOCK_ROW_SIZE - 1) // BLOCK_ROW_SIZE * BLOCK_ROW_SIZE
    
    dense_matrix = torch.nn.functional.pad(dense_matrix, (0, 0, 0, m_ceil - m))
    bsr = dense_matrix.to_sparse_bsr(blocksize=(BLOCK_ROW_SIZE, 1))
    bsr.values_rm = create_bsr_values_rm(bsr)
    bsr.pre_padded_shape = (m, n) # need to set custom shape here because shape is not overrideable in tensor.

    return bsr
