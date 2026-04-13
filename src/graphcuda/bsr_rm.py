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
