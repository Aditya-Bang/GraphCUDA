import torch

def fused_spmm_gemm_relu_dense_torch_impl(
    adjm_dense: torch.Tensor,
    X: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor | None,
    apply_relu: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    adjm_dense is a dense tensor
    X is a dense tensor
    weights is a dense tensor
    bias is optional; broadcastable to Y before activation (e.g. shape (N,) or (1, N))
    Returns (Y, relu_mask) where relu_mask is bool pre-ReLU > 0 if apply_relu, else None.
    """
    Y = adjm_dense @ X @ weights
    if bias is not None:
        Y = Y + bias
    relu_mask: torch.Tensor | None = None
    if apply_relu:
        relu_mask = Y > 0
        Y = torch.relu(Y)
    return Y, relu_mask

def fused_spmm_gemm_relu_sparse_torch_impl(
    adjm_sparse: torch.Tensor,
    X: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor | None,
    apply_relu: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    adjm_sparse is a sparse tensor in CSR or COO format
    X is a dense tensor
    weights is a dense tensor
    bias is optional; broadcastable to Y before activation (e.g. shape (N,) or (1, N))
    Returns (Y, relu_mask) where relu_mask is bool pre-ReLU > 0 if apply_relu, else None.
    """
    Y = torch.sparse.mm(adjm_sparse, X) @ weights
    if bias is not None:
        Y = Y + bias
    relu_mask: torch.Tensor | None = None
    if apply_relu:
        relu_mask = Y > 0
        Y = torch.relu(Y)
    return Y, relu_mask
