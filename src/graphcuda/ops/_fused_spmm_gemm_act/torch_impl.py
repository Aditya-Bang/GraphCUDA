import torch

def dense_torch_impl(adjm_dense: torch.Tensor, X: torch.tensor, weights: torch.Tensor, bias: torch.Tensor | None = None, activation: str = "relu"):
    """
    adjm_dense is a dense tensor
    X is a dense tensor
    weights is a dense tensor
    bias is optional; broadcastable to Y before activation (e.g. shape (N,) or (1, N))
    """
    Y = adjm_dense @ X @ weights
    if bias is not None:
        Y = Y + bias
    if activation == "relu":
        Y = torch.relu(Y)
    return Y

def sparse_torch_impl(adjm_sparse: torch.Tensor, X: torch.tensor, weights: torch.Tensor, bias: torch.Tensor | None = None, activation: str = "relu"):
    """
    adjm_sparse is a sparse tensor in CSR or COO format
    X is a dense tensor
    weights is a dense tensor
    bias is optional; broadcastable to Y before activation (e.g. shape (N,) or (1, N))
    """
    Y = torch.sparse.mm(adjm_sparse, X) @ weights
    if bias is not None:
        Y = Y + bias
    if activation == "relu":
        Y = torch.relu(Y)
    return Y
