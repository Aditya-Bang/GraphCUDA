import torch

def dense_torch_impl(adjm_dense: torch.Tensor, X: torch.tensor, weights: torch.Tensor, activation: str = "relu"):
    """
    adjm_dense is a dense tensor
    X is a dense tensor
    weights is a dense tensor
    """
    Y = adjm_dense @ X @ weights
    if activation == "relu":
        Y = torch.relu(Y)
    return Y

def sparse_torch_impl(adjm_sparse: torch.Tensor, X: torch.tensor, weights: torch.Tensor, activation: str = "relu"):
    """
    adjm_sparse is a sparse tensor in CSR or COO format
    X is a dense tensor
    weights is a dense tensor
    """
    Y = torch.sparse.mm(adjm_sparse, X) @ weights
    if activation == "relu":
        Y = torch.relu(Y)
    return Y
