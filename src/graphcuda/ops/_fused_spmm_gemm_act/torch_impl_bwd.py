import torch


def torch_backward(
    grad_output: torch.Tensor,
    adj: torch.Tensor,
    X: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor | None,
    relu_mask: torch.Tensor | None,
    *,
    activation: str = "relu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Backward for fused ``Y = relu(A @ X @ W + bias)`` (or without relu / bias), matching
    :func:`dense_torch_impl` / :func:`sparse_torch_impl`.

    ``adj`` may be dense ``(M, K1)`` or a sparse tensor (e.g. CSR); behavior follows
    ``@`` vs :func:`torch.sparse.mm`. Returns ``(grad_X, grad_weights, grad_bias)``;
    ``grad_bias`` is None if ``bias`` was None. Adjacency is fixed (no ``grad_adj``).
    """
    # ------------------- Pre-activation gradient -------------------
    if activation == "relu":
        if relu_mask is None:
            raise ValueError("relu_mask is required when activation is 'relu'")
        g = grad_output * relu_mask.to(grad_output.dtype)
    else:
        g = grad_output

    # ------------------- Gradients for X and weights -------------------
    grad_P = g @ weights.transpose(0, 1)
    if adj.is_sparse:
        P = torch.sparse.mm(adj, X)
        grad_X = torch.sparse.mm(adj.transpose(0, 1), grad_P)
    else:
        P = adj @ X
        grad_X = adj.transpose(0, 1) @ grad_P
    grad_weights = P.transpose(0, 1) @ g

    # ------------------- Bias gradient -------------------
    if bias is None:
        grad_bias = None
    elif bias.dim() == 1:
        grad_bias = g.sum(dim=0)
    else:
        grad_bias = g.sum(dim=0, keepdim=True)

    return grad_X, grad_weights, grad_bias
