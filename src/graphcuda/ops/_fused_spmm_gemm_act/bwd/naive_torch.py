import torch


def spmm_gemm_relu_backward_naive_torch_impl(
    grad_output: torch.Tensor,
    adj: torch.Tensor,
    X: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor | None,
    relu_mask: torch.Tensor | None,
    apply_relu: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Backward for fused ``Y = relu(A @ X @ W + bias)`` (or without relu / bias).

    ``adj`` may be dense ``(M, K1)`` or a sparse tensor (e.g. COO); behavior follows
    ``@`` vs :func:`torch.sparse.mm`. Returns ``(grad_X, grad_weights, grad_bias)``;
    ``grad_bias`` is None if ``bias`` was None. Adjacency is fixed (no ``grad_adj``).
    """
    # ------------------- Pre-activation gradient -------------------
    if apply_relu:
        if relu_mask is None:
            raise ValueError("relu_mask is required when apply_relu is True")
        g = grad_output * relu_mask.to(grad_output.dtype)
    else:
        g = grad_output

    # ------------------- Gradients for X and weights -------------------
    grad_P = g @ weights.transpose(0, 1)
    if adj.is_sparse: # only checks if COO
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
