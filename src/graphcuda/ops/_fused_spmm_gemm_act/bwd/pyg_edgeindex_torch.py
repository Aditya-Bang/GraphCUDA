import torch


def spmm_gemm_relu_backward_pyg_edgeindex_torch_impl(
    grad_output: torch.Tensor,
    edge_index: torch.Tensor,          # normalized edge_index from gcn_norm
    edge_weight: torch.Tensor | None,  # normalized edge_weight from gcn_norm
    X: torch.Tensor,                   # [num_nodes, in_channels]
    weights: torch.Tensor,             # [in_channels, out_channels]
    bias: torch.Tensor | None,
    relu_mask: torch.Tensor | None,
    apply_relu: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Backward for Y = relu(A @ X @ W + b), but computed through the equivalent
    factorization Y = relu(A @ (X @ W) + b), which matches PyG's cost structure.

    Returns:
        dX:     [num_nodes, in_channels]
        dW:     [in_channels, out_channels]
        dBias:  [out_channels] or None
    """
    if apply_relu:
        if relu_mask is None:
            raise ValueError("relu_mask is required when apply_relu is True")
        g = grad_output * relu_mask.to(dtype=grad_output.dtype)
    else:
        g = grad_output

    # PyG convention with flow="source_to_target":
    # row = source nodes j, col = target nodes i
    row, col = edge_index

    # dH[j] = sum_{i : j->i} edge_weight[j,i] * g[i]
    dH = torch.zeros(
        (X.size(0), g.size(1)),
        device=g.device,
        dtype=g.dtype,
    )

    msg = g.index_select(0, col)  # gather target-side gradients
    if edge_weight is not None:
        msg = msg * edge_weight.unsqueeze(-1)

    dH.index_add_(0, row, msg)

    dW = X.transpose(0, 1) @ dH
    dX = dH @ weights.transpose(0, 1)

    if bias is None:
        dBias = None
    elif bias.dim() == 1:
        dBias = g.sum(dim=0)
    else:
        dBias = g.sum(dim=0, keepdim=True)

    return dX, dW, dBias
