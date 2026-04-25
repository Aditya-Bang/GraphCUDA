import torch


def spmm_gemm_relu_backward_pyg_csr_torch_impl(
    grad_output: torch.Tensor,
    adj_t_csr: torch.Tensor,           # A^T stored as CSR, not computed on the fly
    X: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor | None,
    relu_mask: torch.Tensor | None,
    apply_relu: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if apply_relu:
        if relu_mask is None:
            raise ValueError("relu_mask is required when apply_relu is True")
        g = grad_output * relu_mask.to(dtype=grad_output.dtype)
    else:
        g = grad_output

    if adj_t_csr.layout != torch.sparse_csr and adj_t_csr.layout != torch.sparse_coo:
        raise TypeError(f"Expected CSR/COO sparse tensor, got {adj_t_csr.layout}")

    dH = torch.sparse.mm(adj_t_csr, g)
    dW = X.transpose(0, 1) @ dH
    dX = dH @ weights.transpose(0, 1)

    if bias is None:
        dBias = None
    elif bias.dim() == 1:
        dBias = g.sum(dim=0)
    else:
        dBias = g.sum(dim=0, keepdim=True)

    return dX, dW, dBias
