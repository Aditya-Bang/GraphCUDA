import pytest
import torch

from graphcuda.utils.bsr_rm import create_bsr_values_rm
from graphcuda.ops._fused_spmm_gemm_relu.fwd.triton_impl_small_n import fused_spmm_gemm_relu_small_n


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_spmm_gemm_relu_identity_16():
    """Identity adjacency: output matches ReLU(X @ W)."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    n = 32

    dense_i = torch.eye(n, dtype=torch.float32, device=device)
    adjm = dense_i.to_sparse_bsr(blocksize=(n, 1))
    adjm.values_rm = create_bsr_values_rm(adjm)

    X = torch.randn(n, n, dtype=torch.float32, device=device)
    weights = torch.randn(n, n, dtype=torch.float32, device=device)

    out = fused_spmm_gemm_relu_small_n(adjm, X, weights, apply_relu=True)
    expected = torch.relu(X @ weights)

    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-5)
