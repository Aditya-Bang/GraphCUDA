from __future__ import annotations

import torch

from . import _fused_spmm_gemm_relu_sm80_cuda as _ext


def _ensure_i32_indices(adjm_bsr_rm: torch.Tensor) -> None:
    if not hasattr(adjm_bsr_rm, "crow_indices_i32"):
        adjm_bsr_rm.crow_indices_i32 = adjm_bsr_rm.crow_indices().to(torch.int32).contiguous()
    if not hasattr(adjm_bsr_rm, "col_indices_i32"):
        adjm_bsr_rm.col_indices_i32 = adjm_bsr_rm.col_indices().to(torch.int32).contiguous()


def fused_spmm_gemm_relu_small_n(
    adjm: torch.Tensor,
    X: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor | None = None,
    apply_relu: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    # ------------------- 1. Input Assertions -------------------
    # a. dims
    assert X.dim() == 2, "X must be 2D"
    assert weights.dim() == 2, "weights must be 2D"
    assert adjm.dim() == 2, "adjm must be 2D"

    # b. layouts
    assert adjm.layout == torch.sparse_bsr, "adjm must use sparse BSR layout"
    assert not X.is_sparse, "X must be a dense tensor"
    assert not weights.is_sparse, "weights must be a dense tensor"
    assert hasattr(adjm, "pre_padded_shape"), "adjm must have pre_padded_shape attribute, construct it via to_sparse_bsr_rm"
    assert hasattr(adjm, "values_rm"), "adjm must have values_rm attribute, construct it via to_sparse_bsr_rm"

    # c. shapes
    M, K1, K2, N = adjm.pre_padded_shape[0], adjm.pre_padded_shape[1], weights.shape[0], weights.shape[1]
    assert K1 == X.shape[0], f"X.shape[0] ({X.shape[0]}) must match adjm.shape[1] ({K1})"
    assert K2 == X.shape[1], f"X.shape[1] ({X.shape[1]}) must match weights.shape[0] ({K2})"
    assert N <= 128, "this SM80 kernel currently supports N <= 128"

    # d. devices
    device = adjm.device
    assert adjm.device.type == "cuda", "adjm must be on CUDA"
    assert X.device == device, "X must be on the same device as adjm"
    assert weights.device == device, "weights must be on the same device as adjm"

    # e. dtypes
    dtype = X.dtype
    assert dtype == torch.float16, "this SM80 kernel is specialized to torch.float16"
    assert weights.dtype == dtype, "weights must have the same dtype as X"
    assert adjm.values_rm.dtype == dtype, "adjm.values_rm must have the same dtype as X"

    # f. bias checks
    if bias is None:
        bias_1d = torch.empty(0, device=device, dtype=dtype)
    else:
        assert not bias.is_sparse, "bias must be a dense tensor"
        assert bias.device == device, "bias must be on the same device as adjm"
        assert bias.dtype == dtype, "bias must have the same dtype as X and weights"
        if bias.dim() == 1:
            assert bias.shape[0] == N, f"bias 1D length must be {N}, got {bias.shape[0]}"
            bias_1d = bias.contiguous()
        elif bias.dim() == 2:
            assert bias.shape == (1, N), f"bias 2D shape must be (1, {N}), got {tuple(bias.shape)}"
            bias_1d = bias.reshape(-1).contiguous()
        else:
            raise AssertionError("bias must be 1D or 2D")

    # g. extra bsr checks
    vals = adjm.values()
    BLOCK_M, adjm_bk = int(vals.shape[-2]), int(vals.shape[-1])
    assert BLOCK_M == 16, "this SM80 kernel currently requires BLOCK_M == 16"
    assert adjm_bk == 1, "adjm must have 1 block per column for this kernel specifically"
    assert adjm.values_rm.dim() == 1, "BSR values row major flattened must be 1D"
    nrow_blocks = (M + BLOCK_M - 1) // BLOCK_M
    assert adjm.crow_indices().numel() == nrow_blocks + 1, "unexpected BSR crow_indices length"

    _ensure_i32_indices(adjm)
    Y, relu_mask = _ext.forward(
        adjm.values_rm.contiguous(),
        adjm.crow_indices_i32,
        adjm.col_indices_i32,
        int(M),
        int(K1),
        X.contiguous(),
        weights.contiguous(),
        bias_1d,
        bool(apply_relu),
    )
    return Y, (relu_mask if apply_relu else None)


fused_spmm_gemm_relu_small_n_cuda = fused_spmm_gemm_relu_small_n
