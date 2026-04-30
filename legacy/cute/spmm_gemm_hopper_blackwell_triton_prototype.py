import os
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


"""
Prototype Hopper/Blackwell-oriented Triton kernel for

    Y = A_sparse @ X @ W

using the selected-XW association:

    Y += A_tile @ (X[selected_rows] @ W)

This is intentionally a *design prototype*, not a drop-in production kernel.
It focuses on three ideas that are useful on Hopper/Blackwell:

1. persistent scheduling over sparse row-blocks
2. architecture-gated warp specialization in tl.range loops
3. small-N-friendly reassociation so the dense K2 reduction happens before
   the sparse projection

Compared with the original kernel, this version is most likely to help when:
- N is small
- K2 is moderate/large
- per-row-block sparse degree is not tiny

Remaining work for a production-quality kernel:
- bucket row-blocks by sparse degree
- migrate X row gathering to TMA gather / descriptor-gather on Hopper+
- use tensor descriptors for W / Y
- add Blackwell-specific maxnreg tuning
- possibly change sparse storage to a bucketed fixed-stride format
"""


def is_cuda() -> bool:
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_hopper() -> bool:
    return is_cuda() and torch.cuda.get_device_capability()[0] == 9


def is_blackwell() -> bool:
    return is_cuda() and torch.cuda.get_device_capability()[0] == 10


def supports_ws() -> bool:
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_K1": bk1, "BLOCK_K2": bk2}, num_warps=nw, num_stages=ns)
    for bk1 in (16, 32, 64)
    for bk2 in (16, 32, 64)
    for nw in (4, 8)
    for ns in (2, 3, 4)
]

if "PYTEST_VERSION" in os.environ:
    _AUTOTUNE_CONFIGS = [
        triton.Config({"BLOCK_K1": 32, "BLOCK_K2": 32}, num_warps=4, num_stages=2)
    ]


@triton.autotune(
    configs=_AUTOTUNE_CONFIGS,
    key=["K2", "N", "WARP_SPECIALIZE"],
)
@triton.jit
def fused_spmm_gemm_relu_selected_xw_persistent_kernel(
    # sizes
    M,
    K1,
    K2,
    N,
    # dense tensors
    x_ptr,
    w_ptr,
    out_ptr,
    relu_mask_ptr,
    # sparse tensors / metadata
    bsr_values_ptr,
    bsr_crow_ptr,
    bsr_col_ptr,
    bias_ptr,
    # strides
    sx_k1,
    sx_k2,
    sw_k2,
    sw_n,
    so_m,
    so_n,
    sm_m,
    sm_n,
    sb_n,
    # persistent scheduling
    NUM_SMS: tl.constexpr,
    # flags
    APPLY_RELU: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
    # tile sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K1: tl.constexpr,
    BLOCK_K2: tl.constexpr,
):
    start_pid = tl.program_id(0)
    num_row_blocks = tl.cdiv(M, BLOCK_M)

    offs_n = tl.arange(0, BLOCK_N)

    for pid_m in tl.range(start_pid, num_row_blocks, NUM_SMS, flatten=True, warp_specialize=WARP_SPECIALIZE):
        row_start = pid_m * BLOCK_M
        offs_m = row_start + tl.arange(0, BLOCK_M)

        crow0 = tl.load(bsr_crow_ptr + pid_m)
        crow1 = tl.load(bsr_crow_ptr + pid_m + 1)
        nnz_blocks = crow1 - crow0

        # The original values_rm layout packs a row-block as a dense [BLOCK_M, nnz_blocks]
        # tile in row-major order, starting at BLOCK_M * crow0.
        values_rowblock_base = bsr_values_ptr + BLOCK_M * crow0

        acc_out = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # Loop over sparse K1 tiles.
        for k1_block in range(0, nnz_blocks, BLOCK_K1):
            offs_k1_local = k1_block + tl.arange(0, BLOCK_K1)
            k1_mask = offs_k1_local < nnz_blocks

            sparse_cols = tl.load(
                bsr_col_ptr + crow0 + offs_k1_local,
                mask=k1_mask,
                other=0,
            )

            # Load A_tile: [BLOCK_M, BLOCK_K1]
            a_ptrs = (
                values_rowblock_base
                + tl.arange(0, BLOCK_M)[:, None] * nnz_blocks
                + offs_k1_local[None, :]
            )
            a_mask = (offs_m[:, None] < M) & k1_mask[None, :]
            a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)

            # Compute G = X[selected_rows, :] @ W, where G has shape [BLOCK_K1, BLOCK_N]
            # Since N is small, this is often a better association than (A @ X) @ W.
            g_tile = tl.zeros((BLOCK_K1, BLOCK_N), dtype=tl.float32)

            for k2_block in tl.range(0, K2, BLOCK_K2, warp_specialize=WARP_SPECIALIZE):
                offs_k2 = k2_block + tl.arange(0, BLOCK_K2)
                k2_mask = offs_k2 < K2

                x_ptrs = (
                    x_ptr
                    + sparse_cols[:, None] * sx_k1
                    + offs_k2[None, :] * sx_k2
                )
                x_mask = k1_mask[:, None] & (sparse_cols[:, None] < K1) & k2_mask[None, :]
                x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

                w_ptrs = (
                    w_ptr
                    + offs_k2[:, None] * sw_k2
                    + offs_n[None, :] * sw_n
                )
                w_mask = k2_mask[:, None] & (offs_n[None, :] < N)
                w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

                g_tile = tl.dot(
                    x_tile,
                    w_tile,
                    acc=g_tile,
                    input_precision="ieee",
                    out_dtype=tl.float32,
                )

            # Accumulate A_tile @ G into output accumulator.
            acc_out = tl.dot(
                a_tile,
                g_tile.to(a_tile.dtype),
                acc=acc_out,
                input_precision="ieee",
                out_dtype=tl.float32,
            )

        if HAS_BIAS:
            bias_vals = tl.load(bias_ptr + offs_n * sb_n, mask=offs_n < N, other=0.0)
            acc_out += bias_vals[None, :]

        out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        if APPLY_RELU:
            relu_mask_vals = acc_out > 0.0
            relu_mask_ptrs = relu_mask_ptr + offs_m[:, None] * sm_m + offs_n[None, :] * sm_n
            tl.store(relu_mask_ptrs, relu_mask_vals, mask=out_mask)
            acc_out = tl.maximum(acc_out, 0.0)

        out_ptrs = out_ptr + offs_m[:, None] * so_m + offs_n[None, :] * so_n
        tl.store(out_ptrs, acc_out, mask=out_mask)


def fused_spmm_gemm_relu_selected_xw_persistent(
    adjm: torch.Tensor,
    X: torch.Tensor,
    weights: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    apply_relu: bool = True,
    warp_specialize: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Prototype launcher.

    The sparse input must match the same conventions as the user's original kernel:
    - BSR layout
    - block shape (BLOCK_M, 1)
    - custom attrs: pre_padded_shape and values_rm
    """
    assert adjm.layout == torch.sparse_bsr, "adjm must be BSR"
    assert hasattr(adjm, "pre_padded_shape"), "adjm must have pre_padded_shape"
    assert hasattr(adjm, "values_rm"), "adjm must have values_rm"
    assert X.dim() == 2 and weights.dim() == 2 and adjm.dim() == 2
    assert X.is_contiguous() and weights.is_contiguous()
    assert X.device.type == "cuda"
    assert X.device == weights.device == adjm.device
    assert X.dtype == weights.dtype

    M, K1 = adjm.pre_padded_shape[0], adjm.pre_padded_shape[1]
    assert K1 == X.shape[0]
    K2, N = weights.shape
    assert X.shape[1] == K2

    vals = adjm.values()
    BLOCK_M = int(vals.shape[-2])
    block_k = int(vals.shape[-1])
    assert block_k == 1, "prototype assumes BSR block width == 1"

    has_bias = bias is not None
    bias_stride_n = 0
    if has_bias:
        assert bias.device == X.device and bias.dtype == X.dtype
        if bias.dim() == 1:
            assert bias.shape[0] == N
            bias_stride_n = bias.stride(0)
        elif bias.dim() == 2:
            assert bias.shape == (1, N)
            bias_stride_n = bias.stride(1)
        else:
            raise AssertionError("bias must be 1D or 2D")

    BLOCK_N = max(triton.next_power_of_2(N), 16)
    Y = torch.empty((M, N), device=X.device, dtype=X.dtype)
    relu_mask = torch.empty((M, N), device=X.device, dtype=torch.bool) if apply_relu else None

    num_sms = torch.cuda.get_device_properties(X.device).multi_processor_count
    ws = bool(warp_specialize and supports_ws())

    grid = lambda META: (min(num_sms, triton.cdiv(M, BLOCK_M)),)

    fused_spmm_gemm_relu_selected_xw_persistent_kernel[grid](
        M,
        K1,
        K2,
        N,
        X,
        weights,
        Y,
        relu_mask if relu_mask is not None else 0,
        adjm.values_rm,
        adjm.crow_indices(),
        adjm.col_indices(),
        bias if has_bias else 0,
        X.stride(0),
        X.stride(1),
        weights.stride(0),
        weights.stride(1),
        Y.stride(0),
        Y.stride(1),
        relu_mask.stride(0) if relu_mask is not None else 0,
        relu_mask.stride(1) if relu_mask is not None else 0,
        bias_stride_n,
        NUM_SMS=num_sms,
        APPLY_RELU=apply_relu,
        HAS_BIAS=has_bias,
        WARP_SPECIALIZE=ws,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    return Y, relu_mask


__all__ = [
    "fused_spmm_gemm_relu_selected_xw_persistent",
    "fused_spmm_gemm_relu_selected_xw_persistent_kernel",
]
