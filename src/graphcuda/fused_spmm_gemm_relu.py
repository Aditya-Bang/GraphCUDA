import triton
import triton.language as tl
import torch


"""
Fused SPMM-GEMM-ReLU operation written in Triton.

Inputs:
    adjm: torch.Tensor. Must be a BSR tensor. Shape (M, K1)
    X: torch.Tensor. Must be a dense tensor. Shape (K1, K2)
    weights: torch.Tensor. Must be a dense tensor. Shape (K2, N)
    apply_relu: bool. Whether to apply ReLU to the output.

Outputs:
    Y: torch.Tensor. Must be a dense tensor. Shape (M, N)

Each triton program computes BLOCK_M x BLOCK_N of the output. Tile sizes BLOCK_N,
BLOCK_K1, BLOCK_K2 (and BLOCK_M) are chosen by triton.autotune.
"""

_FUSED_SPMM_GEMM_RELU_AUTOTUNE_CONFIGS = [
    triton.Config(
        {"BLOCK_N": 64, "BLOCK_K1": 128, "BLOCK_K2": 32},
        num_stages=2,
        num_warps=4,
    ),
    triton.Config(
        {"BLOCK_N": 128, "BLOCK_K1": 128, "BLOCK_K2": 64},
        num_stages=2,
        num_warps=8,
    ),
]


@triton.autotune(
    configs=_FUSED_SPMM_GEMM_RELU_AUTOTUNE_CONFIGS,
    key=["M", "N", "K1", "K2"],
)
@triton.jit
def _fused_spmm_gemm_relu_kernel(
    # sizes
    M,
    K1,
    K2,
    N,
    # ptrs
    x_ptr, # dense X (K1, K2)
    w_ptr, # dense weights (K2, N)
    out_ptr, # dense output (M, N)
    bsr_values_ptr, # BSR adjacency (M, K1)
    bsr_crow_ptr,
    bsr_col_ptr,
    # strides
    sx_k1, sx_k2,
    sw_k2, sw_n,
    so_m, so_n,
    # flags
    apply_relu: tl.constexpr,
    # tile sizes (BLOCK_N, BLOCK_K1, BLOCK_K2 from autotune; BLOCK_M for row tile / grid)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K1: tl.constexpr,
    BLOCK_K2: tl.constexpr,
):
    # ------------------- Compute PIDS -------------------
    pid = tl.program_id(0)
    pid_m, pid_n = pid // BLOCK_N, pid % BLOCK_N
    
    
    
    pass


def fused_spmm_gemm_relu(
    adjm: torch.Tensor,
    X: torch.Tensor,
    weights: torch.Tensor,
    apply_relu: bool = True,
) -> torch.Tensor:
    # ------------------- 1. Input Assertions -------------------
    # a. dims
    assert X.dim() == 2, "X must be 2D"
    assert weights.dim() == 2, "weights must be 2D"
    assert adjm.dim() == 2, "adjm must be 2D"
    
    # b. layouts
    assert adjm.is_sparse, "adjm must be a sparse tensor"
    assert adjm.layout == torch.sparse_bsr, "adjm must use sparse BSR layout"
    assert not X.is_sparse, "X must be a dense tensor"
    assert not weights.is_sparse, "weights must be a dense tensor"
    assert X.is_contiguous(), "X must be contiguous"
    assert weights.is_contiguous(), "weights must be contiguous"
    assert adjm.is_contiguous(), "adjm must be contiguous"
    
    # c. shapes
    M, K1, K2, N = adjm.shape[0], adjm.shape[1], weights.shape[0], weights.shape[1]
    assert K1 == X.shape[0], f"X.shape[0] ({X.shape[0]}) must match adjm.shape[1] ({K1})"
    assert K2 == X.shape[1], f"X.shape[1] ({X.shape[1]}) must match weights.shape[0] ({K2})"
    
    # d. devices
    device = adjm.device
    assert adjm.device.type == "cuda", "adjm must be on CUDA"
    assert X.device == device, "X must be on the same device as adjm"
    assert weights.device == device, "weights must be on the same device as adjm"
    
    # e. dtypes
    dtype = X.dtype
    assert X.dtype == weights.dtype, "X and weights must have the same dtype"
    
    # f. Extra bsr checks
    BLOCK_M, adjm_bk = adjm.blocksize()
    assert BLOCK_M > 0, "BLOCK_M must be positive"
    assert adjm_bk == 1, "adjm must have 1 block per column for this kernel specifically"
    assert adjm.values_rm.dim() == 1, "BSR values row major flattened must be 1D"
    nrow_blocks = (M + BLOCK_M - 1) // BLOCK_M
    assert adjm.crow_indices().numel() == nrow_blocks + 1, "unexpected BSR crow_indices length"

    # ------------------- 2. Triton Kernel Launcher -------------------
    Y = torch.empty((M, N), device=device, dtype=dtype)

    grid = lambda meta: (
        BLOCK_M *
        triton.cdiv(N, meta["BLOCK_N"]),
    )

    _fused_spmm_gemm_relu_kernel[grid](
        M, K1, K2, N,
        X, weights, Y,
        adjm.values_rm, adjm.crow_indices(), adjm.col_indices(),
        X.stride(0), X.stride(1), weights.stride(0), weights.stride(1), Y.stride(0), Y.stride(1),
        apply_relu,
        BLOCK_M,
    )
    return Y
