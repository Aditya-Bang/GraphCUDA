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

Each triton program computes BLOCK_M x N of the output. Tile sizes BLOCK_K1, BLOCK_K2 are chosen by triton.autotune.

TODO: Shared memory calculations for the op to tune BLOCK_K1, BLOCK_K2.
"""

_FUSED_SPMM_GEMM_RELU_AUTOTUNE_CONFIGS = [
    triton.Config(
        {"BLOCK_K1": bk1, "BLOCK_K2": bk2},
        num_stages=ns,
        num_warps=nw,
    )
    for bk1 in (32, 64, 128)
    for bk2 in (64,)
    for nw in (2,4,8)
    for ns in (2,)
]


@triton.autotune(
    configs=_FUSED_SPMM_GEMM_RELU_AUTOTUNE_CONFIGS,
    key=["M", "N", "K1", "K2"],
)
@triton.jit
def _fused_spmm_gemm_relu_kernel_small_n(
    # sizes
    M: tl.constexpr,
    K1: tl.constexpr,
    K2: tl.constexpr,
    N: tl.constexpr,
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
    # tile sizes
    BLOCK_M: tl.constexpr,
    BLOCK_K1: tl.constexpr,
    BLOCK_K2: tl.constexpr,
):
    # TODO: ignoring K1 blocks right now, assume all just one block in K1 dim, fix later.
    # ------------------- Compute PIDS -------------------
    pid_m = tl.program_id(0)
    
    # ------------------- Compute Logical Offsets -------------------
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M))
    offs_n = (tl.arange(0, N))
    
    tile_m_crow = tl.load(bsr_crow_ptr + pid_m)
    tile_m_next_crow = tl.load(bsr_crow_ptr + pid_m + 1)
    K1_TILE_M = tile_m_next_crow - tile_m_crow
    
    tile_m_adjm_bsr_values_offs = BLOCK_M * tile_m_crow
    offs_k1_adjm = tl.arange(0, BLOCK_K1) # will be masked by K1_TILE_M
    offs_m_adjm = tl.arange(0, BLOCK_M)
    
    offs_adjm_bsr_cols = tl.arange(0, BLOCK_K1) # indexes we want to read from BSR col indices, will be masked by K1_TILE_M, need this because arange needs constant end value
    offs_k1_x = tl.load(bsr_col_ptr + tile_m_crow + offs_adjm_bsr_cols, mask=offs_adjm_bsr_cols < K1_TILE_M, other=0)
    
    # ------------------- Loop over K2 blocks (like dense matmul) -------------------
    acc2 = tl.zeros((BLOCK_M, N), dtype=tl.float32)
    
    for k2_block in range(0, K2, BLOCK_K2):
        # ------------------- Compute K1 block indices and offsets -------------------
        offs_k2 = (k2_block + tl.arange(0, BLOCK_K2))
        
        # ------------------- Load adjm values -------------------
        adjm_values_ptrs = (
            bsr_values_ptr
            + tile_m_adjm_bsr_values_offs
            + offs_m_adjm[:, None] * K1_TILE_M
            + offs_k1_adjm[None, :] * 1 # all contiguous in row major order
        )
        adjm_values_mask = (offs_m[:, None] < M) & (offs_k1_adjm[None, :] < K1_TILE_M)
        adjm_values = tl.load(adjm_values_ptrs, mask=adjm_values_mask, other=0.0)
        
        # ------------------- Load x values -------------------
        x_ptrs = (
            x_ptr
            + offs_k1_x[:, None] * sx_k1
            + offs_k2[None, :] * sx_k2
        )
        x_mask = (offs_adjm_bsr_cols[:, None] < K1_TILE_M) & (offs_k1_x[:, None] < K1) & (offs_k2[None, :] < K2)
        x_values = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        # ------------------- Load weights values -------------------
        w_ptrs = (
            w_ptr
            + offs_k2[:, None] * sw_k2
            + offs_n[None, :] * sw_n
        )
        w_mask = (offs_k2[:, None] < K2) & (offs_n[None, :] < N)
        w_values = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        # ------------------- Compute GEMM -------------------
        acc1 = tl.zeros((BLOCK_M, BLOCK_K2), dtype=tl.float32)
        acc1 = tl.dot(adjm_values, x_values, acc=acc1, input_precision="ieee", out_dtype=tl.float32)
        acc2 = tl.dot(acc1, w_values, acc=acc2, input_precision="ieee", out_dtype=tl.float32)
    
    # ------------------- Apply ReLU -------------------
    if apply_relu:
        acc2 = tl.maximum(acc2, 0.0)
    
    # ------------------- Write back to output -------------------
    out_ptrs = (
        out_ptr
        + offs_m[:, None] * so_m
        + offs_n[None, :] * so_n
    )
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc2, mask=out_mask)

def fused_spmm_gemm_relu_small_n(
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
    assert adjm.layout == torch.sparse_bsr, "adjm must use sparse BSR layout"
    assert not X.is_sparse, "X must be a dense tensor"
    assert not weights.is_sparse, "weights must be a dense tensor"
    assert X.is_contiguous(), "X must be contiguous"
    assert weights.is_contiguous(), "weights must be contiguous"
    
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
    vals = adjm.values()
    BLOCK_M, adjm_bk = int(vals.shape[-2]), int(vals.shape[-1])
    assert BLOCK_M > 0, "BLOCK_M must be positive"
    assert adjm_bk == 1, "adjm must have 1 block per column for this kernel specifically"
    assert adjm.values_rm.dim() == 1, "BSR values row major flattened must be 1D"
    nrow_blocks = (M + BLOCK_M - 1) // BLOCK_M
    assert adjm.crow_indices().numel() == nrow_blocks + 1, "unexpected BSR crow_indices length"

    # ------------------- 2. Triton Kernel Launcher -------------------
    Y = torch.empty((M, N), device=device, dtype=dtype)

    grid = lambda meta: (
        triton.cdiv(M, BLOCK_M),
    )

    _fused_spmm_gemm_relu_kernel_small_n[grid](
        M, K1, K2, N,
        X, weights, Y,
        adjm.values_rm, adjm.crow_indices(), adjm.col_indices(),
        X.stride(0), X.stride(1), weights.stride(0), weights.stride(1), Y.stride(0), Y.stride(1),
        apply_relu,
        BLOCK_M,
    )
    return Y
