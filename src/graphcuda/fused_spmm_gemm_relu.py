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
    # TODO: ignoring K1 blocks right now, assume all just one block in K1 dim, fix later.
    # ------------------- Compute PIDS -------------------
    pid = tl.program_id(0)
    pid_m, pid_n = pid // BLOCK_N, pid % BLOCK_N
    
    # ------------------- Compute Logical Offsets -------------------
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    
    # ------------------- Loop over K2 blocks (like dense matmul) -------------------
    acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k2_block in range(0, K2, BLOCK_K2):
        # ------------------- Compute K1 block indices and offsets -------------------
        K1_BLOCKIDX_M = tl.load(bsr_crow_ptr + pid_m)
        K1_BLOCKIDX_M_NEXT = tl.load(bsr_crow_ptr + pid_m + 1)
        K1_ADJM = K1_BLOCKIDX_M_NEXT - K1_BLOCKIDX_M # changes for each row block on adjm, since all have different sizes
        
        offs_k1_adjm = tl.arange(K1_BLOCKIDX_M * BLOCK_M, K1_BLOCKIDX_M_NEXT * BLOCK_M)
        offs_k1_x = tl.load(bsr_col_ptr + tl.arange(K1_BLOCKIDX_M, K1_BLOCKIDX_M_NEXT))
        offs_k2 = (k2_block + tl.arange(0, BLOCK_K2)).to(tl.int64)
        
        # ------------------- Load adjm values -------------------
        adjm_values_ptrs = (
            bsr_values_ptr
            + offs_k1_adjm * 1 # all contiguous in row major order
        ).reshape(BLOCK_M, K1_ADJM)
        # TODO: add masking here
        adjm_values = tl.load(adjm_values_ptrs)
        
        # ------------------- Load x values -------------------
        x_ptrs = (
            x_ptr
            + offs_k1_x[:, None] * sx_k1
            + offs_k2[None, :] * sx_k2
        )
        x_mask = (offs_k1_x[:, None] < K1) & (offs_k2[None, :] < K2)
        x_values = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        # ------------------- Load weights values -------------------
        w_ptrs = (
            w_ptr
            + offs_k2[None, :] * sw_k2
            + offs_n[None, :] * sw_n
        )
        w_values = tl.load(w_ptrs)
        
        # ------------------- Compute GEMM -------------------
        acc1 = tl.zeros((BLOCK_M, BLOCK_K2), dtype=tl.float32)
        acc1 += tl.dot(adjm_values, x_values, acc=acc1, input_precision="ieee", out_dtype=tl.float32)
        acc2 += tl.dot(acc1, w_ptrs, acc=acc2, input_ptr=0, output_ptr=0)
    
    # ------------------- Apply ReLU -------------------
    if apply_relu:
        acc2 = tl.maximum(acc2, 0.0)
    
    # ------------------- Write back to output -------------------
    tl.store(out_ptr + offs_m[:, None] * so_m + offs_n[None, :] * so_n, acc2)

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
