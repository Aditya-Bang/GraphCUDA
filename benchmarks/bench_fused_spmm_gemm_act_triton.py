import argparse
from contextlib import contextmanager

import torch
import triton
import triton.language as tl
import triton.profiler as proton
from typing import List, Tuple

from graphcuda.ops._fused_spmm_gemm_act.torch_impl import dense_torch_impl, sparse_torch_impl
from graphcuda.ops._fused_spmm_gemm_act.triton_impl_small_n import fused_spmm_gemm_relu_small_n
from graphcuda.bsr_rm import create_bsr_values_rm
from graphcuda.fused_spmm_gemm_relu import fused_spmm_gemm_relu


# ------------------------------------------------------------
# Proton Utilities
# ------------------------------------------------------------
@contextmanager
def proton_context(session: int = 0):
    proton.activate(session)
    try:
        yield
    finally:
        proton.deactivate(session)

def show_profile(profile_name):
    import triton.profiler.viewer as proton_viewer

    metric_names = ["time/ms"]
    file_name = f"{profile_name}.hatchet"
    tree, metrics = proton_viewer.parse(metric_names, file_name)
    proton_viewer.print_tree(tree, metrics)



# ------------------------------------------------------------
# Testing Utilities
# ------------------------------------------------------------
def make_inputs(M: int, K1: int, K2: int, N: int, dtype: torch.dtype, adj_density: float = 0.05):
    device = torch.device("cuda")
    mask = torch.rand((M, K1), device=device) < adj_density
    adjm_dense = torch.randn(M, K1, dtype=dtype, device=device)
    adjm_dense.mul_(mask.to(dtype))
    
    X = torch.randn(K1, K2, dtype=dtype, device=device)
    weights = torch.randn(K2, N, dtype=dtype, device=device)
    
    adjm_bsr = adjm_dense.to_sparse_bsr(blocksize=(16, 1))
    adjm_bsr.values_rm = create_bsr_values_rm(adjm_bsr)
    adjm_csr = adjm_dense.to_sparse_csr()
    
    return adjm_dense, adjm_bsr, adjm_csr, X, weights

def validate(adjm_dense, adjm_bsr, adjm_csr, X, weights):
    Y_ref = dense_torch_impl(adjm_dense, X, weights)
    Y_torch_sparse = sparse_torch_impl(adjm_csr, X, weights)
    # Y_triton_naive = fused_spmm_gemm_relu(adjm_bsr, X, weights)
    Y_triton_small_n = fused_spmm_gemm_relu_small_n(adjm_bsr, X, weights)

    passed = torch.allclose(Y_ref, Y_torch_sparse, atol=1e-4, rtol=1e-4)
    print(f"  sparse torch impl: {'✅' if passed else '❌'}. Max abs error: {torch.abs(Y_ref - Y_torch_sparse).max().item()}")
    
    # passed = torch.allclose(Y_ref, Y_triton_naive, atol=1e-4, rtol=1e-4)
    # print(f"  triton naive: {'✅' if passed else '❌'}. Max abs error: {torch.abs(Y_ref - Y_triton_naive).max().item()}")

    passed = torch.allclose(Y_ref, Y_triton_small_n, atol=1e-4, rtol=1e-4)
    print(f"  triton small n: {'✅' if passed else '❌'}. Max abs error: {torch.abs(Y_ref - Y_triton_small_n).max().item()}")


def bench_fn(label, reps, warmup_reps, fn, *args):
    
    session = proton.start(label, hook="triton")
    proton.deactivate(session)

    print(f"Benchmarking {label}")
    for _ in range(warmup_reps):
        fn(*args)
    torch.cuda.synchronize()

    with proton_context(session):
        for _ in range(reps):
            fn(*args)
    torch.cuda.synchronize()

    proton.finalize(session)
    show_profile(label)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    # -------------- Arguments --------------
    def parse_dtype(dtype_str: str) -> torch.dtype:
        if dtype_str == "fp32":
            return torch.float32
        if dtype_str == "fp16":
            return torch.float16
        if dtype_str == "bf16":
            return torch.bfloat16
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=1024)
    parser.add_argument("--K1", type=int, default=1024)
    parser.add_argument("--K2", type=int, default=1024)
    parser.add_argument("--N", type=int, default=1024)
    parser.add_argument("--reps", type=int, default=1000)
    parser.add_argument("--warmup-reps", type=int, default=100)
    parser.add_argument("--dtype", type=str, choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument(
        "--adj-density",
        type=float,
        default=0.05,
        help="Approximate fraction of nonzero entries in the (dense) adjacency before BSR/CSR conversion.",
    )
    args = parser.parse_args()
    dtype = parse_dtype(args.dtype)
    
    torch.manual_seed(0)

    # -------------- Make inputs --------------
    adjm_dense, adjm_bsr, adjm_csr, X, weights = make_inputs(
        args.M, args.K1, args.K2, args.N, dtype, adj_density=args.adj_density
    )

    # -------------- Validate --------------
    validate(adjm_dense, adjm_bsr, adjm_csr, X, weights)
    
    # -------------- Benchmark --------------
    bench_fn("dense_torch_impl", args.reps, args.warmup_reps, torch.compile(dense_torch_impl, dynamic=True), adjm_dense, X, weights)
    bench_fn("sparse_torch_impl", args.reps, args.warmup_reps, torch.compile(sparse_torch_impl, dynamic=True), adjm_csr, X, weights)
    # bench_fn("fused_spmm_gemm_relu", args.reps, args.warmup_reps, fused_spmm_gemm_relu, adjm_bsr, X, weights)
    bench_fn("fused_spmm_gemm_relu_small_n", args.reps, args.warmup_reps, fused_spmm_gemm_relu_small_n, adjm_bsr, X, weights)
