import argparse
from contextlib import contextmanager

import torch
import triton
import triton.language as tl
import triton.profiler as proton
from graphcuda.bsr_rm import create_bsr_values_rm
from graphcuda.ops._fused_spmm_gemm_act.torch_impl import dense_torch_impl, sparse_torch_impl
from graphcuda.ops._fused_spmm_gemm_act.triton_impl_small_n import fused_spmm_gemm_relu_small_n
from graphcuda.ops._fused_spmm_gemm_act.triton_impl_small_n_switch_loop import fused_spmm_gemm_relu_small_n as fused_spmm_gemm_relu_small_n_switch_loop


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
def make_inputs(M: int, K1: int, K2: int, N: int, dtype: torch.dtype, adj_density: float = 0.05, use_bias: bool = False):
    device = torch.device("cuda")
    mask = torch.rand((M, K1), device=device) < adj_density
    adjm_dense = torch.randn(M, K1, dtype=dtype, device=device)
    adjm_dense.mul_(mask.to(dtype))
    
    X = torch.randn(K1, K2, dtype=dtype, device=device)
    weights = torch.randn(K2, N, dtype=dtype, device=device)
    
    adjm_bsr = adjm_dense.to_sparse_bsr(blocksize=(16, 1))
    adjm_bsr.values_rm = create_bsr_values_rm(adjm_bsr)
    adjm_csr = adjm_dense.to_sparse_csr()
    
    bias = torch.randn(1, N, dtype=dtype, device=device) if use_bias else None
    return adjm_dense, adjm_bsr, adjm_csr, X, weights, bias


def validate(adjm_dense, adjm_bsr, adjm_csr, X, weights, bias, apply_relu: bool = True):
    Y_ref, _ = dense_torch_impl(adjm_dense, X, weights, bias, apply_relu)
    Y_torch_sparse, _ = sparse_torch_impl(adjm_csr, X, weights, bias, apply_relu)
    Y_triton_small_n, _ = fused_spmm_gemm_relu_small_n(adjm_bsr, X, weights, bias, apply_relu)
    Y_triton_small_n_switch_loop, _ = fused_spmm_gemm_relu_small_n_switch_loop(
        adjm_bsr, X, weights, bias, apply_relu
    )

    atol = 1e-2
    rtol = 1e-2

    passed = torch.allclose(Y_ref, Y_torch_sparse, atol=atol, rtol=rtol)
    print(f"  sparse torch impl: {'✅' if passed else '❌'}. Max abs error: {torch.abs(Y_ref - Y_torch_sparse).max().item()}")

    passed = torch.allclose(Y_ref, Y_triton_small_n, atol=atol, rtol=rtol)
    print(f"  triton small n: {'✅' if passed else '❌'}. Max abs error: {torch.abs(Y_ref - Y_triton_small_n).max().item()}")
    
    passed = torch.allclose(Y_ref, Y_triton_small_n_switch_loop, atol=atol, rtol=rtol)
    print(f"  triton small n switch loop: {'✅' if passed else '❌'}. Max abs error: {torch.abs(Y_ref - Y_triton_small_n_switch_loop).max().item()}")
    

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
    parser.add_argument(
        "--bias",
        action="store_true",
        help="Use a random (1, N) bias in reference and Triton implementations.",
    )
    parser.add_argument(
        "--apply-relu",
        action="store_true",
        help="Apply ReLU after all operations.",
    )
    args = parser.parse_args()
    dtype = parse_dtype(args.dtype)
    
    torch.manual_seed(0)

    # -------------- Make inputs --------------
    adjm_dense, adjm_bsr, adjm_csr, X, weights, bias = make_inputs(
        args.M, args.K1, args.K2, args.N, dtype, adj_density=args.adj_density, use_bias=args.bias
    )

    # -------------- Validate --------------
    validate(adjm_dense, adjm_bsr, adjm_csr, X, weights, bias, args.apply_relu)

    # -------------- Benchmark --------------
    bench_fn(f"dense_torch_impl", args.reps, args.warmup_reps, torch.compile(dense_torch_impl, dynamic=True), adjm_dense, X, weights, bias, args.apply_relu)
    bench_fn(f"sparse_torch_impl", args.reps, args.warmup_reps, torch.compile(sparse_torch_impl, dynamic=True), adjm_csr, X, weights, bias, args.apply_relu)
    bench_fn(f"fused_spmm_gemm_relu_small_n", args.reps, args.warmup_reps, fused_spmm_gemm_relu_small_n, adjm_bsr, X, weights, bias, args.apply_relu)
    bench_fn(f"fused_spmm_gemm_relu_small_n_switch_loop", args.reps, args.warmup_reps, fused_spmm_gemm_relu_small_n_switch_loop, adjm_bsr, X, weights, bias, args.apply_relu)
