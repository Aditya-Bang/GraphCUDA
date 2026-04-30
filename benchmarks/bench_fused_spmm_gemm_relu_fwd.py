import os
import argparse
from contextlib import contextmanager

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.profiler as proton
from graphcuda.utils.bsr_rm import to_sparse_bsr_rm
from graphcuda.ops._fused_spmm_gemm_relu.fwd.naive_torch import fused_spmm_gemm_relu_dense_torch_impl, fused_spmm_gemm_relu_sparse_torch_impl
from graphcuda.ops._fused_spmm_gemm_relu.fwd.triton_impl_small_n import fused_spmm_gemm_relu_small_n
from graphcuda.ops._fused_spmm_gemm_relu.fwd.triton_impl_small_n_switch_loop import fused_spmm_gemm_relu_small_n_switch_loop
from graphcuda.ops._fused_spmm_gemm_relu.fwd.cuda_impl_small_n import fused_spmm_gemm_relu_small_n_cuda

from torch_geometric.datasets import Planetoid
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import to_dense_adj


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
    
    adjm_bsr = to_sparse_bsr_rm(adjm_dense)
    adjm_csr = adjm_dense.to_sparse_csr()
    
    bias = torch.randn(1, N, dtype=dtype, device=device) if use_bias else None
    return adjm_dense, adjm_bsr, adjm_csr, X, weights, bias


def make_edge_case_inputs(M: int, K1: int, K2: int, N: int, dtype: torch.dtype, use_bias: bool = False):
    device = torch.device("cuda")
    adjm_dense = torch.zeros(M, K1, dtype=dtype, device=device)
    adjm_dense[0].fill_(1)

    X = torch.randn(K1, K2, dtype=dtype, device=device)
    weights = torch.randn(K2, N, dtype=dtype, device=device)
    
    adjm_bsr = to_sparse_bsr_rm(adjm_dense)
    adjm_csr = adjm_dense.to_sparse_csr()
    
    bias = torch.randn(1, N, dtype=dtype, device=device) if use_bias else None
    return adjm_dense, adjm_bsr, adjm_csr, X, weights, bias


def make_cora_inputs(N: int, dtype: torch.dtype, use_bias: bool = False):
    device = torch.device("cuda")
    dataset = Planetoid(root=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data")), name='Cora')
    data = dataset[0]
    data = data.to(device)
    x = data.x.to(dtype=dtype)
    k2 = x.shape[1]
    k2_pad = (k2 + 15) // 16 * 16
    if k2_pad > k2:
        x = F.pad(x, (0, k2_pad - k2))
    edge_index = data.edge_index
    edge_index, edge_weight = gcn_norm(
        edge_index=edge_index,
        edge_weight=None,
        num_nodes=x.size(-2),
        improved=False,
        add_self_loops=True,
        flow="source_to_target",
        dtype=x.dtype,
    )
    adjm_dense = to_dense_adj(edge_index, edge_attr=edge_weight)[0]
    adjm_bsr = to_sparse_bsr_rm(adjm_dense)
    adjm_csr = adjm_dense.to_sparse_csr()
    
    M = adjm_dense.shape[0]
    K1 = adjm_dense.shape[1]
    K2 = x.shape[1]
    
    weights = torch.randn(K2, N, dtype=dtype, device=device)
    bias = torch.randn(1, N, dtype=dtype, device=device) if use_bias else None
    return adjm_dense, adjm_bsr, adjm_csr, x, weights, bias


def validate(adjm_dense, adjm_bsr, adjm_csr, X, weights, bias, apply_relu: bool = True):
    Y_ref, _ = fused_spmm_gemm_relu_dense_torch_impl(adjm_dense, X, weights, bias, apply_relu)
    Y_torch_sparse, _ = fused_spmm_gemm_relu_sparse_torch_impl(adjm_csr, X, weights, bias, apply_relu)
    Y_triton_small_n, _ = fused_spmm_gemm_relu_small_n(adjm_bsr, X, weights, bias, apply_relu)
    Y_triton_small_n_switch_loop, _ = fused_spmm_gemm_relu_small_n_switch_loop(adjm_bsr, X, weights, bias, apply_relu)
    compute_capability = torch.cuda.get_device_capability(X.device)
    is_ampere = compute_capability[0] == 8
    if X.dtype == torch.float16 and is_ampere:
        Y_cuda_small_n, _ = fused_spmm_gemm_relu_small_n_cuda(adjm_bsr, X, weights, bias, apply_relu)
    elif X.dtype == torch.float16:
        print(f"Skipping CUDA impl of fused SpMM-GEMM-ReLU because it requires Ampere (compute capability 8.x), got {compute_capability[0]}.{compute_capability[1]}.")
    else:
        print("Skipping CUDA impl of fused SpMM-GEMM-ReLU because it only supports float16.")

    atol = 1e-2
    rtol = 1e-2

    passed = torch.allclose(Y_ref, Y_torch_sparse, atol=atol, rtol=rtol)
    print(f"  sparse torch impl: {'✅' if passed else '❌'}. Max abs error: {torch.abs(Y_ref - Y_torch_sparse).max().item()}")

    passed = torch.allclose(Y_ref, Y_triton_small_n, atol=atol, rtol=rtol)
    print(f"  triton small n: {'✅' if passed else '❌'}. Max abs error: {torch.abs(Y_ref - Y_triton_small_n).max().item()}")
    
    passed = torch.allclose(Y_ref, Y_triton_small_n_switch_loop, atol=atol, rtol=rtol)
    print(f"  triton small n switch loop: {'✅' if passed else '❌'}. Max abs error: {torch.abs(Y_ref - Y_triton_small_n_switch_loop).max().item()}")

    if X.dtype == torch.float16 and is_ampere:
        passed = torch.allclose(Y_ref, Y_cuda_small_n, atol=atol, rtol=rtol)
        print(f"  cuda small n: {'✅' if passed else '❌'}. Max abs error: {torch.abs(Y_ref - Y_cuda_small_n).max().item()}")


def bench_fn(disable_proton, label, reps, warmup_reps, fn, *args):
    
    if not disable_proton:
        session = proton.start(label, hook="triton")
        proton.deactivate(session)

    print(f"Benchmarking {label}")
    for _ in range(warmup_reps):
        fn(*args)
    torch.cuda.synchronize()

    if not disable_proton:
        with proton_context(session):
            for _ in range(reps):
                fn(*args)
    else:
        for _ in range(reps):
            fn(*args)
    torch.cuda.synchronize()

    if not disable_proton:
        proton.finalize(session)
        show_profile(label)

    print(f"{label}: profiling done")


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
    parser.add_argument("--adj-density", type=float, default=0.05, help="Approximate fraction of nonzero entries in the (dense) adjacency before BSR/CSR conversion.",)
    parser.add_argument("--bias", action="store_true", help="Use a random (1, N) bias in reference and Triton implementations.",)
    parser.add_argument("--apply-relu", action="store_true", help="Apply ReLU after all operations.",)
    parser.add_argument("--use-cora", action="store_true", help="Use Cora dataset inputs.")
    parser.add_argument("--use-edge-case", action="store_true", help="Use edge case inputs.")
    parser.add_argument("--disable-proton", action="store_true", help="Disable Proton profiling.")
    args = parser.parse_args()
    dtype = parse_dtype(args.dtype)
    
    torch.manual_seed(0)

    # -------------- Make inputs --------------
    if not args.use_cora and not args.use_edge_case:
        adjm_dense, adjm_bsr, adjm_csr, X, weights, bias = make_inputs(
            args.M, args.K1, args.K2, args.N, dtype, adj_density=args.adj_density, use_bias=args.bias
        )
    elif args.use_edge_case and not args.use_cora:
        print("Using edge case inputs, ignoring adj-density arguments.")
        adjm_dense, adjm_bsr, adjm_csr, X, weights, bias = make_edge_case_inputs(M=args.M, K1=args.K1, K2=args.K2, N=args.N, dtype=dtype, use_bias=args.bias)
    elif args.use_cora and not args.use_edge_case:
        print("Using Cora dataset inputs, ignoring M, K1, K2, adj-density arguments.")
        adjm_dense, adjm_bsr, adjm_csr, X, weights, bias = make_cora_inputs(N=args.N, dtype=dtype, use_bias=args.bias)
    else:
        raise ValueError("Only one of --use-cora or --use-edge-case can be True.")

    # -------------- Validate --------------
    validate(adjm_dense, adjm_bsr, adjm_csr, X, weights, bias, args.apply_relu)

    # -------------- Benchmark --------------
    bench_fn(args.disable_proton, f"fused_spmm_gemm_relu_dense_torch_impl", args.reps, args.warmup_reps, torch.compile(fused_spmm_gemm_relu_dense_torch_impl, dynamic=True), adjm_dense, X, weights, bias, args.apply_relu)
    bench_fn(args.disable_proton, f"fused_spmm_gemm_relu_sparse_torch_impl", args.reps, args.warmup_reps, torch.compile(fused_spmm_gemm_relu_sparse_torch_impl, dynamic=True), adjm_csr, X, weights, bias, args.apply_relu)
    bench_fn(args.disable_proton, f"fused_spmm_gemm_relu_small_n", args.reps, args.warmup_reps, fused_spmm_gemm_relu_small_n, adjm_bsr, X, weights, bias, args.apply_relu)
    bench_fn(args.disable_proton, f"fused_spmm_gemm_relu_small_n_switch_loop", args.reps, args.warmup_reps, fused_spmm_gemm_relu_small_n_switch_loop, adjm_bsr, X, weights, bias, args.apply_relu)
    compute_capability = torch.cuda.get_device_capability(X.device)
    is_ampere = compute_capability[0] == 8
    if X.dtype == torch.float16 and is_ampere:
        bench_fn(args.disable_proton, f"fused_spmm_gemm_relu_small_n_cuda", args.reps, args.warmup_reps, fused_spmm_gemm_relu_small_n_cuda, adjm_bsr, X, weights, bias, args.apply_relu)
    elif X.dtype == torch.float16:
        print(f"Skipping CUDA impl of fused SpMM-GEMM-ReLU because it requires Ampere (compute capability 8.x), got {compute_capability[0]}.{compute_capability[1]}.")
    else:
        print("Skipping CUDA impl of fused SpMM-GEMM-ReLU because it only supports float16.")
