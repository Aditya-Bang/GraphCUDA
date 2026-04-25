import os
import argparse
from contextlib import contextmanager

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.profiler as proton
from graphcuda.ops._fused_spmm_gemm_act.fwd.naive_torch import fused_spmm_gemm_relu_dense_torch_impl
from graphcuda.ops._fused_spmm_gemm_act.bwd.naive_torch import spmm_gemm_relu_backward_naive_torch_impl
from graphcuda.ops._fused_spmm_gemm_act.bwd.pyg_edgeindex_torch import spmm_gemm_relu_backward_pyg_edgeindex_torch_impl
from graphcuda.ops._fused_spmm_gemm_act.bwd.pyg_csr_torch import spmm_gemm_relu_backward_pyg_csr_torch_impl

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
def make_inputs(M: int, K1: int, K2: int, N: int, dtype: torch.dtype, adj_density: float = 0.05, use_bias: bool = False, apply_relu: bool = True):
    device = torch.device("cuda")
    mask = torch.rand((M, K1), device=device) < adj_density
    adjm_dense = torch.randn(M, K1, dtype=dtype, device=device)
    adjm_dense.mul_(mask.to(dtype))
    
    X = torch.randn(K1, K2, dtype=dtype, device=device)
    weights = torch.randn(K2, N, dtype=dtype, device=device)
    
    adjm_csr = adjm_dense.to_sparse_csr()
    adj_t_csr = adjm_dense.t().to_sparse_csr()
    target, source = adjm_dense.nonzero(as_tuple=True)
    edge_index = torch.stack([source, target], dim=0)
    edge_weight = adjm_dense[target, source]
    
    bias = torch.randn(1, N, dtype=dtype, device=device) if use_bias else None
    Y, relu_mask = fused_spmm_gemm_relu_dense_torch_impl(adjm_dense, X, weights, bias, apply_relu)
    grad_output = torch.randn_like(Y)
    return grad_output, adjm_dense, adjm_csr, adj_t_csr, edge_index, edge_weight, X, weights, bias, relu_mask


def make_cora_inputs(N: int, dtype: torch.dtype, use_bias: bool = False, apply_relu: bool = True):
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
    adjm_dense = to_dense_adj(edge_index.flip(0), edge_attr=edge_weight, max_num_nodes=x.size(-2))[0]
    adjm_csr = adjm_dense.to_sparse_csr()
    adj_t_csr = adjm_dense.t().to_sparse_csr()
    
    M = adjm_dense.shape[0]
    K1 = adjm_dense.shape[1]
    K2 = x.shape[1]
    
    weights = torch.randn(K2, N, dtype=dtype, device=device)
    bias = torch.randn(1, N, dtype=dtype, device=device) if use_bias else None
    Y, relu_mask = fused_spmm_gemm_relu_dense_torch_impl(adjm_dense, x, weights, bias, apply_relu)
    grad_output = torch.randn_like(Y)
    return grad_output, adjm_dense, adjm_csr, adj_t_csr, edge_index, edge_weight, x, weights, bias, relu_mask


def validate(grad_output, adjm_dense, adjm_csr, adj_t_csr, edge_index, edge_weight, X, weights, bias, relu_mask, apply_relu: bool = True):
    grads_ref = spmm_gemm_relu_backward_naive_torch_impl(grad_output, adjm_dense, X, weights, bias, relu_mask, apply_relu)
    grads_naive_sparse = spmm_gemm_relu_backward_naive_torch_impl(grad_output, adjm_csr, X, weights, bias, relu_mask, apply_relu)
    grads_pyg_edgeindex = spmm_gemm_relu_backward_pyg_edgeindex_torch_impl(grad_output, edge_index, edge_weight, X, weights, bias, relu_mask, apply_relu)
    grads_pyg_csr = spmm_gemm_relu_backward_pyg_csr_torch_impl(grad_output, adj_t_csr, X, weights, bias, relu_mask, apply_relu)

    def _print_close(label, ref, actual):
        atol = 1e-2
        rtol = 1e-2

        for i, (ref_i, actual_i) in enumerate(zip(ref, actual)):
            if ref_i is None:
                passed = actual_i is None
                max_abs_error = 0.0
            else:
                passed = torch.allclose(ref_i, actual_i, atol=atol, rtol=rtol)
                max_abs_error = torch.abs(ref_i - actual_i).max().item()
            print(f"  {label} grad {i}: {'✅' if passed else '❌'}. Max abs error: {max_abs_error}")

    _print_close("naive sparse torch impl", grads_ref, grads_naive_sparse)
    _print_close("pyg edgeindex torch impl", grads_ref, grads_pyg_edgeindex)
    _print_close("pyg csr impl", grads_ref, grads_pyg_csr)


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
    parser.add_argument("--adj-density", type=float, default=0.05, help="Approximate fraction of nonzero entries in the (dense) adjacency before CSR conversion.",)
    parser.add_argument("--bias", action="store_true", help="Use a random (1, N) bias in backward implementations.",)
    parser.add_argument("--apply-relu", action="store_true", help="Apply ReLU after all operations.",)
    parser.add_argument("--use-cora", action="store_true", help="Use Cora dataset inputs.")
    args = parser.parse_args()
    dtype = parse_dtype(args.dtype)
    
    torch.manual_seed(0)

    # -------------- Make inputs --------------
    if not args.use_cora:
        grad_output, adjm_dense, adjm_csr, adj_t_csr, edge_index, edge_weight, X, weights, bias, relu_mask = make_inputs(
            args.M, args.K1, args.K2, args.N, dtype, adj_density=args.adj_density, use_bias=args.bias, apply_relu=args.apply_relu
        )
    else:
        print("Using Cora dataset inputs, ignoring M, K1, K2, adj-density arguments.")
        grad_output, adjm_dense, adjm_csr, adj_t_csr, edge_index, edge_weight, X, weights, bias, relu_mask = make_cora_inputs(N=args.N, dtype=dtype, use_bias=args.bias, apply_relu=args.apply_relu)

    # -------------- Validate --------------
    validate(grad_output, adjm_dense, adjm_csr, adj_t_csr, edge_index, edge_weight, X, weights, bias, relu_mask, args.apply_relu)

    # -------------- Benchmark --------------
    bench_fn(f"spmm_gemm_relu_backward_naive_dense_torch_impl", args.reps, args.warmup_reps, torch.compile(spmm_gemm_relu_backward_naive_torch_impl, dynamic=True), grad_output, adjm_dense, X, weights, bias, relu_mask, args.apply_relu)
    bench_fn(f"spmm_gemm_relu_backward_naive_sparse_torch_impl", args.reps, args.warmup_reps, torch.compile(spmm_gemm_relu_backward_naive_torch_impl, dynamic=True), grad_output, adjm_csr, X, weights, bias, relu_mask, args.apply_relu)
    bench_fn(f"spmm_gemm_relu_backward_pyg_edgeindex_torch_impl", args.reps, args.warmup_reps, torch.compile(spmm_gemm_relu_backward_pyg_edgeindex_torch_impl, dynamic=True), grad_output, edge_index, edge_weight, X, weights, bias, relu_mask, args.apply_relu)
    bench_fn(f"spmm_gemm_relu_backward_pyg_csr_torch_impl", args.reps, args.warmup_reps, torch.compile(spmm_gemm_relu_backward_pyg_csr_torch_impl, dynamic=True), grad_output, adj_t_csr, X, weights, bias, relu_mask, args.apply_relu)
