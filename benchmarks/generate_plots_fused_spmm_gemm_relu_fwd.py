import argparse
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import torch

from bench_fused_spmm_gemm_relu_fwd import (
    fused_spmm_gemm_relu_dense_torch_impl,
    fused_spmm_gemm_relu_small_n,
    fused_spmm_gemm_relu_sparse_torch_impl,
    make_inputs,
)


PLOTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "plots"))


@dataclass
class DensityResult:
    adj_density: float
    dense_torch_ms: float
    sparse_torch_ms: float
    triton_small_n_ms: float


def parse_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str == "fp32":
        return torch.float32
    if dtype_str == "fp16":
        return torch.float16
    if dtype_str == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def parse_densities(densities: str) -> list[float]:
    parsed = [float(density.strip()) for density in densities.split(",") if density.strip()]
    if not parsed:
        raise ValueError("--adj-densities must contain at least one value")
    for density in parsed:
        if density <= 0.0 or density > 1.0:
            raise ValueError("Adjacency densities must be in the range (0, 1]")
    return sorted(parsed, reverse=True)


def time_cuda_fn(label: str, fn, reps: int, warmup_reps: int, *args) -> float:
    for _ in range(warmup_reps):
        fn(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(reps):
        fn(*args)
    end.record()

    torch.cuda.synchronize()
    avg_ms = start.elapsed_time(end) / reps
    print(f"  {label}: {avg_ms:.4f} ms")
    return avg_ms


def benchmark_density(
    adj_density: float,
    m: int,
    k1: int,
    k2: int,
    n: int,
    dtype: torch.dtype,
    reps: int,
    warmup_reps: int,
    use_bias: bool,
    apply_relu: bool,
    compile_torch_dense: bool,
    compile_torch_sparse: bool,
) -> DensityResult:
    print(f"\nBenchmarking adj_density={adj_density:g}")
    adjm_dense, adjm_bsr, adjm_csr, x, weights, bias = make_inputs(
        m,
        k1,
        k2,
        n,
        dtype,
        adj_density=adj_density,
        use_bias=use_bias,
    )

    dense_torch_fn = fused_spmm_gemm_relu_dense_torch_impl
    if compile_torch_dense:
        dense_torch_fn = torch.compile(dense_torch_fn, dynamic=True)

    sparse_torch_fn = fused_spmm_gemm_relu_sparse_torch_impl
    if compile_torch_sparse:
        sparse_torch_fn = torch.compile(sparse_torch_fn, dynamic=True)

    dense_torch_ms = time_cuda_fn(
        "pytorch dense spmm-gemm-relu",
        dense_torch_fn,
        reps,
        warmup_reps,
        adjm_dense,
        x,
        weights,
        bias,
        apply_relu,
    )
    sparse_torch_ms = time_cuda_fn(
        "pytorch sparse spmm-gemm-relu",
        sparse_torch_fn,
        reps,
        warmup_reps,
        adjm_csr,
        x,
        weights,
        bias,
        apply_relu,
    )
    triton_small_n_ms = time_cuda_fn(
        "triton fused_spmm_gemm_relu_small_n",
        fused_spmm_gemm_relu_small_n,
        reps,
        warmup_reps,
        adjm_bsr,
        x,
        weights,
        bias,
        apply_relu,
    )

    return DensityResult(
        adj_density=adj_density,
        dense_torch_ms=dense_torch_ms,
        sparse_torch_ms=sparse_torch_ms,
        triton_small_n_ms=triton_small_n_ms,
    )


def plot_results(
    results: list[DensityResult],
    gpu_name: str,
    m: int,
    k1: int,
    k2: int,
    n: int,
    dtype: str,
    reps: int,
    warmup_reps: int,
    output_path: str,
):
    densities = [result.adj_density for result in results]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(
        densities,
        [result.dense_torch_ms for result in results],
        marker="o",
        linewidth=2,
        label="PyTorch dense",
    )
    ax.plot(
        densities,
        [result.sparse_torch_ms for result in results],
        marker="s",
        linewidth=2,
        label="PyTorch sparse",
    )
    ax.plot(
        densities,
        [result.triton_small_n_ms for result in results],
        marker="^",
        linewidth=2,
        label="Triton fused small-n",
    )
    ax.set_xlabel("Adjacency Density")
    ax.set_ylabel("Average Time (ms)")
    ax.set_title("Fused SpMM-GEMM-ReLU Forward Runtime")
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)
    ax.legend()

    metadata = (
        f"GPU: {gpu_name}\n"
        f"M={m}, K1={k1}, K2={k2}, N={n}, dtype={dtype}\n"
        f"reps={reps}, warmup_reps={warmup_reps}"
    )
    ax.text(
        0.02,
        0.98,
        metadata,
        transform=ax.transAxes,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Sweep adjacency densities and plot dense PyTorch, sparse PyTorch, "
            "and Triton fused_spmm_gemm_relu_small_n forward runtimes."
        )
    )
    parser.add_argument("--M", type=int, default=1024)
    parser.add_argument("--K1", type=int, default=1024)
    parser.add_argument("--K2", type=int, default=1024)
    parser.add_argument("--N", type=int, default=16)
    parser.add_argument("--dtype", type=str, choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument("--adj-densities", type=str, default="0.01,0.005,0.002,0.001")
    parser.add_argument("--reps", type=int, default=1000)
    parser.add_argument("--warmup-reps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--apply-relu", action="store_true")
    parser.add_argument(
        "--compile-torch-dense",
        action="store_true",
        help="Use torch.compile for the PyTorch dense implementation, matching bench_fused_spmm_gemm_relu_fwd.py.",
    )
    parser.add_argument(
        "--compile-torch-sparse",
        action="store_true",
        help="Use torch.compile for the PyTorch sparse implementation, matching bench_fused_spmm_gemm_relu_fwd.py.",
    )
    parser.add_argument("--output", type=str, default=None, help="Optional output PNG path.")
    args = parser.parse_args()

    if args.reps <= 0:
        raise ValueError("--reps must be positive")
    if args.warmup_reps < 0:
        raise ValueError("--warmup-reps must be non-negative")
    if not torch.cuda.is_available():
        raise RuntimeError("This plot generator requires CUDA.")

    dtype = parse_dtype(args.dtype)
    densities = parse_densities(args.adj_densities)
    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())

    torch.manual_seed(args.seed)
    results = [
        benchmark_density(
            adj_density=density,
            m=args.M,
            k1=args.K1,
            k2=args.K2,
            n=args.N,
            dtype=dtype,
            reps=args.reps,
            warmup_reps=args.warmup_reps,
            use_bias=args.bias,
            apply_relu=args.apply_relu,
            compile_torch_dense=args.compile_torch_dense,
            compile_torch_sparse=args.compile_torch_sparse,
        )
        for density in densities
    ]

    os.makedirs(PLOTS_DIR, exist_ok=True)
    output_path = args.output
    if output_path is None:
        output_name = (
            "fused_spmm_gemm_relu_fwd_time_"
            f"M{args.M}_K1{args.K1}_K2{args.K2}_N{args.N}_{args.dtype}.png"
        )
        output_path = os.path.join(PLOTS_DIR, output_name)
    else:
        output_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plot_results(
        results=results,
        gpu_name=gpu_name,
        m=args.M,
        k1=args.K1,
        k2=args.K2,
        n=args.N,
        dtype=args.dtype,
        reps=args.reps,
        warmup_reps=args.warmup_reps,
        output_path=output_path,
    )

    print("\nSummary")
    for result in results:
        print(
            f"  density={result.adj_density:g}: "
            f"dense={result.dense_torch_ms:.4f}ms, "
            f"sparse={result.sparse_torch_ms:.4f}ms, "
            f"triton={result.triton_small_n_ms:.4f}ms"
        )
    print(f"\nSaved plot to {output_path}")


if __name__ == "__main__":
    main()
