from __future__ import annotations

import argparse

import torch

from graphcuda.ops._fused_spmm_gemm_relu.fwd.naive_torch import fused_spmm_gemm_relu_dense_torch_impl
from graphcuda.ops._fused_spmm_gemm_relu.fwd.cuda_impl_small_n import fused_spmm_gemm_relu_small_n_cuda
from graphcuda.utils.bsr_rm import to_sparse_bsr_rm


def make_inputs(
    M: int,
    K1: int,
    K2: int,
    N: int,
    dtype: torch.dtype,
    adj_density: float = 0.05,
    use_bias: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    device = torch.device("cuda")
    mask = torch.rand((M, K1), device=device) < adj_density
    adjm_dense = torch.randn(M, K1, dtype=dtype, device=device)
    adjm_dense.mul_(mask.to(dtype))

    X = torch.randn(K1, K2, dtype=dtype, device=device)
    weights = torch.randn(K2, N, dtype=dtype, device=device)
    adjm_bsr = to_sparse_bsr_rm(adjm_dense)

    bias = torch.randn(1, N, dtype=dtype, device=device) if use_bias else None
    return adjm_dense, adjm_bsr, X, weights, bias


def validate(
    adjm_dense: torch.Tensor,
    adjm_bsr: torch.Tensor,
    X: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor | None,
    apply_relu: bool = True,
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> bool:
    Y_ref, relu_mask_ref = fused_spmm_gemm_relu_dense_torch_impl(
        adjm_dense, X, weights, bias, apply_relu
    )
    Y_cuda, relu_mask_cuda = fused_spmm_gemm_relu_small_n_cuda(
        adjm_bsr, X, weights, bias, apply_relu
    )
    torch.cuda.synchronize()

    max_abs_error = torch.abs(Y_ref - Y_cuda).max().item()
    passed = torch.allclose(Y_ref, Y_cuda, atol=atol, rtol=rtol)
    print(
        f"  fused CUDA custom impl: {'PASS' if passed else 'FAIL'}. "
        f"Max abs error: {max_abs_error}"
    )

    if apply_relu:
        assert relu_mask_ref is not None
        assert relu_mask_cuda is not None
        mask_passed = torch.equal(relu_mask_ref, relu_mask_cuda)
        mask_diff = torch.count_nonzero(relu_mask_ref != relu_mask_cuda).item()
        print(
            f"  fused CUDA ReLU mask: {'PASS' if mask_passed else 'FAIL'}. "
            f"Differing entries: {mask_diff}"
        )
        passed = passed and mask_passed

    return passed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the csrc fused SpMM-GEMM-ReLU CUDA forward kernel."
    )
    parser.add_argument("--M", type=int, default=256)
    parser.add_argument("--K1", type=int, default=256)
    parser.add_argument("--K2", type=int, default=64)
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["fp16"],
        default="fp16",
        help="The csrc SM80 forward kernel currently supports fp16 only.",
    )
    parser.add_argument(
        "--adj-density",
        type=float,
        default=0.05,
        help="Approximate fraction of nonzero entries in the dense adjacency before BSR conversion.",
    )
    parser.add_argument("--bias", action="store_true", help="Use a random (1, N) bias.")
    parser.add_argument("--apply-relu", action="store_true", help="Apply ReLU after the fused matmul.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required to validate the csrc forward kernel.")
    if args.N > 128:
        raise SystemExit("The csrc SM80 forward kernel currently supports N <= 128.")

    torch.manual_seed(args.seed)
    dtype = torch.float16

    print(
        "Validating fused SpMM-GEMM-ReLU CUDA forward with "
        f"M={args.M}, K1={args.K1}, K2={args.K2}, N={args.N}, "
        f"density={args.adj_density}, bias={args.bias}, apply_relu={args.apply_relu}"
    )
    adjm_dense, adjm_bsr, X, weights, bias = make_inputs(
        args.M,
        args.K1,
        args.K2,
        args.N,
        dtype,
        adj_density=args.adj_density,
        use_bias=args.bias,
    )

    if not validate(
        adjm_dense,
        adjm_bsr,
        X,
        weights,
        bias,
        apply_relu=args.apply_relu,
        atol=args.atol,
        rtol=args.rtol,
    ):
        raise SystemExit(1)
