#include "fused_spmm_gemm_relu_sm80_kernel.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cstdint>
#include <limits>

namespace graphcuda {
namespace ops {
namespace {

using namespace nvcuda;

constexpr int kBlockM = 16;
constexpr int kChunkK1 = 64;
constexpr int kChunkK2 = 64;
constexpr int kWarpCount = 4;
constexpr int kThreads = 32 * kWarpCount;
constexpr int kTilesPerPass = 4;
constexpr int kMaxPasses = 2;   // supports N <= 128 with 4 warps x 2 passes.
struct SharedStorage {
    alignas(16) int32_t cols[kChunkK1];
    alignas(16) half a[kBlockM * kChunkK1];                  // [16, 64]
    alignas(16) half x[kWarpCount][kChunkK1 * 16];           // per-warp [64, 16]
    alignas(16) float acc1_f[kWarpCount][kBlockM * 16];      // per-warp [16, 16]
    alignas(16) half acc1_h[kWarpCount][kBlockM * 16];       // per-warp [16, 16]
    alignas(16) half w[kWarpCount][16 * 16];                 // per-warp [16, 16]
};

template <bool HasBias, bool ApplyRelu>
__global__ __launch_bounds__(kThreads)
void fused_spmm_gemm_relu_sm80_kernel(
    const half* __restrict__ bsr_values_rm,
    const int32_t* __restrict__ bsr_crow,
    const int32_t* __restrict__ bsr_col,
    int M,
    int K1,
    int K2,
    int N,
    const half* __restrict__ X,
    const half* __restrict__ W,
    const half* __restrict__ bias,
    half* __restrict__ Y,
    bool* __restrict__ relu_mask) {

    __shared__ SharedStorage smem;

    const int tid = static_cast<int>(threadIdx.x);
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    const int row_block = static_cast<int>(blockIdx.x);
    const int row_base = row_block * kBlockM;
    const int n_tiles = (N + 15) >> 4;

    const int row_start = bsr_crow[row_block];
    const int row_end = bsr_crow[row_block + 1];
    const int row_nnz = row_end - row_start;
    const int values_base = row_start * kBlockM;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frags[kMaxPasses];
#pragma unroll
    for (int pass = 0; pass < kMaxPasses; ++pass) {
        wmma::fill_fragment(c_frags[pass], 0.0f);
    }

    for (int k1_base = 0; k1_base < row_nnz; k1_base += kChunkK1) {
        const int actual_k1 = min(kChunkK1, row_nnz - k1_base);

        for (int i = tid; i < kChunkK1; i += kThreads) {
            smem.cols[i] = (i < actual_k1) ? bsr_col[row_start + k1_base + i] : 0;
        }

        for (int idx = tid; idx < kBlockM * kChunkK1; idx += kThreads) {
            const int r = idx / kChunkK1;
            const int c = idx % kChunkK1;
            half v = __float2half(0.0f);
            if (c < actual_k1 && (row_base + r) < M) {
                v = bsr_values_rm[values_base + r * row_nnz + k1_base + c];
            }
            smem.a[idx] = v;
        }
        __syncthreads();

        for (int k2_base = 0; k2_base < K2; k2_base += kChunkK2) {
            // Each warp stages one 64x16 slice of X for the first WMMA stage.
            for (int idx = lane; idx < kChunkK1 * 16; idx += 32) {
                const int r = idx / 16;
                const int c = idx % 16;
                const int x_row = (r < actual_k1) ? smem.cols[r] : 0;
                const int x_col = k2_base + warp_id * 16 + c;
                half v = __float2half(0.0f);
                if (r < actual_k1 && x_row < K1 && x_col < K2) {
                    v = X[x_row * K2 + x_col];
                }
                smem.x[warp_id][idx] = v;
            }
            __syncthreads();

            // Stage 1: acc1[:, warp_id*16:(warp_id+1)*16] = A(16x64) @ X_tile(64x16)
            {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc1_frag;
                wmma::fill_fragment(acc1_frag, 0.0f);

#pragma unroll
                for (int kk = 0; kk < kChunkK1; kk += 16) {
                    const half* a_ptr = &smem.a[kk];
                    const half* x_ptr = &smem.x[warp_id][kk * 16];
                    wmma::load_matrix_sync(a_frag, a_ptr, kChunkK1);
                    wmma::load_matrix_sync(b_frag, x_ptr, 16);
                    wmma::mma_sync(acc1_frag, a_frag, b_frag, acc1_frag);
                }

                wmma::store_matrix_sync(smem.acc1_f[warp_id], acc1_frag, 16, wmma::mem_row_major);
                __syncwarp();
                for (int idx = lane; idx < kBlockM * 16; idx += 32) {
                    smem.acc1_h[warp_id][idx] = __float2half_rn(smem.acc1_f[warp_id][idx]);
                }
            }
            __syncthreads();

            // Stage 2: each warp owns one 16-column output tile per pass.
#pragma unroll
            for (int pass = 0; pass < kMaxPasses; ++pass) {
                const int tile_idx = pass * kTilesPerPass + warp_id;
                if (tile_idx < n_tiles) {
#pragma unroll
                    for (int src_tile = 0; src_tile < kWarpCount; ++src_tile) {
                        // W rows correspond to the 16 columns produced by acc1_h[src_tile].
                        for (int idx = lane; idx < 16 * 16; idx += 32) {
                            const int r = idx / 16;
                            const int c = idx % 16;
                            const int gk = k2_base + src_tile * 16 + r;
                            const int gn = tile_idx * 16 + c;
                            half v = __float2half(0.0f);
                            if (gk < K2 && gn < N) {
                                v = W[gk * N + gn];
                            }
                            smem.w[warp_id][idx] = v;
                        }
                        __syncwarp();

                        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a2_frag;
                        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b2_frag;
                        wmma::load_matrix_sync(a2_frag, smem.acc1_h[src_tile], 16);
                        wmma::load_matrix_sync(b2_frag, smem.w[warp_id], 16);
                        wmma::mma_sync(c_frags[pass], a2_frag, b2_frag, c_frags[pass]);
                        __syncwarp();
                    }
                }
            }
            __syncthreads();
        }
    }

    // Epilogue: bias + ReLU + store.
#pragma unroll
    for (int pass = 0; pass < kMaxPasses; ++pass) {
        const int tile_idx = pass * kTilesPerPass + warp_id;
        if (tile_idx < n_tiles) {
            wmma::store_matrix_sync(smem.acc1_f[warp_id], c_frags[pass], 16, wmma::mem_row_major);
            __syncwarp();

            for (int idx = lane; idx < kBlockM * 16; idx += 32) {
                const int r = idx / 16;
                const int c = idx % 16;
                const int gm = row_base + r;
                const int gn = tile_idx * 16 + c;
                if (gm < M && gn < N) {
                    float v = smem.acc1_f[warp_id][idx];
                    if constexpr (HasBias) {
                        v += __half2float(bias[gn]);
                    }
                    if constexpr (ApplyRelu) {
                        const bool keep = v > 0.0f;
                        relu_mask[gm * N + gn] = keep;
                        v = keep ? v : 0.0f;
                    }
                    Y[gm * N + gn] = __float2half_rn(v);
                }
            }
        }
    }
}

void check_inputs(
    const torch::Tensor& bsr_values_rm,
    const torch::Tensor& bsr_crow_i32,
    const torch::Tensor& bsr_col_i32,
    int64_t M,
    int64_t K1,
    const torch::Tensor& X,
    const torch::Tensor& W,
    const torch::Tensor& bias) {

    TORCH_CHECK(bsr_values_rm.is_cuda(), "bsr_values_rm must be CUDA");
    TORCH_CHECK(bsr_crow_i32.is_cuda(), "bsr_crow_i32 must be CUDA");
    TORCH_CHECK(bsr_col_i32.is_cuda(), "bsr_col_i32 must be CUDA");
    TORCH_CHECK(X.is_cuda(), "X must be CUDA");
    TORCH_CHECK(W.is_cuda(), "W must be CUDA");
    TORCH_CHECK(!bias.defined() || bias.is_cuda(), "bias must be CUDA when defined");

    TORCH_CHECK(bsr_values_rm.scalar_type() == torch::kFloat16, "bsr_values_rm must be float16");
    TORCH_CHECK(X.scalar_type() == torch::kFloat16, "X must be float16");
    TORCH_CHECK(W.scalar_type() == torch::kFloat16, "W must be float16");
    TORCH_CHECK(!bias.defined() || bias.numel() == 0 || bias.scalar_type() == torch::kFloat16,
                "bias must be float16 when defined");

    TORCH_CHECK(bsr_crow_i32.scalar_type() == torch::kInt32, "bsr_crow_i32 must be int32");
    TORCH_CHECK(bsr_col_i32.scalar_type() == torch::kInt32, "bsr_col_i32 must be int32");

    TORCH_CHECK(bsr_values_rm.is_contiguous(), "bsr_values_rm must be contiguous");
    TORCH_CHECK(bsr_crow_i32.is_contiguous(), "bsr_crow_i32 must be contiguous");
    TORCH_CHECK(bsr_col_i32.is_contiguous(), "bsr_col_i32 must be contiguous");
    TORCH_CHECK(X.is_contiguous(), "X must be contiguous");
    TORCH_CHECK(W.is_contiguous(), "W must be contiguous");
    TORCH_CHECK(!bias.defined() || bias.numel() == 0 || bias.is_contiguous(), "bias must be contiguous");

    TORCH_CHECK(X.dim() == 2, "X must be 2D");
    TORCH_CHECK(W.dim() == 2, "W must be 2D");
    TORCH_CHECK(M >= 0 && K1 >= 0, "M and K1 must be non-negative");

    TORCH_CHECK(X.size(0) == K1, "X.size(0) must equal K1");
    TORCH_CHECK(X.size(1) == W.size(0), "X.size(1) must equal W.size(0)");
    TORCH_CHECK(W.size(1) <= 128, "This SM80 kernel currently supports N <= 128");

    TORCH_CHECK(!bias.defined() || bias.numel() == 0 || (bias.dim() == 1 && bias.size(0) == W.size(1)),
                "bias must be empty or 1D of shape [N]");

    TORCH_CHECK(M <= std::numeric_limits<int>::max(), "M is too large for this kernel");
    TORCH_CHECK(K1 <= std::numeric_limits<int>::max(), "K1 is too large for this kernel");
    TORCH_CHECK(X.size(1) <= std::numeric_limits<int>::max(), "K2 is too large for this kernel");
    TORCH_CHECK(W.size(1) <= std::numeric_limits<int>::max(), "N is too large for this kernel");
}

template <bool HasBias, bool ApplyRelu>
void launch_kernel(
    const torch::Tensor& bsr_values_rm,
    const torch::Tensor& bsr_crow_i32,
    const torch::Tensor& bsr_col_i32,
    int M,
    int K1,
    int K2,
    int N,
    const torch::Tensor& X,
    const torch::Tensor& W,
    const torch::Tensor& bias,
    const torch::Tensor& Y,
    const torch::Tensor& relu_mask,
    cudaStream_t stream) {

    const dim3 grid((M + kBlockM - 1) / kBlockM);
    const dim3 block(kThreads);

    const half* bias_ptr = nullptr;
    if constexpr (HasBias) {
        bias_ptr = reinterpret_cast<const half*>(bias.data_ptr<at::Half>());
    }

    bool* relu_ptr = nullptr;
    if constexpr (ApplyRelu) {
        relu_ptr = relu_mask.data_ptr<bool>();
    }

    fused_spmm_gemm_relu_sm80_kernel<HasBias, ApplyRelu><<<grid, block, 0, stream>>>(
        reinterpret_cast<const half*>(bsr_values_rm.data_ptr<at::Half>()),
        bsr_crow_i32.data_ptr<int32_t>(),
        bsr_col_i32.data_ptr<int32_t>(),
        M,
        K1,
        K2,
        N,
        reinterpret_cast<const half*>(X.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(W.data_ptr<at::Half>()),
        bias_ptr,
        reinterpret_cast<half*>(Y.data_ptr<at::Half>()),
        relu_ptr);
}

}  // namespace

std::vector<torch::Tensor> fused_spmm_gemm_relu_sm80_forward_cuda(
    torch::Tensor bsr_values_rm,
    torch::Tensor bsr_crow_i32,
    torch::Tensor bsr_col_i32,
    int64_t M,
    int64_t K1,
    torch::Tensor X,
    torch::Tensor W,
    torch::Tensor bias,
    bool apply_relu) {

    check_inputs(bsr_values_rm, bsr_crow_i32, bsr_col_i32, M, K1, X, W, bias);

    c10::cuda::CUDAGuard device_guard(X.device());
    const auto stream = at::cuda::getCurrentCUDAStream();

    auto Y = torch::empty({M, W.size(1)}, X.options());
    torch::Tensor relu_mask;
    if (apply_relu) {
        relu_mask = torch::empty({M, W.size(1)}, X.options().dtype(torch::kBool));
    } else {
        relu_mask = torch::empty({0}, X.options().dtype(torch::kBool));
    }

    const int M_i = static_cast<int>(M);
    const int K1_i = static_cast<int>(K1);
    const int K2_i = static_cast<int>(X.size(1));
    const int N_i = static_cast<int>(W.size(1));
    const bool has_bias = bias.defined() && bias.numel() != 0;

    if (has_bias) {
        if (apply_relu) {
            launch_kernel<true, true>(
                bsr_values_rm, bsr_crow_i32, bsr_col_i32,
                M_i, K1_i, K2_i, N_i,
                X, W, bias, Y, relu_mask, stream.stream());
        } else {
            launch_kernel<true, false>(
                bsr_values_rm, bsr_crow_i32, bsr_col_i32,
                M_i, K1_i, K2_i, N_i,
                X, W, bias, Y, relu_mask, stream.stream());
        }
    } else {
        if (apply_relu) {
            launch_kernel<false, true>(
                bsr_values_rm, bsr_crow_i32, bsr_col_i32,
                M_i, K1_i, K2_i, N_i,
                X, W, bias, Y, relu_mask, stream.stream());
        } else {
            launch_kernel<false, false>(
                bsr_values_rm, bsr_crow_i32, bsr_col_i32,
                M_i, K1_i, K2_i, N_i,
                X, W, bias, Y, relu_mask, stream.stream());
        }
    }

    AT_CUDA_CHECK(cudaGetLastError());
    return {Y, relu_mask};
}

}  // namespace ops
}  // namespace graphcuda
