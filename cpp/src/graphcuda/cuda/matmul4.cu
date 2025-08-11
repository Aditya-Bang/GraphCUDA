#include "matmul.cuh"

// compile-time warp width for indexing (keep constexpr)
constexpr int THREADS_PER_WARP = 32;

__device__ inline bool is_aligned_16(const void* p) {
    return (reinterpret_cast<uintptr_t>(p) & 0xF) == 0;
}

// Helper: safe float4 reader (handles tails)
__device__ inline float4 safe_load4(const float* ptr, int stride, int idx0, int limit0, int limit1) {
  // ptr points to row-major array with row stride `stride`.
  // We attempt to read elements at positions idx0..idx0+3 in the row `limit0` (row index).
  // limit1 is the number of columns in that row (to check bounds).
  float4 out;
  if (idx0 + 3 < limit1) {
    out = reinterpret_cast<const float4*>(&ptr[idx0])[0];
  } else {
    // scalar fallback per element
    out.x = (idx0 + 0 < limit1) ? ptr[idx0 + 0] : 0.0f;
    out.y = (idx0 + 1 < limit1) ? ptr[idx0 + 1] : 0.0f;
    out.z = (idx0 + 2 < limit1) ? ptr[idx0 + 2] : 0.0f;
    out.w = (idx0 + 3 < limit1) ? ptr[idx0 + 3] : 0.0f;
  }
  return out;
}

// Cooperative vectorized store (vector of 4)
__device__ inline void safe_store4(float* ptr, int idx0, int limit, const float4 &v) {
  if (idx0 + 3 < limit && ((reinterpret_cast<uintptr_t>(&ptr[idx0]) & 0xF) == 0)) {
    reinterpret_cast<float4*>(&ptr[idx0])[0] = v;
  } else {
    if (idx0 + 0 < limit) ptr[idx0 + 0] = v.x;
    if (idx0 + 1 < limit) ptr[idx0 + 1] = v.y;
    if (idx0 + 2 < limit) ptr[idx0 + 2] = v.z;
    if (idx0 + 3 < limit) ptr[idx0 + 3] = v.w;
  }
}

/*
 Reworked warp-tiling SGEMM kernel.
 Template parameters:
  BM,BN,BK : block tile sizes (rows x cols x k)
  WM,WN    : warp tile sizes (rows x cols) computed by each warp
  WNITER   : how many subtiles across WN (affects inner loops)
  TM,TN    : per-thread micro-tiles inside warp subtiles
  NUM_THREADS_PER_BLOCK: number of threads per CTA (block)
*/
template <int BM, int BN, int BK, int WM, int WN, int WNITER, int TM, int TN, int NUM_THREADS_PER_BLOCK>
__global__ void __launch_bounds__(NUM_THREADS_PER_BLOCK) sgemmWarptiling(
                                                          int M, int N, int K, float alpha,
                                                          const float* __restrict__ A,
                                                          const float* __restrict__ B,
                                                          float beta,
                                                          float* __restrict__ C
                                                          ) {
  const unsigned int blockRow = blockIdx.y;
  const unsigned int blockCol = blockIdx.x;

  const unsigned int warpIdx = threadIdx.x / THREADS_PER_WARP;
  const unsigned int warpCol = warpIdx % (BN / WN);
  const unsigned int warpRow = warpIdx / (BN / WN);

  constexpr unsigned int WMITER = (WM * WN) / (THREADS_PER_WARP * TM * TN * WNITER);
  constexpr unsigned int WSUBM = WM / WMITER; // sub-tile rows per warp
  constexpr unsigned int WSUBN = WN / WNITER; // sub-tile cols per warp

  const uint threadIdxInWarp = threadIdx.x % THREADS_PER_WARP; // [0, 31]
  const uint threadColInWarpSubtile = threadIdxInWarp % (WSUBN / TN); // i%(16/4)
  const uint threadRowInWarpSubtile = threadIdxInWarp / (WSUBN / TN); // i/4

  // Shared memory (dynamic): As [BM * BK], Bs [BK * BN]
  extern __shared__ float shmem[];
  float* As = shmem;
  float* Bs = shmem + (BM * BK);

  // register memory
  float regM[WMITER * TM] = {0.0};
  float regN[WNITER * TN] = {0.0};
  float threadResults[WMITER * TM * WNITER * TN] = {0.0};

  for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // Cooperative load As (BM x BK) into shared memory using linear-stride
    // We will write As in transposed layout that favors later reading:
    for (int currIdx = threadIdx.x; currIdx < CEIL_DIV(BM * BK, 4); currIdx += NUM_THREADS_PER_BLOCK) {
      // current block accessing memory in A is (blockRow, bkIdx/BK)
      // start index in A for current block (row major) is blockRow * BM * K + bkIdx
      unsigned int block_element_idx = currIdx * 4;
      unsigned int a_block_row = block_element_idx / BK; // row inside BM
      unsigned int a_block_col = block_element_idx % BK; // col inside BK
      int global_r = blockRow * BM + (int)a_block_row;
      int global_c = bkIdx + (int)a_block_col;
      float4 v; // load 4 elements
      if (global_r < M) {
        if (global_c + 3 < K && is_aligned_16(&A[global_r * K + global_c])) {
          v = reinterpret_cast<const float4*>(&A[global_r * K + global_c])[0];
        } else {
          v.x = (global_c + 0 < K) ? A[global_r * K + global_c + 0] : 0.0f;
          v.y = (global_c + 1 < K) ? A[global_r * K + global_c + 1] : 0.0f;
          v.z = (global_c + 2 < K) ? A[global_r * K + global_c + 2] : 0.0f;
          v.w = (global_c + 3 < K) ? A[global_r * K + global_c + 3] : 0.0f;
        }
      } else {
        v.x = v.y = v.z = v.w = 0.0f;
      }

      // store As as As[ (k_col) * BM + row ] so that reading over row is coalesced.
      // Each thread can load multiple float4 elements
      if (a_block_row < (unsigned int)BM) {
        As[(a_block_col + 0) * BM + a_block_row] = v.x;
        As[(a_block_col + 1) * BM + a_block_row] = v.y;
        As[(a_block_col + 2) * BM + a_block_row] = v.z;
        As[(a_block_col + 3) * BM + a_block_row] = v.w;
      }
    }

    // Cooperative load Bs (BK x BN) into shared memory using linear-stride
    for (int currIdx = threadIdx.x; currIdx < CEIL_DIV(BK * BN, 4); currIdx += NUM_THREADS_PER_BLOCK) {
      // current block accessing memory in B is (bkIdx/BK, blockCol)
      // start index in B for current block (row major) is bkIdx * N + blockCol * BN
      unsigned int block_element_idx = currIdx * 4;
      unsigned int b_block_row = block_element_idx / BN; // row inside BK
      unsigned int b_block_col = block_element_idx % BN; // col inside BN
      int global_r = bkIdx + (int)b_block_row;
      int global_c = blockCol * BN + (int)b_block_col;
      float4 v; // load 4 elements
      if (global_r < K) {
        if (global_c + 3 < N && is_aligned_16(&B[global_r * N + global_c])) {
          v = reinterpret_cast<const float4*>(&B[global_r * N + global_c])[0];
        } else {
          v.x = (global_c + 0 < N) ? B[global_r * N + global_c + 0] : 0.0f;
          v.y = (global_c + 1 < N) ? B[global_r * N + global_c + 1] : 0.0f;
          v.z = (global_c + 2 < N) ? B[global_r * N + global_c + 2] : 0.0f;
          v.w = (global_c + 3 < N) ? B[global_r * N + global_c + 3] : 0.0f;
        }
      } else {
        v.x = v.y = v.z = v.w = 0.0f;
      }

      // store Bs as Bs[ (k_col) * BM + row ] so that reading over row is coalesced.
      // Each thread can load multiple float4 elements
      if (b_block_row < (unsigned int)BK) { // TODO: remove if statement here
        Bs[(b_block_row + 0) * BM + b_block_col] = v.x;
        Bs[(b_block_row + 1) * BM + b_block_col] = v.y;
        Bs[(b_block_row + 2) * BM + b_block_col] = v.z;
        Bs[(b_block_row + 3) * BM + b_block_col] = v.w;
      }
    }

    __syncthreads();

    
    for (unsigned int dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // load regM for this thread, i.e. parts of warp tile, each in seperate warp subtile
      for (unsigned int subWarpTileRow = 0; subWarpTileRow < WMITER; ++subWarpTileRow) {
        for (unsigned int tm = 0; tm < TM; ++tm) {
          // As inverted so, need column major access
          // get element at (As_r, dotIdx)
          unsigned int As_r = warpRow * WM + subWarpTileRow * WSUBM + threadRowInWarpSubtile * TM + tm;
          regM[subWarpTileRow * TM + tm] = As[(dotIdx * BM) + As_r];
        }
      }

      // load regN for this thread, i.e. parts of warp tile, each in seperate warp subtile
      for (unsigned int subWarpTileCol = 0; subWarpTileCol < WNITER; ++subWarpTileCol) {
        for (unsigned int tn = 0; tn < TN; ++tn) {
          // get element (dotIdx, Bs_c)
          unsigned int Bs_c = warpCol * WN + subWarpTileCol * WSUBN + threadColInWarpSubtile * TN + tn;
          regN[subWarpTileCol * TN + tn] = Bs[dotIdx * BN + Bs_c];
        }
      }

      // actual calculation for each TMxTN tile in each warp subtile
      for (unsigned int subWarpTileRow = 0; subWarpTileRow < WMITER; ++subWarpTileRow) {
        for (unsigned int subWarpTileCol = 0; subWarpTileCol < WNITER; ++subWarpTileCol) {
          for (unsigned int tm = 0; tm < TM; ++tm) {
            float a_val = regM[subWarpTileRow * TM + tm];
            for (unsigned int tn = 0; tn < TN; ++tn) {
              float b_val = regN[subWarpTileCol * TN + tn];
              // thread results has row major layout, only stores elements computed by this thread for the warp tile, will need to be expanded
              threadResults[(subWarpTileRow * TM + tm) * (WNITER * TN) + (subWarpTileCol * TN + tn)] += a_val * b_val;
            }
          }
        }
      }
    }
    __syncthreads();
  }

  // write out results -> fix this
  for (unsigned int subWarpTileRow = 0; subWarpTileRow < WMITER; ++subWarpTileRow) {
    for (unsigned int subWarpTileCol = 0; subWarpTileCol < WNITER; ++subWarpTileCol) {
      for (unsigned int tm = 0; tm < TM; ++tm) {
        int globalRow = blockRow * BM + warpRow * WM + subWarpTileRow * WSUBM + threadRowInWarpSubtile * TM + tm;
        if (globalRow >= M) continue;
        float* crow = &C[globalRow * N + blockCol * BN + warpCol * WN + subWarpTileCol * WSUBN + threadColInWarpSubtile * TN];
        for (unsigned int tn = 0; tn < TN; tn += 4) {
          int globalCol = blockCol * BN + warpCol * WN + subWarpTileCol * WSUBN + threadColInWarpSubtile * TN + tn;
          if (globalCol >= N) continue;
          float4 oldv = safe_load4(crow, N, tn, N, N); // load existing C values
          float4 newv;
          newv.x = alpha * threadResults[(subWarpTileRow * TM + tm) * (WNITER * TN) + (subWarpTileCol * TN + tn + 0)] + beta * oldv.x;
          newv.y = alpha * threadResults[(subWarpTileRow * TM + tm) * (WNITER * TN) + (subWarpTileCol * TN + tn + 1)] + beta * oldv.y;
          newv.z = alpha * threadResults[(subWarpTileRow * TM + tm) * (WNITER * TN) + (subWarpTileCol * TN + tn + 2)] + beta * oldv.z;
          newv.w = alpha * threadResults[(subWarpTileRow * TM + tm) * (WNITER * TN) + (subWarpTileCol * TN + tn + 3)] + beta * oldv.w;
          safe_store4(crow, tn, N, newv);
        }
      }
    }
  }
}


// PyTorch wrapper
torch::Tensor matmul4(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "A.cols must match B.rows");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    A = A.contiguous();
    B = B.contiguous();

    const int M = static_cast<int>(A.size(0));
    const int K = static_cast<int>(A.size(1));
    const int N = static_cast<int>(B.size(1));

    auto C = torch::zeros({M, N}, A.options()); // make torch::empty instead?

    const float *A_ptr = A.data_ptr<float>();
    const float *B_ptr = B.data_ptr<float>();
    float *C_ptr = C.data_ptr<float>();

    // tuned values (example for modern NVIDIA GPUs)
    constexpr int NUM_THREADS_PER_BLOCK = 128; // threads per block (divisible by 32)
    constexpr int BK = 16;
    constexpr int TM = 8;
    constexpr int TN = 4;
    // warp tile
    constexpr int WM = 64;
    constexpr int WN = 64;
    constexpr int WNITER = 1; // keep simple example

    // block tile
    constexpr int BM = 128;
    constexpr int BN = 128;

    // Sanity static asserts (compile-time)
    static_assert(NUM_THREADS_PER_BLOCK % THREADS_PER_WARP == 0, "NUM_THREADS_PER_BLOCK must be multiple of warp size");
    static_assert(BM % WM == 0 && BN % WN == 0, "block tile must be multiple of warp tile");
    static_assert(WM % TM == 0 && WN % TN == 0, "warp tile must be multiple of per-thread microtile");

    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 block(NUM_THREADS_PER_BLOCK);

    // shared memory bytes: BM*BK + BK*BN floats
    size_t shared_bytes = (size_t)BM * (size_t)BK * sizeof(float) + (size_t)BK * (size_t)BN * sizeof(float);

    // instantiate kernel (these template args must match launcher constants)
    sgemmWarptiling<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS_PER_BLOCK>
        <<<grid, block, shared_bytes>>>(M, N, K, 1.0f, A_ptr, B_ptr, 0.0f, C_ptr);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "sgemmWarptiling launch failed: %s\n", cudaGetErrorString(err));
    }

    // Optionally synchronize here while debugging:
    // cudaDeviceSynchronize();

    return C;
}
