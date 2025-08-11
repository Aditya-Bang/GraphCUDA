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
__global__ void sgemmWarptiling(int M, int N, int K, float alpha,
                                const float* __restrict__ A,
                                const float* __restrict__ B,
                                float beta,
                                float* __restrict__ C) {
  static_assert((BM % WM) == 0, "BM must be divisible by WM");
  static_assert((BN % WN) == 0, "BN must be divisible by WN");
  static_assert((WM % TM) == 0, "WM must be divisible by TM");
  static_assert((WN % TN) == 0, "WN must be divisible by TN");
  static_assert((NUM_THREADS_PER_BLOCK % THREADS_PER_WARP) == 0, "NUM_THREADS_PER_BLOCK must be multiple of warp size");

  // derived quantities
  const int WARPS_PER_BLOCK = NUM_THREADS_PER_BLOCK / THREADS_PER_WARP;
  const int WARPS_PER_ROW = BN / WN;
  const int WARPS_PER_COL = BM / WM;

  // tile coords for this CTA
  const unsigned int cRow = blockIdx.y;
  const unsigned int cCol = blockIdx.x;

  // warp index within block and its grid position (warpRow, warpCol)
  const unsigned int warpId = (threadIdx.x / THREADS_PER_WARP);
  const unsigned int warpCol = warpId % WARPS_PER_ROW;
  const unsigned int warpRow = warpId / WARPS_PER_ROW;

  // thread index inside warp
  const unsigned int lane = threadIdx.x & (THREADS_PER_WARP - 1);

  // thread's sub-tile coordinates inside a warp tile:
  // we assign each lane to compute a TM x TN microtile (or part of it)
  const unsigned int tilesPerWarpRow = WM / TM; // #microtiles along M per warp
  const unsigned int tilesPerWarpCol = WN / TN; // #microtiles along N per warp
  // thread local coords (we map lane to (threadRowInWarp, threadColInWarp))
  const unsigned int threadColInWarp = lane % tilesPerWarpCol;
  const unsigned int threadRowInWarp = lane / tilesPerWarpCol;

  // Shared memory (dynamic): As [BM * BK], Bs [BK * BN]
  extern __shared__ float shmem[];
  float* As = shmem;                         // size BM * BK
  float* Bs = shmem + (BM * BK);            // size BK * BN

  // compute base global pointers for the block tile
  const int A_block_row = cRow * BM;   // rows of A covered by this block tile
  const int B_block_col = cCol * BN;   // cols of B covered by this block tile
  const int C_block_row = cRow * BM;
  const int C_block_col = cCol * BN;

  // each warp writes out WM x WN starting at:
  const int warpC_row0 = C_block_row + warpRow * WM;
  const int warpC_col0 = C_block_col + warpCol * WN;

  // per-thread accumulator: we let each thread accumulate TM x TN microtile
  float accum[ TM * TN /* 64 */ ]; // use fixed max to avoid VLA; we'll index carefully
  // zero the needed portion
  #pragma unroll
  for (int i = 0; i < TM * TN; ++i) accum[i] = 0.0f;

  // cooperative loading parameters
  const unsigned int threadsInBlock = blockDim.x;
  const unsigned int linearTid = threadIdx.x; // simple since 1D block

  // vectorization: we will load float4 chunks when possible.
  // Number of float elements in shared buffers:
  const int AsElems = BM * BK;
  const int BsElems = BK * BN;

  // outer K-loop (panel)
  for (int bk = 0; bk < K; bk += BK) {
    // Cooperative load As (BM x BK) into shared memory using linear-stride
    // We will write As in transposed layout that favors later reading:
    // store As as As[ (k_col) * BM + row ] so that reading over row is coalesced.
    const int AsFloat4 = CEIL_DIV(BM * BK, 4);
    for (unsigned int idx = linearTid; idx < (unsigned int)AsFloat4; idx += threadsInBlock) {
      unsigned int elementIndex = idx * 4; // flat index in As natural row-major (row major over BM rows, then BK)
      unsigned int a_r = elementIndex / BK;           // row inside BM
      unsigned int a_c = elementIndex % BK;           // col inside BK
      const int global_r = A_block_row + (int)a_r;
      const int global_c = bk + (int)a_c;
      float4 v;
      if (global_r < M) {
        if (global_c + 3 < K && is_aligned_16(&A[global_r * K + global_c])) {
          v = reinterpret_cast<const float4*>(&A[global_r * K + global_c])[0];
        } else if (global_c + 3 < K) {
          // pointer not 16-byte aligned: do scalar loads to avoid fault
          v.x = A[global_r * K + global_c + 0];
          v.y = A[global_r * K + global_c + 1];
          v.z = A[global_r * K + global_c + 2];
          v.w = A[global_r * K + global_c + 3];
        } else {
          // tail: load scalars cautiously
          v.x = (global_c + 0 < K) ? A[global_r * K + global_c + 0] : 0.0f;
          v.y = (global_c + 1 < K) ? A[global_r * K + global_c + 1] : 0.0f;
          v.z = (global_c + 2 < K) ? A[global_r * K + global_c + 2] : 0.0f;
          v.w = (global_c + 3 < K) ? A[global_r * K + global_c + 3] : 0.0f;
        }
      } else {
        v.x = v.y = v.z = v.w = 0.0f;
      }
      // write transposed into As: for each of the four cols
      // As[ (a_c + offset)*BM + a_r ] = v.component
      if (a_r < (unsigned int)BM) {
        unsigned int base_c = a_c;
        if (base_c + 0 < (unsigned int)BK) As[(base_c + 0) * BM + a_r] = v.x;
        if (base_c + 1 < (unsigned int)BK) As[(base_c + 1) * BM + a_r] = v.y;
        if (base_c + 2 < (unsigned int)BK) As[(base_c + 2) * BM + a_r] = v.z;
        if (base_c + 3 < (unsigned int)BK) As[(base_c + 3) * BM + a_r] = v.w;
      }
    }

    // Cooperative load Bs (BK x BN) into shared memory (row-major)
    const int BsFloat4 = CEIL_DIV(BK * BN, 4);
    for (unsigned int idx = linearTid; idx < (unsigned int)BsFloat4; idx += threadsInBlock) {
      unsigned int elementIndex = idx * 4;
      unsigned int b_r = elementIndex / BN;   // 0..BK-1
      unsigned int b_c = elementIndex % BN;   // 0..BN-1
      const int global_r = bk + (int)b_r;
      const int global_c = B_block_col + (int)b_c;
      float4 v;
      if (global_r < K) {
        if (global_c + 3 < N && is_aligned_16(&B[global_r * N + global_c])) {
          v = reinterpret_cast<const float4*>(&B[global_r * N + global_c])[0];
        } else if (global_c + 3 < N) {
          v.x = B[global_r * N + global_c + 0];
          v.y = B[global_r * N + global_c + 1];
          v.z = B[global_r * N + global_c + 2];
          v.w = B[global_r * N + global_c + 3];
        } else {
          v.x = (global_c + 0 < N) ? B[global_r * N + global_c + 0] : 0.0f;
          v.y = (global_c + 1 < N) ? B[global_r * N + global_c + 1] : 0.0f;
          v.z = (global_c + 2 < N) ? B[global_r * N + global_c + 2] : 0.0f;
          v.w = (global_c + 3 < N) ? B[global_r * N + global_c + 3] : 0.0f;
        }
      } else {
        v.x = v.y = v.z = v.w = 0.0f;
      }
      // store into Bs (row-major)
      if (b_r < (unsigned int)BK) {
        if (b_c + 0 < (unsigned int)BN) Bs[b_r * BN + b_c + 0] = v.x;
        if (b_c + 1 < (unsigned int)BN) Bs[b_r * BN + b_c + 1] = v.y;
        if (b_c + 2 < (unsigned int)BN) Bs[b_r * BN + b_c + 2] = v.z;
        if (b_c + 3 < (unsigned int)BN) Bs[b_r * BN + b_c + 3] = v.w;
      }
    }

    __syncthreads();

    // processing: each warp computes WM x WN block using shared As/Bs
    // We iterate over k dimension BK inside shared-memory tile
    for (int d = 0; d < BK; ++d) {
      // For each lane, load TM values from As (As stored transposed as [d*BM + row])
      // and TN values from Bs (Bs stored row-major as [d*BN + col])
      // Then multiply-accumulate into accum[tm*TN + tn]
      for (int tm = 0; tm < TM; ++tm) {
        int rowInWarpTile = threadRowInWarp * TM + tm;       // 0..WM-1 per warp
        int rowInAs = warpRow * WM + rowInWarpTile;          // row inside BM
        float a_val = 0.0f;
        if (rowInAs < BM) {
          // As layout: (d * BM) + rowInAs
          a_val = As[d * BM + rowInAs];
        } else {
          a_val = 0.0f;
        }
        for (int tn = 0; tn < TN; ++tn) {
          int colInWarpTile = threadColInWarp * TN + tn;     // 0..WN-1 per warp
          int colInBs = warpCol * WN + colInWarpTile;        // col inside BN
          float b_val = 0.0f;
          if (colInBs < BN) {
            b_val = Bs[d * BN + colInBs];
          } else {
            b_val = 0.0f;
          }
          accum[tm * TN + tn] += a_val * b_val;
        }
      }
    }

    A += BK;         // move BK columns right
    B += BK * N;     // move BK rows down
    __syncthreads();
  } // end bk loop

  // Write back warp accumulators into C
  // Each lane writes its TM x TN microtile at:
  // base = warpC_row0 + threadRowInWarp*TM, warpC_col0 + threadColInWarp*TN
  for (int tm = 0; tm < TM; ++tm) {
    int globalRow = warpC_row0 + threadRowInWarp * TM + tm;
    if (globalRow >= M) continue;
    // pointer to row start in C
    float* crow = &C[globalRow * N + warpC_col0 + threadColInWarp * TN];
    for (int tn = 0; tn < TN; tn += 4) {
      int globalCol = warpC_col0 + threadColInWarp * TN + tn;
      // try vector write if possible and aligned
      uintptr_t addr = reinterpret_cast<uintptr_t>(&crow[tn]);
      float4 oldv;
      if (globalCol + 3 < N && ((globalCol & 3) == 0) && (addr % 16 == 0)) {
        // safe to read 4 floats
        oldv = reinterpret_cast<float4*>(&crow[tn])[0];
        float4 newv;
        newv.x = alpha * accum[tm * TN + tn + 0] + beta * oldv.x;
        newv.y = alpha * accum[tm * TN + tn + 1] + beta * oldv.y;
        newv.z = alpha * accum[tm * TN + tn + 2] + beta * oldv.z;
        newv.w = alpha * accum[tm * TN + tn + 3] + beta * oldv.w;
        reinterpret_cast<float4*>(&crow[tn])[0] = newv;
      } else {
        // scalar fallback
        for (int i = 0; i < 4; ++i) {
          int c = tn + i;
          int gc = warpC_col0 + threadColInWarp * TN + c;
          if (gc < N) {
            crow[c] = alpha * accum[tm * TN + c] + beta * crow[c];
          }
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
