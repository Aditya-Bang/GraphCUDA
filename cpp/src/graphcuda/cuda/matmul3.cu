#include "matmul.cuh"


template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemmVectorize(int M, int N, int K, float alpha,
                               const float* __restrict__ A,
                               const float* __restrict__ B, float beta,
                               float* __restrict__ C) {
  static_assert(BM % TM == 0, "BM must be divisible by TM");
  static_assert(BN % TN == 0, "BN must be divisible by TN");
  static_assert((BK % 4) == 0, "BK must be multiple of 4 for float4 loads");
  static_assert((TN % 4) == 0, "TN must be multiple of 4 for float4 stores");

  // block tile coordinates
  const unsigned int cRow = blockIdx.y;
  const unsigned int cCol = blockIdx.x;

  // thread tile coordinates inside the block
  const unsigned int threadTileCol = threadIdx.x; // 0 .. BN/TN - 1
  const unsigned int threadTileRow = threadIdx.y; // 0 .. BM/TM - 1

  const unsigned int threadsPerBlock = blockDim.x * blockDim.y;
  const unsigned int linearTid = threadTileRow * blockDim.x + threadTileCol;

  // local offsets in the BMxBN block this thread will compute
  const unsigned int rowBaseInBlock = threadTileRow * TM;
  const unsigned int colBaseInBlock = threadTileCol * TN;

  // Shared memory layout (dynamic): As [BM * BK], Bs [BK * BN]
  extern __shared__ float shmem[];
  float* As = shmem;                     // size BM * BK
  float* Bs = shmem + (BM * BK);        // size BK * BN

  // global pointers to block tile starts
  const int A_block_row = cRow * BM; // rows of A covered by this block
  const int B_block_col = cCol * BN; // cols of B covered by this block
  const int C_block_row = cRow * BM;
  const int C_block_col = cCol * BN;

  // per-thread accumulators
  float accum[TM * TN];
  #pragma unroll
  for (int i = 0; i < TM * TN; ++i) accum[i] = 0.0f;

  // iterate over K in BK chunks
  for (int bk = 0; bk < K; bk += BK) {
    // --- Load As (BM x BK) into shared memory, transposing as we write
    // We'll load 4 floats at a time from global A (as float4) when possible.
    // We treat As in shared mem as transposed: As[ a_c * BM + a_r ] so that
    // later access As[dotIdx * BM + localRow] is coalesced.
    const int AsSize = BM * BK / 4; // number of float4 elements (since BK multiple of 4)
    for (unsigned int idx = linearTid; idx < (unsigned int)AsSize; idx += threadsPerBlock) {
      // compute which float4 this is: idx enumerates across (BM * BK / 4)
      // map idx -> (a_r, a_c4) where a_c4 indexes groups of 4 in BK
      unsigned int a_r = (idx * 4) % BM;              // row inside BM
      unsigned int tmp = (idx * 4) / BM;              // how many full rows we've passed
      unsigned int a_c4 = tmp;                        // index of 4-wide group in BK
      // Equivalent mapping simpler: iterate linear across a_r major then a_c groups:
      // But to keep simple and predictable, compute more robustly:
      // Let's compute by row-major linear index L = idx*4
      unsigned int L = idx * 4;
      unsigned int a_row = L / BK; // 0..BM-1
      unsigned int a_col = (L % BK) / 1; // 0..BK-1 (we will load 4 from a_col .. a_col+3)
      // To avoid complex mapping mistakes, we'll compute a_r and a_c like this:
      // totalElements = BM * BK; elementIndex = idx * 4;
      unsigned int elementIndex = idx * 4;
      unsigned int a_r2 = elementIndex / BK;        // row 0..BM-1
      unsigned int a_c_start = elementIndex % BK;   // col 0..BK-4 step
      // Now do safe float4 load from A global: row = A_block_row + a_r2, col = bk + a_c_start
      const int global_r = A_block_row + (int)a_r2;
      const int global_c = bk + (int)a_c_start;
      // load float4 from A if in-bounds, else pack scalars
      float4 avals;
      if (global_r < M) {
        if (global_c + 3 < K) {
          // safe to do vector load
          avals = reinterpret_cast<const float4*>(&A[global_r * K + global_c])[0];
        } else {
          // partial tail: load scalar-wise
          avals.x = (global_c + 0 < K) ? A[global_r * K + global_c + 0] : 0.0f;
          avals.y = (global_c + 1 < K) ? A[global_r * K + global_c + 1] : 0.0f;
          avals.z = (global_c + 2 < K) ? A[global_r * K + global_c + 2] : 0.0f;
          avals.w = (global_c + 3 < K) ? A[global_r * K + global_c + 3] : 0.0f;
        }
      } else {
        avals.x = avals.y = avals.z = avals.w = 0.0f;
      }
      // write them transposed into As: As[ (a_c)*BM + a_r ] layout, for each of 4 cols
      // for cols c = a_c_start .. a_c_start+3
      // ensure a_r2 < BM
      if (a_r2 < (unsigned int)BM) {
        unsigned int base_c = a_c_start;
        // each write: As[ (col) * BM + row ] = val
        As[(base_c + 0) * BM + a_r2] = avals.x;
        As[(base_c + 1) * BM + a_r2] = avals.y;
        As[(base_c + 2) * BM + a_r2] = avals.z;
        As[(base_c + 3) * BM + a_r2] = avals.w;
      }
    }

    // --- Load Bs (BK x BN) into shared memory
    // We'll also try to load float4 chunks across columns (BN dimension) when possible.
    const int BsSize = (BK * BN) / 4; // number of float4 elements
    for (unsigned int idx = linearTid; idx < (unsigned int)BsSize; idx += threadsPerBlock) {
      unsigned int elementIndex = idx * 4;
      unsigned int b_r = elementIndex / BN;        // 0..BK-1
      unsigned int b_c_start = elementIndex % BN;  // 0..BN-4
      const int global_r = bk + (int)b_r;
      const int global_c = B_block_col + (int)b_c_start;
      float4 bvals;
      if (global_r < K) {
        if (global_c + 3 < N) {
          bvals = reinterpret_cast<const float4*>(&B[global_r * N + global_c])[0];
        } else {
          bvals.x = (global_c + 0 < N) ? B[global_r * N + global_c + 0] : 0.0f;
          bvals.y = (global_c + 1 < N) ? B[global_r * N + global_c + 1] : 0.0f;
          bvals.z = (global_c + 2 < N) ? B[global_r * N + global_c + 2] : 0.0f;
          bvals.w = (global_c + 3 < N) ? B[global_r * N + global_c + 3] : 0.0f;
        }
      } else {
        bvals.x = bvals.y = bvals.z = bvals.w = 0.0f;
      }
      // store into Bs row-major: Bs[ b_r * BN + b_c_start + offset ] = value
      Bs[b_r * BN + b_c_start + 0] = bvals.x;
      Bs[b_r * BN + b_c_start + 1] = bvals.y;
      Bs[b_r * BN + b_c_start + 2] = bvals.z;
      Bs[b_r * BN + b_c_start + 3] = bvals.w;
    }

    __syncthreads();

    // advance A/B pointers for next BK panel (we used indices so no pointer arithmetic needed)
    // compute partial results: loop over dot dimension BK
    for (int d = 0; d < BK; ++d) {
      // load TM elements from As: note As is transposed so we index As[d * BM + rowInBlock]
      #pragma unroll
      for (int tm = 0; tm < TM; ++tm) {
        // local row inside BM
        float aval = As[d * BM + (rowBaseInBlock + tm)];
        // load TN elements from Bs: Bs[d * BN + colBaseInBlock + tn]
        #pragma unroll
        for (int tn = 0; tn < TN; ++tn) {
          float bval = Bs[d * BN + (colBaseInBlock + tn)];
          accum[tm * TN + tn] += aval * bval;
        }
      }
    }

    __syncthreads();
  } // end for bk

  // Write back accumulators into C.
  // We vectorize stores in groups of 4 when global column index is 4-aligned and enough columns remain.
  for (int tm = 0; tm < TM; ++tm) {
    const int rowInBlock = rowBaseInBlock + tm;
    const int globalRow = C_block_row + rowInBlock;
    if (globalRow >= M) continue;
    // base pointer to this row in C
    float* crow = &C[globalRow * N + C_block_col + colBaseInBlock];
    for (int tn = 0; tn < TN; tn += 4) {
      const int globalCol = C_block_col + colBaseInBlock + tn;
      if (globalCol + 3 < N && ((globalCol & 3) == 0)) {
        // safe to do vectorized read-modify-write
        float4 old = reinterpret_cast<float4*>(&crow[tn])[0];
        float4 neu;
        neu.x = alpha * accum[tm * TN + tn + 0] + beta * old.x;
        neu.y = alpha * accum[tm * TN + tn + 1] + beta * old.y;
        neu.z = alpha * accum[tm * TN + tn + 2] + beta * old.z;
        neu.w = alpha * accum[tm * TN + tn + 3] + beta * old.w;
        reinterpret_cast<float4*>(&crow[tn])[0] = neu;
      } else {
        // fallback scalar writes (handle edges / misalignment)
        for (int i = 0; i < 4; ++i) {
          int col = tn + i;
          int globalColI = C_block_col + colBaseInBlock + col;
          if (globalColI < N) {
            crow[col] = alpha * accum[tm * TN + col] + beta * crow[col];
          }
        }
      }
    }
  }
}

// PyTorch wrapper
torch::Tensor matmul3(torch::Tensor A, torch::Tensor B) {
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

    constexpr int BK = 8;   // must be multiple of 4
    constexpr int TM = 8;
    constexpr int TN = 8;   // must be multiple of 4

    int BM = (M >= 128) ? 128 : 64;
    int BN = (N >= 128) ? 128 : 64;

    // Ensure divisibility
    if (BM % TM != 0) BM = ((BM + TM - 1) / TM) * TM;
    if (BN % TN != 0) BN = ((BN + TN - 1) / TN) * TN;

    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 block(BN / TN, BM / TM); // block.x = BN/TN, block.y = BM/TM

    // shared memory bytes: BM*BK + BK*BN floats
    size_t shared_bytes = (size_t)BM * (size_t)BK * sizeof(float) + (size_t)BK * (size_t)BN * sizeof(float);

    if (BM == 128 && BN == 128) {
        sgemmVectorize<128, 128, BK, TM, TN><<<grid, block, shared_bytes>>>(M, N, K, 1.0f, A_ptr, B_ptr, 0.0f, C_ptr);
    } else {
        sgemmVectorize<64, 64, BK, TM, TN><<<grid, block, shared_bytes>>>(M, N, K, 1.0f, A_ptr, B_ptr, 0.0f, C_ptr);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "sgemmVectorize launch failed: %s\n", cudaGetErrorString(err));
    }

    // Optionally synchronize here while debugging:
    // cudaDeviceSynchronize();

    return C;
}
