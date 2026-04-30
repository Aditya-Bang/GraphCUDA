# FA4-style CuTe DSL port for fused SpMM-GEMM

This note explains the design choices in `spmm_gemm_cute_fa4_style.py`.

## 1. Why the sparse format changed

Your Triton kernel reads a row-block from `crow/col/values_rm` directly:

- `crow` gives the number of scalar-column blocks in a row block
- `col` gives the gathered rows of `X`
- `values_rm` stores a dense `(BLOCK_M, nnz_row_block)` tile in row-major order

That format is perfectly fine in Triton, but it is awkward for a FlashAttention-4 style CuTe kernel because the producer / consumer pipeline wants fixed-shape tiles.

The CuTe rewrite therefore introduces a **pack format**:

- `pack_count[row_tile]`
- `pack_cols[row_tile, pack_id, :]` with fixed width `BLOCK_K1`
- `pack_vals[row_tile, pack_id, :, :]` with shape `(BLOCK_M, BLOCK_K1)`

Tail packs are zero-padded. This keeps the main loop uniform and avoids the masked/full-list complexity from FA block-sparse attention.

## 2. Algebraic change

The Triton kernel computes

`Y += (A_pack @ X_pack_k2) @ W_k2`

inside the nested `(K1_pack, K2_tile)` loop.

The CuTe port instead computes

`G_pack = X_gathered @ W`

followed by

`Y += A_pack @ G_pack`

for each sparse pack.

This is better aligned with the FA pipeline because:

- the irregular gather only touches `X`
- `W` becomes the single dense operand that is worth feeding through TMA
- the intermediate is only `(BLOCK_K1, N_PAD)`, which is small in the small-`N` regime

## 3. Hopper mapping

The Hopper path uses a simple FA3/FA4-style split:

- **producer warps**
  - load sparse values `A_pack`
  - gather `X` rows for each `K2` tile
  - issue TMA for `W` tiles
- **consumer warpgroup**
  - run `XW` GEMM over the `K2` pipeline
  - materialize the small intermediate `G`
  - run `A @ G`
  - apply bias / ReLU / store

That is the key optimization transplant from flash attention: persistent execution + warp specialization + async pipelining.

## 4. Blackwell mapping

The Blackwell class is intentionally only a structural port. The pieces you want to swap in from `flash_fwd_sm100.py` are:

- `sm100_utils_basic.make_trivial_tiled_mma(...)`
- `tcgen05` operand modes / CTA groups
- `PipelineTmaUmma` for `W`
- `PipelineAsyncUmma` (or equivalent) for the gathered `X` path
- the same persistent CTA scheduling style FA4 uses on SM100

Unlike FA4, this kernel does **not** need softmax warps, correction warps, or tensor-memory score staging. The work split can therefore stay much simpler.

## 5. What I would benchmark first

1. `BLOCK_K1 in {64, 128}`
2. `BLOCK_K2 in {32, 64, 128}`
3. `N_PAD = 64` for `N <= 64`
4. persistent launch with `grid_x = min(num_row_tiles, sm_count)`
5. caching the packed sparse metadata across repeated launches

## 6. Most likely API-sensitive spots

These are the lines I expect you may need to adjust against your local `flash-attn` + `cutlass-dsl` revision:

- the exact `make_trivial_tiled_mma` operand-major-mode pairing for the 2nd GEMM
- the `copy_utils.tma_get_copy_fn(...)` signature for `W`
- the accumulator -> shared-memory store path in `_store_acc_g_to_smem`
- the preferred store path for the output tile (`TMA S2G` vs direct cooperative store)

Everything else is the main algorithmic rewrite.

## 7. How to integrate in the flash-attn repo

I would place the new file under `flash_attn/cute/spmm_gemm_fwd.py`, then:

- keep `pack_bsr_scalar_cols_for_cute(...)` on the Python side
- create a thin interface wrapper next to the other CuTe entrypoints
- add a reference test against your current Triton kernel for a range of
  `(M, K1, K2, N, avg_nnz_per_row_tile)`
- benchmark Hopper first, then swap in the SM100 helper / pipeline pieces

