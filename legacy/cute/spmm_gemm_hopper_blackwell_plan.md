# Optimizing fused SPMM-GEMM for Hopper / Blackwell

This note is based on:
- the current Triton kernel provided by the user
- the Triton fused attention tutorial
- the FlashAttention CuTe SM90 kernel
- the current FlashAttention CuTe SM100/Blackwell kernel
- Triton / CUTLASS / NVIDIA Hopper and Blackwell docs

## 1. What matters most before warp specialization

Your current kernel computes

```
Y = A_sparse @ X @ W
```

with loop order roughly:

```
for k1_block in sparse_cols:
    load A[:, k1_block]
    for k2_block in K2:
        acc1 = A[:, k1_block] @ X[k1_block, k2_block]
        acc2 += acc1 @ W[k2_block, :]
```

This has two important implications:

1. The same `W[k2_block, :]` tile is reloaded for every sparse `k1_block`.
2. The second GEMM (`acc1 @ W`) is executed once per `(k1_block, k2_block)` pair.

On Hopper / Blackwell, that is often not the best association when `N` is small.

---

## 2. The most promising algebraic rewrite for small N

Because this is a **small-N** kernel, the most important rewrite to test is:

```
Y = A_sparse @ (X @ W)
```

but done *locally* only for the selected sparse rows.

For one sparse tile:

```
cols = sparse column indices for this row-block
G = X[cols, :] @ W        # shape [BLOCK_K1, N]
Y += A_tile @ G           # shape [BLOCK_M, N]
```

This replaces:

- `BLOCK_M x BLOCK_K1 @ BLOCK_K1 x BLOCK_K2`
- then `BLOCK_M x BLOCK_K2 @ BLOCK_K2 x N`

with:

- `BLOCK_K1 x K2 @ K2 x N`
- then `BLOCK_M x BLOCK_K1 @ BLOCK_K1 x N`

When `N << BLOCK_M`, this can be dramatically cheaper.

### Rule of thumb

Prefer `A @ (XW_selected)` when:
- `N` is small (your kernel is explicitly small-N)
- `K2` is moderate / large
- `BLOCK_M` is not tiny

Prefer `(A X) @ W` when:
- per-row-block sparse degree is tiny
- `A` reuse dominates and recomputing / reloading `A` would hurt more than repeating the projection

In practice, I would implement **both** and dispatch using a simple heuristic based on:

- `avg_nnz_per_row_block`
- `K2`
- `N`

---

## 3. Why FlashAttention tricks transfer well here

FlashAttention SM90/SM100 is not just “better tiling.”
It adds:

- producer / consumer warp specialization
- async pipelines (TMA / cp.async)
- persistent work scheduling
- per-role register budgeting
- architecture-specific memory paths (SMEM on SM90, Tensor Memory on SM100)

For this kernel, the analogous mapping is:

- **Producer warp(s)**: load sparse metadata + gather rows from `X` + load `W`
- **Consumer warp group(s)**: tensor-core MMA on dense subproblems
- **Scheduler**: persistent row-block scheduler, ideally degree-aware
- **Epilogue warp(s)**: bias + ReLU + mask + store

The big difference from attention is that your bottleneck is **irregular gather** rather than softmax.
That means the *metadata format* matters more than it does in attention.

---

## 4. Hopper path (best realistic Triton path)

If you want to stay in Triton first, do these in order:

### 4.1 Persistent row-block scheduler

Launch

```python
grid = (min(NUM_SMS, num_row_blocks),)
```

and let each CTA iterate over multiple row-blocks:

```python
for tile_id in tl.range(start_pid, num_row_blocks, NUM_SMS, flatten=True, warp_specialize=WARP_SPECIALIZE):
    ...
```

Benefits:
- amortizes launch / prologue overhead
- improves L2 locality on `W`
- makes degree bucketing easier

### 4.2 Degree bucketing

Your current scheduler assigns one row-block per program, but row-block cost is proportional to its sparse degree.
Bucket row-blocks by `crow[i+1] - crow[i]`, e.g.:

- short rows: 1–32
- medium rows: 33–128
- long rows: 129+

and launch separate kernels / autotune spaces.

This is the closest analogue to FlashAttention’s varlen tile scheduler.

### 4.3 Use descriptors for dense tensors first

Even if the sparse path remains manual, use tensor descriptors for:

- `W`
- `Y`
- optionally `X` when loading contiguous column ranges

This reduces address-generation pressure and lets Triton use TMA-backed loads/stores on Hopper+.

### 4.4 Reorder the inner loops to maximize the expensive reuse

Two candidate Triton variants:

#### Variant A: keep `A` outer (close to your current kernel)

Pros:
- reuse `A` tile across all `k2`
- good when sparse degree is tiny

Cons:
- reloads `W` for every sparse `k1` tile
- repeats the second GEMM for every `(k1, k2)` pair

#### Variant B: make `k2` outer

```python
for k2_block in K2:
    mid = 0
    load W[k2_block]
    for k1_block in sparse_cols:
        mid += A_tile @ X_gathered
    acc += mid @ W[k2_block]
```

Pros:
- loads each `W` tile once
- does projection (`mid @ W`) once per `k2_block`

Cons:
- reloads `A` for every `k2_block`

On Hopper, I would benchmark both. The winner depends on row-block degree.

### 4.5 Register pressure tuning

Like the fused-attention Triton kernel, add architecture-gated tuning of:

- `num_warps`
- `num_stages`
- `maxnreg` (especially on Blackwell)
- `warp_specialize`

Typical search dimensions:

- `BLOCK_K1`: 16, 32, 64
- `BLOCK_K2`: 32, 64, 128
- `num_warps`: 4, 8
- `num_stages`: 2, 3, 4
- `warp_specialize`: False / True on SM90+ only

### 4.6 Change sparse storage if you want TMA for A

Your current `values_rm` layout is effectively CSR-like with per-row-block variable stride.
That makes it awkward for TMA because the logical leading dimension changes per row-block.

If you want FlashAttention-style producer pipelines for `A`, switch from packed CSR/BSR values to one of:

- **bucketed Block-ELL** per degree bucket
- **padded block-CSR by bucket**
- **sorted / run-length encoded rows** if many column indices are contiguous

This is probably the single biggest data-layout change needed to unlock Hopper-style async loading on the sparse side.

---

## 5. Blackwell path (fastest possible path)

For Blackwell, the true endgame is not classic Triton but a CuTe DSL kernel using:

- TMA gather/load pipelines
- tcgen05 MMA
- tensor memory for dense MMA stages
- explicit warp-role partitioning

### 5.1 Recommended Blackwell decomposition

For Blackwell I would target this dataflow:

```
for row_block in persistent_schedule:
    read sparse metadata
    for sparse k1_tile in row_block:
        # producer: TMA gather rows of X[cols, :] and load W (tile or whole)
        G = X_selected @ W        # dense GEMM over K2, result shape [BK1, N]
        Y += A_tile @ G           # dense GEMM over BK1
```

Why this is attractive on SM100:
- `N` is small, so `G` is compact
- the dense phase `X_selected @ W` is where tcgen05 helps the most
- `G` is tiny enough to communicate through shared / tensor memory
- the irregular part is isolated to the producer-side gather

### 5.2 Warp roles I would use

A much simpler version of the FA4 SM100 warp partition is enough here:

- **1 load warp**: sparse metadata + TMA gather for `X` + TMA load for `W`
- **1–2 MMA warp groups**: dense MMA for `X_selected @ W`
- **1 warp / warp group for reduction+epilogue**: `A_tile @ G`, bias, ReLU, store
- optional extra warp group only if you split the output tile across `M`

Unlike FA4, you do not need dedicated softmax / correction warps.
Spend that budget on load + MMA.

### 5.3 When to use Tensor Memory

Use tensor memory only for the dense MMA intermediates (`G` or other narrow dense tiles).
Do **not** try to force sparse metadata or CSR-like values through tensor memory.
That part should stay producer-side in regular registers / shared memory.

### 5.4 2-CTA / cluster features

I would only consider 2-CTA tcgen05 / clusterized kernels if:

- `K2` is very large
- `N` is not extremely tiny
- you have enough work per row-block to amortize cluster synchronization

For the very small-N regime, a single-CTA persistent kernel is often the better first target.

---

## 6. Very important: TMA gather is a near-perfect match for X row gathering

For Hopper+ / Blackwell+, Triton’s TMA gather path supports gathering multiple rows from a 2D tensor descriptor with a 1D tensor of row offsets.
That is almost exactly your `X[offs_k1_x, k2:k2+BLOCK_K2]` access pattern.

So the most architecture-specific optimization for your kernel is:

- keep `X` as a 2D descriptor `[K1, K2]`
- use the sparse column indices for the row-offset vector
- gather the needed `BLOCK_K1` rows of `X` asynchronously into shared memory

This is much better than synthesizing many scalar global addresses in every MMA warp.

---

## 7. Why hardware sparse MMAs do NOT solve this directly

Blackwell and Hopper expose sparse MMA instructions in CUTLASS examples, but those kernels use **structured sparse compression + metadata**, not arbitrary BSR/CSR sparsity.

So unless your sparse matrix can be converted to the hardware-expected structured format, you should assume:

- dense tensor cores for the dense subproblems
- explicit metadata-driven gather for the sparse side

That is the correct mental model for this kernel.

---

## 8. Concrete implementation plan

### Step 1 — minimal Triton upgrade

Keep the current sparse format and add:
- persistent grid over row-blocks
- degree bucketing
- descriptor/TMA loads for `W` and stores for `Y`
- autotune `warp_specialize`
- test both `(A X) W` and `A (XW_selected)` associations

### Step 2 — Hopper-focused format upgrade

Change sparse storage to a bucketed fixed-stride format so that:
- `A` values are loadable with predictable tiles
- metadata loads are cheaper
- producer warps can prefetch better

### Step 3 — Blackwell CuTe DSL kernel

Build a dedicated SM100 kernel with:
- TMA gather for `X`
- TMA load for `W`
- tcgen05 dense MMA for `X_selected @ W`
- small dense follow-up `A_tile @ G`
- persistent row-block scheduler with degree-aware ordering

---

## 9. Pseudocode sketches

### 9.1 Hopper Triton sketch (persistent)

```python
for row_block in tl.range(start_pid, num_row_blocks, NUM_SMS,
                          flatten=True, warp_specialize=WARP_SPECIALIZE):
    crow0 = load(crow[row_block])
    crow1 = load(crow[row_block + 1])
    nnz = crow1 - crow0

    acc = zeros([BLOCK_M, BLOCK_N], fp32)

    # candidate: selected XW path
    for k1_block in range(0, nnz, BLOCK_K1):
        cols = load(col[crow0 + k1_block : ...])
        a = load(A_values[row_block, :, k1_block : ...])

        g = zeros([BLOCK_K1, BLOCK_N], fp32)
        for k2_block in tl.range(0, K2, BLOCK_K2, warp_specialize=WARP_SPECIALIZE):
            x = gather_rows_of_X(cols, k2_block)
            w = load_W_tile(k2_block)
            g += dot(x, w)

        acc += dot(a, g)

    bias_relu_store(acc)
```

### 9.2 Blackwell CuTe DSL sketch

```python
producer warp:
    load sparse metadata
    tma_gather X[cols, k2_tile]
    tma_load W[k2_tile, n_tile]

consumer mma warpgroup:
    G_partial += X_selected @ W_tile   # tcgen05

epilogue / follow-up warpgroup:
    Y_partial += A_tile @ G            # dense small GEMM
    apply bias / relu
    store
```

---

## 10. What I would benchmark first

1. **Unfused baseline**:
   - `G = X @ W`
   - `Y = A @ G`

2. **Current fused kernel**

3. **Persistent + degree-bucketed Triton kernel**

4. **Selected-XW fused Triton kernel** (`A @ (XW_selected)`)

5. **If Blackwell matters most**: CuTe DSL prototype with TMA gather for `X`

The unfused baseline matters because, for small `N`, materializing `X @ W` may be much cheaper than repeating dense work inside the fused sparse kernel.

