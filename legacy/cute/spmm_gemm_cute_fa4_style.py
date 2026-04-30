"""
FA4-style CuTe DSL rewrite of the fused small-N SpMM-GEMM-ReLU kernel
from the Triton implementation in `Pasted text (3).txt`.

This file is intentionally written in the same *style* as FlashAttention-4:
- persistent CTA scheduling over row tiles
- producer / consumer warp specialization
- separate async pipelines for sparse-value tiles, gathered X tiles, and dense W tiles
- Hopper path built around SM90 WGMMA + TMA for W
- Blackwell path sketched with the same decomposition but tcgen05 / UMMA helpers

Important:
1. The sparse side is repacked before launch. The original BSR `values_rm + crow + col`
   format works for Triton, but for a FA4-like CuTe kernel it is much easier and faster
   to pre-pack each row-block into fixed-width BK1 packs.
2. The code below is designed to live next to the official `flash_attn/cute`
   implementation and re-use the same helper stack (`quack`, `flash_attn.cute.*`,
   CUTLASS CuTe DSL). A few helper calls are the most version-sensitive pieces and are
   isolated behind small methods / comments.
3. The algebra used here is:

       Y[m_tile] += A_pack @ (X[gathered_rows] @ W)

   rather than the Triton loop nest

       Y += (A_pack @ X_tile) @ W_tile

   because for your small-N regime this keeps the intermediate small (BK1 x N_PAD)
   and fits the FA4 producer/consumer pattern more naturally.

This is a forward kernel only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple, Optional

import cuda.bindings.driver as cuda
import torch

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.base_dsl.arch import Arch
from cutlass.cute.nvgpu import cpasync, warpgroup
import cutlass.cute.nvgpu.tcgen05 as tcgen05
from cutlass import pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils import LayoutEnum
import cutlass.utils.hopper_helpers as sm90_utils_basic
import cutlass.utils.blackwell_helpers as sm100_utils_basic

from quack import copy_utils, layout_utils, sm90_utils

from flash_attn.cute import pipeline as pipeline_custom
from flash_attn.cute.cute_dsl_utils import assume_tensor_aligned
from flash_attn.cute import utils


# --------------------------------------------------------------------------------------
# Packing helpers
# --------------------------------------------------------------------------------------


@dataclass
class PackedScalarBsrPacks:
    """FA4-style sparse metadata for the fused kernel.

    Each BSR row-block (size BLOCK_M x 1 in the original adjacency) is repacked into a
    fixed number of BK1-wide sparse packs.

    For row-block r and pack p:
      pack_cols[r, p, :]  -> K1 row indices gathered from X
      pack_vals[r, p, :, :] -> sparse values tile of shape (BLOCK_M, BK1)

    Invalid tail positions are zero-padded in both cols and vals, so the kernel can run
    a uniform loop without a separate masked path.
    """

    block_m: int
    block_k1: int
    m: int
    k1: int
    num_row_tiles: int
    max_packs_per_row: int
    pack_count: torch.Tensor       # [num_row_tiles] int32
    pack_cols: torch.Tensor        # [num_row_tiles, max_packs_per_row, BK1] int32
    pack_vals: torch.Tensor        # [num_row_tiles, max_packs_per_row, BLOCK_M, BK1] dtype



def _round_up(x: int, mult: int) -> int:
    return ((x + mult - 1) // mult) * mult



def pack_bsr_scalar_cols_for_cute(adjm: torch.Tensor, block_k1: int) -> PackedScalarBsrPacks:
    """Repack the original BSR(block_m x 1) adjacency into fixed BK1 packs.

    This is the sparse analogue of FA4's block list preprocessing. It converts the
    variable-length `crow/col/values_rm` representation into a tile-friendly layout.
    """

    assert adjm.layout == torch.sparse_bsr
    assert hasattr(adjm, "pre_padded_shape")
    assert hasattr(adjm, "values_rm")

    vals = adjm.values()
    block_m = int(vals.shape[-2])
    block_k = int(vals.shape[-1])
    assert block_k == 1, "This port assumes the same scalar-column BSR layout as your Triton kernel"

    m, k1 = map(int, adjm.pre_padded_shape[:2])
    crow = adjm.crow_indices().detach().cpu().to(torch.int64)
    cols = adjm.col_indices().detach().cpu().to(torch.int64)
    values_rm = adjm.values_rm.detach().cpu()

    num_row_tiles = (m + block_m - 1) // block_m
    pack_counts_host = []
    max_packs_per_row = 0

    for rb in range(num_row_tiles):
        nnz_rb = int(crow[rb + 1] - crow[rb])
        num_packs = (nnz_rb + block_k1 - 1) // block_k1
        pack_counts_host.append(num_packs)
        max_packs_per_row = max(max_packs_per_row, num_packs)

    pack_count = torch.tensor(pack_counts_host, dtype=torch.int32, device=adjm.device)
    pack_cols = torch.zeros(
        (num_row_tiles, max_packs_per_row, block_k1),
        dtype=torch.int32,
        device=adjm.device,
    )
    pack_vals = torch.zeros(
        (num_row_tiles, max_packs_per_row, block_m, block_k1),
        dtype=adjm.dtype,
        device=adjm.device,
    )

    # Host-side fill; expected to be amortized across many launches for a fixed sparsity pattern.
    for rb in range(num_row_tiles):
        start = int(crow[rb])
        end = int(crow[rb + 1])
        nnz_rb = end - start
        if nnz_rb == 0:
            continue

        dense_vals_rb = values_rm[block_m * start : block_m * end].view(block_m, nnz_rb)
        cols_rb = cols[start:end]
        num_packs = pack_counts_host[rb]

        for p in range(num_packs):
            lo = p * block_k1
            hi = min(lo + block_k1, nnz_rb)
            width = hi - lo

            pack_cols[rb, p, :width] = cols_rb[lo:hi].to(torch.int32).to(adjm.device)
            pack_vals[rb, p, :, :width] = dense_vals_rb[:, lo:hi].to(adjm.device)
            # tail stays zero-padded

    return PackedScalarBsrPacks(
        block_m=block_m,
        block_k1=block_k1,
        m=m,
        k1=k1,
        num_row_tiles=num_row_tiles,
        max_packs_per_row=max_packs_per_row,
        pack_count=pack_count,
        pack_cols=pack_cols,
        pack_vals=pack_vals,
    )


# --------------------------------------------------------------------------------------
# Shared runtime helpers
# --------------------------------------------------------------------------------------


class EpilogueArgs(NamedTuple):
    bias: Optional[cute.Tensor]
    relu_mask: Optional[cute.Tensor]
    has_bias: bool
    apply_relu: bool
    n_actual: int
    n_padded: int



def _get_arch() -> Arch:
    from cutlass.cutlass_dsl import BaseDSL

    return BaseDSL._get_dsl().get_arch_enum()


# --------------------------------------------------------------------------------------
# Hopper (SM90) implementation
# --------------------------------------------------------------------------------------


class FusedSpmmGemmForwardSm90:
    """Hopper forward kernel in the style of flash_attn/cute/flash_fwd_sm90.py.

    Warp layout:
      warps 0..3  : producer warps
          - warp 0 issues TMA for W tiles
          - all producer warps cooperatively load A tiles and gather X rows
      warps 4..7  : consumer warpgroup running WGMMA
    """

    def __init__(
        self,
        *,
        block_m: int,
        block_k1: int,
        block_k2: int,
        n_actual: int,
        num_stages: int = 2,
        persistent: bool = True,
    ):
        assert block_m in (64, 128), f"Expected BLOCK_M in {{64, 128}} for the fast path, got {block_m}"
        assert block_k1 in (64, 128), f"Expected BLOCK_K1 in {{64, 128}} for the fast path, got {block_k1}"
        assert block_k2 in (32, 64, 128), f"Unexpected BLOCK_K2={block_k2}"
        self.block_m = block_m
        self.block_k1 = block_k1
        self.block_k2 = block_k2
        self.n_actual = n_actual
        # Pad N to a tensor-core-friendly tile. Using >=64 keeps the 2nd GEMM natural on SM90.
        self.n_padded = max(64, _round_up(n_actual, 64))
        self.num_stages = num_stages
        self.persistent = persistent

        self.arch = _get_arch()
        assert self.arch >= Arch.sm_90 and self.arch <= Arch.sm_90a, "Use SM90 path only on Hopper"

        self.num_load_warps = 4
        self.num_consumer_warps = 4
        self.num_threads = cute.arch.WARP_SIZE * (self.num_load_warps + self.num_consumer_warps)
        self.num_producer_threads = self.num_load_warps * cute.arch.WARP_SIZE
        self.num_mma_threads = self.num_consumer_warps * cute.arch.WARP_SIZE
        self.buffer_align_bytes = 1024

        # Conservative register split. This mirrors FA's setmaxregister idea but for a simpler kernel.
        self.num_mma_regs = 216
        self.num_producer_regs = 40

    def _get_tiled_mmas(self, dtype):
        # XW: (BK1 x BK2) @ (BK2 x N_PAD) -> (BK1 x N_PAD)
        tiled_mma_xw = sm90_utils_basic.make_trivial_tiled_mma(
            dtype,
            dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.MN,
            Float32,
            atom_layout_mnk=(self.block_k1 // 64, 1, 1),
            tiler_mn=(64, self.n_padded),
        )

        # AG: (BM x BK1) @ (BK1 x N_PAD) -> (BM x N_PAD)
        # API-sensitive note:
        #   depending on the exact cutlass-dsl helper revision you have, you may need to flip
        #   the second operand major mode and use a transpose view of sG.
        tiled_mma_ag = sm90_utils_basic.make_trivial_tiled_mma(
            dtype,
            dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.MN,
            Float32,
            atom_layout_mnk=(self.block_m // 64, 1, 1),
            tiler_mn=(64, self.n_padded),
        )
        return tiled_mma_xw, tiled_mma_ag

    def _setup_layouts(self, mPackVals: cute.Tensor, mX: cute.Tensor, mW: cute.Tensor, mY: cute.Tensor):
        self.sA_layout = sm90_utils.make_smem_layout(
            mPackVals.element_type, LayoutEnum.ROW_MAJOR, (self.block_m, self.block_k1), None
        )
        self.sX_layout = sm90_utils.make_smem_layout(
            mX.element_type, LayoutEnum.ROW_MAJOR, (self.block_k1, self.block_k2), self.num_stages
        )
        self.sW_layout = sm90_utils.make_smem_layout(
            mW.element_type, LayoutEnum.ROW_MAJOR, (self.block_k2, self.n_padded), self.num_stages
        )
        self.sG_layout = sm90_utils.make_smem_layout(
            mW.element_type, LayoutEnum.ROW_MAJOR, (self.block_k1, self.n_padded), None
        )
        self.sO_layout = sm90_utils.make_smem_layout(
            mY.element_type, LayoutEnum.ROW_MAJOR, (self.block_m, self.n_padded), None
        )

    def _get_shared_storage_cls(self, dtype):
        sA_struct = cute.struct.Align[
            cute.struct.MemRange[dtype, cute.cosize(self.sA_layout)], self.buffer_align_bytes
        ]
        sX_struct = cute.struct.Align[
            cute.struct.MemRange[dtype, cute.cosize(self.sX_layout)], self.buffer_align_bytes
        ]
        sW_struct = cute.struct.Align[
            cute.struct.MemRange[dtype, cute.cosize(self.sW_layout)], self.buffer_align_bytes
        ]
        sG_struct = cute.struct.Align[
            cute.struct.MemRange[dtype, cute.cosize(self.sG_layout)], self.buffer_align_bytes
        ]
        sO_struct = cute.struct.Align[
            cute.struct.MemRange[dtype, cute.cosize(self.sO_layout)], self.buffer_align_bytes
        ]

        mbar_ptr_A_struct = cute.struct.MemRange[cutlass.Int64, 1 * 2]
        mbar_ptr_X_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
        mbar_ptr_W_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]

        @cute.struct
        class SharedStorage:
            mbar_ptr_A: mbar_ptr_A_struct
            mbar_ptr_X: mbar_ptr_X_struct
            mbar_ptr_W: mbar_ptr_W_struct
            sW: sW_struct
            sA: sA_struct
            sX: sX_struct
            sG: sG_struct
            sO: sO_struct

        return SharedStorage

    @cute.jit
    def __call__(
        self,
        mPackCount: cute.Tensor,   # [num_row_tiles]
        mPackCols: cute.Tensor,    # [num_row_tiles, max_packs, BK1]
        mPackVals: cute.Tensor,    # [num_row_tiles, max_packs, BM, BK1]
        mX: cute.Tensor,           # [K1, K2]
        mW: cute.Tensor,           # [K2, N_PAD]
        mY: cute.Tensor,           # [M, N_PAD]
        mBias: Optional[cute.Tensor] = None,       # [N_PAD]
        mReluMask: Optional[cute.Tensor] = None,   # [M, N_PAD]
        stream: cuda.CUstream = None,
    ):
        mPackVals, mX, mW, mY = [assume_tensor_aligned(t) for t in (mPackVals, mX, mW, mY)]
        if const_expr(mBias is not None):
            mBias = assume_tensor_aligned(mBias)
        if const_expr(mReluMask is not None):
            mReluMask = assume_tensor_aligned(mReluMask)

        self._setup_layouts(mPackVals, mX, mW, mY)
        tiled_mma_xw, tiled_mma_ag = self._get_tiled_mmas(mX.element_type)
        SharedStorage = self._get_shared_storage_cls(mX.element_type)

        # TMA for W and O. W is the dense operand worth giving to TMA.
        gmem_tiled_copy_W = cpasync.CopyBulkTensorTileG2SOp()
        tma_atom_W, tma_tensor_W = cpasync.make_tiled_tma_atom(
            gmem_tiled_copy_W,
            mW,
            cute.select(self.sW_layout, mode=[0, 1]),
            (self.block_k2, self.n_padded),
            1,
        )

        gmem_tiled_copy_O = cpasync.CopyBulkTensorTileS2GOp()
        tma_atom_O, tma_tensor_O = cpasync.make_tiled_tma_atom(
            gmem_tiled_copy_O,
            mY,
            cute.select(self.sO_layout, mode=[0, 1]),
            (self.block_m, self.n_padded),
            1,
        )

        hardware_info = cutlass.utils.HardwareInfo()
        sm_count = hardware_info.get_device_multiprocessor_count()
        num_row_tiles = cute.size(mPackCount.shape[0])
        grid_x = cutlass.min(sm_count, num_row_tiles) if self.persistent else num_row_tiles

        self.kernel(
            mPackCount,
            mPackCols,
            mPackVals,
            mX,
            tma_tensor_W,
            tma_tensor_O,
            mBias,
            mReluMask,
            tma_atom_W,
            tma_atom_O,
            tiled_mma_xw,
            tiled_mma_ag,
            SharedStorage,
        ).launch(
            grid=(grid_x, 1, 1),
            block=[self.num_threads, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        mPackCount: cute.Tensor,
        mPackCols: cute.Tensor,
        mPackVals: cute.Tensor,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mY: cute.Tensor,
        mBias: Optional[cute.Tensor],
        mReluMask: Optional[cute.Tensor],
        tma_atom_W: cute.CopyAtom,
        tma_atom_O: cute.CopyAtom,
        tiled_mma_xw: cute.TiledMma,
        tiled_mma_ag: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_W)
            cpasync.prefetch_descriptor(tma_atom_O)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        ThreadCooperativeGroup = partial(pipeline.CooperativeGroup, pipeline.Agent.Thread)
        tma_warp = ThreadCooperativeGroup(1)
        load_threads = ThreadCooperativeGroup(self.num_producer_threads)
        mma_warps = ThreadCooperativeGroup(self.num_mma_threads // cute.arch.WARP_SIZE)

        pipeline_a = pipeline_custom.PipelineCpAsync.create(
            barrier_storage=storage.mbar_ptr_A.data_ptr(),
            num_stages=1,
            producer_group=load_threads,
            consumer_group=mma_warps,
            defer_sync=True,
            elect_one_release=True,
            syncwarp_before_release=False,
        )
        pipeline_x = pipeline_custom.PipelineCpAsync.create(
            barrier_storage=storage.mbar_ptr_X.data_ptr(),
            num_stages=self.num_stages,
            producer_group=load_threads,
            consumer_group=mma_warps,
            defer_sync=True,
            elect_one_release=True,
            syncwarp_before_release=False,
        )
        pipeline_w = pipeline_custom.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_ptr_W.data_ptr(),
            num_stages=self.num_stages,
            producer_group=tma_warp,
            consumer_group=mma_warps,
            tx_count=cute.size_in_bytes(mW.element_type, (self.block_k2, self.n_padded)),
            defer_sync=True,
        )

        pipeline_init_arrive(cluster_shape_mn=(1, 1), is_relaxed=True)

        sA = storage.sA.get_tensor(self.sA_layout.outer, swizzle=self.sA_layout.inner)
        sX = storage.sX.get_tensor(self.sX_layout.outer, swizzle=self.sX_layout.inner)
        sW = storage.sW.get_tensor(self.sW_layout.outer, swizzle=self.sW_layout.inner)
        sG = storage.sG.get_tensor(self.sG_layout.outer, swizzle=self.sG_layout.inner)
        sO = storage.sO.get_tensor(self.sO_layout.outer, swizzle=self.sO_layout.inner)

        pipeline_init_wait(cluster_shape_mn=(1, 1))

        if warp_idx < self.num_load_warps:
            cute.arch.setmaxregister_decrease(self.num_producer_regs)
            self.load(
                mPackCount,
                mPackCols,
                mPackVals,
                mX,
                mW,
                sA,
                sX,
                sW,
                tma_atom_W,
                pipeline_a,
                pipeline_x,
                pipeline_w,
            )
        else:
            cute.arch.setmaxregister_increase(self.num_mma_regs)
            self.mma(
                mPackCount,
                mPackVals,
                mX,
                mY,
                mBias,
                mReluMask,
                sA,
                sX,
                sW,
                sG,
                sO,
                tma_atom_O,
                tiled_mma_xw,
                tiled_mma_ag,
                pipeline_a,
                pipeline_x,
                pipeline_w,
                tidx - self.num_producer_threads,
            )

    @cute.jit
    def _load_a_tile(self, mPackVals, sA, row_tile, pack_id, tidx):
        linear = tidx
        num_threads = self.num_producer_threads
        total = self.block_m * self.block_k1
        while linear < total:
            r = linear // self.block_k1
            c = linear - r * self.block_k1
            sA[r, c] = mPackVals[row_tile, pack_id, r, c]
            linear += num_threads

    @cute.jit
    def _load_x_tile(self, mPackCols, mX, sX, row_tile, pack_id, k2_block, stage, tidx):
        linear = tidx
        num_threads = self.num_producer_threads
        k2_start = k2_block * self.block_k2
        total = self.block_k1 * self.block_k2
        k2_dim = cute.size(mX.shape[1])
        while linear < total:
            r = linear // self.block_k2
            c = linear - r * self.block_k2
            x_row = Int32(mPackCols[row_tile, pack_id, r])
            k2 = k2_start + c
            sX[stage, r, c] = mX[x_row, k2] if k2 < k2_dim else mX.element_type(0)
            linear += num_threads
        cute.arch.cp_async_commit_group()

    @cute.jit
    def _load_w_tile(self, mW, sW, tma_atom_W, pipeline_w, producer_state, k2_block):
        # This follows the FA4 TMA pattern: define a tiled TMA closure over the K2 dimension.
        gW = cute.local_tile(mW, (self.block_k2, self.n_padded), (None, 0))
        tma_load_W_fn, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_W, 0, cute.make_layout(1), gW, sW
        )
        tma_load_W_fn = copy_utils.tma_producer_copy_fn(tma_load_W_fn, pipeline_w)
        tma_load_W_fn(src_idx=k2_block, producer_state=producer_state)

    @cute.jit
    def load(
        self,
        mPackCount,
        mPackCols,
        mPackVals,
        mX,
        mW,
        sA,
        sX,
        sW,
        tma_atom_W,
        pipeline_a,
        pipeline_x,
        pipeline_w,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        row_tile = cute.arch.block_idx()[0]
        num_row_tiles = cute.size(mPackCount.shape[0])

        a_phase = Int32(1)
        while row_tile < num_row_tiles:
            pack_count = Int32(mPackCount[row_tile])
            pack_id = Int32(0)
            while pack_id < pack_count:
                # Load sparse values tile once for this pack.
                pipeline_a.producer_acquire_w_index_phase(0, a_phase)
                self._load_a_tile(mPackVals, sA, row_tile, pack_id, tidx)
                pipeline_a.producer_commit_w_index(0)
                a_phase ^= 1

                num_k2_tiles = cute.ceil_div(cute.size(mX.shape[1]), self.block_k2)
                x_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_stages)
                w_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_stages)

                k2_block = Int32(0)
                while k2_block < num_k2_tiles:
                    pipeline_x.producer_acquire(x_state)
                    self._load_x_tile(mPackCols, mX, sX, row_tile, pack_id, k2_block, x_state.index, tidx)
                    pipeline_x.producer_commit(x_state)

                    if warp_idx == 0:
                        pipeline_w.producer_acquire(w_state)
                        self._load_w_tile(mW, sW, tma_atom_W, pipeline_w, w_state, k2_block)
                    x_state.advance()
                    if warp_idx == 0:
                        w_state.advance()
                    k2_block += 1

                pack_id += 1

            row_tile += cute.arch.grid_dim()[0]

    @cute.jit
    def _store_acc_g_to_smem(self, acc_g, sG, tiled_mma_xw, tidx):
        # Mirrors FA's "acc_S -> P" store path.
        smem_copy_atom_G = utils.get_smem_store_atom(self.arch.major * 10 + self.arch.minor, sG.element_type)
        smem_thr_copy_G = cute.make_tiled_copy_C(smem_copy_atom_G, tiled_mma_xw).get_slice(tidx)
        tGsG = smem_thr_copy_G.partition_D(sG)
        tGrG_acc = layout_utils.reshape_acc_to_frgA(acc_g)
        tGrG = cute.make_rmem_tensor_like(tGrG_acc, sG.element_type)
        utils.cvt_f16(tGrG_acc, tGrG)
        tGrGs = smem_thr_copy_G.retile(tGrG)
        cute.copy(smem_thr_copy_G, tGrGs, tGsG)
        cute.arch.fence_view_async_shared()
        cute.arch.sync_warp()

    @cute.jit
    def _epilogue_store(self, acc_y, mY, mBias, mReluMask, sO, tma_atom_O, row_tile, tidx):
        # Elementwise epilogue in-register, then cooperative store.
        y_mn = layout_utils.reshape_acc_to_mn(acc_y)
        num_rows = self.block_m
        num_cols = self.n_padded
        base_row = row_tile * self.block_m
        linear = tidx
        total = num_rows * num_cols
        while linear < total:
            r = linear // num_cols
            c = linear - r * num_cols
            row = base_row + r
            val = y_mn[r, c]
            if const_expr(mBias is not None):
                val = val + Float32(mBias[c])
            active = val > 0
            if const_expr(mReluMask is not None):
                if row < cute.size(mY.shape[0]) and c < self.n_actual:
                    mReluMask[row, c] = active
            val = val if const_expr(mReluMask is None) else (val if active else Float32(0))
            if row < cute.size(mY.shape[0]):
                sO[r, c] = val.to(mY.element_type)
            linear += self.num_mma_threads

        # TMA store of the contiguous output tile.
        if tidx == 0:
            # Version-sensitive note:
            # Depending on the exact CUTLASS DSL build, you may prefer an explicit
            # cpasync S2G closure here instead of direct store. The tile and storage
            # layout are already arranged for TMA store.
            pass
        cute.arch.sync_warp()

        linear = tidx
        total = num_rows * self.n_actual
        base_row = row_tile * self.block_m
        while linear < total:
            r = linear // self.n_actual
            c = linear - r * self.n_actual
            row = base_row + r
            if row < cute.size(mY.shape[0]):
                mY[row, c] = sO[r, c]
            linear += self.num_mma_threads

    @cute.jit
    def mma(
        self,
        mPackCount,
        mPackVals,
        mX,
        mY,
        mBias,
        mReluMask,
        sA,
        sX,
        sW,
        sG,
        sO,
        tma_atom_O,
        tiled_mma_xw,
        tiled_mma_ag,
        pipeline_a,
        pipeline_x,
        pipeline_w,
        tidx,
    ):
        warp_group_idx = cute.arch.make_warp_uniform(tidx // 128)
        warp_group_thread_layout = cute.make_layout(
            self.num_consumer_warps // 4, stride=128
        )

        # One consumer warpgroup on Hopper for this kernel.
        wg_mma_xw = tiled_mma_xw.get_slice(warp_group_thread_layout(warp_group_idx))
        wg_mma_ag = tiled_mma_ag.get_slice(warp_group_thread_layout(warp_group_idx))

        # XW fragments: (BK1 x BK2) @ (BK2 x N_PAD) -> (BK1 x N_PAD)
        acc_g, tGrX, tGrW = sm90_utils.partition_fragment_ABC(
            wg_mma_xw,
            (self.block_k1, self.n_padded, self.block_k2),
            sX,
            sW,
        )
        mma_xw_fn = partial(sm90_utils.gemm_w_idx, tiled_mma_xw, acc_g, tGrX, tGrW)

        # AG fragments: (BM x BK1) @ (BK1 x N_PAD) -> (BM x N_PAD)
        # API-sensitive note:
        #   if your helper stack expects the 2nd operand transposed for OperandMajorMode.MN,
        #   use `layout_utils.transpose_view(sG)` here and flip the stored layout in
        #   `_store_acc_g_to_smem` accordingly.
        acc_y, tYrA, tYrG = sm90_utils.partition_fragment_ABC(
            wg_mma_ag,
            (self.block_m, self.n_padded, self.block_k1),
            sA,
            sG,
        )
        mma_ag_fn = partial(sm90_utils.gemm_w_idx, tiled_mma_ag, acc_y, tYrA, tYrG)

        row_tile = cute.arch.block_idx()[0]
        num_row_tiles = cute.size(mPackCount.shape[0])
        a_phase = Int32(0)

        # Accumulate Y across sparse packs.
        acc_y.fill(0.0)

        while row_tile < num_row_tiles:
            pack_count = Int32(mPackCount[row_tile])
            pack_id = Int32(0)
            output_initialized = False

            while pack_id < pack_count:
                pipeline_a.consumer_wait_w_index_phase(0, a_phase)
                x_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_stages)
                w_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_stages)
                num_k2_tiles = cute.ceil_div(cute.size(mX.shape[1]), self.block_k2)

                k2_block = Int32(0)
                while k2_block < num_k2_tiles:
                    pipeline_x.consumer_wait(x_state, pipeline_x.consumer_try_wait(x_state))
                    pipeline_w.consumer_wait(w_state, pipeline_w.consumer_try_wait(w_state))
                    mma_xw_fn(B_idx=x_state.index, zero_init=(k2_block == 0), wg_wait=0)
                    pipeline_x.consumer_release(x_state)
                    pipeline_w.consumer_release(w_state)
                    x_state.advance()
                    w_state.advance()
                    k2_block += 1

                self._store_acc_g_to_smem(acc_g, sG, tiled_mma_xw, tidx)
                mma_ag_fn(B_idx=0, zero_init=not output_initialized, wg_wait=0)
                output_initialized = True
                pipeline_a.consumer_release_w_index(0)
                a_phase ^= 1
                pack_id += 1

            self._epilogue_store(acc_y, mY, mBias, mReluMask, sO, tma_atom_O, row_tile, tidx)
            acc_y.fill(0.0)
            row_tile += cute.arch.grid_dim()[0]


# --------------------------------------------------------------------------------------
# Blackwell (SM100) implementation
# --------------------------------------------------------------------------------------


class FusedSpmmGemmForwardSm100:
    """Blackwell forward kernel following the FA4 decomposition.

    This is intentionally much closer to `flash_fwd_sm100.py` at the structural level:
    - persistent schedule
    - dedicated load warp(s)
    - UMMA / tcgen05 compute warp
    - optional epilogue warp

    Because the exact tcgen05 helper APIs continue to move between CUTLASS DSL drops,
    this class is provided as a concrete architectural port with the API-sensitive pieces
    isolated. The sparse packing, scheduling, and pack-loop structure are the important part.
    """

    def __init__(
        self,
        *,
        block_m: int,
        block_k1: int,
        block_k2: int,
        n_actual: int,
        q_stage: int = 2,
        persistent: bool = True,
        use_2cta_instrs: bool = False,
    ):
        self.block_m = block_m
        self.block_k1 = block_k1
        self.block_k2 = block_k2
        self.n_actual = n_actual
        self.n_padded = max(64, _round_up(n_actual, 64))
        self.q_stage = q_stage
        self.persistent = persistent
        self.use_2cta_instrs = use_2cta_instrs

        self.arch = _get_arch()
        assert self.arch >= Arch.sm_100 and self.arch <= Arch.sm_110f, "Use SM100 path only on Blackwell"

        # Much simpler warp-role split than FA4 because we do not have softmax/correction phases.
        self.load_warp_ids = (0, 1)
        self.mma_warp_id = 2
        self.epilogue_warp_ids = (3,)
        self.empty_warp_ids = (4, 5, 6, 7)
        self.threads_per_cta = cute.arch.WARP_SIZE * len(
            (*self.load_warp_ids, self.mma_warp_id, *self.epilogue_warp_ids, *self.empty_warp_ids)
        )

    def _get_tiled_mmas(self, dtype):
        cta_group = tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE

        # XW and AG are both dense GEMMs. We keep the same decomposition as SM90,
        # just swapping WGMMA helper selection for tcgen05 / UMMA.
        tiled_mma_xw = sm100_utils_basic.make_trivial_tiled_mma(
            dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.MN,
            Float32,
            cta_group,
            (self.block_k1, self.n_padded),
        )
        tiled_mma_ag = sm100_utils_basic.make_trivial_tiled_mma(
            dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.MN,
            Float32,
            cta_group,
            (self.block_m, self.n_padded),
        )
        return tiled_mma_xw, tiled_mma_ag

    @cute.jit
    def __call__(
        self,
        mPackCount: cute.Tensor,
        mPackCols: cute.Tensor,
        mPackVals: cute.Tensor,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mY: cute.Tensor,
        mBias: Optional[cute.Tensor] = None,
        mReluMask: Optional[cute.Tensor] = None,
        stream: cuda.CUstream = None,
    ):
        # Host-side / launch side notes:
        # - reuse the same packed sparse format as SM90
        # - reuse the same algebra A @ (X_gathered @ W)
        # - in practice, you would mirror the FA4 host path:
        #     * TMA for W and O
        #     * PipelineTmaUmma for W
        #     * PipelineAsyncUmma for gathered X / sparse A
        #     * optional cluster launch control if row tiles are highly skewed
        # - unlike FA4, we do not need softmax/correction/tmem staging; one MMA warp is enough
        raise NotImplementedError(
            "SM100 port structure is provided here, but the exact tcgen05 / UMMA helper calls "
            "need to be synchronized with your local flash-attn + cutlass-dsl snapshot. "
            "Start from the SM90 class above and swap in the helper/pipeline pieces from "
            "flash_attn/cute/flash_fwd_sm100.py."
        )


# --------------------------------------------------------------------------------------
# User-facing wrapper
# --------------------------------------------------------------------------------------



def fused_spmm_gemm_relu_cute(
    adjm: torch.Tensor,
    X: torch.Tensor,
    weights: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    apply_relu: bool = True,
    *,
    block_k1: int = 64,
    block_k2: int = 64,
    persistent: bool = True,
    sm100: Optional[bool] = None,
):
    """User-facing wrapper keeping your original API shape.

    Returns:
        Y, relu_mask, packed_sparse

    `packed_sparse` is returned so you can cache it across repeated launches if the
    sparsity pattern is static.
    """

    assert adjm.layout == torch.sparse_bsr
    assert X.dim() == 2 and weights.dim() == 2
    assert X.is_cuda and weights.is_cuda and adjm.is_cuda
    assert X.dtype == weights.dtype
    assert X.shape[0] == adjm.pre_padded_shape[1]
    assert X.shape[1] == weights.shape[0]

    packed = pack_bsr_scalar_cols_for_cute(adjm, block_k1)

    m = packed.m
    k2 = int(X.shape[1])
    n = int(weights.shape[1])
    device = X.device
    dtype = X.dtype

    n_padded = max(64, _round_up(n, 64))
    if n_padded != n:
        W_pad = torch.zeros((k2, n_padded), device=device, dtype=dtype)
        W_pad[:, :n] = weights
        weights = W_pad
        if bias is not None:
            bias_pad = torch.zeros((n_padded,), device=device, dtype=dtype)
            bias_pad[:n] = bias.reshape(-1)[:n]
            bias = bias_pad
    Y = torch.empty((m, n_padded), device=device, dtype=dtype)
    relu_mask = torch.empty((m, n_padded), device=device, dtype=torch.bool) if apply_relu else None

    arch_major, _ = torch.cuda.get_device_capability(device)
    use_sm100 = (arch_major >= 10) if sm100 is None else sm100

    if use_sm100:
        kernel = FusedSpmmGemmForwardSm100(
            block_m=packed.block_m,
            block_k1=block_k1,
            block_k2=block_k2,
            n_actual=n,
            persistent=persistent,
        )
    else:
        kernel = FusedSpmmGemmForwardSm90(
            block_m=packed.block_m,
            block_k1=block_k1,
            block_k2=block_k2,
            n_actual=n,
            persistent=persistent,
        )

    kernel(
        packed.pack_count,
        packed.pack_cols,
        packed.pack_vals,
        X,
        weights,
        Y,
        bias.reshape(-1) if bias is not None else None,
        relu_mask if apply_relu else None,
    )

    if apply_relu:
        Y[:, :n] = torch.where(relu_mask[:, :n], Y[:, :n], torch.zeros_like(Y[:, :n]))

    return Y[:, :n], (relu_mask[:, :n] if relu_mask is not None else None), packed

