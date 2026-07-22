"""
EP Dispatch with TDM + Warp Specialization — gfx1250 Example

=== Hardware Constraints (gfx1250) ===
  Each WGP has 2 TDM engines, one per SA (Shader Array):
    TDM engine 0 ← SA0 (warp 0, warp 1)
    TDM engine 1 ← SA1 (warp 2, warp 3)
  Both TDM engines must be utilized to achieve max Global↔LDS bandwidth.

  gfx1250 native wave32 (WARP_SIZE=32).

=== Warp Assignment (4 warps per block) ===
  Warp 0: Control plane — atomic_add to reserve remote slots, write slot_table to LDS
  Warp 1: TDM engine 0 — load first-half tokens (Global→LDS), then TDM store to remote
  Warp 2: VMEM — load weight/scale to LDS, then store weight/idx to remote
  Warp 3: TDM engine 1 — load second-half tokens (Global→LDS), then TDM store to remote

=== Two-Phase Pipeline ===
  Phase 1 (parallel):
    Warp 0: atomic_add → write slot_table to LDS → signal
    Warp 1: TDM load first-half tokens → LDS → signal
    Warp 2: buffer_load weights → LDS → signal
    Warp 3: TDM load second-half tokens → LDS → signal
    All wait (slot_table ready)

  Phase 2 (TDM store):
    Warp 1,3: read slot_table → build TDM store descriptor → LDS→P2P remote
    Warp 0,2: store weight/idx/metadata to remote (small data, VMEM path)

  Bulk token data goes through TDM throughout (0 VGPR): Global→LDS→P2P remote

=== LDS Layout ===
  ┌─────────────────────────────────┐  0
  │ slot_ctr[npes]                  │  i32, per-PE slot counter (start offset, atomically incremented)
  ├─────────────────────────────────┤  align 128 (TDM)
  │ tok_buf_w1[tpb_w1][hidden_dim]  │  warp 1 TDM load/store zone
  ├─────────────────────────────────┤  align 128 (TDM)
  │ tok_buf_w3[tpb_w3][hidden_dim]  │  warp 3 TDM load/store zone
  ├─────────────────────────────────┤  align 16
  │ wt_buf[TPB][topk]               │  f32 weights
  └─────────────────────────────────┘
  p2p_bases: warp 1/3 直接 buffer_load, 不走 LDS
"""

from __future__ import annotations

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import T, arith, range_constexpr
from flydsl.expr.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
)
from flydsl.expr.math import cttz
from flydsl.expr.rocdl import ballot
from flydsl.expr import rocdl, tdm_ops
from flydsl.expr.typing import Int32, Int64
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr, check_smem_capacity
from flydsl._mlir.dialects import arith as arith_d
from flydsl._mlir.dialects import memref as memref_d
from flydsl.expr import primitive as fly_prim
from flydsl.expr.gpu import AddressSpace
from flydsl.expr.typing import PointerType
from mori.ops.dispatch_combine_v2.flydsl_prims import atomic_add_global


WGP_BARRIER_ID = -1
WARP_SIZE = 32  # gfx1250 native wave32
_BALLOT_INT = T.i64 if WARP_SIZE == 64 else T.i32


def _ceildiv(a, b):
    return (a + b - 1) // b


def _clamp_block_tok_count(inp_cur_tok, tok_start, tpb):
    remaining = inp_cur_tok - tok_start
    return arith.select(
        remaining > fx.Int32(0),
        arith.select(remaining < fx.Int32(tpb), remaining, fx.Int32(tpb)),
        fx.Int32(0),
    )




def _global_tensor_from_addr(addr, outer, inner, elem_bytes):
    elem_ty = T.i16() if elem_bytes == 2 else T.f32()
    ptr_ty = PointerType.get(elem_ty, AddressSpace.Global)
    ptr = fly_prim.inttoptr(ptr_ty, addr)
    layout = fly_prim.make_layout((outer, inner), (inner, 1))
    return fly_prim.make_view(ptr, layout)

def _lds_atomic_add_i32(slot_ctr_view, dest_pe_k):
    idx = arith.index_cast(T.index(), dest_pe_k)
    old = memref_d.atomic_rmw(arith_d.AtomicRMWKind.addi, arith.constant(1), slot_ctr_view, [idx])
    return fx.Int32(old)


def build_ep_dispatch_tdm_kernel(
    npes: int = 8,
    experts_per_rank: int = 1,
    experts_per_token: int = 8,
    hidden_dim: int = 7168,
    elem_bytes: int = 2,
    tokens_per_block: int = 8,
    max_recv: int = 256,
    warp_size: int = 32,
):
    NUM_WARPS = 4
    BLOCK_THREADS = NUM_WARPS * warp_size

    topk = experts_per_token
    TPB = tokens_per_block
    nbytes = hidden_dim * elem_bytes
    toks_per_iter = warp_size // topk
    topk_mask = (1 << topk) - 1

    max_tpb_w1 = TPB // 2
    max_tpb_w3 = TPB - max_tpb_w1

    max_slot_iters = _ceildiv(TPB * topk, warp_size)
    max_wt_iters = _ceildiv(TPB * topk, warp_size)

    warp_shift = int(math.log2(warp_size))
    block_num = _ceildiv(max_recv, TPB)

    # ------------------------------------------------------------------
    # LDS Layout
    # ------------------------------------------------------------------
    lds = SmemAllocator(None, arch="gfx1250", global_sym_name="dispatch_tdm_smem")

    slot_ctr_off = lds._align(lds.ptr, 16)
    lds.ptr = slot_ctr_off + npes * 4

    tok_buf_w1_off = lds._align(lds.ptr, 128)
    tok_buf_w1_bytes = max_tpb_w1 * nbytes
    lds.ptr = tok_buf_w1_off + tok_buf_w1_bytes

    tok_buf_w3_off = lds._align(lds.ptr, 128)
    tok_buf_w3_bytes = max_tpb_w3 * nbytes
    lds.ptr = tok_buf_w3_off + tok_buf_w3_bytes

    wt_buf_off = lds._align(lds.ptr, 16)
    wt_buf_bytes = TPB * topk * 4
    lds.ptr = wt_buf_off + wt_buf_bytes

    total_lds = lds._align(lds.ptr, 128)
    check_smem_capacity(total_lds, "gfx1250")

    # ------------------------------------------------------------------
    # Kernel
    # ------------------------------------------------------------------
    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def ep_dispatch_tdm(
        addr_inp_tok: Int64,
        addr_inp_idx: Int64,
        addr_inp_wts: Int64,
        addr_p2p_out_tok: Int64,
        addr_p2p_tok_off: Int64,
        inp_cur_tok: Int32,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        warp_id = tid >> warp_shift
        lane = tid & (warp_size - 1)

        tok_start = bid * fx.Int32(TPB)
        block_tok_count = _clamp_block_tok_count(inp_cur_tok, tok_start, TPB)

        smem_base = lds.get_base()

        total_work = block_tok_count * fx.Int32(topk)
        tpb_w1 = block_tok_count >> 1
        tpb_w3 = block_tok_count - tpb_w1

        # ---- Warp 0: Control plane ----
        if warp_id == 0:
            rsrc_idx = create_buffer_resource_from_addr(addr_inp_idx)
            rsrc_p2p_tok_off = create_buffer_resource_from_addr(addr_p2p_tok_off)

            # Phase 1: count per-PE unique token demand via ballot, batch atomic_add
            pe_counts = [fx.Int32(0) for _ in range(npes)]

            for it in range_constexpr(max_slot_iters):
                work_id = fx.Int32(it * warp_size + lane)
                tok_global = tok_start + work_id // fx.Int32(topk)
                valid = (work_id < total_work) & (tok_global < inp_cur_tok)

                idx_offset = tok_global * fx.Int32(topk) + (work_id % fx.Int32(topk))
                dest_expert = valid.select(
                    buffer_load(rsrc_idx, idx_offset, vec_width=1, dtype=T.i32()),
                    fx.Int32(0),
                )
                dest_pe = dest_expert // fx.Int32(experts_per_rank)

                for pe in range_constexpr(npes):
                    mask = ballot(_BALLOT_INT(), valid & (dest_pe == pe))
                    for g in range_constexpr(toks_per_iter):
                        group_bits = (mask >> (g * topk)) & topk_mask
                        pe_counts[pe] = pe_counts[pe] + arith.select(
                            group_bits != fx.Int32(0), fx.Int32(1), fx.Int32(0),
                        )

            # One remote atomic per PE to reserve slot range
            for pe in range_constexpr(npes):
                if lane == pe:
                    counter_addr = buffer_load(
                        rsrc_p2p_tok_off, pe, vec_width=1, dtype=T.i64(),
                    )
                    start = atomic_add_global(counter_addr, pe_counts[pe])
                    SmemPtr(
                        smem_base, slot_ctr_off + pe * 4,
                        T.i32(), shape=(1,),
                    ).store(start, [0])

            rocdl.s_wait_dscnt(0)
            rocdl.s_barrier_signal(WGP_BARRIER_ID)

        # ---- Warp 1: TDM engine 0 — first-half tokens ----
        if warp_id == 1:
            # Phase 1: TDM load Global -> LDS
            tok_buf_w1_memref = SmemPtr(
                smem_base, tok_buf_w1_off, T.i16(),
                shape=(tok_buf_w1_bytes // 2,),
            ).get()

            desc_load_w1 = tdm_ops.make_tensor_descriptor_2d(
                global_ptr=_global_tensor_from_addr(addr_inp_tok, inp_cur_tok, hidden_dim, elem_bytes),
                lds_memref=tok_buf_w1_memref,
                global_offset=(tok_start, 0),
                tensor_shape=(inp_cur_tok, hidden_dim),
                strides=(hidden_dim, 1),
                tile_shape=(max_tpb_w1, hidden_dim),
                elem_bytes=elem_bytes,
                num_warps=1,
                oob_outer_bound=tpb_w1,
            )
            tdm_ops.tensor_load_2d(desc_load_w1)

            # Prefetch routing indices while TDM load is in flight (4 VGPRs)
            rsrc_idx_w1 = create_buffer_resource_from_addr(addr_inp_idx)
            rsrc_p2p_w1 = create_buffer_resource_from_addr(addr_p2p_out_tok)
            w1_dest_pe = []
            for t in range_constexpr(max_tpb_w1):
                tok_global_t = tok_start + t
                valid_lane = (tok_global_t < inp_cur_tok) & (lane < topk)
                dest_expert = valid_lane.select(
                    buffer_load(
                        rsrc_idx_w1, tok_global_t * topk + lane,
                        vec_width=1, dtype=T.i32(),
                    ),
                    fx.Int32(-1),
                )
                w1_dest_pe.append(valid_lane.select(
                    dest_expert // experts_per_rank, fx.Int32(-1),
                ))

            # Arrive (so the WG split barrier reaches its full 4-wave count),
            # then wait for slot_ctr (warp 0) + TDM load.
            rocdl.s_barrier_signal(WGP_BARRIER_ID)
            rocdl.s_barrier_wait(WGP_BARRIER_ID)
            tdm_ops.tensor_wait(0)

            # Phase 2: ballot dedup + TDM store (dest_pe already in VGPRs)
            for t in range_constexpr(max_tpb_w1):
                tok_global_t = tok_start + t
                valid_lane = (tok_global_t < inp_cur_tok) & (lane < topk)
                dest_pe = w1_dest_pe[t]

                for pe in range_constexpr(npes):
                    pe_mask = ballot(_BALLOT_INT(), valid_lane & (dest_pe == pe))
                    # `pe_mask` is wave-uniform (ballot), so this branch is uniform.
                    # The TDM store is a wave-level DMA and MUST be issued wave-
                    # uniformly (like the load) — not from a single master lane, or
                    # it never fires. All store params here are uniform (pe is a
                    # constexpr, slot_id is the placeholder 0), so every lane issues
                    # the same one store.
                    if pe_mask != fx.Int32(0):
                        # slot_id = memref.atomic_rmw("addi", slot_ctr_view, 1, [pe])
                        slot_id = fx.Int32(0)  # placeholder

                        remote_base = buffer_load(
                            rsrc_p2p_w1, pe, vec_width=1, dtype=T.i64(),
                        )
                        remote_addr = remote_base + fx.Int64(slot_id) * nbytes

                        tok_row_memref = SmemPtr(
                            smem_base,
                            tok_buf_w1_off + t * nbytes,
                            T.i16(),
                            shape=(hidden_dim,),
                        ).get()

                        desc_store = tdm_ops.make_tensor_descriptor_2d(
                            global_ptr=_global_tensor_from_addr(remote_addr, 1, hidden_dim, elem_bytes),
                            lds_memref=tok_row_memref,
                            global_offset=(0, 0),
                            tensor_shape=(1, hidden_dim),
                            strides=(hidden_dim, 1),
                            tile_shape=(1, hidden_dim),
                            elem_bytes=elem_bytes,
                            num_warps=1,
                            for_store=True,
                        )
                        tdm_ops.tensor_store_2d(desc_store)

            tdm_ops.tensor_wait(0)

        # ---- Warp 2: Weight load (VMEM, no TDM usage) ----
        if warp_id == 2:
            # Phase 1: buffer_load weights -> LDS
            rsrc_wts = create_buffer_resource_from_addr(addr_inp_wts)
            for it in range_constexpr(max_wt_iters):
                wt_work_id = fx.Int32(it * warp_size + lane)
                tok_global_wt = tok_start + wt_work_id // fx.Int32(topk)
                wt_valid = (wt_work_id < total_work) & (tok_global_wt < inp_cur_tok)
                if wt_valid:
                    wt_global_off = tok_global_wt * fx.Int32(topk) + (
                        wt_work_id % fx.Int32(topk)
                    )
                    wt_val = buffer_load(
                        rsrc_wts, wt_global_off, vec_width=1, dtype=T.f32()
                    )
                    SmemPtr(
                        smem_base, wt_buf_off,
                        T.f32(), shape=(TPB * topk,),
                    ).store(wt_val, [wt_work_id])

            # Arrive at the WG split barrier (weight load done), then wait.
            rocdl.s_barrier_signal(WGP_BARRIER_ID)
            rocdl.s_barrier_wait(WGP_BARRIER_ID)

        # ---- Warp 3: TDM engine 1 — second-half tokens ----
        if warp_id == 3:
            # Phase 1: TDM load Global -> LDS
            tok_buf_w3_memref = SmemPtr(
                smem_base, tok_buf_w3_off, T.i16(),
                shape=(tok_buf_w3_bytes // 2,),
            ).get()

            desc_load_w3 = tdm_ops.make_tensor_descriptor_2d(
                global_ptr=_global_tensor_from_addr(addr_inp_tok, inp_cur_tok, hidden_dim, elem_bytes),
                lds_memref=tok_buf_w3_memref,
                global_offset=(tok_start + tpb_w1, 0),
                tensor_shape=(inp_cur_tok, hidden_dim),
                strides=(hidden_dim, 1),
                tile_shape=(max_tpb_w3, hidden_dim),
                elem_bytes=elem_bytes,
                num_warps=1,
                oob_outer_bound=tpb_w3,
            )
            tdm_ops.tensor_load_2d(desc_load_w3)

            # Prefetch routing indices while TDM load is in flight (4 VGPRs)
            rsrc_idx_w3 = create_buffer_resource_from_addr(addr_inp_idx)
            rsrc_p2p_w3 = create_buffer_resource_from_addr(addr_p2p_out_tok)
            w3_dest_pe = []
            for t in range_constexpr(max_tpb_w3):
                tok_global_t = tok_start + tpb_w1 + t
                valid_lane = (tok_global_t < inp_cur_tok) & (lane < topk)
                dest_expert = valid_lane.select(
                    buffer_load(
                        rsrc_idx_w3, tok_global_t * topk + lane,
                        vec_width=1, dtype=T.i32(),
                    ),
                    fx.Int32(-1),
                )
                w3_dest_pe.append(valid_lane.select(
                    dest_expert // experts_per_rank, fx.Int32(-1),
                ))

            # Arrive (so the WG split barrier reaches its full 4-wave count),
            # then wait for slot_ctr (warp 0) + TDM load.
            rocdl.s_barrier_signal(WGP_BARRIER_ID)
            rocdl.s_barrier_wait(WGP_BARRIER_ID)
            tdm_ops.tensor_wait(0)

            # Phase 2: ballot dedup + TDM store (dest_pe already in VGPRs)
            for t in range_constexpr(max_tpb_w3):
                tok_global_t = tok_start + tpb_w1 + t
                valid_lane = (tok_global_t < inp_cur_tok) & (lane < topk)
                dest_pe = w3_dest_pe[t]

                for pe in range_constexpr(npes):
                    pe_mask = ballot(_BALLOT_INT(), valid_lane & (dest_pe == pe))
                    # Wave-uniform TDM store (see warp 1 note): issued by all lanes.
                    if pe_mask != fx.Int32(0):
                        # slot_id = memref.atomic_rmw("addi", slot_ctr_view, 1, [pe])
                        slot_id = fx.Int32(0)  # placeholder

                        remote_base = buffer_load(
                            rsrc_p2p_w3, pe, vec_width=1, dtype=T.i64(),
                        )
                        remote_addr = remote_base + fx.Int64(slot_id) * nbytes

                        tok_row_memref = SmemPtr(
                            smem_base,
                            tok_buf_w3_off + t * nbytes,
                            T.i16(),
                            shape=(hidden_dim,),
                        ).get()

                        desc_store = tdm_ops.make_tensor_descriptor_2d(
                            global_ptr=_global_tensor_from_addr(remote_addr, 1, hidden_dim, elem_bytes),
                            lds_memref=tok_row_memref,
                            global_offset=(0, 0),
                            tensor_shape=(1, hidden_dim),
                            strides=(hidden_dim, 1),
                            tile_shape=(1, hidden_dim),
                            elem_bytes=elem_bytes,
                            num_warps=1,
                            for_store=True,
                        )
                        tdm_ops.tensor_store_2d(desc_store)

            tdm_ops.tensor_wait(0)

    @flyc.jit
    def run(
        addr_inp_tok: Int64,
        addr_inp_idx: Int64,
        addr_inp_wts: Int64,
        addr_p2p_out_tok: Int64,
        addr_p2p_tok_off: Int64,
        inp_cur_tok: Int32,
        stream=fx.Stream(None),
    ):
        if not lds.finalized:
            from flydsl.compiler.kernel_function import CompilationContext
            from flydsl._mlir import ir

            ctx = CompilationContext.get_current()
            with ir.InsertionPoint(ctx.gpu_module_body):
                lds.finalize()

        ep_dispatch_tdm(
            addr_inp_tok,
            addr_inp_idx,
            addr_inp_wts,
            addr_p2p_out_tok,
            addr_p2p_tok_off,
            inp_cur_tok,
        ).launch(
            grid=(block_num, 1, 1),
            block=[BLOCK_THREADS, 1, 1],
            stream=stream,
        )

    return run


# ==========================================================================
# Design Notes
# ==========================================================================
#
# 1. Bulk data goes through TDM throughout (0 VGPR):
#
#    Phase 1: Global --TDM load--> LDS(tok_buf)      [0 VGPR]
#    Phase 2: LDS(tok_buf) --TDM store--> P2P remote  [0 VGPR]
#
#    Compared to existing kernel:
#    buffer_load(Global) -> VGPR -> buffer_store(P2P)  [occupies 4+ VGPRs]
#
# 2. TDM Engine load balancing:
#    ┌──────────┬─────────────────────────────────────────────────────┐
#    │ TDM Eng  │ Phase 1 (load)              │ Phase 2 (store)      │
#    ├──────────┼─────────────────────────────┼──────────────────────┤
#    │ TDM 0    │ Warp 1: load 1st-half tok   │ Warp 1: store 1st   │
#    │ TDM 1    │ Warp 3: load 2nd-half tok   │ Warp 3: store 2nd   │
#    │ None     │ Warp 0: atomic/slot         │ Warp 0: wt/idx P2P  │
#    │ None     │ Warp 2: weight load         │ Warp 2: wt/idx P2P  │
#    └──────────┴─────────────────────────────┴──────────────────────┘
#
# 3. Phase 2 TDM store granularity:
#    Each unique (token, dest_pe) -> 1 tensor_store_2d call
#    tile = (1, hidden_dim) = 1 x 7168 x 2B = 14 KB
#    Each warp issues at most tpb_w1 x npes = 4 x 8 = 32 stores
#    In practice far fewer (most tokens go to only 1-2 PEs)
#
# 4. Future optimizations:
#    - Pipeline: Phase 1 TDM load and Phase 2 TDM store
#      can overlap when hidden_dim is chunked (double-buffering)
#    - Gather store: multiple tokens to the same dest_pe can use
#      tensor_store_gather to merge into a single descriptor
#    - Validation: need to confirm TDM store reachability to XGMI P2P mapped addrs
