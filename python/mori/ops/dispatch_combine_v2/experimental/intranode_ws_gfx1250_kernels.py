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
  ├─────────────────────────────────┤  align 16
  │ p2p_bases[npes]                 │  i64, remote shmem_tok P2P base addrs
  ├─────────────────────────────────┤  align 128 (TDM)
  │ tok_buf_w1[tpb_w1][hidden_dim]  │  warp 1 TDM load/store zone
  ├─────────────────────────────────┤  align 128 (TDM)
  │ tok_buf_w3[tpb_w3][hidden_dim]  │  warp 3 TDM load/store zone
  ├─────────────────────────────────┤  align 16
  │ wt_buf[TPB][topk]               │  f32 weights
  └─────────────────────────────────┘
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
from flydsl.expr.rocdl import ballot
from flydsl.expr import rocdl, tdm_ops
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr, check_smem_capacity
from mori.ops.dispatch_combine_v2.flydsl_prims import atomic_add_global


WGP_BARRIER_ID = -1


def _ceildiv(a, b):
    return (a + b - 1) // b


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
    n_i32 = nbytes // 4

    toks_per_iter = warp_size // topk
    topk_mask = (1 << topk) - 1

    max_tpb_w1 = TPB // 2
    max_tpb_w3 = TPB - max_tpb_w1

    max_slot_iters = _ceildiv(TPB * topk, warp_size)
    max_wt_iters = _ceildiv(TPB * topk, warp_size)

    warp_shift = int(math.log2(warp_size))

    # ------------------------------------------------------------------
    # LDS Layout
    # ------------------------------------------------------------------
    lds = SmemAllocator(None, arch="gfx1250", global_sym_name="dispatch_tdm_smem")

    slot_ctr_off = lds._align(lds.ptr, 16)
    lds.ptr = slot_ctr_off + npes * 4

    p2p_bases_off = lds._align(lds.ptr, 16)
    lds.ptr = p2p_bases_off + npes * 8

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
        addr_inp_tok: fx.Int64,
        addr_inp_idx: fx.Int64,
        addr_inp_wts: fx.Int64,
        addr_p2p_out_tok: fx.Int64,
        addr_p2p_tok_off: fx.Int64,
        inp_cur_tok: fx.Int32,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        warp_id = tid >> warp_shift
        lane = tid & (warp_size - 1)

        tok_start = bid * TPB
        block_tok_count = arith.maxsi(
            arith.minsi(fx.Int32(TPB), inp_cur_tok - tok_start),
            fx.Int32(0),
        )

        smem_base = lds.get_base()

        total_work = block_tok_count * topk
        tpb_w1 = block_tok_count >> 1
        tpb_w3 = block_tok_count - tpb_w1

        # ---- Warp 0: Control plane ----
        if warp_id == 0:
            rsrc_idx = create_buffer_resource_from_addr(addr_inp_idx)
            rsrc_p2p_tok_off = create_buffer_resource_from_addr(addr_p2p_tok_off)
            rsrc_p2p_out_tok = create_buffer_resource_from_addr(addr_p2p_out_tok)

            # Phase 1: count per-PE demand via ballot, batch atomic_add
            pe_counts = [fx.Int32(0) for _ in range(npes)]

            for it in range_constexpr(max_slot_iters):
                work_id = it * warp_size + lane
                tok_global = tok_start + work_id // topk
                valid = (work_id < total_work) & (tok_global < inp_cur_tok)

                idx_offset = tok_global * topk + (work_id % topk)
                dest_expert = valid.select(
                    buffer_load(rsrc_idx, idx_offset, vec_width=1, dtype=T.i32()),
                    fx.Int32(0),
                )
                dest_pe = dest_expert // experts_per_rank

                for pe in range_constexpr(npes):
                    mask = ballot(valid & (dest_pe == pe))
                    # Count unique tokens, not raw lanes:
                    # each token occupies topk consecutive lanes in the ballot
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

            # Load P2P base addresses
            if lane < npes:
                p2p_addr = buffer_load(
                    rsrc_p2p_out_tok, lane, vec_width=1, dtype=T.i64()
                )
                SmemPtr(
                    smem_base, p2p_bases_off + lane * 8,
                    T.i64(), shape=(1,),
                ).store(p2p_addr, [0])

            rocdl.s_barrier_signal(WGP_BARRIER_ID)

            # Phase 2: store weight/idx to remote (small data, VMEM path)
            # (omitted: same as existing dispatch kernel's weight/idx P2P write)

        # ---- Warp 1: TDM engine 0 — first-half tokens ----
        if warp_id == 1:
            # Phase 1: TDM load Global -> LDS
            tok_buf_w1_memref = SmemPtr(
                smem_base, tok_buf_w1_off, T.i16(),
                shape=(tok_buf_w1_bytes // 2,),
            ).get()

            desc_load_w1 = tdm_ops.make_tensor_descriptor_2d(
                global_ptr=addr_inp_tok,
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
            tdm_ops.tensor_wait(0)

            # Wait for warp 0's slot_ctr
            rocdl.s_barrier_wait(WGP_BARRIER_ID)

            # Phase 2: read routing index, dedup, LDS atomic slot alloc, TDM store
            rsrc_idx_w1 = create_buffer_resource_from_addr(addr_inp_idx)
            for t in range_constexpr(max_tpb_w1):
                tok_global_t = tok_start + t
                if tok_global_t < inp_cur_tok:
                    for k in range_constexpr(topk):
                        dest_expert = buffer_load(
                            rsrc_idx_w1, tok_global_t * topk + k,
                            vec_width=1, dtype=T.i32(),
                        )
                        dest_pe_k = dest_expert // experts_per_rank

                        # Dedup: only first occurrence of dest_pe within this token
                        is_first = fx.Boolean(1)
                        for prev in range_constexpr(k):
                            prev_expert = buffer_load(
                                rsrc_idx_w1, tok_global_t * topk + prev,
                                vec_width=1, dtype=T.i32(),
                            )
                            prev_pe = prev_expert // experts_per_rank
                            if prev_pe == dest_pe_k:
                                is_first = fx.Boolean(0)

                        if is_first:
                            # LDS atomic increment to get slot_id
                            slot_ctr_view = SmemPtr(
                                smem_base, slot_ctr_off,
                                T.i32(), shape=(npes,),
                            ).get()
                            # slot_id = memref.atomic_rmw("addi", slot_ctr_view, 1, [dest_pe_k])
                            slot_id = fx.Int32(0)  # placeholder

                            remote_base = SmemPtr(
                                smem_base, p2p_bases_off,
                                T.i64(), shape=(npes,),
                            ).load([dest_pe_k])
                            remote_addr = remote_base + fx.Int64(slot_id) * nbytes

                            tok_row_memref = SmemPtr(
                                smem_base,
                                tok_buf_w1_off + t * nbytes,
                                T.i16(),
                                shape=(hidden_dim,),
                            ).get()

                            desc_store = tdm_ops.make_tensor_descriptor_2d(
                                global_ptr=remote_addr,
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
                wt_work_id = it * warp_size + lane
                tok_global_wt = tok_start + wt_work_id // topk
                wt_valid = (wt_work_id < total_work) & (tok_global_wt < inp_cur_tok)
                if wt_valid:
                    wt_global_off = tok_global_wt * topk + (wt_work_id % topk)
                    wt_val = buffer_load(
                        rsrc_wts, wt_global_off, vec_width=1, dtype=T.f32()
                    )
                    SmemPtr(
                        smem_base, wt_buf_off + wt_work_id * 4,
                        T.f32(), shape=(1,),
                    ).store(arith.bitcast(T.i32(), wt_val), [0])

            # Wait for warp 0's slot_table
            rocdl.s_barrier_wait(WGP_BARRIER_ID)

            # Phase 2: store weight/idx to remote (small data, VMEM path)
            # (omitted: same as existing dispatch kernel's weight/idx P2P write)

        # ---- Warp 3: TDM engine 1 — second-half tokens ----
        if warp_id == 3:
            # Phase 1: TDM load Global -> LDS
            tok_buf_w3_memref = SmemPtr(
                smem_base, tok_buf_w3_off, T.i16(),
                shape=(tok_buf_w3_bytes // 2,),
            ).get()

            desc_load_w3 = tdm_ops.make_tensor_descriptor_2d(
                global_ptr=addr_inp_tok,
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
            tdm_ops.tensor_wait(0)

            # Wait for warp 0's slot_ctr
            rocdl.s_barrier_wait(WGP_BARRIER_ID)

            # Phase 2: read routing index, dedup, LDS atomic slot alloc, TDM store
            rsrc_idx_w3 = create_buffer_resource_from_addr(addr_inp_idx)
            for t in range_constexpr(max_tpb_w3):
                tok_global_t = tok_start + tpb_w1 + t
                if tok_global_t < inp_cur_tok:
                    for k in range_constexpr(topk):
                        dest_expert = buffer_load(
                            rsrc_idx_w3, tok_global_t * topk + k,
                            vec_width=1, dtype=T.i32(),
                        )
                        dest_pe_k = dest_expert // experts_per_rank

                        is_first = fx.Boolean(1)
                        for prev in range_constexpr(k):
                            prev_expert = buffer_load(
                                rsrc_idx_w3, tok_global_t * topk + prev,
                                vec_width=1, dtype=T.i32(),
                            )
                            prev_pe = prev_expert // experts_per_rank
                            if prev_pe == dest_pe_k:
                                is_first = fx.Boolean(0)

                        if is_first:
                            slot_ctr_view = SmemPtr(
                                smem_base, slot_ctr_off,
                                T.i32(), shape=(npes,),
                            ).get()
                            # slot_id = memref.atomic_rmw("addi", slot_ctr_view, 1, [dest_pe_k])
                            slot_id = fx.Int32(0)  # placeholder

                            remote_base = SmemPtr(
                                smem_base, p2p_bases_off,
                                T.i64(), shape=(npes,),
                            ).load([dest_pe_k])
                            remote_addr = remote_base + fx.Int64(slot_id) * nbytes

                            tok_row_memref = SmemPtr(
                                smem_base,
                                tok_buf_w3_off + t * nbytes,
                                T.i16(),
                                shape=(hidden_dim,),
                            ).get()

                            desc_store = tdm_ops.make_tensor_descriptor_2d(
                                global_ptr=remote_addr,
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

    lds.finalized = False
    lds.finalize()

    return ep_dispatch_tdm


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
