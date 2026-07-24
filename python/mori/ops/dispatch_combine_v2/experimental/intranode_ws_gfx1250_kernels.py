"""
EP Dispatch with TDM + Warp Specialization — gfx1250 Example

=== Hardware Constraints (gfx1250) ===
  Each WGP has 2 TDM engines, one per SA (Shader Array):
    TDM engine 0 ← SA0 (warp 0, warp 1)
    TDM engine 1 ← SA1 (warp 2, warp 3)
  Both TDM engines must be utilized to achieve max Global↔LDS bandwidth.

  gfx1250 native wave32 (WARP_SIZE=32).

=== Warp Assignment (4 warps per block) ===
  Warp 0: batch1 control — ballot-count first B1 tokens' per-PE demand,
          one remote atomic_add per PE to reserve slots, seed slot_ctr_A.
  Warp 1: TDM engine 0 — batch1 first-half tokens, then batch2 loop.
  Warp 2: batch2 control — ballot-count remaining tokens' per-PE demand,
          one remote atomic_add per PE, seed slot_ctr_B (empty when TPB==B1).
  Warp 3: TDM engine 1 — batch1 second-half tokens, then batch2 loop.

=== Two-Batch Pipeline (single WGP barrier) ===
  Phase 0 (parallel, before barrier):
    Warp 0: count batch1 → remote reserve → slot_ctr_A
    Warp 2: count batch2 → remote reserve → slot_ctr_B   (overlaps warp1/3 load)
    Warp 1: TDM load batch1 first-half tokens → LDS
    Warp 3: TDM load batch1 second-half tokens → LDS
    all signal + wait   (slot_ctr_A and slot_ctr_B both seeded)

  Phase 1 (warp 1/3):
    store batch1 (slot_ctr_A): per (token,PE) TDM store token + inline weight
    for each batch2 iter: TDM load 4 tokens → store (slot_ctr_B)

  Per-token slot is allocated on the fly by warp1/3 via a 4-token-batched LDS
  atomic on slot_ctr_[A|B]. Because that slot id lives only in warp1/3 VGPRs and
  is atomic-order-dependent, the SAME warp writes the token embedding (TDM) and
  its weight vector (VMEM) to the reserved slot — no extra barrier / LDS table.

=== LDS Layout (independent of TPB) ===
  slot_ctr_A[npes]                 i32  batch1 per-PE counter (seeded to remote base)
  slot_ctr_B[npes]                 i32  batch2 per-PE counter (double-buffered)
  tok_buf_w1[TOKS_PER_WAVE][hidden] i16 warp 1 TDM load/store zone (fixed 4 tokens)
  tok_buf_w3[TOKS_PER_WAVE][hidden] i16 warp 3 TDM load/store zone (fixed 4 tokens)
  weights go Global→VGPR→remote VMEM directly (no LDS staging).
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
from flydsl.expr.typing import Int32, Int64
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr, check_smem_capacity
from flydsl._mlir.dialects import arith as arith_d
from flydsl._mlir.dialects import memref as memref_d
from flydsl.expr import primitive as fly_prim
from flydsl.expr.gpu import AddressSpace
from flydsl.expr.typing import PointerType
from mori.ops.dispatch_combine_v2.flydsl_prims import atomic_add_global
import mori.cco.device.flydsl as cco


WGP_BARRIER_ID = -1
WARP_SIZE = 32  # gfx1250 native wave32
# Always i64: the gfx1250 backend lacks the ballot.i32 predicate-lowering pattern
# (a divergent `ballot(a & b)` there crashes ISel with an un-selectable i32
# AMDGPUISD::SETCC). All FlyDSL examples use i64 ballot; the low 32 bits carry the
# wave32 lane mask.
_BALLOT_INT = T.i64


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


def build_ep_dispatch_tdm_kernel(
    npes: int = 8,
    experts_per_rank: int = 1,
    experts_per_token: int = 8,
    hidden_dim: int = 7168,
    elem_bytes: int = 2,
    tokens_per_block: int = 8,
    max_recv: int = 256,
    warp_size: int = 32,
    off_tok_off: int = 0,
    off_out_tok: int = 0,
    off_out_wts: int = 0,
):
    NUM_WARPS = 4
    BLOCK_THREADS = NUM_WARPS * warp_size

    topk = experts_per_token
    TPB = tokens_per_block
    nbytes = hidden_dim * elem_bytes
    topk_mask = (1 << topk) - 1
    warp_shift = int(math.log2(warp_size))

    # Fixed per-warp/iter token granularity (decoupled from TPB): one wave (32
    # lanes) covers TOKS_PER_WAVE tokens × topk experts.
    TOKS_PER_WAVE = warp_size // topk          # 4 for wave32 / topk8
    B1_TOKS = 2 * TOKS_PER_WAVE                 # batch1 = both warps' first wave
    B2_TOKS = max(TPB - B1_TOKS, 0)            # batch2 = the rest
    b2_iters = _ceildiv(B2_TOKS, B1_TOKS)      # compile-time batch2 loop bound
    count_iters_b1 = _ceildiv(B1_TOKS * topk, warp_size)
    count_iters_b2 = _ceildiv(B2_TOKS * topk, warp_size)

    tok_buf_i16 = TOKS_PER_WAVE * hidden_dim   # i16 elems per warp buffer

    # ------------------------------------------------------------------
    # LDS Layout
    # ------------------------------------------------------------------
    lds = SmemAllocator(None, arch="gfx1250", global_sym_name="dispatch_tdm_smem")

    slot_ctr_a_off = lds._align(lds.ptr, 16)
    lds.ptr = slot_ctr_a_off + npes * 4

    slot_ctr_b_off = lds._align(lds.ptr, 16)
    lds.ptr = slot_ctr_b_off + npes * 4

    tok_buf_w1_off = lds._align(lds.ptr, 128)
    lds.ptr = tok_buf_w1_off + TOKS_PER_WAVE * nbytes

    tok_buf_w3_off = lds._align(lds.ptr, 128)
    lds.ptr = tok_buf_w3_off + TOKS_PER_WAVE * nbytes

    total_lds = lds._align(lds.ptr, 128)
    check_smem_capacity(total_lds, "gfx1250")

    # ------------------------------------------------------------------
    # Kernel
    # ------------------------------------------------------------------
    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def ep_dispatch_tdm(
        arena: Int64,
        addr_inp_tok: Int64,
        addr_inp_idx: Int64,
        addr_inp_wts: Int64,
        inp_cur_tok: Int32,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        warp_id = tid >> warp_shift
        lane = tid & (warp_size - 1)

        tok_start = bid * fx.Int32(TPB)
        block_tok_count = _clamp_block_tok_count(inp_cur_tok, tok_start, TPB)

        smem_base = lds.get_base()

        g_of_lane = lane // fx.Int32(topk)
        k_of_lane = lane % fx.Int32(topk)

        def _pe_ballot(valid_b, dest_pe, pe):
            # Fold validity + PE match into a single i32 value (sentinel = warp_size
            # for non-participating lanes) and ballot on one `< warp_size` compare.
            # Passing a compound `setcc & setcc` i1 directly makes the gfx1250
            # backend emit an un-selectable divergent AMDGPUISD::SETCC; the FlyDSL
            # comm example (`dup_per_lane < 64`) uses this select-fold form instead.
            dup = (dest_pe == fx.Int32(pe)).select(
                valid_b.select(lane, fx.Int32(warp_size)),
                fx.Int32(warp_size),
            )
            return ballot(_BALLOT_INT(), dup < fx.Int32(warp_size))

        # -------- control plane: count + remote reserve + seed slot_ctr --------
        def do_count(tok_lo, tok_hi, n_iters, slot_ctr_off, arena_h):
            rsrc_idx = create_buffer_resource_from_addr(addr_inp_idx)
            win = cco.Window(arena_h)
            n_work = (tok_hi - tok_lo) * topk
            pe_counts = [fx.Int32(0) for _ in range(npes)]

            for it in range_constexpr(n_iters):
                work_id = fx.Int32(it * warp_size) + lane
                local_tok = fx.Int32(tok_lo) + work_id // fx.Int32(topk)
                tok_global = tok_start + local_tok
                valid = (work_id < fx.Int32(n_work)) & (tok_global < inp_cur_tok)
                idx_off = tok_global * fx.Int32(topk) + (work_id % fx.Int32(topk))
                dest_expert = valid.select(
                    buffer_load(rsrc_idx, idx_off, vec_width=1, dtype=T.i32()),
                    fx.Int32(0),
                )
                dest_pe = dest_expert // fx.Int32(experts_per_rank)
                for pe in range_constexpr(npes):
                    mask = _pe_ballot(valid, dest_pe, pe)
                    for g in range_constexpr(TOKS_PER_WAVE):
                        bits = (mask >> (g * topk)) & topk_mask
                        pe_counts[pe] = pe_counts[pe] + arith.select(
                            bits != 0, fx.Int32(1), fx.Int32(0)
                        )

            for pe in range_constexpr(npes):
                if lane == fx.Int32(pe):
                    counter_addr = fx.Int64(win.lsa_ptr(pe, off_tok_off))
                    start = atomic_add_global(counter_addr, pe_counts[pe])
                    SmemPtr(
                        smem_base, slot_ctr_off + pe * 4, T.i32(), shape=(1,)
                    ).store(start, [0])
            rocdl.s_wait_dscnt(0)

        # -------- TDM load G=TOKS_PER_WAVE tokens Global -> LDS ----------------
        def do_load(tok_buf_off, warp_tok_start, local_start):
            memref = SmemPtr(
                smem_base, tok_buf_off, T.i16(), shape=(tok_buf_i16,)
            ).get()
            cnt = _clamp_block_tok_count(
                block_tok_count, fx.Int32(local_start), TOKS_PER_WAVE
            )
            desc = tdm_ops.make_tensor_descriptor_2d(
                global_ptr=_global_tensor_from_addr(
                    addr_inp_tok, inp_cur_tok, hidden_dim, elem_bytes
                ),
                lds_memref=memref,
                global_offset=(warp_tok_start, 0),
                tensor_shape=(inp_cur_tok, hidden_dim),
                strides=(hidden_dim, 1),
                tile_shape=(TOKS_PER_WAVE, hidden_dim),
                elem_bytes=elem_bytes,
                num_warps=1,
                # ABSOLUTE global outer extent; descriptor subtracts the tile
                # start internally. Clamp to the block boundary so a middle block
                # never loads the next block's rows.
                oob_outer_bound=warp_tok_start + cnt,
            )
            tdm_ops.tensor_load_2d(desc)

        # -------- data plane: 4-token batched slot alloc + TDM/weight store ----
        def do_stores(tok_buf_off, warp_tok_start, local_start, slot_ctr_off, arena_h):
            rsrc_idx = create_buffer_resource_from_addr(addr_inp_idx)
            rsrc_wts = create_buffer_resource_from_addr(addr_inp_wts)
            win = cco.Window(arena_h)
            slot_ctr_view = SmemPtr(
                smem_base, slot_ctr_off, T.i32(), shape=(npes,)
            ).get()

            local_index = fx.Int32(local_start) + g_of_lane
            tok_g_global = tok_start + local_index
            valid_lane = local_index < block_tok_count
            wt_off = tok_g_global * fx.Int32(topk) + k_of_lane
            raw_expert = buffer_load(rsrc_idx, wt_off, vec_width=1, dtype=T.i32())
            dest_pe = valid_lane.select(
                raw_expert // fx.Int32(experts_per_rank), fx.Int32(-1)
            )
            wt_val = buffer_load(rsrc_wts, wt_off, vec_width=1, dtype=T.f32())

            for pe in range_constexpr(npes):
                pe_mask = ballot(
                    _BALLOT_INT(), valid_lane & (dest_pe == fx.Int32(pe))
                )
                if pe_mask != 0:
                    h_i32 = []
                    count = fx.Int32(0)
                    for g in range_constexpr(TOKS_PER_WAVE):
                        bits = (pe_mask >> (g * topk)) & topk_mask
                        h_g = arith.select(
                            bits != 0, fx.Int32(1), fx.Int32(0)
                        )
                        h_i32.append(h_g)
                        count = count + h_g
                    rank = []
                    acc = fx.Int32(0)
                    for g in range_constexpr(TOKS_PER_WAVE):
                        rank.append(acc)
                        acc = acc + h_i32[g]

                    addend = arith.select(lane == fx.Int32(0), count, fx.Int32(0))
                    idx = arith.index_cast(T.index(), fx.Int32(pe))
                    old = memref_d.atomic_rmw(
                        arith_d.AtomicRMWKind.addi, addend, slot_ctr_view, [idx]
                    )
                    base = fx.Int32(rocdl.readlane(T.i32(), old, fx.Int32(0)))

                    remote_tok_base = fx.Int64(win.lsa_ptr(pe, off_out_tok))
                    remote_wt_base = fx.Int64(win.lsa_ptr(pe, off_out_wts))
                    rsrc_wt_remote = create_buffer_resource_from_addr(remote_wt_base)

                    for g in range_constexpr(TOKS_PER_WAVE):
                        g_bits = (pe_mask >> (g * topk)) & topk_mask
                        if g_bits != 0:  # wave-uniform
                            slot = base + rank[g]
                            remote_addr = remote_tok_base + fx.Int64(slot) * nbytes
                            row_memref = SmemPtr(
                                smem_base,
                                tok_buf_off + g * nbytes,
                                T.i16(),
                                shape=(hidden_dim,),
                            ).get()
                            desc_store = tdm_ops.make_tensor_descriptor_2d(
                                global_ptr=_global_tensor_from_addr(
                                    remote_addr, 1, hidden_dim, elem_bytes
                                ),
                                lds_memref=row_memref,
                                global_offset=(0, 0),
                                tensor_shape=(1, hidden_dim),
                                strides=(hidden_dim, 1),
                                tile_shape=(1, hidden_dim),
                                elem_bytes=elem_bytes,
                                num_warps=1,
                                for_store=True,
                            )
                            tdm_ops.tensor_store_2d(desc_store)
                            wt_end = (g_of_lane == fx.Int32(g)).select(
                                fx.Int32(1), fx.Int32(0)
                            )
                            for _wt in range(fx.Int32(0), wt_end):
                                buffer_store(
                                    wt_val,
                                    rsrc_wt_remote,
                                    slot * fx.Int32(topk) + k_of_lane,
                                )

        # ---- Warp 0: batch1 control ----
        if warp_id == 0:
            do_count(0, B1_TOKS, count_iters_b1, slot_ctr_a_off, arena)
            rocdl.s_barrier_signal(WGP_BARRIER_ID)

        # ---- Warp 2: batch2 control ----
        if warp_id == 2:
            if B2_TOKS > 0:
                do_count(B1_TOKS, TPB, count_iters_b2, slot_ctr_b_off, arena)
            rocdl.s_barrier_signal(WGP_BARRIER_ID)

        # ---- Warp 1: TDM engine 0 ----
        if warp_id == 1:
            do_load(tok_buf_w1_off, tok_start, 0)
            rocdl.s_barrier_signal(WGP_BARRIER_ID)
            rocdl.s_barrier_wait(WGP_BARRIER_ID)
            tdm_ops.tensor_wait(0)
            do_stores(tok_buf_w1_off, tok_start, 0, slot_ctr_a_off, arena)
            tdm_ops.tensor_wait(0)
            for i in range_constexpr(b2_iters):
                local_start = B1_TOKS + i * B1_TOKS
                wstart = tok_start + fx.Int32(local_start)
                do_load(tok_buf_w1_off, wstart, local_start)
                tdm_ops.tensor_wait(0)
                do_stores(tok_buf_w1_off, wstart, local_start, slot_ctr_b_off, arena)
                tdm_ops.tensor_wait(0)

        # ---- Warp 3: TDM engine 1 ----
        if warp_id == 3:
            w3_start = tok_start + fx.Int32(TOKS_PER_WAVE)
            do_load(tok_buf_w3_off, w3_start, TOKS_PER_WAVE)
            rocdl.s_barrier_signal(WGP_BARRIER_ID)
            rocdl.s_barrier_wait(WGP_BARRIER_ID)
            tdm_ops.tensor_wait(0)
            do_stores(tok_buf_w3_off, w3_start, TOKS_PER_WAVE, slot_ctr_a_off, arena)
            tdm_ops.tensor_wait(0)
            for i in range_constexpr(b2_iters):
                local_start = B1_TOKS + i * B1_TOKS + TOKS_PER_WAVE
                wstart = tok_start + fx.Int32(local_start)
                do_load(tok_buf_w3_off, wstart, local_start)
                tdm_ops.tensor_wait(0)
                do_stores(tok_buf_w3_off, wstart, local_start, slot_ctr_b_off, arena)
                tdm_ops.tensor_wait(0)

    @flyc.jit
    def run(
        arena: Int64,
        addr_inp_tok: Int64,
        addr_inp_idx: Int64,
        addr_inp_wts: Int64,
        inp_cur_tok: Int32,
        stream=fx.Stream(None),
    ):
        if not lds.finalized:
            from flydsl.compiler.kernel_function import CompilationContext
            from flydsl._mlir import ir

            ctx = CompilationContext.get_current()
            with ir.InsertionPoint(ctx.gpu_module_body):
                lds.finalize()

        grid_blocks = (inp_cur_tok + fx.Int32(TPB - 1)) // fx.Int32(TPB)

        ep_dispatch_tdm(
            arena,
            addr_inp_tok,
            addr_inp_idx,
            addr_inp_wts,
            inp_cur_tok,
        ).launch(
            grid=(grid_blocks, 1, 1),
            block=[BLOCK_THREADS, 1, 1],
            stream=stream,
        )

    return run
