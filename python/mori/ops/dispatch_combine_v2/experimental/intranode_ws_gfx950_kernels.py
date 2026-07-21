# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""FlyDSL LL (Low-Latency) intranode dispatch kernel — block-per-token cooperative
dispatch with load-once-write-many, LDS ping-pong pipelined routing, and 3-warp-group
division of labor.

Port of src/ops/dispatch_combine/intranode_ll.hpp to FlyDSL + CCO.
"""
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, range_constexpr, T
from flydsl.expr.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
)
from flydsl.expr.rocdl import (
    ballot,
    readlane,
    s_waitcnt,
    s_barrier,
)
from flydsl.expr.typing import Int32, Int64

from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm_d

import mori.cco.device.flydsl as cco

from .. import flydsl_prims as P

import os as _os


def _detect_wave_size():
    v = _os.environ.get("MORI_WAVE_SIZE")
    if v:
        return int(v)
    try:
        from mori.jit.config import detect_gpu_arch

        return 32 if detect_gpu_arch().startswith("gfx12") else 64
    except Exception:
        return 64


WAVE = _detect_wave_size()
LANE_MASK = WAVE - 1
LOG2_WAVE = WAVE.bit_length() - 1
_BALLOT_INT = T.i64 if WAVE == 64 else T.i32
_MAX_GPUS_PER_NODE = 8
# TokenRoute LDS layout: numDests(1) + srcTokenId(1) + destRank[8] + recvSlot[8] = 18 i32
_ROUTE_I32 = 1 + 1 + _MAX_GPUS_PER_NODE + _MAX_GPUS_PER_NODE  # 18
_ROUTE_BYTES = _ROUTE_I32 * 4  # 72
_LDS_TOTAL = _ROUTE_BYTES * 2  # 144 bytes for ping-pong


def _I32():
    return ir.IntegerType.get_signless(32)


def _I64():
    return ir.IntegerType.get_signless(64)


def _lds_ptr(base_i64, byte_offset):
    """IntToPtrOp into LDS address space (3) at base + byte_offset."""
    addr = _llvm_d.AddOp(
        arith.unwrap(base_i64),
        arith.unwrap(byte_offset),
        ir.Attribute.parse("#llvm.overflow<none>"),
    ).result
    return _llvm_d.IntToPtrOp(_llvm_d.PointerType.get(address_space=3), addr).result


def _lds_store_i32(base_i64, index, val):
    """Store i32 val at LDS base + index*4."""
    off = _llvm_d.MulOp(
        arith.unwrap(index),
        _llvm_d.ConstantOp(_I32(), ir.IntegerAttr.get(_I32(), 4)).result,
        ir.Attribute.parse("#llvm.overflow<none>"),
    ).result
    off64 = _llvm_d.ZExtOp(_I64(), off).res
    ptr = _lds_ptr(base_i64, off64)
    _llvm_d.StoreOp(arith.unwrap(val), ptr, alignment=4)


def _lds_load_i32(base_i64, index):
    """Load i32 from LDS base + index*4."""
    off = _llvm_d.MulOp(
        arith.unwrap(index),
        _llvm_d.ConstantOp(_I32(), ir.IntegerAttr.get(_I32(), 4)).result,
        ir.Attribute.parse("#llvm.overflow<none>"),
    ).result
    off64 = _llvm_d.ZExtOp(_I64(), off).res
    ptr = _lds_ptr(base_i64, off64)
    return _llvm_d.LoadOp(_I32(), ptr, alignment=4).res


def _lds_barrier():
    """s_waitcnt lgkmcnt(0) + s_barrier — full LDS fence + workgroup sync."""
    # gfx9 s_waitcnt encoding: vmcnt[3:0] | expcnt[6:4] | lgkmcnt[12:8]
    # vmcnt=15 (don't wait), expcnt=7 (don't wait), lgkmcnt=0 (wait all LDS)
    s_waitcnt(0x007F)
    s_barrier()

def _readlane_ptr(addr_i64, src_lane):
    """Broadcast a 64-bit pointer from src_lane to all lanes via readlane.

    Split i64 → lo/hi i32, readlane each, recombine.
    """
    raw = arith.unwrap(addr_i64)
    lo = _llvm_d.TruncOp(_I32(), raw).res
    hi_shifted = _llvm_d.LShrOp(raw, _llvm_d.ConstantOp(
        _I64(), ir.IntegerAttr.get(_I64(), 32)
    ).result).result
    hi = _llvm_d.TruncOp(_I32(), hi_shifted).res
    bcast_lo = readlane(T.i32(), fx.Int32(lo), src_lane)
    bcast_hi = readlane(T.i32(), fx.Int32(hi), src_lane)
    lo64 = _llvm_d.ZExtOp(_I64(), arith.unwrap(bcast_lo)).res
    hi64 = _llvm_d.ZExtOp(_I64(), arith.unwrap(bcast_hi)).res
    hi64_shifted = _llvm_d.ShlOp(
        hi64,
        _llvm_d.ConstantOp(_I64(), ir.IntegerAttr.get(_I64(), 32)).result,
        ir.Attribute.parse("#llvm.overflow<none>"),
    ).result
    return fx.Int64(_llvm_d.OrOp(hi64_shifted, lo64).result)

def _popcnt(ballot_val):
    """Population count of a ballot result (i64 for wave64, i32 for wave32)."""
    return fx.Int32(arith.unwrap(arith.popcount(ballot_val)))

def _masked_popcnt(ballot_val, lane):
    """popcount(ballot & ((1 << lane) - 1)) — compact position within active set."""
    if WAVE == 64:
        one = fx.Int64(1)
        shifted = one << fx.Int64(lane)
        mask = shifted - fx.Int64(1)
        masked = ballot_val & mask
        return fx.Int32(arith.unwrap(arith.popcount(masked)))
    else:
        one = fx.Int32(1)
        shifted = one << lane
        mask = shifted - fx.Int32(1)
        masked = ballot_val & mask
        return arith.popcount(masked)


def make_dispatch_ws(
    *,
    rank,
    npes,
    experts_per_rank,
    experts_per_token,
    hidden_dim,
    hidden_elem_size,
    max_tok_per_rank,
    max_recv,
    block_num,
    warp_num_per_block,
    off_tok_off,
    off_recv_num,
    off_tis,
    off_out_idx,
    off_out_wts,
    off_out_tok,
    off_out_scales=0,
    scale_dim=0,
    scale_type_size=0,
    enable_signal=True,
    fp4=False,
):
    nbytes = hidden_dim // 2 if fp4 else hidden_dim * hidden_elem_size
    n_i32 = nbytes // 4
    sentinel_val = npes * max_recv
    scale_bytes = scale_dim * scale_type_size
    scale_num_i32 = (scale_bytes + 3) // 4
    enable_scales = scale_bytes > 0

    # Data-plane geometry: copy warps are warps 2..warp_num_per_block-1
    num_copy_warps = warp_num_per_block - 2
    assert num_copy_warps >= 1, "LL dispatch needs >= 3 warps per block"

    # WarpLoadBroadcastStore constants
    VEC_BYTES = 16
    elems_per_vec = VEC_BYTES // hidden_elem_size if not fp4 else VEC_BYTES
    # In i32 units for the copy loop
    i32_per_vec = 4  # 16 bytes = 4 i32
    COPY_UNROLL = 2
    base_i32_per_warp = COPY_UNROLL * WAVE * i32_per_vec  # per warp in main loop

    @flyc.kernel(known_block_size=[warp_num_per_block * WAVE, 1, 1])
    def ep_dispatch_ws_gfx950(
        arena: Int64,
        addr_inp_tok: Int64,
        addr_inp_idx: Int64,
        addr_inp_wts: Int64,
        addr_tok_map: Int64,
        addr_dest_pe_ctr: Int64,
        addr_disp_bar: Int64,
        addr_total_recv: Int64,
        addr_inp_scales: Int64,
        my_lsa_rank: Int32,
        inp_cur_tok: Int32,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid & LANE_MASK
        warp = tid >> LOG2_WAVE
        grid_dim_x = fx.Int32(block_num)

        window = cco.Window(arena)
        lds_raw = fx.get_dyn_shared()
        # Convert LDS pointer to i64 for arithmetic
        lds_base = fx.Int64(
            _llvm_d.PtrToIntOp(_I64(), arith.unwrap(lds_raw)).result
        )
        # routeBuf[0] at lds_base, routeBuf[1] at lds_base + _ROUTE_BYTES
        lds_route0 = lds_base
        lds_route1 = lds_base + fx.Int64(_ROUTE_BYTES)

        rsrc_inp_idx = create_buffer_resource_from_addr(addr_inp_idx)
        rsrc_inp_wts = create_buffer_resource_from_addr(addr_inp_wts)
        rsrc_tok_map = create_buffer_resource_from_addr(addr_tok_map)
        rsrc_dest_ctr = create_buffer_resource_from_addr(addr_dest_pe_ctr)
        rsrc_disp_bar = create_buffer_resource_from_addr(addr_disp_bar)

        # ── Per-warp peer base pointer prefetch (into VGPR, one per lane) ──
        # Warp 0 lanes < npes: weights/indices/tis base pointers
        # Warp 1 lanes < npes: scales base pointer (if enabled)
        # Warps 2+ lanes < npes: dispOut base pointer

        wts_base_reg = fx.Int64(0)
        idx_base_reg = fx.Int64(0)
        tis_base_reg = fx.Int64(0)
        scales_base_reg = fx.Int64(0)
        disp_out_base_reg = fx.Int64(0)

        did_alloc = fx.Int32(0)
        warp_slot = fx.Int32(0)
        expert_slot_for_me = fx.Int32(-1)

        if bid < inp_cur_tok:
            if warp == 0:
                if lane < npes:
                    wts_base_reg = fx.Int64(window.lsa_ptr(lane, off_out_wts))
                    idx_base_reg = fx.Int64(window.lsa_ptr(lane, off_out_idx))
                    tis_base_reg = fx.Int64(window.lsa_ptr(lane, off_tis))

                token_id = bid

                peer_tok_off = fx.Int64(0)
                if lane < npes:
                    peer_tok_off = fx.Int64(window.lsa_ptr(lane, off_tok_off))

                warp_target_pe = fx.Int32(-1)
                if lane < experts_per_token:
                    expert_id = buffer_load(
                        rsrc_inp_idx,
                        token_id * experts_per_token + lane,
                        vec_width=1,
                        dtype=T.i32(),
                    )
                    if expert_id >= 0:
                        warp_target_pe = expert_id // experts_per_rank

                for k in range_constexpr(_MAX_GPUS_PER_NODE):
                    pe_k = readlane(T.i32(), warp_target_pe, k)
                    if pe_k == lane:
                        expert_slot_for_me = fx.Int32(k)

                if lane < npes:
                    if expert_slot_for_me >= 0:
                        did_alloc = fx.Int32(1)
                        warp_slot = P.atomic_add_global(peer_tok_off, fx.Int32(1))

                active_mask = ballot(_BALLOT_INT(), did_alloc != 0)
                compact_pos = _masked_popcnt(active_mask, lane)
                num_dests = _popcnt(active_mask)

                if lane == 0:
                    _lds_store_i32(lds_route0, fx.Int32(0), num_dests)
                    _lds_store_i32(lds_route0, fx.Int32(1), token_id)
                if did_alloc != 0:
                    _lds_store_i32(lds_route0, fx.Int32(2) + compact_pos, lane)
                    _lds_store_i32(
                        lds_route0, fx.Int32(2 + _MAX_GPUS_PER_NODE) + compact_pos, warp_slot
                    )

            elif warp == 1:
                if const_expr(enable_scales):
                    if lane < npes:
                        scales_base_reg = fx.Int64(window.lsa_ptr(lane, off_out_scales))
            else:  # warp >= 2
                if lane < npes:
                    disp_out_base_reg = fx.Int64(window.lsa_ptr(lane, off_out_tok))

            _lds_barrier()

            # WriteDispDestTokIdMap — after barrier so warp 0 reaches sync faster
            if warp == 0:
                token_id_p = bid
                if lane < experts_per_token:
                    buffer_store(
                        fx.Int32(sentinel_val),
                        rsrc_tok_map,
                        token_id_p * experts_per_token + lane,
                    )
                if did_alloc != 0:
                    P.atomic_add_global(
                        fx.Int64(addr_dest_pe_ctr) + fx.Int64(lane) * fx.Int64(4),
                        fx.Int32(1),
                    )
                    src_tok_encoded = rank * max_tok_per_rank + token_id_p
                    peer_tis = _readlane_ptr(tis_base_reg, lane)
                    buffer_store(
                        src_tok_encoded,
                        create_buffer_resource_from_addr(peer_tis),
                        warp_slot,
                    )
                    tok_map_entry = lane * max_recv + warp_slot
                    buffer_store(
                        tok_map_entry,
                        rsrc_tok_map,
                        token_id_p * experts_per_token + expert_slot_for_me,
                    )

            # ══════════════════════════════════════════════════════════════
            #  Main Loop: one token per iteration, block-strided
            # ══════════════════════════════════════════════════════════════
            for token_idx in range(bid, inp_cur_tok, grid_dim_x):
            ping_pong = (token_idx // grid_dim_x) & 1
            # Select route buffer for current and next
            cur_route = arith.select(ping_pong == 0, lds_route0, lds_route1)
            nxt_route = arith.select(ping_pong == 0, lds_route1, lds_route0)

            # ── Warp 0: route next token + weights/indices for current ─────
            if warp == 0:
                next_token_idx = token_idx + grid_dim_x
                has_next = next_token_idx < inp_cur_tok

                # Route next token → nxt_route
                if has_next:
                    next_tok = next_token_idx

                    peer_tok_off_next = fx.Int64(0)
                    if lane < npes:
                        peer_tok_off_next = fx.Int64(window.lsa_ptr(lane, off_tok_off))

                    warp_target_pe_n = fx.Int32(-1)
                    if lane < experts_per_token:
                        expert_id_n = buffer_load(
                            rsrc_inp_idx,
                            next_tok * experts_per_token + lane,
                            vec_width=1,
                            dtype=T.i32(),
                        )
                        if expert_id_n >= 0:
                            warp_target_pe_n = expert_id_n // experts_per_rank

                    expert_slot_n = fx.Int32(-1)
                    for k in range_constexpr(_MAX_GPUS_PER_NODE):
                        pe_k_n = readlane(T.i32(), warp_target_pe_n, k)
                        if pe_k_n == lane:
                            expert_slot_n = fx.Int32(k)

                    did_alloc_n = fx.Int32(0)
                    warp_slot_n = fx.Int32(0)
                    if lane < npes:
                        if expert_slot_n >= 0:
                            did_alloc_n = fx.Int32(1)
                            warp_slot_n = P.atomic_add_global(
                                peer_tok_off_next, fx.Int32(1)
                            )

                    active_mask_n = ballot(_BALLOT_INT(), did_alloc_n != 0)
                    compact_pos_n = _masked_popcnt(active_mask_n, lane)
                    num_dests_n = _popcnt(active_mask_n)

                    if lane == 0:
                        _lds_store_i32(nxt_route, fx.Int32(0), num_dests_n)
                        _lds_store_i32(nxt_route, fx.Int32(1), next_tok)
                    if did_alloc_n != 0:
                        _lds_store_i32(
                            nxt_route, fx.Int32(2) + compact_pos_n, lane
                        )
                        _lds_store_i32(
                            nxt_route,
                            fx.Int32(2 + _MAX_GPUS_PER_NODE) + compact_pos_n,
                            warp_slot_n,
                        )

                    # WriteDispDestTokIdMap for next token
                    if lane < experts_per_token:
                        buffer_store(
                            fx.Int32(sentinel_val),
                            rsrc_tok_map,
                            next_tok * experts_per_token + lane,
                        )
                    if did_alloc_n != 0:
                        P.atomic_add_global(
                            fx.Int64(addr_dest_pe_ctr) + fx.Int64(lane) * fx.Int64(4),
                            fx.Int32(1),
                        )
                        src_enc_n = rank * max_tok_per_rank + next_tok
                        peer_tis_n = _readlane_ptr(tis_base_reg, lane)
                        buffer_store(
                            src_enc_n,
                            create_buffer_resource_from_addr(peer_tis_n),
                            warp_slot_n,
                        )
                        tok_map_n = lane * max_recv + warp_slot_n
                        buffer_store(
                            tok_map_n,
                            rsrc_tok_map,
                            next_tok * experts_per_token + expert_slot_n,
                        )

                # Scatter weights/indices for CURRENT token from cur_route
                r_num_dests = fx.Int32(_lds_load_i32(cur_route, fx.Int32(0)))
                r_src_tok = fx.Int32(_lds_load_i32(cur_route, fx.Int32(1)))

                if r_num_dests > 0:
                    # Pre-read route into VGPRs: lane d holds destRank[d] / recvSlot[d]
                    r_dest_rank = fx.Int32(0)
                    r_recv_slot = fx.Int32(0)
                    if lane < r_num_dests:
                        r_dest_rank = fx.Int32(
                            _lds_load_i32(cur_route, fx.Int32(2) + lane)
                        )
                        r_recv_slot = fx.Int32(
                            _lds_load_i32(
                                cur_route, fx.Int32(2 + _MAX_GPUS_PER_NODE) + lane
                            )
                        )

                    warp_weight = fx.Float32(0.0)
                    warp_index = fx.Int32(0)
                    if lane < experts_per_token:
                        warp_weight = buffer_load(
                            rsrc_inp_wts,
                            r_src_tok * experts_per_token + lane,
                            vec_width=1,
                            dtype=T.f32(),
                        )
                        warp_index = buffer_load(
                            rsrc_inp_idx,
                            r_src_tok * experts_per_token + lane,
                            vec_width=1,
                            dtype=T.i32(),
                        )

                    for d in range_constexpr(_MAX_GPUS_PER_NODE):
                        if d < r_num_dests:
                            d_rank = readlane(T.i32(), r_dest_rank, d)
                            d_slot = readlane(T.i32(), r_recv_slot, d)
                            if lane < experts_per_token:
                                dest_off = d_slot * experts_per_token + lane
                                peer_wts = _readlane_ptr(wts_base_reg, d_rank)
                                buffer_store(
                                    arith.bitcast(T.i32(), warp_weight),
                                    create_buffer_resource_from_addr(peer_wts),
                                    dest_off,
                                )
                                peer_idx = _readlane_ptr(idx_base_reg, d_rank)
                                buffer_store(
                                    warp_index,
                                    create_buffer_resource_from_addr(peer_idx),
                                    dest_off,
                                )

            # ── Warp 1: scales for current token ────────────────────────────
            if warp == 1:
                if const_expr(enable_scales):
                    s_num_dests = fx.Int32(_lds_load_i32(cur_route, fx.Int32(0)))
                    s_src_tok = fx.Int32(_lds_load_i32(cur_route, fx.Int32(1)))

                    if s_num_dests > 0:
                        # Batch LDS → VGPR: lane d holds destRank[d] / recvSlot[d]
                        s_dest_rank = fx.Int32(0)
                        s_recv_slot = fx.Int32(0)
                        if lane < s_num_dests:
                            s_dest_rank = fx.Int32(
                                _lds_load_i32(cur_route, fx.Int32(2) + lane)
                            )
                            s_recv_slot = fx.Int32(
                                _lds_load_i32(
                                    cur_route, fx.Int32(2 + _MAX_GPUS_PER_NODE) + lane
                                )
                            )

                        # Build per-lane dst pointer via readlane
                        scale_dst_reg = fx.Int64(0)
                        for d in range_constexpr(_MAX_GPUS_PER_NODE):
                            if d < s_num_dests:
                                sd_rank = readlane(T.i32(), s_dest_rank, d)
                                sd_slot = readlane(T.i32(), s_recv_slot, d)
                                if lane == d:
                                    scale_dst_reg = (
                                        _readlane_ptr(scales_base_reg, sd_rank)
                                        + fx.Int64(sd_slot) * fx.Int64(scale_bytes)
                                    )

                        # WarpLoadBroadcastStore for scales (i32 granularity)
                        src_scale_base = addr_inp_scales + fx.Int64(s_src_tok) * fx.Int64(
                            scale_bytes
                        )
                        rsrc_scale_src = create_buffer_resource_from_addr(src_scale_base)
                        for k_off in range(lane, scale_num_i32, WAVE):
                            sv = buffer_load(
                                rsrc_scale_src,
                                k_off,
                                vec_width=1,
                                dtype=T.i32(),
                            )
                            for d in range_constexpr(_MAX_GPUS_PER_NODE):
                                if d < s_num_dests:
                                    dst_addr = _readlane_ptr(scale_dst_reg, d)
                                    buffer_store(
                                        sv,
                                        create_buffer_resource_from_addr(dst_addr),
                                        k_off,
                                    )

            # ── Data plane: warps 2..N-1 broadcast-copy token payload ──────
            if warp >= 2:
                c_num_dests = fx.Int32(_lds_load_i32(cur_route, fx.Int32(0)))
                c_src_tok = fx.Int32(_lds_load_i32(cur_route, fx.Int32(1)))

                if c_num_dests > 0:
                    copy_warp_rank = warp - 2

                    # Compute this warp's slice of the hidden dim (in i32 units)
                    warps_needed = (n_i32 + base_i32_per_warp - 1) // base_i32_per_warp
                    if const_expr(warps_needed <= num_copy_warps):
                        elems_per_warp = base_i32_per_warp
                    else:
                        elems_per_warp = (n_i32 + num_copy_warps - 1) // num_copy_warps
                    warp_elem_offset = copy_warp_rank * elems_per_warp
                    warp_elem_count_raw = arith.select(
                        warp_elem_offset < n_i32,
                        arith.select(
                            elems_per_warp < n_i32 - warp_elem_offset,
                            fx.Int32(elems_per_warp),
                            fx.Int32(n_i32) - fx.Int32(warp_elem_offset),
                        ),
                        fx.Int32(0),
                    )

                    if warp_elem_count_raw > 0:
                        src_tok_addr = addr_inp_tok + fx.Int64(c_src_tok) * fx.Int64(
                            nbytes
                        ) + fx.Int64(warp_elem_offset * 4)
                        rsrc_src = create_buffer_resource_from_addr(src_tok_addr)

                        # Build per-lane dst pointer via readlane from route
                        warp_dest_rank = fx.Int32(0)
                        warp_recv_slot = fx.Int32(0)
                        if lane < c_num_dests:
                            warp_dest_rank = fx.Int32(
                                _lds_load_i32(cur_route, fx.Int32(2) + lane)
                            )
                            warp_recv_slot = fx.Int32(
                                _lds_load_i32(
                                    cur_route,
                                    fx.Int32(2 + _MAX_GPUS_PER_NODE) + lane,
                                )
                            )

                        # Build per-dest register: lane i holds pointer to dest i
                        copy_dst_reg = fx.Int64(0)
                        for i in range_constexpr(_MAX_GPUS_PER_NODE):
                            if i < c_num_dests:
                                d = arith.select(
                                    copy_warp_rank & 1 != 0,
                                    c_num_dests - 1 - fx.Int32(i),
                                    fx.Int32(i),
                                )
                                rank_d = readlane(T.i32(), warp_dest_rank, d)
                                slot_d = readlane(T.i32(), warp_recv_slot, d)
                                if lane == i:
                                    peer_out = _readlane_ptr(
                                        disp_out_base_reg, rank_d
                                    )
                                    copy_dst_reg = peer_out + fx.Int64(
                                        slot_d
                                    ) * fx.Int64(nbytes) + fx.Int64(
                                        warp_elem_offset * 4
                                    )

                        # Pre-build per-dest buffer resource (loop-invariant)
                        dst_rsrcs = []
                        for d in range_constexpr(_MAX_GPUS_PER_NODE):
                            dst_addr = _readlane_ptr(copy_dst_reg, d)
                            dst_rsrcs.append(create_buffer_resource_from_addr(dst_addr))

                        # WarpLoadBroadcastStore — vec4 (16B) load-once, multi-store
                        lane_i32_off = lane * i32_per_vec
                        _LANE_STRIDE_I32 = WAVE * i32_per_vec

                        # Main loop: 2x unrolled
                        if const_expr(COPY_UNROLL == 2):
                            _MAIN_STRIDE = 2 * _LANE_STRIDE_I32
                            safe_end = (warp_elem_count_raw // _MAIN_STRIDE) * _MAIN_STRIDE
                            for chunk in range(
                                lane_i32_off, safe_end, _MAIN_STRIDE
                            ):
                                va = buffer_load(
                                    rsrc_src, chunk, vec_width=4, dtype=T.i32()
                                )
                                vb = buffer_load(
                                    rsrc_src,
                                    chunk + _LANE_STRIDE_I32,
                                    vec_width=4,
                                    dtype=T.i32(),
                                )
                                for d in range_constexpr(_MAX_GPUS_PER_NODE):
                                    if d < c_num_dests:
                                        buffer_store(va, dst_rsrcs[d], chunk)
                                        buffer_store(
                                            vb, dst_rsrcs[d], chunk + _LANE_STRIDE_I32
                                        )

                            # Tail: single-stream
                            for chunk in range(
                                lane_i32_off + safe_end,
                                warp_elem_count_raw,
                                _LANE_STRIDE_I32,
                            ):
                                va = buffer_load(
                                    rsrc_src, chunk, vec_width=4, dtype=T.i32()
                                )
                                for d in range_constexpr(_MAX_GPUS_PER_NODE):
                                    if d < c_num_dests:
                                        buffer_store(va, dst_rsrcs[d], chunk)

            _lds_barrier()

        if const_expr(enable_signal):
            global_warp_id = bid * warp_num_per_block + warp

            if global_warp_id == 0:
                if lane == 0:
                    buffer_store(
                        arith.constant(0),
                        create_buffer_resource_from_addr(addr_total_recv),
                        0,
                    )
                P.fence_system_release()

            fx.barrier()
            if tid == 0:
                P.atomic_add_global(fx.Int64(addr_disp_bar), arith.constant(1))

            local_recv_num = fx.Int64(window.lsa_ptr(my_lsa_rank, off_recv_num))
            for dest_pe in range(lane, npes, WAVE):
                if global_warp_id == 0:
                    P.spin_until_eq_i32(fx.Int64(addr_disp_bar), block_num)
                    P.fence_system_acquire()
                    buffer_store(arith.constant(0), rsrc_disp_bar, 0)
                    signal_value = (
                        buffer_load(
                            rsrc_dest_ctr, dest_pe, vec_width=1, dtype=T.i32()
                        )
                        + 1
                    )
                    peer_recv_num = fx.Int64(window.lsa_ptr(dest_pe, off_recv_num))
                    recv_num_remote_addr = peer_recv_num + fx.Int64(rank) * fx.Int64(4)
                    P.spin_until_eq_i32(recv_num_remote_addr, 0)
                    P.store_i32_system(
                        recv_num_remote_addr, arith.constant(0), signal_value
                    )

            for src_pe in range(lane, npes, WAVE):
                if global_warp_id == 0:
                    recv_num_src_addr = local_recv_num + fx.Int64(src_pe) * fx.Int64(4)
                    signal_value = P.spin_until_gt_i32(recv_num_src_addr, 0)
                    peer_recv_count = signal_value - 1
                    P.store_i32_system(
                        recv_num_src_addr, arith.constant(0), arith.constant(0)
                    )
                    P.atomic_add_global(fx.Int64(addr_total_recv), peer_recv_count)
                    buffer_store(arith.constant(0), rsrc_dest_ctr, src_pe)

            if global_warp_id == 0:
                if lane == 0:
                    local_tok_off = fx.Int64(
                        window.lsa_ptr(my_lsa_rank, off_tok_off)
                    )
                    P.store_i32_system(
                        local_tok_off, arith.constant(0), arith.constant(0)
                    )

    @flyc.jit
    def run(
        arena: Int64,
        addr_inp_tok: Int64,
        addr_inp_idx: Int64,
        addr_inp_wts: Int64,
        addr_tok_map: Int64,
        addr_dest_pe_ctr: Int64,
        addr_disp_bar: Int64,
        addr_total_recv: Int64,
        addr_inp_scales: Int64,
        my_lsa_rank: Int32,
        inp_cur_tok: Int32,
        stream=fx.Stream(None),
    ):
        ep_dispatch_ws_gfx950(
            arena,
            addr_inp_tok,
            addr_inp_idx,
            addr_inp_wts,
            addr_tok_map,
            addr_dest_pe_ctr,
            addr_disp_bar,
            addr_total_recv,
            addr_inp_scales,
            my_lsa_rank,
            inp_cur_tok,
        ).launch(
            grid=(block_num, 1, 1),
            block=[warp_num_per_block * WAVE, 1, 1],
            stream=stream,
            dynamic_shared_memory=_LDS_TOTAL,
        )

    return run
