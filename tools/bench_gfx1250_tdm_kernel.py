#!/usr/bin/env python3
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
"""gfx1250 experimental TDM dispatch bench (dispatch-only).

Arena / token caps / env knobs match ``tests/python/ops/dispatch_combine_v2/
bench_dispatch_combine.py``; only the launch path uses
``intranode_ws_gfx1250_kernels`` instead of ``intranode_kernels.make_dispatch``.

    torchrun --standalone --nproc_per_node=4 tools/bench_gfx1250_tdm_kernel.py
    MODE=eager|graph|both  SWEEP=128,512,2048  HIDDEN=7168 TOPK=8 EPR=32 DTYPE=bf16
"""
from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
import types
from pathlib import Path

import torch
import torch.distributed as dist

import flydsl.expr as fx
from mori.cco import Communicator
from mori.tensor_utils import from_gpu_ptr
import mori.cco.device.flydsl as cco  # noqa: F401

_ROOT = Path(__file__).resolve().parents[1]
_DCV2 = _ROOT / "python" / "mori" / "ops" / "dispatch_combine_v2"
_KERNEL_PATH = _DCV2 / "experimental" / "intranode_ws_gfx1250_kernels.py"

sys.path.insert(0, str(_ROOT / "examples" / "cco" / "python"))
from cco_example_common import set_device, sync  # noqa: E402

import mori as _mori_pkg

if "mori.ops" not in sys.modules:
    _ops = types.ModuleType("mori.ops")
    _repo_ops = os.path.join(
        os.environ.get("MORI_REPO", str(_ROOT)), "python", "mori", "ops"
    )
    _wheel_ops = str(pathlib.Path(_mori_pkg.__file__).parent / "ops")
    _ops.__path__ = (
        [_repo_ops, _wheel_ops] if os.path.isdir(_repo_ops) else [_wheel_ops]
    )
    sys.modules["mori.ops"] = _ops

from mori.ops.dispatch_combine_v2 import (  # noqa: E402
    EpDispatchCombineConfig,
    EpDispatchCombineOp,
)
from mori.ops.dispatch_combine_v2 import tuning_configs as _tc  # noqa: E402

# ---- env (aligned with bench_dispatch_combine.py) ----
HIDDEN = int(os.environ.get("HIDDEN", 7168))
K = int(os.environ.get("TOPK", 8))
EPR = int(os.environ.get("EPR", 32))
DISP_BLOCK = int(os.environ.get("DISP_BLOCK", os.environ.get("BLOCK_NUM", 64)))
WARP_NUM = int(os.environ.get("WARP_NUM", 16))
AUTO = int(os.environ.get("AUTO", 0))
WARMUP = int(os.environ.get("WARMUP", 10))
ITERS = int(os.environ.get("ITERS", 50))
MODE = os.environ.get("MODE", "both")
DTYPE = os.environ.get("DTYPE", "bf16")
SWEEP = [int(x) for x in os.environ.get("SWEEP", "128,512,2048").split(",")]

# TDM kernel geometry (not used by intranode make_dispatch)
TPB = int(os.environ.get("TPB", 8))
WARP_SIZE = int(os.environ.get("WARP_SIZE", 32))

_FP8_DT = (
    torch.float8_e4m3fn
    if _tc._topology()[1] in (90500, 120500)
    else torch.float8_e4m3fnuz
)
_DT = {
    "bf16": (torch.bfloat16, 2),
    "f32": (torch.float32, 4),
    "fp8": (_FP8_DT, 1),
    "fp4": (torch.float4_e2m1fn_x2, 1),
}
TOK_DT, ESZ = _DT[DTYPE]
TOK_NB = HIDDEN // 2 if DTYPE == "fp4" else HIDDEN * ESZ


class Dist:
    def __init__(self):
        self.rank = int(os.environ["RANK"])
        self.world = int(os.environ["WORLD_SIZE"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        if not dist.is_initialized():
            dist.init_process_group(backend="gloo")
        torch.cuda.set_device(self.local_rank)

    def bcast_uid(self, uid):
        objs = [uid if self.rank == 0 else None]
        dist.broadcast_object_list(objs, src=0)
        return objs[0]

    def allreduce_sum(self, value):
        t = torch.tensor([value], dtype=torch.int64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return int(t.item())

    def shutdown(self):
        if dist.is_initialized():
            dist.destroy_process_group()


def _load_kernel_module():
    spec = importlib.util.spec_from_file_location(
        "mori.ops.dispatch_combine_v2.flydsl_prims", _DCV2 / "flydsl_prims.py"
    )
    flydsl_prims = importlib.util.module_from_spec(spec)
    flydsl_prims.__package__ = "mori.ops.dispatch_combine_v2"
    sys.modules["mori.ops.dispatch_combine_v2.flydsl_prims"] = flydsl_prims
    sys.modules["flydsl_prims"] = flydsl_prims
    spec.loader.exec_module(flydsl_prims)

    spec = importlib.util.spec_from_file_location(
        "intranode_ws_gfx1250_kernels", _KERNEL_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    if DTYPE not in ("bf16", "f32"):
        raise SystemExit(
            "TDM experimental kernel supports DTYPE=bf16|f32 only (same 2B/4B elem_bytes)."
        )

    d = Dist()
    rank, npes = d.rank, d.world
    set_device(d.local_rank)
    dev = torch.device("cuda", d.local_rank)

    num_experts = npes * EPR
    max_tok = max(SWEEP)
    M = max_tok
    max_recv = npes * M

    g = torch.Generator(device="cpu").manual_seed(1234 + rank)
    inp = (
        torch.randn(max_tok, HIDDEN, generator=g, dtype=torch.float32)
        .to(TOK_DT)
        .to(dev)
    )
    idx = torch.randint(0, num_experts, (max_tok, K), generator=g, dtype=torch.int32).to(
        dev
    )
    wts = torch.rand(max_tok, K, generator=g, dtype=torch.float32).to(dev)

    uid = Communicator.get_unique_id() if rank == 0 else None
    uid = d.bcast_uid(uid)

    win_bytes = max_recv * TOK_NB + npes * M * TOK_NB + (1 << 24)
    with Communicator.init(
        npes, rank, uid, per_rank_vmm=2 * win_bytes + (1 << 28)
    ) as comm:
        _cfg_kwargs = dict(
            rank=rank,
            world_size=npes,
            hidden_dim=HIDDEN,
            max_num_inp_token_per_rank=M,
            num_experts_per_rank=EPR,
            num_experts_per_token=K,
            data_type=TOK_DT,
            combine_mode="gather",
            quant_type="none",
        )
        if not AUTO:
            _cfg_kwargs.update(
                dispatch_block_num=DISP_BLOCK,
                warp_num_per_block=WARP_NUM,
                schedule=None,
            )
        cfg = EpDispatchCombineConfig(**_cfg_kwargs)
        op = EpDispatchCombineOp(cfg, comm)
        op.reset()
        arena = op.arena
        recv_cap = cfg.effective_max_recv

        if rank == 0:
            _geom = (
                "geom=AUTO(tuned schedule)"
                if AUTO
                else f"block disp={DISP_BLOCK} x{WARP_NUM}w"
            )
            print(
                f"# gfx1250 TDM dispatch  EP{npes} hidden={HIDDEN} topk={K} "
                f"experts={num_experts} recv_cap={recv_cap} tpb={TPB} "
                f"{_geom}  iters={ITERS} mode={MODE} dtype={DTYPE}",
                flush=True,
            )
            if AUTO:
                print(
                    f"# AUTO schedule={cfg.schedule} "
                    f"disp_default=({cfg.dispatch_block_num},{cfg.warp_num_per_block})",
                    flush=True,
                )

        mod = _load_kernel_module()
        disp_kern = mod.build_ep_dispatch_tdm_kernel(
            npes=npes,
            experts_per_rank=EPR,
            experts_per_token=K,
            hidden_dim=HIDDEN,
            elem_bytes=ESZ,
            tokens_per_block=TPB,
            max_recv=recv_cap,
            warp_size=WARP_SIZE,
            off_tok_off=arena.offset("tok_off"),
            off_out_tok=arena.offset("disp_out"),
            off_out_wts=arena.offset("out_wts"),
        )
        if rank == 0:
            print("# compiled OK (TDM experimental)", flush=True)

        def run_disp(ct):
            disp_kern(
                arena.handle,
                inp.data_ptr(),
                idx.data_ptr(),
                wts.data_ptr(),
                ct,
                fx.Stream(torch.cuda.current_stream()),
            )

        def reset_arena():
            arena.zero()
            sync()

        def time_eager(ct):
            for _ in range(WARMUP):
                reset_arena()
                run_disp(ct)
                sync()
                comm.barrier()
            reset_arena()
            sync()
            comm.barrier()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(ITERS):
                reset_arena()
                run_disp(ct)
            e.record()
            torch.cuda.synchronize()
            comm.barrier()
            return s.elapsed_time(e) / ITERS * 1000

        def time_graph(ct):
            for _ in range(WARMUP):
                reset_arena()
                run_disp(ct)
                sync()
                comm.barrier()
            reset_arena()
            sync()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                reset_arena()
                run_disp(ct)
            torch.cuda.synchronize()
            comm.barrier()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(ITERS):
                gr.replay()
            e.record()
            torch.cuda.synchronize()
            comm.barrier()
            return s.elapsed_time(e) / ITERS * 1000

        def bw(nb, us):
            return nb / (1000**3) / (us / 1e6) if us > 0 else 0.0

        def read_recv():
            return int(
                from_gpu_ptr(arena.local_ptr("tok_off"), (1,), torch.int32)
                .cpu()
                .item()
            )

        def verify(ct):
            reset_arena()
            run_disp(ct)
            sync()
            comm.barrier()
            recv = read_recv()
            has_data = 0
            if recv > 0:
                n = min(recv, recv_cap)
                tok_data = from_gpu_ptr(
                    arena.local_ptr("disp_out"), (n, HIDDEN), TOK_DT
                )
                has_data = int((tok_data != 0).any().item())
            ok = recv > 0 and has_data and recv <= recv_cap
            errs = d.allreduce_sum(0 if ok else 1)
            if rank == 0:
                print(
                    f"# correctness ct={ct}: {'PASS' if errs == 0 else 'FAIL'} "
                    f"(recv={recv}, cap={recv_cap}, has_data={'yes' if has_data else 'no'})",
                    flush=True,
                )
            return errs == 0

        verify(min(SWEEP))

        eager = MODE in ("eager", "both")
        graph = MODE in ("graph", "both")
        for ct in SWEEP:
            reset_arena()
            run_disp(ct)
            sync()
            comm.barrier()
            recv = read_recv()
            payload = recv * TOK_NB

            dp_e = time_eager(ct) if eager else 0.0
            dp_g = time_graph(ct) if graph else 0.0

            if rank == 0:
                parts = [
                    f"tok/rank {ct:5d}  recv {recv:6d}  "
                    f"payload {payload / 1e6:7.2f}MB"
                ]
                if eager:
                    parts.append(
                        f"| EAGER {dp_e:8.2f}us / {bw(payload, dp_e):6.1f}GB/s"
                    )
                if graph:
                    parts.append(
                        f"| GRAPH {dp_g:8.2f}us / {bw(payload, dp_g):6.1f}GB/s"
                    )
                print("  ".join(parts), flush=True)

    d.shutdown()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback

        traceback.print_exc()
        print(f"BENCH FAILED: {type(exc).__name__}: {exc}", flush=True)
        raise SystemExit(1) from exc
