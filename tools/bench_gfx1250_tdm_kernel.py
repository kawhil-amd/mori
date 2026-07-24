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
"""Multi-GPU perf bench for the gfx1250 TDM EP-dispatch kernel.

Real torchrun bootstrap, real CCO communicator + SymmArena, real cross-GPU
dispatch, barrier-synchronized timing, correctness verification.

    torchrun --standalone --nproc_per_node=8 tools/bench_gfx1250_tdm_kernel.py
    MODE=eager|graph|both  SWEEP=64,128,256  HIDDEN=7168 TOPK=8 EPR=1
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

import flydsl.expr as fx
from mori.cco import Communicator
from mori.tensor_utils import from_gpu_ptr
from mori.ops.dispatch_combine_v2.dispatch_combine_op import SymmArena

_ROOT = Path(__file__).resolve().parents[1]
_DCV2 = _ROOT / "python" / "mori" / "ops" / "dispatch_combine_v2"
_KERNEL_PATH = _DCV2 / "experimental" / "intranode_ws_gfx1250_kernels.py"

sys.path.insert(0, str(_ROOT / "examples" / "cco" / "python"))
from cco_example_common import set_device, sync  # noqa: E402


class Dist:
    """Minimal torchrun/gloo bootstrap."""

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


HIDDEN = int(os.environ.get("HIDDEN", 7168))
TOPK = int(os.environ.get("TOPK", 8))
EPR = int(os.environ.get("EPR", 1))
ELEM_BYTES = int(os.environ.get("ELEM_BYTES", 2))
TPB = int(os.environ.get("TPB", 8))
MAX_RECV = int(os.environ.get("MAX_RECV", 256))
WARP_SIZE = int(os.environ.get("WARP_SIZE", 32))
WARMUP = int(os.environ.get("WARMUP", 10))
ITERS = int(os.environ.get("ITERS", 50))
MODE = os.environ.get("MODE", "both")
SWEEP = [int(x) for x in os.environ.get("SWEEP", "64,128,256").split(",")]


def _ceildiv(a, b):
    return (a + b - 1) // b


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
    d = Dist()
    rank, npes = d.rank, d.world
    set_device(d.local_rank)
    dev = torch.device("cuda", d.local_rank)

    num_experts = npes * EPR
    block_num = _ceildiv(MAX_RECV, TPB)
    max_tok = block_num * TPB
    nbytes = HIDDEN * ELEM_BYTES
    sweep = [min(c, max_tok) for c in SWEEP]

    if rank == 0:
        print(
            f"# gfx1250 TDM ep_dispatch  npes={npes} hidden={HIDDEN} topk={TOPK} "
            f"epr={EPR} elem_bytes={ELEM_BYTES} tpb={TPB} max_recv={MAX_RECV} "
            f"blocks={block_num} iters={ITERS} mode={MODE}",
            flush=True,
        )

    g = torch.Generator(device="cpu").manual_seed(1234 + rank)
    tok_dt = torch.bfloat16 if ELEM_BYTES == 2 else torch.float32
    inp = (
        torch.randn(max_tok, HIDDEN, generator=g, dtype=torch.float32)
        .to(tok_dt)
        .to(dev)
    )
    idx = torch.randint(
        0, num_experts, (max_tok, TOPK), generator=g, dtype=torch.int32
    ).to(dev)
    wts = torch.rand(max_tok, TOPK, generator=g, dtype=torch.float32).to(dev)

    uid = Communicator.get_unique_id() if rank == 0 else None
    uid = d.bcast_uid(uid)

    arena_total = MAX_RECV * nbytes + MAX_RECV * TOPK * 4 + (1 << 20)
    with Communicator.init(
        npes, rank, uid, per_rank_vmm=2 * arena_total + (1 << 24)
    ) as comm:
        regions = [
            ("tok_off", 4),
            ("out_wts", MAX_RECV * TOPK * 4),
            ("out_tok", MAX_RECV * nbytes),
        ]
        arena = SymmArena(comm, regions)
        arena.zero()

        mod = _load_kernel_module()
        disp_kern = mod.build_ep_dispatch_tdm_kernel(
            npes=npes,
            experts_per_rank=EPR,
            experts_per_token=TOPK,
            hidden_dim=HIDDEN,
            elem_bytes=ELEM_BYTES,
            tokens_per_block=TPB,
            max_recv=MAX_RECV,
            warp_size=WARP_SIZE,
            off_tok_off=arena.offset("tok_off"),
            off_out_tok=arena.offset("out_tok"),
            off_out_wts=arena.offset("out_wts"),
        )
        if rank == 0:
            print("# compiled OK", flush=True)

        def run_disp(ct):
            disp_kern(
                arena.handle,
                inp.data_ptr(),
                idx.data_ptr(),
                wts.data_ptr(),
                ct,
                fx.Stream(torch.cuda.current_stream()),
            )

        def time_eager(ct):
            for _ in range(WARMUP):
                arena.zero()
                sync()
                run_disp(ct)
                sync()
                comm.barrier()
            arena.zero()
            sync()
            comm.barrier()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(ITERS):
                arena.zero()
                run_disp(ct)
            e.record()
            torch.cuda.synchronize()
            comm.barrier()
            return s.elapsed_time(e) / ITERS * 1000

        def time_graph(ct):
            for _ in range(WARMUP):
                arena.zero()
                sync()
                run_disp(ct)
                sync()
                comm.barrier()
            arena.zero()
            sync()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                arena.zero()
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

        def verify(ct):
            arena.zero()
            sync()
            run_disp(ct)
            sync()
            comm.barrier()
            recv = int(
                from_gpu_ptr(arena.local_ptr("tok_off"), (1,), torch.int32)
                .cpu()
                .item()
            )
            has_data = 0
            if recv > 0:
                tok_data = from_gpu_ptr(
                    arena.local_ptr("out_tok"), (recv, HIDDEN), tok_dt
                )
                has_data = int((tok_data != 0).any().item())
            errs = d.allreduce_sum(0 if (recv > 0 and has_data) else 1)
            if rank == 0:
                print(
                    f"# correctness ct={ct}: {'PASS' if errs == 0 else 'FAIL'} "
                    f"(recv={recv}, has_data={'yes' if has_data else 'no'})",
                    flush=True,
                )
            return errs == 0

        verify(min(sweep))

        eager = MODE in ("eager", "both")
        graph = MODE in ("graph", "both")
        for ct in sweep:
            arena.zero()
            sync()
            run_disp(ct)
            sync()
            comm.barrier()
            recv = int(
                from_gpu_ptr(arena.local_ptr("tok_off"), (1,), torch.int32)
                .cpu()
                .item()
            )
            payload = recv * nbytes

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
