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
"""Single-GPU perf bench for the gfx1250 TDM EP-dispatch kernel.

Mirrors the structure of tests/python/ops/dispatch_combine_v2/bench_dispatch_combine.py
(env-var knobs, token sweep, eager + CUDA-graph timing, bandwidth report) but is
STANDALONE: the TDM kernel takes raw device addresses (no cco / no torch.distributed),
so we fake the P2P peer layout inside a single GPU — each of the `npes` "peers" is a
local slice of one output buffer, and the per-peer slot counters are local i64 cells.

This is a perf + smoke harness (the experimental kernel hard-codes slot_id=0, so
there is no meaningful multi-slot correctness to check); it verifies the kernel
launches and moves token bytes, and reports Global->LDS->P2P dispatch bandwidth.

    python3 tools/bench_gfx1250_tdm_kernel.py
    MODE=eager|graph|both  SWEEP=64,128,256  HIDDEN=7168 TOPK=8 NPES=8
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DCV2 = REPO_ROOT / "python" / "mori" / "ops" / "dispatch_combine_v2"
KERNEL_PATH = DCV2 / "experimental" / "intranode_ws_gfx1250_kernels.py"


def _fix_rocm_toolkit_path() -> None:
    """MLIR looks for ld.lld at ``$ROCM_PATH/llvm/bin``; the default /opt/rocm may
    point at the runtime-only sdk_core package (no top-level ``llvm/`` symlink),
    which breaks the final ELF link. Point ROCM_PATH at the _rocm_sdk_devel prefix
    (which has ``llvm/`` + ``amdgcn/``) when the current toolkit lacks llvm/bin."""
    cur = os.environ.get("ROCM_PATH", "/opt/rocm")
    if os.path.exists(os.path.join(cur, "llvm", "bin", "ld.lld")):
        return
    import flydsl  # noqa: F401

    site = Path(flydsl.__file__).resolve().parents[1]
    devel = site / "_rocm_sdk_devel"
    if (devel / "llvm" / "bin" / "ld.lld").exists():
        os.environ["ROCM_PATH"] = str(devel)
        print(f"# [bench] ROCM_PATH -> {devel} (toolkit fix)", flush=True)


_fix_rocm_toolkit_path()

import torch  # noqa: E402

import flydsl.compiler as flyc  # noqa: E402
import flydsl.expr as fx  # noqa: E402


def _load_kernel_module():
    """Load the kernel file by path, stubbing its ``mori`` dependency (the compiled
    mori pybind .so need not be present just to build the FlyDSL kernel)."""
    spec = importlib.util.spec_from_file_location(
        "mori.ops.dispatch_combine_v2.flydsl_prims", DCV2 / "flydsl_prims.py"
    )
    flydsl_prims = importlib.util.module_from_spec(spec)
    flydsl_prims.__package__ = "mori.ops.dispatch_combine_v2"
    sys.modules["mori.ops.dispatch_combine_v2.flydsl_prims"] = flydsl_prims
    sys.modules["flydsl_prims"] = flydsl_prims
    spec.loader.exec_module(flydsl_prims)

    spec = importlib.util.spec_from_file_location(
        "intranode_ws_gfx1250_kernels", KERNEL_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- knobs (env-var driven, like the cco bench) ----
NPES = int(os.environ.get("NPES", 8))
HIDDEN = int(os.environ.get("HIDDEN", 7168))
TOPK = int(os.environ.get("TOPK", 8))
EPR = int(os.environ.get("EPR", 1))  # experts_per_rank
ELEM_BYTES = int(os.environ.get("ELEM_BYTES", 2))  # 2 = bf16
TPB = int(os.environ.get("TPB", 8))  # tokens_per_block
MAX_RECV = int(os.environ.get("MAX_RECV", 256))
WARP_SIZE = int(os.environ.get("WARP_SIZE", 32))  # gfx1250 native wave32
WARMUP = int(os.environ.get("WARMUP", 10))
ITERS = int(os.environ.get("ITERS", 50))
MODE = os.environ.get("MODE", "both")  # eager | graph | both
SEED = int(os.environ.get("SEED", 1234))


def _ceildiv(a, b):
    return (a + b - 1) // b


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA/HIP device not available", flush=True)
        return 1
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)

    block_num = _ceildiv(MAX_RECV, TPB)
    max_tok = block_num * TPB  # kernel processes at most this many tokens
    nbytes = HIDDEN * ELEM_BYTES  # per-token transport bytes
    num_experts = NPES * EPR

    sweep = [int(x) for x in os.environ.get("SWEEP", "64,128,256").split(",")]
    sweep = [min(c, max_tok) for c in sweep]

    print(
        f"# gfx1250 TDM ep_dispatch  npes={NPES} hidden={HIDDEN} topk={TOPK} "
        f"epr={EPR} elem_bytes={ELEM_BYTES} tpb={TPB} max_recv={MAX_RECV} "
        f"blocks={block_num} iters={ITERS} mode={MODE}",
        flush=True,
    )

    # ---- compile the kernel (user's direct pattern) ----
    mod = _load_kernel_module()
    run = mod.build_ep_dispatch_tdm_kernel(
        npes=NPES,
        experts_per_rank=EPR,
        experts_per_token=TOPK,
        hidden_dim=HIDDEN,
        elem_bytes=ELEM_BYTES,
        tokens_per_block=TPB,
        max_recv=MAX_RECV,
        warp_size=WARP_SIZE,
    )
    compiled = flyc.compile(
        run,
        fx.Int64(0),
        fx.Int64(0),
        fx.Int64(0),
        fx.Int64(0),
        fx.Int64(0),
        fx.Int32(0),
        fx.Stream(None),
    )
    print("# compiled OK", flush=True)

    # ---- inputs ----
    g = torch.Generator(device="cpu").manual_seed(SEED)
    tok_dt = torch.bfloat16 if ELEM_BYTES == 2 else torch.float32
    inp_tok = (
        torch.randn(max_tok, HIDDEN, generator=g, dtype=torch.float32)
        .to(tok_dt)
        .to(dev)
    )
    inp_idx = torch.randint(
        0, num_experts, (max_tok, TOPK), generator=g, dtype=torch.int32
    ).to(dev)
    inp_wts = torch.rand(max_tok, TOPK, generator=g, dtype=torch.float32).to(dev)

    # ---- fake P2P peer layout on a single GPU ----
    # out_tok[pe] is peer pe's receive buffer; the kernel writes slot 0 of each.
    out_tok = torch.zeros(NPES, MAX_RECV, HIDDEN, dtype=tok_dt, device=dev)
    p2p_out_tok = torch.tensor(
        [out_tok[pe].data_ptr() for pe in range(NPES)],
        dtype=torch.int64,
        device=dev,
    )
    # per-peer slot counters (i64); the table holds the ADDRESS of each counter.
    counters = torch.zeros(NPES, dtype=torch.int64, device=dev)
    p2p_tok_off = torch.tensor(
        [counters.data_ptr() + pe * 8 for pe in range(NPES)],
        dtype=torch.int64,
        device=dev,
    )

    def launch(ct):
        compiled(
            inp_tok.data_ptr(),
            inp_idx.data_ptr(),
            inp_wts.data_ptr(),
            p2p_out_tok.data_ptr(),
            p2p_tok_off.data_ptr(),
            ct,
            fx.Stream(torch.cuda.current_stream()),
        )

    def bw(nbytes_moved, us):
        return nbytes_moved / (1000**3) / (us / 1e6) if us > 0 else 0.0

    def time_eager(ct):
        counters.zero_()
        for _ in range(WARMUP):
            launch(ct)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(ITERS):
            launch(ct)
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / ITERS * 1000  # us

    def time_graph(ct):
        counters.zero_()
        for _ in range(WARMUP):
            launch(ct)
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            launch(ct)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(ITERS):
            gr.replay()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / ITERS * 1000  # us

    # ---- smoke check: one launch writes token bytes into the peer buffers ----
    out_tok.zero_()
    counters.zero_()
    launch(min(sweep))
    torch.cuda.synchronize()
    moved = int((out_tok != 0).any(dim=-1).sum().item())  # #rows that got data
    recv = int(counters.sum().item())
    print(
        f"# smoke ct={min(sweep)}: {'PASS' if moved > 0 else 'FAIL'} "
        f"(nonzero_rows={moved}, slot_counter_sum={recv})",
        flush=True,
    )

    eager = MODE in ("eager", "both")
    graph = MODE in ("graph", "both")
    for ct in sweep:
        payload = ct * nbytes  # token bytes dispatched Global->LDS->P2P
        e_us = time_eager(ct) if eager else 0.0
        g_us = time_graph(ct) if graph else 0.0
        parts = [f"tok {ct:5d}  payload {payload/1e6:8.2f}MB"]
        if eager:
            parts.append(f"| EAGER {e_us:8.2f}us / {bw(payload, e_us):7.1f}GB/s")
        if graph:
            parts.append(f"| GRAPH {g_us:8.2f}us / {bw(payload, g_us):7.1f}GB/s")
        print("  ".join(parts), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        import traceback

        traceback.print_exc()
        print(f"BENCH FAILED: {type(exc).__name__}: {exc}", flush=True)
        raise SystemExit(1) from exc
