#!/usr/bin/env python3
"""Compile-only check for intranode_ws_gfx1250_kernels.py via FlyDSL."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.typing import Int32, Int64

REPO_ROOT = Path(__file__).resolve().parents[1]
DCV2 = REPO_ROOT / "python" / "mori" / "ops" / "dispatch_combine_v2"
KERNEL_PATH = DCV2 / "experimental" / "intranode_ws_gfx1250_kernels.py"


def _load_kernel_module():
    spec = importlib.util.spec_from_file_location(
        "flydsl_prims",
        DCV2 / "flydsl_prims.py",
    )
    flydsl_prims = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["flydsl_prims"] = flydsl_prims
    spec.loader.exec_module(flydsl_prims)
    sys.modules["mori.ops.dispatch_combine_v2.flydsl_prims"] = flydsl_prims

    spec = importlib.util.spec_from_file_location(
        "intranode_ws_gfx1250_kernels",
        KERNEL_PATH,
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@flyc.jit
def compile_only(
    arena: Int64,
    addr_inp_tok: Int64,
    addr_inp_idx: Int64,
    addr_inp_wts: Int64,
    inp_cur_tok: Int32,
    stream=fx.Stream(None),
):
    mod = _load_kernel_module()
    run_fn = mod.build_ep_dispatch_tdm_kernel()
    run_fn(
        arena,
        addr_inp_tok,
        addr_inp_idx,
        addr_inp_wts,
        inp_cur_tok,
        stream,
    )


def main() -> int:
    print(f"Compiling kernel from: {KERNEL_PATH}")
    compiled = flyc.compile(
        compile_only,
        fx.Int64(0),
        fx.Int64(0),
        fx.Int64(0),
        fx.Int64(0),
        fx.Int32(8),
    )
    print("COMPILE SUCCESS")
    print(compiled)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        import traceback

        traceback.print_exc()
        print(f"COMPILE FAILED: {type(exc).__name__}: {exc}")
        raise SystemExit(1) from exc
