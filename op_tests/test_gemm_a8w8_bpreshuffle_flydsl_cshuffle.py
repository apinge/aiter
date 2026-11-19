# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942 FlyDSL preshuffle GEMM: the C-shuffle epilogue and gemm_a8w8_bpreshuffle(out=...)
must match the default paths bit for bit."""

import argparse
import itertools

import pandas as pd
import torch
import torch.nn.functional as F

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl import gemm_kernels
from aiter.ops.flydsl.utils import is_flydsl_available
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

# The C-shuffle epilogue is selected on gfx942 only.
SUPPORTED_GFX = ["gfx942"]
TILE_M, TILE_N, TILE_K = 64, 256, 128


def run_torch(xq, wq, x_scale, w_scale, dtype=dtypes.bf16):
    out = F.linear(xq.to(dtypes.fp32) * x_scale, wq.to(dtypes.fp32) * w_scale)
    return out.to(dtype)


def _flydsl(xq, wsh, x_scale, w_scale, waves_per_eu, cshuffle):
    out = torch.empty(xq.shape[0], wsh.shape[0], dtype=dtypes.bf16)
    saved = gemm_kernels._CSHUFFLE_TILES_GFX942
    if not cshuffle:
        gemm_kernels._CSHUFFLE_TILES_GFX942 = set()
    try:
        return gemm_kernels.flydsl_preshuffle_gemm_a8(
            xq, wsh, x_scale, w_scale, out, TILE_M, TILE_N, TILE_K, 0, waves_per_eu
        )
    finally:
        gemm_kernels._CSHUFFLE_TILES_GFX942 = saved


@benchmark()
def test_gemm_a8w8_bpreshuffle_flydsl_cshuffle(m, n, k, waves_per_eu):
    x = torch.randn(m, k, dtype=dtypes.bf16)
    w = torch.randn(n, k, dtype=dtypes.bf16) * 0.05
    xq, x_scale = aiter.per_token_quant_hip(x, quant_dtype=dtypes.fp8)
    wq, w_scale = aiter.per_token_quant_hip(w, quant_dtype=dtypes.fp8)
    wsh = shuffle_weight(wq, (16, 16))
    ref = run_torch(xq, wq, x_scale, w_scale)

    # The model's AG consumer writes each block into a row slice of a larger buffer.
    big = torch.full((m + 8, n), float("nan"), dtype=dtypes.bf16)
    candidates = {
        "flydsl": lambda: _flydsl(xq, wsh, x_scale, w_scale, waves_per_eu, False),
        "flydsl_cshuffle": lambda: _flydsl(
            xq, wsh, x_scale, w_scale, waves_per_eu, True
        ),
        "bpreshuffle": lambda: aiter.gemm_a8w8_bpreshuffle(
            xq, wsh, x_scale, w_scale, None, dtypes.bf16
        ),
        "bpreshuffle_out": lambda: aiter.gemm_a8w8_bpreshuffle(
            xq, wsh, x_scale, w_scale, None, dtypes.bf16, out=big[:m]
        ),
    }

    flops = 2 * m * n * k
    nbytes = m * k * xq.element_size() + n * k * wq.element_size() + m * n * 2
    ret = {"gfx": get_gfx()}
    outs = {}
    for name, fn in candidates.items():
        out, us = run_perftest(fn)
        outs[name] = out
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = checkAllclose(
            ref.to(dtypes.fp32),
            out.to(dtypes.fp32),
            rtol=1e-2,
            atol=1e-2,
            msg=f"{name}: a8w8 bpreshuffle",
        )
    assert torch.equal(outs["flydsl_cshuffle"], outs["flydsl"]), "C-shuffle epilogue"
    assert torch.equal(outs["bpreshuffle_out"], outs["bpreshuffle"]), "out="
    assert big[m:].isnan().all(), "out= wrote past its rows"
    return ret


def check_dispatch_gate():
    """On gfx942 only fp8 -> bf16 64x256x128 with waves_per_eu 0 or 2 (sync copy, lds_stage 2,
    no xcd swizzle) asks the compiler for the C-shuffle epilogue."""
    m, n, k = 256, 2560, 3072
    x = torch.randn(m, k, dtype=dtypes.bf16)
    w = torch.randn(n, k, dtype=dtypes.bf16) * 0.05
    xq, x_scale = aiter.per_token_quant_hip(x, quant_dtype=dtypes.fp8)
    wq, w_scale = aiter.per_token_quant_hip(w, quant_dtype=dtypes.fp8)
    wsh = shuffle_weight(wq, (16, 16))
    real = gemm_kernels._get_compile_fn()
    asked = []
    gemm_kernels._flydsl_compile_fn = lambda **kw: (
        asked.append(kw.get("cshuffle", False)) or real(**kw)
    )
    try:
        # (tile_m, tile_n, xcd_swizzle, lds_stage, waves_per_eu, out dtype, cshuffle expected)
        for tm, tn, xcd, lds, wpe, odt, expect in [
            (64, 256, 0, 2, 0, dtypes.bf16, True),
            (64, 256, 0, 2, 2, dtypes.bf16, True),
            (64, 256, 0, 2, 1, dtypes.bf16, False),
            (128, 256, 0, 2, 0, dtypes.bf16, False),
            (64, 256, 4, 2, 0, dtypes.bf16, False),
            (64, 256, 0, 1, 0, dtypes.bf16, False),
            (64, 256, 0, 2, 0, dtypes.fp16, False),
        ]:
            asked.clear()
            out = torch.empty(m, n, dtype=odt)
            gemm_kernels.flydsl_preshuffle_gemm_a8(
                xq, wsh, x_scale, w_scale, out, tm, tn, TILE_K, 0, wpe, xcd, lds
            )
            assert asked[0] == expect, (tm, tn, xcd, lds, wpe, odt)
    finally:
        gemm_kernels._flydsl_compile_fn = real


def main():
    if get_gfx() not in SUPPORTED_GFX or not is_flydsl_available():
        aiter.logger.warning(
            "FlyDSL C-shuffle epilogue unsupported on %s; skipping", get_gfx()
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    # Qwen3.8-Flash TP2 sequence-parallel prefill: qkv / in_proj_qkvz AG blocks
    # (N=6656/8192, K=2560) and o_proj / out_proj RS chunks (N=2560, K=3072); odd M included.
    parser.add_argument(
        "-s",
        "--mnk",
        type=dtypes.str2tuple,
        nargs="*",
        default=[
            (8176, 8192, 2560),
            (8176, 6656, 2560),
            (6000, 8192, 2560),
            (1089, 6656, 2560),
            (4088, 2560, 3072),
            (3000, 2560, 3072),
            (2049, 2560, 3072),
        ],
        help="shape (m, n, k), e.g. -s 4088,2560,3072",
    )
    parser.add_argument(
        "-w",
        "--waves_per_eu",
        type=int,
        nargs="*",
        default=[0, 2],
        help="FlyDSL waves_per_eu of the 64x256x128 tile",
    )
    args = parser.parse_args()

    check_dispatch_gate()
    df = [
        test_gemm_a8w8_bpreshuffle_flydsl_cshuffle(m, n, k, wpe)
        for (m, n, k), wpe in itertools.product(args.mnk, args.waves_per_eu)
    ]
    df = pd.DataFrame(df)
    aiter.logger.info(
        "gemm_a8w8_bpreshuffle FlyDSL C-shuffle summary (markdown):\n%s",
        df.to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
