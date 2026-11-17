# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942 PTPC FP8 MoE with fused quant (aiter.ops.flydsl.kernels.moe_fused_quant_gfx942) vs
aiter.fused_moe.

prefill_moe must be bit-exact with fused_moe's FlyDSL prefill path (its gate-up kernel only
fuses the per-token fp8 requant); decode_moe_small must match within the bf16 atomic
accumulation noise of aiter's batch1 down kernel. Shapes are Qwen3.8 flash-next
(H=2560, 512 routed + 1 fused shared expert, top-10 + shared): inter 320 is TP2, inter 192
is TP4 (160 padded to 192).
"""

import argparse
import itertools

import pandas as pd
import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942"]
H, E, TOPK_ROUTED, SHARED_ID = 2560, 513, 10, 512
FP8 = dtypes.fp8


def _ptpc_weight(e, n, k, gen):
    w = torch.randn(e, n, k, generator=gen) * 0.05
    scale = w.abs().amax(dim=-1, keepdim=True) / 240.0
    return (w / scale).clamp(-240, 240).to(FP8), scale.float()


def make_weights(inter):
    gen = torch.Generator(device="cuda").manual_seed(0)
    w13, s13 = _ptpc_weight(E, 2 * inter, H, gen)
    w2, s2 = _ptpc_weight(E, H, inter, gen)
    w13, w2 = shuffle_weight(w13, (16, 16)), shuffle_weight(w2, (16, 16))
    w13.is_shuffled = True
    w2.is_shuffled = True
    return w13, s13, w2, s2


def make_inputs(m, seed=1):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, H, generator=gen, dtype=dtypes.bf16)
    probs = torch.softmax(torch.randn(m, 512, generator=gen), -1)
    w, ids = torch.topk(probs, TOPK_ROUTED, dim=-1)
    ids = torch.cat([ids, torch.full((m, 1), SHARED_ID)], 1)
    w = torch.cat([w, torch.rand(m, 1, generator=gen)], 1)
    return x, w.float().contiguous(), ids.to(dtypes.i32).contiguous()


def run_fused_moe(x, w, ids, w13, s13, w2, s2):
    # Reference: the production aiter MoE that the fused-quant path replaces.
    return fused_moe(
        hidden_states=x,
        w1=w13,
        w2=w2,
        topk_weight=w,
        topk_ids=ids,
        quant_type=QuantType.per_Token,
        activation=ActivationType.Silu,
        w1_scale=s13,
        w2_scale=s2,
    )


def _roofline(ret, name, us, m, inter, experts_touched, err):
    flops = 2 * m * (TOPK_ROUTED + 1) * 3 * H * inter
    nbytes = (
        experts_touched * 3 * H * inter + 2 * m * H * 2
    )  # fp8 weights + bf16 in/out
    ret[f"{name} us"] = us
    ret[f"{name} TFLOPS"] = flops / us / 1e6
    ret[f"{name} TB/s"] = nbytes / us / 1e6
    ret[f"{name} err"] = err


@benchmark()
def test_prefill(m, inter):
    from aiter.ops.flydsl.kernels.moe_fused_quant_gfx942.moe import prefill_moe

    w13, s13, w2, s2 = make_weights(inter)
    x, w, ids = make_inputs(m)
    ref = run_fused_moe(x, w, ids, w13, s13, w2, s2)
    candidates = {
        "fused_moe": lambda: run_fused_moe(x, w, ids, w13, s13, w2, s2),
        "fused_quant": lambda: prefill_moe(
            hidden_states=x,
            w13=w13,
            w13_scale=s13,
            w2=w2,
            w2_scale=s2,
            topk_weights=w,
            topk_ids=ids,
        ),
    }
    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        out, us = run_perftest(fn, num_iters=21)
        # Bit-exact contract: same sort, quant formula, GEMM tiles and reduction order.
        err = checkAllclose(
            ref.to(dtypes.fp32),
            out.to(dtypes.fp32),
            rtol=0,
            atol=0,
            tol_err_ratio=0,
            msg=f"{name}: prefill m={m} inter={inter}",
        )
        _roofline(ret, name, us, m, inter, E, err)
    return ret


@benchmark()
def test_decode(m, inter):
    from aiter.ops.flydsl.kernels.moe_fused_quant_gfx942 import decode_small

    w13, s13, w2, s2 = make_weights(inter)
    x, w, ids = make_inputs(m)
    ref = run_fused_moe(x, w, ids, w13, s13, w2, s2)

    def fused_quant():
        return decode_small.decode_moe_small(
            hidden_states=x,
            w13=w13,
            w13_scale=s13,
            w2=w2,
            w2_scale=s2,
            topk_weights=w,
            topk_ids=ids,
        )

    # Decode runs under CUDA graphs: capture once and check the replayed result.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out_graph = fused_quant()
    graph.replay()
    checkAllclose(
        ref.to(dtypes.fp32),
        out_graph.to(dtypes.fp32),
        rtol=2e-2,
        atol=2e-2,
        msg=f"fused_quant graph replay: decode m={m} inter={inter}",
    )
    candidates = {
        "fused_moe": lambda: run_fused_moe(x, w, ids, w13, s13, w2, s2),
        "fused_quant": fused_quant,
    }
    ret = {"gfx": get_gfx()}
    touched = int(torch.unique(ids).numel())
    for name, fn in candidates.items():
        out, us = run_perftest(fn)
        # batch1 down accumulates the top-k in bf16 atomics: ~3e-3 relative noise.
        err = checkAllclose(
            ref.to(dtypes.fp32),
            out.to(dtypes.fp32),
            rtol=2e-2,
            atol=2e-2,
            msg=f"{name}: decode m={m} inter={inter}",
        )
        _roofline(ret, name, us, m, inter, touched, err)
    return ret


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "moe_fused_quant_gfx942 unsupported on %s; skipping", get_gfx()
        )
        return
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-i",
        "--inter",
        type=int,
        nargs="*",
        default=[320, 192],
        help="per-rank intermediate size: 320 = TP2, 192 = TP4",
    )
    parser.add_argument(
        "-m",
        "--prefill_tokens",
        type=int,
        nargs="*",
        default=[3000, 6000, 12000, 16384],
        help="prefill token counts",
    )
    parser.add_argument(
        "--decode_tokens",
        type=int,
        nargs="*",
        default=[1, 4, 8, 12],
        help="decode token counts (only those decode_small_supported accepts run)",
    )
    args = parser.parse_args()

    from aiter.ops.flydsl.kernels.moe_fused_quant_gfx942.decode_small import (
        decode_small_supported,
    )

    df = [
        test_prefill(m, i)
        for i, m in itertools.product(args.inter, args.prefill_tokens)
    ]
    aiter.logger.info(
        "moe_fused_quant prefill summary (markdown):\n%s",
        pd.DataFrame(df).to_markdown(index=False),
    )
    df = [
        test_decode(m, i)
        for i, m in itertools.product(args.inter, args.decode_tokens)
        if decode_small_supported(m, i)
    ]
    aiter.logger.info(
        "moe_fused_quant decode summary (markdown):\n%s",
        pd.DataFrame(df).to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
