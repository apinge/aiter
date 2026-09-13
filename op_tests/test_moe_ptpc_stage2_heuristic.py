# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""FP8 PTPC stage2 defaults must match shuffle_weight's 16-element K pack.

Run with a rebuilt CK MoE module on gfx942:
    pytest -q op_tests/test_moe_ptpc_stage2_heuristic.py

Calls the CK default dispatcher directly (empty kernelName), without a tuning
CSV. Compare against the same quantized operands evaluated in FP32. K=64/192
also exercise shapes that cannot use a KPerBlock=128 replacement.
"""

import pytest
import torch

import aiter
from aiter.fused_moe import moe_sorting
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.shuffle import shuffle_weight


def _quantize(x):
    scale = x.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / 240
    return (x.float() / scale).clamp(-240, 240).to(torch.float8_e4m3fnuz), scale


@pytest.mark.parametrize("inter_dim", [64, 128, 192, 256])
@pytest.mark.parametrize("block_m", [16, 32, 64, 128, 256])
@pytest.mark.parametrize("parameter_wrapped", [False, True])
def test_ptpc_stage2_default_preshuffle(inter_dim, block_m, parameter_wrapped):
    if not torch.cuda.is_available() or get_gfx() != "gfx942":
        pytest.skip("This regression covers the gfx942 FP8 PTPC CK heuristic")

    torch.manual_seed(42)
    m, n, e, topk = 3, 128, 17, 3
    xq, xs = _quantize(torch.randn(m, topk, inter_dim, device="cuda") * 0.25)
    wq, ws = _quantize(torch.randn(e, n, inter_dim, device="cuda") * 0.02)
    w2 = shuffle_weight(wq, (16, 16))
    if parameter_wrapped:
        w2 = torch.nn.Parameter(w2, requires_grad=False)
        assert not hasattr(w2, "is_shuffled")
    w1 = torch.empty(e, 2 * inter_dim, n, dtype=wq.dtype, device="cuda")
    ids = torch.tensor(
        [[1, 3, 7], [2, 5, 11], [0, 6, 13]], device="cuda", dtype=torch.int32
    )
    weights = torch.tensor([[0.2, 0.3, 0.5]] * m, device="cuda")
    sorted_ids, sorted_weights, sorted_experts, valid, out = moe_sorting(
        ids, weights, e, n, torch.bfloat16, block_size=block_m
    )
    aiter.ck_moe_stage2_fwd(
        xq,
        w1,
        w2,
        sorted_ids,
        sorted_experts,
        valid,
        out,
        topk,
        kernelName="",
        w2_scale=ws,
        a2_scale=xs,
        block_m=block_m,
        sorted_weights=sorted_weights,
        quant_type=aiter.QuantType.per_Token,
        activation=aiter.ActivationType.Silu,
    )

    ref = torch.zeros(m, n, device="cuda")
    for row in range(m):
        for slot in range(topk):
            expert = ids[row, slot].item()
            ref[row] += (
                (xq[row, slot].float() * xs[row, slot])
                @ (wq[expert].float() * ws[expert]).T
            ) * weights[row, slot]
    relative_l2 = ((out.float() - ref).norm() / ref.norm()).item()
    assert torch.isfinite(out).all(), "stage2 produced non-finite output"
    assert relative_l2 < 0.02, (
        f"K={inter_dim}, block_m={block_m}, parameter_wrapped={parameter_wrapped}, "
        f"relative_l2={relative_l2}"
    )
