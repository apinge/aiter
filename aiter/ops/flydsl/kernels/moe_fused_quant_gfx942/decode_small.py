# SPDX-License-Identifier: MIT
"""Small-M decode MoE for gfx942 PTPC FP8: aiter's batch1 gate-up / down kernels with the
output clear fused into gate-up (one fill kernel less per layer), used where batch1 beats
the tuned whole-graph path.

Measured on MI308X (Qwen3.8 flash-next, E=513, top-11, H=2560), us per call:
  inter 320: M<=8 batch1+fused clear 14.4-61.2 vs tuned 16.2-63.4; M=12 tie; M>=16 slower.
  inter 192: M<=12 11.1-58.4 vs 14.2-68.4; M>=16 slower.
"""

import torch

from aiter import QuantType
from aiter.ops.flydsl import fused_moe_gfx942 as _f

# Largest M where the fused-clear batch1 path wins, per padded intermediate size.
_MAX_M = {320: 8, 192: 12}


def decode_small_supported(m: int, inter: int) -> bool:
    return m <= _MAX_M.get(inter, 0)


def decode_moe_small(
    *, hidden_states, w13, w13_scale, w2, w2_scale, topk_weights, topk_ids
):
    """[M, H] bf16 = sum_k w_k expert_k(x), silu, PTPC fp8 weights, bf16 activations.
    Capturable in a CUDA graph (no host sync)."""
    prob = _f._Problem.from_inputs(
        hidden_states, w13, w2, topk_ids, QuantType.per_Token
    )
    tw = topk_weights if topk_weights.dtype == torch.float32 else topk_weights.float()
    gemm1_out = _f._gateup_output(hidden_states, prob)
    out = torch.empty(
        [prob.batch, prob.model_dim],
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    act = _f._activation_scalars("silu", 1.0, 1.0, None)
    gate_up = _f._get_compiled_kernel(
        N=prob.gateup_dim,
        K=prob.hidden_dim,
        weight_dtype_str="fp8",
        quant_type_str=prob.quant_type,
        TOPK=prob.topk,
        BLOCK_TILE_SIZE_M=16,
        BLOCK_TILE_SIZE_N=32,
        stage="gateup",
        alg="batch1",
        E=None,
        activation_str="silu",
        fused_down_clear=True,
    )
    # With fused_down_clear the gate-up kernel zeroes `out` (its 5th argument) for down.
    _f._launch(
        gate_up,
        hidden_states,
        w13,
        gemm1_out,
        topk_ids,
        out,
        w13_scale,
        prob.batch,
        *act
    )
    down = _f._get_compiled_kernel(
        N=prob.model_dim,
        K=prob.inter_dim,
        weight_dtype_str="fp8",
        quant_type_str=prob.quant_type,
        TOPK=prob.topk,
        BLOCK_TILE_SIZE_M=16,
        BLOCK_TILE_SIZE_N=64,
        stage="down",
        alg="batch1",
        E=None,
    )
    _f._launch(down, gemm1_out, w2, out, topk_ids, tw, w2_scale, prob.batch, *act)
    return out
