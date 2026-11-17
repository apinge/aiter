# SPDX-License-Identifier: MIT
"""PTPC FP8 MoE prefill on gfx942 with the fused gate-up + per-route fp8 quant stage
(moe_gemm_2stage_gfx942 gemm1 with FUSE_QUANT).

Same contract and numerics as aiter.fused_moe's FlyDSL whole-graph prefill path
(aiter/ops/flydsl/fused_moe_gfx942.py::_run_prefill): the stage-1 kernel covers a full
intermediate row per CTA, so it quantizes its own output and the separate per-token quant
of the [M * topk, inter] activations disappears.
"""

import csv
import functools
import os

import torch

import aiter
from aiter.fused_moe import AUX_SORT_OPUS, moe_sorting
from aiter.ops.flydsl.fused_moe_gfx942 import (
    Config,
    _activation_scalars,
    _get_compiled_kernel,
    _launch,
)
from aiter.ops.flydsl.kernels.moe_gemm_2stage_gfx942 import sorted_sum

from ..moe_gemm_2stage_gfx942.gemm1 import compile_moe_gemm1

_GATEUP_BLOCK_M = 64
_GATEUP_TILE_K = 128
_TUNED_CSV = os.path.join(
    os.path.dirname(aiter.__file__),
    "configs",
    "model_configs",
    "qwen3_8_flash_next_fp8_ptpc_tuned_fmoe.csv",
)
_FLYDSL_PREFIX = "impl__flydsl_gfx942__"


@functools.cache
def _tuned_rows(csv_path: str):
    rows = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            name = r["kernelName1"]
            if not name.startswith(_FLYDSL_PREFIX):
                continue
            key = (
                int(r["model_dim"]),
                int(r["inter_dim"]),
                int(r["expert"]),
                int(r["topk"]),
            )
            rows.setdefault(key, {})[int(r["token"])] = name[len(_FLYDSL_PREFIX) :]
    return rows


def down_config(
    *, m: int, model_dim: int, inter: int, experts: int, topk: int
) -> Config:
    """The tuned whole-graph config of the smallest token bucket >= m (largest if none)."""
    table = _tuned_rows(_TUNED_CSV).get((model_dim, inter, experts, topk))
    if not table:
        raise KeyError(
            f"no tuned FlyDSL prefill row for {(model_dim, inter, experts, topk)}"
        )
    buckets = sorted(table)
    pick = next((b for b in buckets if b >= m), buckets[-1])
    cfg = Config.from_string(table[pick])
    if not cfg.use_prefill or cfg.BLOCK_M != _GATEUP_BLOCK_M:
        raise ValueError(
            f"tuned row {table[pick]!r} is not a BLOCK_M=64 prefill config"
        )
    return cfg


# Occupancy per intermediate size: inter 192 needs 260 VGPRs by default (1 wave/SIMD); capped
# at 2 waves/EU it fits 256 with no spills: gate-up + quant 0.828 -> 0.750 ms at M=12000
# (bit-exact). inter 320 holds 256 + 182 AGPRs, so no second wave fits.
_WAVES_PER_EU = {192: 2}


def _gateup_quant_kernel(*, inter: int, model_dim: int, topk: int, experts: int):
    return compile_moe_gemm1(
        WAVES_PER_EU=_WAVES_PER_EU.get(inter, 0),
        N=2 * inter,
        K=model_dim,
        weight_dtype="fp8",
        weight_quant_type="ptpc",
        TOPK=topk,
        BLOCK_TILE_SIZE_M=_GATEUP_BLOCK_M,
        BLOCK_TILE_SIZE_N=2 * inter,
        tile_k=_GATEUP_TILE_K,
        alg="prefill_1x4",
        E=experts,
        act_quant_type="ptpc",
        activation="silu",
        METADATA_TILE_SIZE_M=_GATEUP_BLOCK_M,
        FUSE_QUANT=True,
    )


def prefill_moe(
    *,
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    config: Config | None = None,
) -> torch.Tensor:
    """[M, model_dim] bf16 out = sum_k w_k * expert_k(x), bit-exact with aiter.fused_moe's
    FlyDSL prefill path for the same tuned config."""
    m, model_dim = hidden_states.shape
    experts, n13, _ = w13.shape
    inter = n13 // 2
    topk = topk_ids.shape[1]
    if config is None:
        config = down_config(
            m=m, model_dim=model_dim, inter=inter, experts=experts, topk=topk
        )
    (
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        cur_out,
        _,
        loc_ids,
    ) = moe_sorting(
        topk_ids,
        topk_weights,
        experts,
        model_dim,
        torch.bfloat16,
        config.BLOCK_M,
        None,
        None,
        0,
        output_aux=AUX_SORT_OPUS,
    )
    x_q, x_scale = aiter.get_hip_quant(aiter.QuantType.per_Token)(
        hidden_states, scale=None, quant_dtype=w13.dtype, num_rows=None
    )
    x_scale = x_scale.to(torch.float32).contiguous()

    device = w13.device
    down_in = torch.empty(m * topk, inter, dtype=w2.dtype, device=device)
    down_in_scale = torch.empty(m * topk, 1, dtype=torch.float32, device=device)
    task_num = int(sorted_expert_ids.shape[0])
    _launch(
        _gateup_quant_kernel(
            inter=inter, model_dim=model_dim, topk=topk, experts=experts
        ),
        x_q,
        w13,
        down_in,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        w13_scale,
        x_scale,
        m,
        task_num,
        *_activation_scalars("silu", 1.0, 1.0, None),
        down_in.data_ptr(),
        down_in_scale.data_ptr(),
    )
    return _run_down(
        down_in=down_in,
        down_in_scale=down_in_scale,
        w2=w2,
        w2_scale=w2_scale,
        sorted_ids=sorted_ids,
        sorted_weights=sorted_weights,
        sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids,
        loc_ids=loc_ids,
        out=cur_out,
        m=m,
        topk=topk,
        experts=experts,
        model_dim=model_dim,
        inter=inter,
        config=config,
    )


def _run_down(
    *,
    down_in,
    down_in_scale,
    w2,
    w2_scale,
    sorted_ids,
    sorted_weights,
    sorted_expert_ids,
    num_valid_ids,
    loc_ids,
    out,
    m,
    topk,
    experts,
    model_dim,
    inter,
    config,
):
    """aiter's prefill down stage + sorted top-k sum into `out`."""
    pad = config.down_output_padding_bytes
    row_size = model_dim + (pad // out.element_size() if pad is not None else 0)
    gemm2_out = torch.empty(
        [sorted_expert_ids.shape[0] * config.BLOCK_M, row_size],
        dtype=out.dtype,
        device=out.device,
    )
    down_tile_n = {"default": 128, "1x4_64x256": 256, "8x1": 128, "8x1_compact": 128}[
        config.down_path
    ]
    kernel = _get_compiled_kernel(
        N=model_dim,
        K=inter,
        weight_dtype_str="fp8",
        quant_type_str="ptpc",
        TOPK=topk,
        BLOCK_TILE_SIZE_M=config.BLOCK_M,
        BLOCK_TILE_SIZE_N=down_tile_n,
        stage="down",
        alg="prefill_1x4",
        E=experts,
        USE_ATOMIC_WRITE=False,
        act_quant_type_str="ptpc",
        down_path=config.down_path,
        down_output_padding_bytes=pad,
        METADATA_TILE_SIZE_M=config.BLOCK_M,
    )
    task_num = int(sorted_expert_ids.shape[0])
    args = (
        down_in,
        w2,
        gemm2_out,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        w2_scale,
        down_in_scale,
        m,
        task_num,
    )
    if config.down_path == "default":
        _launch(kernel, *args, *_activation_scalars("silu", 1.0, 1.0, None))
    elif config.down_path == "8x1_compact":
        from aiter.ops.flydsl.kernels.moe_gemm_2stage_gfx942.gemm2_8x1_compact import (
            allocate_task_buffers,
        )

        full_tasks, tail_tasks, task_counts = allocate_task_buffers(
            sorted_expert_ids, experts
        )
        _launch(
            kernel,
            *args,
            full_tasks,
            tail_tasks,
            task_counts,
            full_tasks.shape[0],
            tail_tasks.shape[0],
        )
    else:
        _launch(kernel, *args)
    sorted_sum(topk, model_dim, pad)(loc_ids, gemm2_out, out, m)
    return out
