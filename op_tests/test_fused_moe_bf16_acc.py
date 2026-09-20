# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Accuracy test for Qwen3.5 125B BF16 FlyDSL and CK MoE kernels."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any

CONFIG_CSV = (
    Path(__file__).resolve().parents[1]
    / "aiter/configs/model_configs/qwen3_8_flash_next_bf16_tuned_fmoe.csv"
)

# Bind fused_moe's process-lifetime config cache before importing aiter.
os.environ["AITER_CONFIG_FMOE"] = str(CONFIG_CSV)

import torch

import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe, fused_topk, get_2stage_cfgs, get_padded_M
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import checkAllclose


torch.set_default_device("cuda")

ACTIVATION = ActivationType.Silu
QUANT_TYPE = QuantType.No
DTYPE = torch.bfloat16
DIFF_THR = 0.001

TP_CONFIGS = {
    640: "qwen3_5_125b_tp1",
    320: "qwen3_5_125b_tp2",
    192: "qwen3_5_125b_tp4",
}


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    if denominator.item() == 0:
        return 0.0
    similarity = 2 * (x * y).sum() / denominator
    return float(1 - similarity)


def load_selected_cases(
    backends: set[str],
    models: set[str] | None,
    tokens: set[int] | None,
) -> list[dict[str, Any]]:
    cases = []
    with CONFIG_CSV.open(newline="") as handle:
        for row in csv.DictReader(handle):
            inter_dim = int(row["inter_dim"])
            model = TP_CONFIGS.get(inter_dim)
            if model is None or (models is not None and model not in models):
                continue

            token = int(row["token"])
            if tokens is not None and token not in tokens:
                continue

            kernel_name1 = row["kernelName1"]
            backend = (
                "flydsl"
                if kernel_name1.startswith("impl__flydsl_gfx942__")
                else "ck"
            )
            if backend not in backends:
                continue

            cases.append(
                {
                    "model": model,
                    "backend": backend,
                    "token": token,
                    "hidden_size": int(row["model_dim"]),
                    "inter_dim": inter_dim,
                    "expert": int(row["expert"]),
                    "topk": int(row["topk"]),
                    "kernel_name1": kernel_name1,
                    "kernel_name2": row["kernelName2"],
                }
            )
    return sorted(
        cases,
        key=lambda case: (
            -case["inter_dim"],
            case["token"],
            case["backend"],
        ),
    )


def get_torch_ref(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    batch_size, hidden_size = hidden_states.shape
    num_experts, n1, _ = w1.shape
    inter_dim = n1 // 2
    output = torch.zeros(
        batch_size,
        hidden_size,
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    expert_mask = torch.nn.functional.one_hot(
        topk_ids.to(torch.long), num_classes=num_experts
    ).permute(2, 1, 0)
    for expert_id in range(num_experts):
        topk_slot, token_id = torch.where(expert_mask[expert_id])
        if token_id.numel() == 0:
            continue
        gate_proj = w1[expert_id, :inter_dim].t()
        up_proj = w1[expert_id, inter_dim:].t()
        down_proj = w2[expert_id].t()
        hidden = hidden_states[token_id]
        expert_output = (
            torch.nn.functional.silu(hidden @ gate_proj) * (hidden @ up_proj)
        ) @ down_proj
        output.index_add_(
            0,
            token_id,
            (expert_output * topk_weight[token_id, topk_slot, None]).to(
                output.dtype
            ),
        )
    return output


def build_weights(expert: int, inter_dim: int, hidden_size: int):
    torch.manual_seed(42)
    w1_ref = torch.randn(expert, inter_dim * 2, hidden_size, dtype=DTYPE)
    w2_ref = torch.randn(expert, hidden_size, inter_dim, dtype=DTYPE)
    w1_kernel = shuffle_weight(w1_ref.clone(), layout=(16, 16))
    w2_kernel = shuffle_weight(w2_ref.clone(), layout=(16, 16))
    return w1_kernel, w2_kernel, w1_ref, w2_ref


def build_inputs(token: int, expert: int, topk: int, hidden_size: int):
    torch.manual_seed(token)
    hidden_states = (
        torch.randn(token, hidden_size, dtype=DTYPE, device="cuda") + 1
    ) * 0.001
    score = torch.randn(token, expert, dtype=DTYPE, device="cuda")
    topk_weight, topk_ids = fused_topk(
        hidden_states,
        score,
        topk,
        renormalize=True,
    )
    return hidden_states, topk_weight, topk_ids


def _partial_kernel_name(func) -> str:
    return str(getattr(func, "keywords", {}).get("kernelName", ""))


def validate_dispatch(case: dict[str, Any]) -> None:
    metadata = get_2stage_cfgs(
        get_padded_M(case["token"]),
        case["hidden_size"],
        case["inter_dim"],
        case["expert"],
        case["topk"],
        DTYPE,
        DTYPE,
        DTYPE,
        QUANT_TYPE,
        True,
        ACTIVATION,
        False,
        0,
        0,
        True,
        opus_weights_shuffled=True,
    )

    if case["backend"] == "flydsl":
        assert metadata.full_impl is not None, (
            f"{case['model']} token={case['token']} did not select FlyDSL"
        )
        return

    assert metadata.full_impl is None, (
        f"{case['model']} token={case['token']} unexpectedly selected FlyDSL"
    )
    actual_kernel1 = _partial_kernel_name(metadata.stage1)
    actual_kernel2 = _partial_kernel_name(metadata.stage2)
    assert actual_kernel1 == case["kernel_name1"], (
        f"stage1 mismatch: expected={case['kernel_name1']!r} "
        f"actual={actual_kernel1!r}"
    )
    assert actual_kernel2 == case["kernel_name2"], (
        f"stage2 mismatch: expected={case['kernel_name2']!r} "
        f"actual={actual_kernel2!r}"
    )


def run_case(
    case: dict[str, Any],
    weights: tuple[torch.Tensor, ...],
) -> dict[str, Any]:
    w1, w2, w1_ref, w2_ref = weights
    hidden_states, topk_weight, topk_ids = build_inputs(
        case["token"],
        case["expert"],
        case["topk"],
        case["hidden_size"],
    )
    ref_out = get_torch_ref(
        hidden_states,
        w1_ref,
        w2_ref,
        topk_weight,
        topk_ids,
    )

    validate_dispatch(case)
    out = fused_moe(
        hidden_states,
        w1,
        w2,
        topk_weight,
        topk_ids,
        activation=ACTIVATION,
        quant_type=QUANT_TYPE,
    )
    torch.cuda.synchronize()

    assert out is not None, f"{case['backend']} returned no output"
    assert out.shape == ref_out.shape, f"{out.shape=} {ref_out.shape=}"
    assert torch.isfinite(out).all(), f"{case['backend']} output has NaN/Inf"

    mismatch_ratio = checkAllclose(
        ref_out,
        out,
        rtol=1e-2,
        atol=1e-2,
        msg=(
            f"{case['model']} token={case['token']} "
            f"{case['backend']} vs Torch"
        ),
    )
    diff = calc_diff(ref_out, out)
    assert mismatch_ratio == 0, (
        f"{case['model']} token={case['token']} {case['backend']} "
        f"mismatch_ratio={mismatch_ratio:.6f}"
    )
    assert diff <= DIFF_THR, (
        f"{case['model']} token={case['token']} {case['backend']} "
        f"diff={diff:.6f} > {DIFF_THR}"
    )
    aiter.logger.info(
        "%s token=%d backend=%s diff=%.6f mismatch_ratio=%.6f PASS",
        case["model"],
        case["token"],
        case["backend"],
        diff,
        mismatch_ratio,
    )
    return {**case, "diff": diff, "mismatch_ratio": mismatch_ratio, "status": "PASS"}


parser = argparse.ArgumentParser(
    description="Qwen3.5 125B BF16 FlyDSL/CK MoE accuracy test"
)
parser.add_argument(
    "--backend",
    choices=["flydsl", "ck"],
    nargs="*",
    default=["flydsl", "ck"],
)
parser.add_argument(
    "-m",
    "--model",
    choices=list(TP_CONFIGS.values()),
    nargs="*",
    default=None,
)
parser.add_argument("-t", "--tokenNum", type=int, nargs="*", default=None)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("skip: CUDA is not available")
        sys.exit(0)
    if get_gfx() != "gfx942":
        print(f"skip: unsupported platform {get_gfx()!r}; expected gfx942")
        sys.exit(0)
    if not CONFIG_CSV.is_file():
        parser.error(f"config CSV does not exist: {CONFIG_CSV}")

    args = parser.parse_args()
    cases = load_selected_cases(
        set(args.backend),
        set(args.model) if args.model else None,
        set(args.tokenNum) if args.tokenNum else None,
    )
    if not cases:
        parser.error("no BF16 FlyDSL or CK rows matched the requested filters")

    all_results = []
    for inter_dim in sorted({case["inter_dim"] for case in cases}, reverse=True):
        group = [case for case in cases if case["inter_dim"] == inter_dim]
        first = group[0]
        weights = build_weights(
            first["expert"], first["inter_dim"], first["hidden_size"]
        )
        for case in group:
            all_results.append(run_case(case, weights))
        del weights
        torch.cuda.empty_cache()

    import pandas as pd

    result_df = pd.DataFrame(all_results)
    print(result_df.to_markdown(index=False))
    print("\nMax error by TP/backend:")
    print(
        result_df.groupby(["model", "backend"], as_index=False)
        .agg(
            cases=("token", "count"),
            max_diff=("diff", "max"),
            max_mismatch_ratio=("mismatch_ratio", "max"),
        )
        .to_markdown(index=False)
    )
