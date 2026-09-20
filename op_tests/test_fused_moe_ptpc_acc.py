# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""FP8 PTPC accuracy: Qwen3.8 Flash Next FlyDSL and Qwen3.5 asmjit."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch

import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe, fused_topk
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import checkAllclose

torch.set_default_device("cuda")

ACTIVATION = ActivationType.Silu
QUANT_TYPE = QuantType.per_Token
DTYPE = torch.bfloat16
FP8_DTYPE = torch.float8_e4m3fnuz
DIFF_THR = 0.002
# Preserve the source asmjit test's BF16-reference criterion, independently of
# the stricter existing Flash Next criterion. No activation quant in this ref.
ASMJIT_DIFF_THR = 0.02
CONFIG_CSV = (
    Path(__file__).resolve().parents[1]
    / "aiter/configs/model_configs/qwen3_8_flash_next_fp8_ptpc_tuned_fmoe.csv"
)

TP_CONFIGS = {
    640: "qwen3_8_flash_next_tp1",
    320: "qwen3_8_flash_next_tp2",
    192: "qwen3_8_flash_next_tp4",
}
ASMJIT_CONFIGS = {
    (4096, 128, 513, 11): "qwen3_5_397b_tp8",
    (3072, 128, 257, 9): "qwen3_5_122b_tp8",
    (2048, 128, 257, 9): "qwen3_5_35b_tp4",
}
ASMJIT_CSVS = [
    CONFIG_CSV.with_name(f"qwen3_5_{model}_fp8_ptpc_tuned_fmoe.csv")
    for model in ("397b", "122b", "35b")
]


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    if denominator.item() == 0:
        return 0.0
    similarity = 2 * (x * y).sum() / denominator
    return float(1 - similarity)


def load_selected_cases(
    config_csv: Path,
    models: set[str] | None,
    tokens: set[int] | None,
) -> list[dict[str, Any]]:
    cases = []
    with config_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            inter_dim = int(row["inter_dim"])
            kernel_name = row["kernelName1"]
            if kernel_name.startswith("impl__asmjit_gfx942__"):
                backend = "asmjit"
                shape = tuple(
                    int(row[key])
                    for key in ("model_dim", "inter_dim", "expert", "topk")
                )
                model = ASMJIT_CONFIGS.get(shape)
            elif kernel_name.startswith("impl__flydsl_gfx942__"):
                backend = "flydsl"
                model = TP_CONFIGS.get(inter_dim)
            else:
                continue
            if model is None or (models is not None and model not in models):
                continue

            token = int(row["token"])
            if tokens is not None and token not in tokens:
                continue

            config_string = kernel_name.split("__", 2)[2]

            cases.append(
                {
                    "model": model,
                    "backend": backend,
                    "config": config_string,
                    "token": token,
                    "hidden_size": int(row["model_dim"]),
                    "inter_dim": inter_dim,
                    "expert": int(row["expert"]),
                    "topk": int(row["topk"]),
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
            (expert_output * topk_weight[token_id, topk_slot, None]).to(output.dtype),
        )
    return output


def quant_expert_weights(w_bf16: torch.Tensor):
    quant = aiter.get_torch_quant(QUANT_TYPE)
    w_fp8, w_scale = quant(w_bf16, quant_dtype=FP8_DTYPE)
    w_ref = (w_fp8.to(DTYPE) * w_scale).to(DTYPE)
    w_kernel = shuffle_weight(w_fp8.clone(), layout=(16, 16))
    return w_kernel, w_scale, w_ref


def build_weights(expert: int, inter_dim: int, hidden_size: int):
    torch.manual_seed(42)
    w1_bf16 = torch.randn(expert, inter_dim * 2, hidden_size, dtype=DTYPE)
    w2_bf16 = torch.randn(expert, hidden_size, inter_dim, dtype=DTYPE)
    w1_kernel, w1_scale, w1_ref = quant_expert_weights(w1_bf16)
    w2_kernel, w2_scale, w2_ref = quant_expert_weights(w2_bf16)
    return w1_kernel, w2_kernel, w1_scale, w2_scale, w1_ref, w2_ref


def build_inputs(
    token: int,
    expert: int,
    topk: int,
    hidden_size: int,
    *,
    shared_expert: bool = False,
):
    torch.manual_seed(token)
    hidden_states = (
        torch.randn(token, hidden_size, dtype=DTYPE, device="cuda") + 1
    ) * 0.001
    if shared_expert:
        # Keep the source asmjit test's signed routing weights. The final expert
        # is always present once per token; its weight is not assumed to be 1.
        topk_weight = torch.randn(token, topk, dtype=torch.float32, device="cuda")
        count = token * (topk - 1)
        routed = torch.randperm(expert - 1, dtype=torch.int32, device="cuda")
        routed = routed.repeat((count + expert - 2) // (expert - 1))[:count]
        shared = torch.full((token, 1), expert - 1, dtype=torch.int32, device="cuda")
        topk_ids = torch.cat((routed.reshape(token, topk - 1), shared), dim=1)
        assert ((topk_ids == expert - 1).sum(dim=1) == 1).all()
        return hidden_states, topk_weight, topk_ids
    score = torch.randn(token, expert, dtype=DTYPE, device="cuda")
    topk_weight, topk_ids = fused_topk(
        hidden_states,
        score,
        topk,
        renormalize=True,
    )
    return hidden_states, topk_weight, topk_ids


def run_backend(
    case: dict[str, Any],
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
) -> torch.Tensor:
    if case["backend"] == "asmjit":
        import aiter.fused_moe_asmjit_aot as asmjit

        # Use normal merged-CSV dispatch, not a directly forced configuration.
        # Observe the actual call so a CK/default fallback cannot pass this test.
        with patch.object(
            asmjit, "fused_moe_asmjit_aot", wraps=asmjit.fused_moe_asmjit_aot
        ) as launched:
            output = fused_moe(
                hidden_states,
                w1,
                w2,
                topk_weight,
                topk_ids,
                activation=ACTIVATION,
                quant_type=QUANT_TYPE,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
            )
        assert (
            launched.call_count == 1
        ), "default dispatch did not call asmjit exactly once"
        assert (
            launched.call_args.args[-1] == case["config"]
        ), "default dispatch selected a different asmjit config"
        return output

    from aiter.ops.flydsl.fused_moe_gfx942 import run_flydsl_moe_gfx942

    common_args = (
        hidden_states,
        w1,
        w2,
        topk_weight,
        topk_ids,
        ACTIVATION,
        QUANT_TYPE,
        w1_scale,
        w2_scale,
        None,
        None,
        0,
    )
    return run_flydsl_moe_gfx942(
        *common_args,
        config_string=case["config"],
    )


def run_case(
    case: dict[str, Any],
    weights: tuple[torch.Tensor, ...],
) -> dict[str, Any]:
    w1, w2, w1_scale, w2_scale, w1_ref, w2_ref = weights
    hidden_states, topk_weight, topk_ids = build_inputs(
        case["token"],
        case["expert"],
        case["topk"],
        case["hidden_size"],
        shared_expert=case["backend"] == "asmjit",
    )
    ref_out = get_torch_ref(
        hidden_states,
        w1_ref,
        w2_ref,
        topk_weight,
        topk_ids,
    )
    out = run_backend(
        case,
        hidden_states,
        w1,
        w2,
        topk_weight,
        topk_ids,
        w1_scale,
        w2_scale,
    )
    torch.cuda.synchronize()

    assert out is not None, f"{case['backend']} returned no output"
    assert out.shape == ref_out.shape, f"{out.shape=} {ref_out.shape=}"
    assert torch.isfinite(out).all(), f"{case['backend']} output has NaN/Inf"

    if case["backend"] == "asmjit":
        # Diagnostic only: the source asmjit acceptance metric is calc_diff.
        # Avoid printing a misleading checkAllclose failure for this metric.
        mismatch_ratio = float(
            (~torch.isclose(ref_out, out, rtol=1e-2, atol=1e-2)).float().mean()
        )
    else:
        mismatch_ratio = checkAllclose(
            ref_out,
            out,
            rtol=1e-2,
            atol=1e-2,
            msg=(
                f"{case['model']} token={case['token']} " f"{case['backend']} vs Torch"
            ),
        )
    diff = calc_diff(ref_out, out)
    assert case["backend"] == "asmjit" or mismatch_ratio == 0, (
        f"{case['model']} token={case['token']} {case['backend']} "
        f"mismatch_ratio={mismatch_ratio:.6f}"
    )
    diff_thr = ASMJIT_DIFF_THR if case["backend"] == "asmjit" else DIFF_THR
    assert diff <= diff_thr, (
        f"{case['model']} token={case['token']} {case['backend']} "
        f"diff={diff:.6f} > {diff_thr}"
    )
    aiter.logger.info(
        "%s token=%d backend=%s config=%s diff=%.6f mismatch_ratio=%.6f PASS",
        case["model"],
        case["token"],
        case["backend"],
        case["config"],
        diff,
        mismatch_ratio,
    )
    return {**case, "diff": diff, "mismatch_ratio": mismatch_ratio, "status": "PASS"}


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--config",
    type=Path,
    default=None,
    help="Select cases from one CSV; default: Flash Next and all three Qwen3.5 CSVs",
)
parser.add_argument(
    "-m",
    "--model",
    choices=list(TP_CONFIGS.values()) + list(ASMJIT_CONFIGS.values()),
    nargs="*",
    default=None,
)
parser.add_argument("-t", "--tokenNum", type=int, nargs="*", default=None)
parser.add_argument("--backend", choices=("all", "flydsl", "asmjit"), default="all")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("skip: CUDA is not available")
        sys.exit(0)
    if get_gfx() != "gfx942":
        print(f"skip: unsupported platform {get_gfx()!r}; expected gfx942")
        sys.exit(0)

    args = parser.parse_args()
    config_files = [args.config] if args.config else [CONFIG_CSV, *ASMJIT_CSVS]
    cases = []
    for config_file in config_files:
        if not config_file.is_file():
            parser.error(f"config CSV does not exist: {config_file}")
        cases.extend(
            load_selected_cases(
                config_file,
                set(args.model) if args.model else None,
                set(args.tokenNum) if args.tokenNum else None,
            )
        )
    cases = [case for case in cases if args.backend in ("all", case["backend"])]
    if not cases:
        parser.error("no whole-graph rows matched the requested filters")

    all_results = []
    for model in sorted({case["model"] for case in cases}):
        group = [case for case in cases if case["model"] == model]
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
    print("\nMax error by TP:")
    print(
        result_df.groupby("model", as_index=False)
        .agg(
            cases=("token", "count"),
            max_diff=("diff", "max"),
            max_mismatch_ratio=("mismatch_ratio", "max"),
        )
        .to_markdown(index=False)
    )
