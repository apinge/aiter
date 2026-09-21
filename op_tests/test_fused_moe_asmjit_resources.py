# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host-side coverage of every shipped Qwen3.5 asmjit config and CO dependency."""

import csv
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import aiter.fused_moe_asmjit_aot as backend
from aiter import ActivationType, QuantType
from aiter.fused_moe_registry import FusedMoeRequest, resolve_fused_moe_impl
from csrc.cpp_itfs import hsaco_tools

ROOT = Path(__file__).resolve().parents[1]
CASES = []
for model in ("397b", "122b", "35b"):
    path = ROOT / (
        f"aiter/configs/model_configs/qwen3_5_{model}_fp8_ptpc_tuned_fmoe.csv"
    )
    with path.open() as handle:
        CASES.extend(csv.DictReader(handle))


def trace_case(monkeypatch, row, **request_overrides):
    """Run the real Python schedule on meta tensors, recording HIP launches."""
    launches = []

    def fake_kernel(prefix, constexpr_args=()):
        filename = (
            Path(prefix).name + "".join(f"-{k}={v}" for k, v in constexpr_args) + ".co"
        )

        def launch(grid, block, *args):
            launches.append((filename, grid, block))

        return launch

    def fake_sort(ids, weights, experts, hidden, dtype, block_m, *args):
        count = ids.numel() + experts * block_m - ids.shape[1]
        return (
            torch.empty(count, device="meta", dtype=torch.int32),
            torch.empty(count, device="meta", dtype=torch.float32),
            torch.empty(
                (count + block_m - 1) // block_m, device="meta", dtype=torch.int32
            ),
            torch.empty(2, device="meta", dtype=torch.int32),
            torch.empty((ids.shape[0], hidden), device="meta", dtype=dtype),
        )

    def fake_quant(tensor, *, scale, quant_dtype, num_rows):
        return tensor.to(quant_dtype), torch.empty(
            (*tensor.shape[:-1], 1), device="meta"
        )

    monkeypatch.setattr(hsaco_tools, "get_kernel", fake_kernel)
    monkeypatch.setattr(backend, "get_gfx", lambda: "gfx942")
    monkeypatch.setattr(backend, "moe_sorting", fake_sort)
    monkeypatch.setattr(backend.aiter, "get_hip_quant", lambda _: fake_quant)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda *args: SimpleNamespace(multi_processor_count=80),
    )
    b, h, inter, e, k = (
        int(row[key]) for key in ("token", "model_dim", "inter_dim", "expert", "topk")
    )
    w1 = torch.empty((e, 2 * inter, h), device="meta", dtype=torch.float8_e4m3fnuz)
    w2 = torch.empty((e, h, inter), device="meta", dtype=torch.float8_e4m3fnuz)
    w1.is_shuffled = w2.is_shuffled = True
    request = FusedMoeRequest(
        hidden_states=torch.empty((b, h), device="meta", dtype=torch.bfloat16),
        w1=w1,
        w2=w2,
        topk_ids=torch.empty((b, k), device="meta", dtype=torch.int32),
        topk_weight=torch.empty((b, k), device="meta", dtype=torch.float32),
        w1_scale=torch.empty((e, 2 * inter, 1), device="meta"),
        w2_scale=torch.empty((e, h, 1), device="meta"),
        activation=ActivationType.Silu,
        quant_type=QuantType.per_Token,
    )
    request = replace(request, **request_overrides)
    impl = resolve_fused_moe_impl(row["kernelName1"])
    assert impl is not None
    output = impl(request)
    assert output.shape == (b, h) and output.dtype == torch.bfloat16
    assert len(launches) == (3 if "_True_True_" in row["kernelName1"] else 2)
    return launches


@pytest.mark.parametrize("row", CASES, ids=lambda r: f"{r['model_dim']}-{r['token']}")
def test_csv_resources(monkeypatch, row):
    for filename, grid, block in trace_case(monkeypatch, row):
        path = ROOT / "hsa/gfx942/fmoe_asmjit" / filename
        assert path.is_file(), path
        with path.open("rb") as handle:
            assert handle.read(4) == b"\x7fELF", path
        assert all(v > 0 for v in grid + block)


def test_shape_coverage():
    assert len(CASES) == 48
    for hidden in (4096, 3072, 2048):
        rows = [row for row in CASES if int(row["model_dim"]) == hidden]
        assert {int(row["token"]) for row in rows} == {2**i for i in range(16)}
        assert all(row["gfx"] == "gfx942" and row["cu_num"] == "80" for row in rows)


def test_decode_limit():
    assert backend._get_decode_max_batch(513, 11, 4096, 128) == 128
    assert backend._get_decode_max_batch(257, 9, 3072, 128) == 32
    assert backend._get_decode_max_batch(257, 9, 2048, 128) == 32


@pytest.mark.parametrize(
    "config",
    [
        "16_True_False_False",
        "64_True_True_False",
        "128_True_True_False",
        "128_True_True_True",
    ],
)
def test_config_round_trip(config):
    assert backend.Config.from_string(config).to_string() == config


@pytest.mark.parametrize(
    "config",
    [
        "16_1_False_False",
        "16_True_False",
        "32_True_False_False",
        "64_True_False_False",
        "16_True_False_True",
        "64_True_True_True",
    ],
)
def test_invalid_config(config):
    with pytest.raises(ValueError):
        backend.Config.from_string(config)


@pytest.mark.parametrize(
    "overrides",
    [
        {"doweight_stage1": True},
        {"hidden_pad": 128},
        {"intermediate_pad": 128},
        {"bias1": torch.empty(1, device="meta")},
        {"bias2": torch.empty(1, device="meta")},
        {"expert_mask": torch.empty(1, device="meta")},
        {"num_local_tokens": torch.empty(1, device="meta")},
        {"a1_scale": torch.empty(1, device="meta")},
        {"a2_scale": torch.empty(1, device="meta")},
        {"dtype": torch.float16},
        {"gate_mode": "interleave"},
        {"w1_scale": None},
        {"w2_scale": torch.empty(1, device="meta", dtype=torch.bfloat16)},
        {"w2": torch.empty((513, 4096, 128), device="meta")},
    ],
)
def test_unsupported_request(monkeypatch, overrides):
    with pytest.raises(ValueError):
        trace_case(monkeypatch, CASES[0], **overrides)
