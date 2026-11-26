# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Compare router configs using 48 checkpoint weights and paired HIP graphs."""

import argparse
import gc
import json
import os
import re
import statistics
import struct
from pathlib import Path

import torch

import aiter
from aiter import tuned_gemm as tg
from aiter.jit.core import AITER_CONFIGS

VERIFY_M = [
    4,
    8,
    12,
    16,
    20,
    24,
    28,
    32,
    40,
    48,
    56,
    64,
    72,
    80,
    88,
    96,
    104,
    112,
    120,
    128,
]


def load_weights(model_path, prefix):
    index = json.loads((model_path / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    keys = [
        key
        for key in index
        if key.startswith(prefix) and key.endswith(".mlp.gate.weight")
    ]
    keys.sort(key=lambda key: int(re.search(r"\.layers\.(\d+)\.", key).group(1)))
    if len(keys) != 48:
        raise ValueError(f"Expected 48 target router weights, found {len(keys)}")
    headers = {}
    tensors = []
    for key in keys:
        shard = index[key]
        with (model_path / shard).open("rb") as stream:
            header_size = struct.unpack("<Q", stream.read(8))[0]
            if shard not in headers:
                headers[shard] = json.loads(stream.read(header_size))
            entry = headers[shard][key]
            if entry["shape"] != [512, 2560] or entry["dtype"] != "BF16":
                raise ValueError(f"Unsupported router weight: {key}: {entry}")
            start, end = entry["data_offsets"]
            stream.seek(8 + header_size + start)
            data = bytearray(stream.read(end - start))
        tensors.append(
            torch.frombuffer(data, dtype=torch.bfloat16).reshape(512, 2560).clone()
        )
    return torch.stack(tensors).contiguous()


def load_config(path):
    os.environ["AITER_CONFIG_GEMM_BF16"] = str(path)
    AITER_CONFIGS.get_config_file.cache_clear()
    tg.get_GEMM_A16W16_config_.cache_clear()
    tg.get_GEMM_A16W16_config.cache_clear()
    clear = getattr(
        getattr(tg, "get_GEMM_A16W16_tuned_config", None), "cache_clear", None
    )
    if clear is not None:
        clear()


def capture(config_path, inputs, weights):
    load_config(config_path)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            for layer in range(48):
                tg.tgemm.mm(inputs[layer], weights[layer], otype=torch.bfloat16)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outputs = [
            tg.tgemm.mm(inputs[layer], weights[layer], otype=torch.bfloat16)
            for layer in range(48)
        ]
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    return graph, outputs


def timing(graph, replays):
    for _ in range(3):
        graph.replay()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / (48 * replays)


def check(outputs, reference):
    values = torch.stack(outputs).float()
    difference = (values - reference).abs()
    violations = (difference > 0.02 + 0.02 * reference.abs()).sum().item()
    if not torch.isfinite(values).all().item() or violations:
        raise ValueError(f"Numerical check failed: {violations} tolerance violations")
    return {
        "relative_l2": ((values - reference).norm() / reference.norm()).item(),
        "violations": violations,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--router-prefix", default="model.language_model.layers.")
    parser.add_argument(
        "--baseline-csv",
        type=Path,
        required=True,
        help="Baseline BF16 tuning CSV path or colon-separated CSV paths",
    )
    parser.add_argument(
        "--candidate-csv",
        type=Path,
        required=True,
        help="Candidate BF16 tuning CSV path or colon-separated CSV paths",
    )
    parser.add_argument("--m", type=int, nargs="+", default=VERIFY_M)
    parser.add_argument(
        "--devices",
        type=int,
        nargs="+",
        default=[0, 1],
        help="HIP ordinals after visibility masking",
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--replays", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rounds < 1 or args.replays < 1:
        parser.error("rounds and replays must be positive")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    host_weights = load_weights(args.model_path, args.router_prefix)
    results = []
    for device in args.devices:
        torch.cuda.set_device(device)
        properties = torch.cuda.get_device_properties(device)
        if (
            not properties.gcnArchName.startswith("gfx942")
            or properties.multi_processor_count != 80
        ):
            raise ValueError("These configs require gfx942 / 80 CU")
        weights = host_weights.to(device)
        reference_weights = weights.float()
        for m in args.m:
            generator = torch.Generator().manual_seed(20261125 + m)
            inputs = (
                torch.randn((48, m, 2560), generator=generator)
                .to(torch.bfloat16)
                .to(device)
            )
            reference = torch.bmm(inputs.float(), reference_weights.transpose(1, 2))
            baseline_graph, baseline_outputs = capture(
                args.baseline_csv, inputs, weights
            )
            baseline_error = check(baseline_outputs, reference)
            candidate_graph, candidate_outputs = capture(
                args.candidate_csv, inputs, weights
            )
            candidate_error = check(candidate_outputs, reference)
            config = tg.get_GEMM_A16W16_config(
                m, 512, 2560, False, "torch.bfloat16", "torch.bfloat16", False, False
            )
            baseline_samples, candidate_samples, samples = [], [], []
            for round_id in range(args.rounds):
                order = "BCCB" if round_id % 2 == 0 else "CBBC"
                for position, label in enumerate(order):
                    value = timing(
                        baseline_graph if label == "B" else candidate_graph,
                        args.replays,
                    )
                    (baseline_samples if label == "B" else candidate_samples).append(
                        value
                    )
                    samples.append(
                        {
                            "round": round_id + 1,
                            "position": position,
                            "backend": label,
                            "us": value,
                        }
                    )
            baseline_us = statistics.median(baseline_samples)
            candidate_us = statistics.median(candidate_samples)
            result = {
                "gpu_ordinal": device,
                "M": m,
                "N": 512,
                "K": 2560,
                "baseline_us": baseline_us,
                "selected_us": candidate_us,
                "improvement_pct": 100 * (baseline_us - candidate_us) / baseline_us,
                "selected_libtype": config["libtype"],
                "selected_solidx": int(config["solidx"]),
                "baseline_error": baseline_error,
                "selected_error": candidate_error,
                "samples": samples,
            }
            results.append(result)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(
                    {
                        "torch": torch.__version__,
                        "hip": torch.version.hip,
                        "aiter_module": aiter.__file__,
                        "results": results,
                    },
                    indent=2,
                )
                + "\n"
            )
            print(
                f"GPU{device} M{m}: {baseline_us:.4f} -> {candidate_us:.4f} us ({result['improvement_pct']:.2f}%)",
                flush=True,
            )
            del (
                baseline_graph,
                baseline_outputs,
                candidate_graph,
                candidate_outputs,
                inputs,
                reference,
            )
            gc.collect()
            torch.cuda.empty_cache()
        del weights, reference_weights
        if tg.extensions_created:
            torch.cuda.synchronize()
            aiter.hipb_destroy_extension()
            tg.extensions_created = False
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
