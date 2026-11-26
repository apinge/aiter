# Qwen3.8 Flash Next TP2 MTP router GEMM tuning

With EAGLE/MTP steps=3, topk=1 and draft_tokens=4, target verify runs four token rows per graph request slot. HTTP concurrency up to 32 therefore reaches router M120/M128, beyond ordinary decode M1–32. This change retains the existing small-M Opus configs, adds Opus 10305 at M72/M80, and selects hipBLASLt solution 90312 at M88/M96/M104/M112/M120/M128.

The router is replicated across TP ranks: `X[M,2560] @ W[512,2560].T -> Y[M,512]`, BF16 input/weight/output, no bias, scales or preshuffle. N512 is the routed-expert projection; the shared-expert gate is separate.

## Results

Measured on MI308X, gfx942, 80 CU, physical GPUs 2 and 3. The baseline is `qwen38-flash-next-dev` commit `1cf62092f39fac89d9cdf54e2dc8978cbb4591f2`. Each candidate was evaluated through this branch's production `tgemm.mm` dispatch using all 48 checkpoint router weights. Each GPU ran three paired BCCB/CBBC/BCCB rounds, six samples per backend, with 100 graph replays per sample. Every graph contains 48 GEMMs with distinct layer weights (120 MiB total), and the result is divided by 48. Inputs are generated BF16 activations, not captured serving hidden states.

The table uses the mean of the two GPU medians. A configuration change is accepted only when all three rounds improve on both GPUs. Unchanged rows use the measured baseline for both columns.

| Verify M | Graph request slots | Baseline | Selected | Baseline µs | Selected µs | Improvement |
| ---: | ---: | --- | --- | ---: | ---: | ---: |
| 4 | 1 | opus | opus 10300 | 4.2346 | 4.2346 | 0.00% |
| 8 | 2 | opus | opus 10300 | 4.2803 | 4.2803 | 0.00% |
| 12 | 3 | opus | opus 10300 | 4.2857 | 4.2857 | 0.00% |
| 16 | 4 | opus | opus 10300 | 5.0046 | 5.0046 | 0.00% |
| 20 | 5 | opus | opus 10300 | 4.3498 | 4.3498 | 0.00% |
| 24 | 6 | opus | opus 10300 | 4.3547 | 4.3547 | 0.00% |
| 28 | 7 | opus | opus 10300 | 4.3594 | 4.3594 | 0.00% |
| 32 | 8 | opus | opus 10300 | 4.9366 | 4.9366 | 0.00% |
| 40 | 10 | opus | opus 10305 | 5.3525 | 5.3525 | 0.00% |
| 48 | 12 | opus | opus 10305 | 5.3748 | 5.3748 | 0.00% |
| 56 | 14 | opus | opus 10305 | 5.3986 | 5.3986 | 0.00% |
| 64 | 16 | opus | opus 10305 | 5.4034 | 5.4034 | 0.00% |
| 72 | 18 | torch | opus 10305 | 9.9212 | 5.4297 | 45.27% |
| 80 | 20 | torch | opus 10305 | 9.9503 | 5.4372 | 45.36% |
| 88 | 22 | torch | hipblaslt 90312 | 9.9812 | 8.0215 | 19.63% |
| 96 | 24 | torch | hipblaslt 90312 | 9.7794 | 7.9046 | 19.17% |
| 104 | 26 | torch | hipblaslt 90312 | 10.5719 | 8.0048 | 24.28% |
| 112 | 28 | torch | hipblaslt 90312 | 10.4458 | 8.0400 | 23.03% |
| 120 | 30 | torch | hipblaslt 90312 | 9.9881 | 8.0571 | 19.33% |
| 128 | 32 | torch | hipblaslt 90312 | 9.8073 | 7.9444 | 18.99% |

The retained baseline buckets through M64 already use Opus. The five WKC tile variants were compared independently of profiler ranking. M72/M80 favor Opus 10305; larger tested buckets favor the explicitly tuned hipBLASLt solution. `90312` launches the Tensile kernel containing `MT32x32x256_MI32x32x1`.

An upstream AITER checkout was used to enumerate hipBLASLt solutions: 2,791 per target C32 shape, followed by an accurate pass over the 20 fastest. Its framework path was not used to claim an improvement. The candidates were validated in this fork against the same PyTorch/hipBLASLt stack. Solution 90312 was additionally tested across every verify M in the table.

## Correctness and coverage

All 240 candidate/GPU/shape checks passed against FP32 reference with `rtol=0.02, atol=0.02`, no tolerance violations and finite outputs. BF16 accumulation/rounding may differ from baseline, so this is not a bitwise-equivalence claim.

The new model config has eight optimized rows and 56 exact-M baseline guards covering other integer M in 65–128. AITER tries exact M, a multiple-of-16 M and a power-of-two M; without guards, adding M80 or M128 would change dispatch for untested neighbors. Guard rows preserve Torch and intentionally leave timing/error/performance fields empty. The existing configs through M64 are unchanged. The normal loader was audited for M1–256: only M72/M80/M88/M96/M104/M112/M120/M128 changed, and 248 M retained baseline dispatch. The cross-family config-collision suite passed all 14 tests.

The library tested is `libhipblaslt.so.1.4`, SHA256 `39e8f4d3db3d690f964e8971b421fae8cd64a9e6398a6ea688354be8bd8fc122`, with PyTorch `2.12.0+rocm7.14.0.git6bbd260`, HIP `7.14.60850`. Revalidate solution IDs when the hipBLASLt library changes.

## Reproduce

Run from the repository root with the matching checkpoint and two idle MI308X GPUs:

```bash
rocm-smi --showuse --showmemuse
HIP_VISIBLE_DEVICES=2,3 CUDA_VISIBLE_DEVICES=2,3 \
PYTHONPATH="$PWD" GPU_ARCHS=gfx942 \
python3 op_tests/bench_qwen38_mtp_router_gemm.py \
  --model-path /path/to/Qwen3.8-Flash-Next-PTPC-FP8-PLE-BF16 \
  --rounds 3 --replays 100 --output /tmp/qwen38_router_graph.json
```

The script reads only the checkpoint index, safetensors headers and router tensors. It compares the committed baseline and selected config CSVs through the normal runtime loader. `--m 72 80 120 128` narrows the shapes. `--devices` uses ordinals after visibility masking. The default router prefix is `model.language_model.layers.`; use `--router-prefix` if checkpoint names differ.

The new config is discovered automatically under `aiter/configs/model_configs/` on the next process start. An explicit `AITER_CONFIG_GEMM_BF16` overrides automatic discovery; the complete router-only configs below can be used for operator A/B. Do not use a router-only CSV as a serving baseline for other projections.

## Evidence

- [Software, hardware and method](perf/qwen38_tp2_mtp_router/manifest.json)
- [20-shape comparison](perf/qwen38_tp2_mtp_router/comparison.csv)
- [All candidate comparisons](perf/qwen38_tp2_mtp_router/candidate_comparison.csv)
- [Paired timing samples for accepted changes](perf/qwen38_tp2_mtp_router/paired_samples.csv)
- [Baseline router config](perf/qwen38_tp2_mtp_router/baseline_router.csv)
- [Selected router config](perf/qwen38_tp2_mtp_router/selected_router.csv)

These are operator graph results. They do not include a full model or serving TPOT measurement; the 48-router graph omits the attention, MoE and communication work between layers.
