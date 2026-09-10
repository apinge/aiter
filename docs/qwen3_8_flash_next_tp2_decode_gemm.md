# Qwen3.8 Flash Next TP2 Decode BF16 GEMMs

The model configuration
[`qwen3_8_flash_next_tp2_decode_bf16_tuned_gemm.csv`](../aiter/configs/model_configs/qwen3_8_flash_next_tp2_decode_bf16_tuned_gemm.csv)
adds measured BF16 GEMM choices for MI308X (`gfx942`, 80 CUs).

Without an explicit `AITER_CONFIG_GEMM_BF16` override, AITER discovers this file
through the existing `model_configs/*bf16_tuned_gemm*.csv` loader and merges it
with the other BF16 configurations. No logging flag is required.
`AITER_LOG_TUNED_CONFIG=1` only enables configuration-hit messages.
Restart existing inference processes to refresh their lookup caches and captured
graphs. The inference framework must already route the relevant linear layers
through `aiter.tuned_gemm.tgemm`, as SGLang does with `SGLANG_USE_AITER=1`.

## Shapes

The measured SGLang TP2 workload uses decode graph buckets
`M = 1, 2, 4, 8, 12, 16, 24, 32`. These cover logical request batches 1 through
32 with graph padding. The eight weight shapes below give 64 configuration rows.
All rows use BF16 inputs and outputs, no bias, no scaling, and no preshuffle.

| Projection | N | K | Calls per model forward |
|---|---:|---:|---:|
| GDN QKVZ | 8192 | 2560 | 36 |
| GDN/QSA output | 2560 | 3072 | 48 |
| QSA QKV and gate | 6656 | 2560 | 12 |
| MoE router | 512 | 2560 | 48 |
| QSA index QK | 640 | 2560 | 12 |
| PLE key | 10240 | 2560 | 1 |
| GDN BA | 48 | 2560 | 36 |
| PLE value | 2560 | 2560 | 1 |

Runtime selection is keyed by hardware, shape, and dtype, not by model name or
TP degree. TP2 identifies the workload used to obtain these shapes. Other
workloads with identical keys can also use these entries.

## Selection And Validation

Candidates were generated with `csrc/gemm_a16w16/gemm_tuner.py`: a representative
all-backend search, an ASM/Opus/Triton/skinny/Torch bucket sweep, and hipBLASLt
searches for the three main decode projections. This was not an exhaustive
FlyDSL search for every shape.

All 64 shapes passed independent tests using checkpoint weights reconstructed
for TP rank 1 and generated BF16 activations. Outputs were checked against an
FP32 reference rounded to BF16, with `rtol=0.02`, `atol=0.02`, and no mismatched
elements allowed. The largest relative L2 error against the FP32 reference was
0.0018101, including BF16 output rounding.

Independent graph-replay measurements rotated input/weight buffers and used
multiple timing samples. Only backend changes improving the median by at least
3% were retained: 34 Opus and 13 hipBLASLt entries. The remaining 17 entries
explicitly retain the original Torch/skinny choice. Keeping those entries avoids
an unmeasured fallback to a different newly tuned M bucket.

The added keys had no collisions with the existing default BF16 configuration
sources. In a fresh process with both `AITER_CONFIG_GEMM_BF16` and
`AITER_LOG_TUNED_CONFIG` unset, all 64 default lookup results matched the
validated backend, solution ID, and split-K choice. Existing source CSVs were
unchanged.

## Model Measurements

Measured with SGLang `benchmark.one_batch`, TP2, graph decode, 1024 input tokens,
96 output tokens, and seed 42. Input hashes matched across baseline and tuned
runs for all eight batches. Each entry is the median of 95 decode steps on the
same pair of GPUs, with no concurrent tuner.

| Batch | Baseline ms/step | Tuned ms/step | Reduction |
|---|---:|---:|---:|
| 1 | 15.242 | 14.799 | 2.91% |
| 2 | 16.977 | 16.556 | 2.48% |
| 4 | 19.208 | 18.688 | 2.71% |
| 8 | 23.826 | 23.074 | 3.15% |
| 12 | 28.097 | 27.520 | 2.05% |
| 16 | 31.298 | 30.914 | 1.23% |
| 24 | 38.743 | 38.203 | 1.39% |
| 32 | 44.445 | 44.032 | 0.93% |

These are model-side decode latencies from one matched-input A/B sweep, not
HTTP serving TPOT or confidence intervals over independent restarts. The
sub-percent result at batch 32 is particularly sensitive to measurement noise.
Full task accuracy, perplexity, and long-context regression tests were not run.

Environment: SGLang `21d0d512ea452a59490aa6585a42d721ef9fb18d`, AITER base
`c16d44b93a528b2a4bfd6d8d3409116d465872a9`, PyTorch
`2.12.0+rocm7.2.4.gitcf5ea6e.post2`, and hipBLASLt package
`1.2.2.70204-93~22.04`. hipBLASLt solution IDs require revalidation when the
library version changes.
