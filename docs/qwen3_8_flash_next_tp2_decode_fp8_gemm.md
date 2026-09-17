# Qwen3.8 Flash Next TP2 Decode FP8 GEMMs

The model configuration
[`qwen3_8_flash_next_tp2_decode_a8w8_bpreshuffle_tuned_gemm.csv`](../aiter/configs/model_configs/qwen3_8_flash_next_tp2_decode_a8w8_bpreshuffle_tuned_gemm.csv)
adds measured FP8 (`torch.float8_e4m3fnuz`, per-token activation / per-channel
weight) GEMM choices for MI308X (`gfx942`, 80 CUs). It is the FP8 counterpart of
the BF16 table added in
[`qwen3_8_flash_next_tp2_decode_gemm.md`](qwen3_8_flash_next_tp2_decode_gemm.md);
before this file, `a8w8_bpreshuffle_tuned_gemm` had **no rows at all with
`K = 2560`**, so every dense FP8 projection in this model ran the default kernel.

Without an explicit `AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE` override, AITER
discovers this file through the existing
`model_configs/*a8w8_bpreshuffle_tuned_gemm*.csv` loader and merges it with the
other FP8 preshuffle configurations. No logging flag is required.
`AITER_LOG_TUNED_CONFIG=1` only enables configuration-hit messages. Restart
existing inference processes to refresh their lookup caches and captured graphs.

## Shapes

The measured SGLang TP2 workload uses decode graph buckets
`M = 1, 2, 4, 8, 12, 16, 24, 32`. These cover logical request batches 1 through
32 with graph padding. Three FP8 weight shapes carry the dense decode path:

| Projection | N | K | Calls per model forward |
|---|---:|---:|---:|
| GDN `in_proj_qkvz` | 8192 | 2560 | 36 |
| GDN `out_proj` / QSA `o_proj` | 2560 | 3072 | 48 |
| QSA `qkv_proj` and gate | 6656 | 2560 | 12 |

All 24 `(M, N, K)` combinations were searched. **Only the 6 that improved by at
least 3% are written to the CSV**; the other 18 are deliberately absent so those
shapes keep the default kernel (see below).

Runtime selection is keyed by hardware, shape, and dtype, not by model name or
TP degree. TP2 identifies the workload used to obtain these shapes. Other
workloads with identical keys can also use these entries.

## Selection And Validation

Candidates were generated with
`csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_tune.py`, searching
**10176 candidates** (159 ck + 265 cktile per shape) across the 24 shapes:

```
python3 csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_tune.py \
  -i untuned.csv -o tuned.csv -o2 profile.csv \
  --libtype ck,cktile --mp 2 --shape_grouped --timeout 600
```

**The FlyDSL candidates were excluded deliberately.** On this toolchain the
FlyDSL preshuffle GEMM pipeline aborts the tuning worker with
`LLVM ERROR: Do not know how to expand this operator's operand!` while expanding
`llvm.amdgcn.raw.ptr.buffer.load.lds` with an s8192 LDS store. Because the crash
kills the worker rather than the individual candidate, a `--shape_grouped` run
loses the whole group to the timeout path in `aiter/utility/mp_tuner.py`,
discarding the ck and cktile results that had already been measured. Restricting
`--libtype` to `ck,cktile` avoids the crash entirely. The `asm` family bails out
for FP8 (it only accepts `torch.int8`), so it contributes no candidates. This
table therefore represents a ck + cktile search only, not an exhaustive one.

All 24 shapes were checked against an FP32 reference with `rtol=0.05`,
`atol=0.05`. The tuner reported `errRatio = 0.0` for every retained candidate,
and the independent A/B benchmark reported a maximum elementwise error of 0.0
for both the default and the tuned kernels. All six retained entries are
`cktile`; no `hipblaslt` entry is produced by this tuner.

Independent A/B measurement ran the three projections through
`aiter.gemm_a8w8_bpreshuffle` with 201 timed iterations after 20 warmups,
switching only the CSV that `AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE` points at, and
was repeated twice end to end. **Only shapes that improved by at least 3% in
both runs were retained.** One shape (`M=32, N=2560, K=3072`) showed −28.3% in
the first run and −0.04% in the second; it was treated as a measurement outlier
and is not in the CSV.

A third run with the final 6-row CSV confirmed the intended behavior: the six
retained shapes improve 4.1–21.6%, and the other eighteen stay within ±1.5% of
the default, i.e. the `get_padded_m` retry loop does not make an unlisted `M`
match a listed one.

`python3 -m unittest op_tests.tuning_tests.test_config_shape_collision` passes
(13 tests); the added keys had no collisions with existing FP8 preshuffle
configuration sources.

## Operator Measurements

Unit: us. Lower is better. `origin` is the default kernel (no matching CSV row),
`tuned` is the retained entry. `origin` is the mean of the two baseline runs.

### GDN `out_proj` / QSA `o_proj` (N=2560, K=3072, 48 calls/forward)

| M | origin (us) | tuned (us) | Reduction |
|---|---:|---:|---:|
| 1 | 7.270 | 7.281 | -0.16% |
| 2 | 7.317 | 7.332 | -0.20% |
| **4** | **7.397** | **7.083** | **4.24%** |
| **8** | **7.398** | **7.097** | **4.07%** |
| **12** | **7.412** | **7.086** | **4.39%** |
| 16 | 6.466 | 6.536 | -1.08% |
| **24** | **7.865** | **7.503** | **4.60%** |
| 32 | 6.966 | 6.990 | -0.36% |

### QSA `qkv_proj` and gate (N=6656, K=2560, 12 calls/forward)

| M | origin (us) | tuned (us) | Reduction |
|---|---:|---:|---:|
| 1 | 8.617 | 8.747 | -1.51% |
| 2 | 8.686 | 8.697 | -0.12% |
| 4 | 8.774 | 8.773 | 0.02% |
| 8 | 9.001 | 9.044 | -0.48% |
| 12 | 9.184 | 9.201 | -0.18% |
| 16 | 9.228 | 9.334 | -1.15% |
| **24** | **11.678** | **10.867** | **6.95%** |
| 32 | 11.873 | 11.978 | -0.88% |

### GDN `in_proj_qkvz` (N=8192, K=2560, 36 calls/forward)

| M | origin (us) | tuned (us) | Reduction |
|---|---:|---:|---:|
| 1 | 8.939 | 8.945 | -0.07% |
| 2 | 8.991 | 9.041 | -0.56% |
| 4 | 9.043 | 9.037 | 0.07% |
| 8 | 9.274 | 9.283 | -0.10% |
| 12 | 9.452 | 9.472 | -0.21% |
| 16 | 9.492 | 9.609 | -1.23% |
| **24** | **14.209** | **11.147** | **21.55%** |
| 32 | 14.464 | 14.581 | -0.80% |

The gains concentrate at `M = 24`, which is the one decode bucket that is
neither a power of two nor close to one. At `M = 24` the default kernel costs
about as much as at `M = 32` (14.209 vs 14.464 us at N=8192) while `M = 16`
costs 9.492 us, which is consistent with the default path rounding `M` up to 32.
The retained `cktile` entries do not, which is where the 21.55% comes from.

The eighteen non-retained shapes are within ±1.5% of the default. Leaving them
out of the CSV keeps the default path for them, so a future improvement to that
path is not blocked by a pinned entry.

## Model Measurements

Measured with `sglang.bench_serving`, TP2, 12000 input tokens, 350 output
tokens, 32 prompts, one run per point on the same pair of GPUs.

| Concurrency | Baseline TPOT (ms) | Tuned TPOT (ms) | Baseline tok/s | Tuned tok/s |
|---|---:|---:|---:|---:|
| 1 | 9.26 | 9.30 | 2923.07 | 2911.50 |
| 8 | 20.59 | 21.08 | 7294.91 | 7198.95 |
| 24 | 43.80 | 43.22 | 8564.99 | 8677.18 |

**These model-level differences are below the measurement resolution of this
setup and should not be read as a result.** Repeated runs of the same
configuration on this machine vary by about 2%, and the expected effect is
smaller than that. Multiplying the operator gains by the call counts gives an
upper bound of 15 us/step at `M = 4/8/12` and 137 us/step at `M = 24`, i.e.
0.07% of a 20 ms step and 0.31% of a 44 ms step. The concurrency-24 row is the
only one that moves in the predicted direction, and it is the only bucket where
all three projections have a retained entry. The concurrency-8 row moves the
other way by a similar margin, which is what noise at this scale looks like.

The operator table above, not this one, is the evidence for the change.

Accuracy: gsm8k (1319 questions, chat API, 16384 max tokens) scored 0.9764
before and 0.9749 after, on the same server build. That is a difference of 2
questions out of 1319, within the sampling noise of a single run (the standard
error over 1319 samples is about 0.004), and the tuner reported an elementwise
error of 0.0 for every retained kernel. It is one run per side, not a
multi-restart confidence interval.

Full task accuracy beyond gsm8k, perplexity, and long-context regression tests
were not run.

Environment: SGLang `1dcc6866843e4ce61fda455c1d121d92c559a21e` (a branch off
`21d0d512ea452a59490aa6585a42d721ef9fb18d` carrying local QSA and
hyper-connection decode changes, which shift the absolute step times but not the
A/B), AITER base `c16d44b93a528b2a4bfd6d8d3409116d465872a9`, PyTorch
`2.12.0+rocm7.2.4.gitcf5ea6e.post2`, ROCm 7.2.4, MI308X (`gfx942`, 80 CUs).
The model-level rows additionally require the FlyDSL PTPC MoE kernels, since
without them the MoE dominates and the dense GEMM share changes. CK and CKTile
kernel selections require revalidation when those libraries change.
