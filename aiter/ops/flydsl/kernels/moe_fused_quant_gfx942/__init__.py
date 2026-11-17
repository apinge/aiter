# SPDX-License-Identifier: MIT
"""gfx942 PTPC FP8 MoE paths for tensor-parallel serving: prefill with the per-token fp8
requant fused into gate-up (moe.prefill_moe) and small-M decode with the output clear fused
into gate-up (decode_small.decode_moe_small)."""
