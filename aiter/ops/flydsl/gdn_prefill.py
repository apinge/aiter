# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL GDN prefill front end (gfx942): qkvz -> causal conv + SiLU -> q/k/v (q/k L2-normed)."""

from __future__ import annotations

import torch

from .kernels.gdn_prefill_front import CHUNK_TOKENS, create_gdn_prefill_front_kernel
from .kernels.tensor_shim import _run_compiled

__all__ = ["flydsl_gdn_prefill_front"]


def flydsl_gdn_prefill_front(
    qkvz: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state: torch.Tensor,
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_k_heads: int,
    num_v_heads: int,
    max_seqlen: int,
    num_tokens: int | None = None,
):
    """Returns q, k (L2-normalised) [1, T, H, 128] and v [1, T, HV, 128], all contiguous bf16.

    Updates conv_state[cache_indices[s]] in place with the last 3 raw inputs of each
    sequence (initial state used when has_initial_state[s] and the sequence is shorter).
    """
    D = 128
    H, HV = num_k_heads, num_v_heads
    T = qkvz.shape[0] if num_tokens is None else num_tokens
    assert qkvz.dtype == torch.bfloat16 and qkvz.stride(-1) == 1
    assert qkvz.shape[1] >= 2 * H * D + HV * D
    assert conv_weight.is_contiguous() and conv_weight.shape[-1] == 4
    assert conv_state.dtype == torch.bfloat16 and conv_state.stride(-1) == 1
    assert conv_state.stride(-2) == 3 and conv_state.shape[-1] == 3
    N = cache_indices.numel()
    q = qkvz.new_empty(1, T, H, D)
    k = qkvz.new_empty(1, T, H, D)
    v = qkvz.new_empty(1, T, HV, D)
    if T == 0 or N == 0:
        return q, k, v
    idx = cache_indices if cache_indices.dtype == torch.int32 else cache_indices.to(torch.int32)
    hi = has_initial_state.to(torch.int32) if has_initial_state is not None else torch.zeros(
        N, dtype=torch.int32, device=qkvz.device
    )
    cu = query_start_loc if query_start_loc.dtype == torch.int32 else query_start_loc.to(torch.int32)
    max_chunks = max(1, (int(max_seqlen) + CHUNK_TOKENS - 1) // CHUNK_TOKENS)
    exe = create_gdn_prefill_front_kernel(H, HV)
    _run_compiled(
        exe, qkvz, conv_weight, conv_state, idx, hi, cu, q, k, v,
        qkvz.stride(0), conv_state.stride(0), max_chunks, N, torch.cuda.current_stream(),
    )
    return q, k, v
