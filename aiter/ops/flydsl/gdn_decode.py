# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL GDN T=1 decode entry points (gfx942), named and ordered like the Triton ones.

fused_recurrent_gated_delta_rule_packed_decode_flydsl
    Drop-in for ``fused_recurrent_gated_delta_rule_packed_decode``: packed conv output
    in, o [B, 1, HV, V] out, SSM state updated in place.
fused_qkvzba_conv1d_gated_delta_rule_decode_flydsl
    Same recurrence, but it starts from the in_proj outputs: the q/k/v/z/b/a unpack and
    the causal conv1d update (with the indexed conv-state shift) run in the same kernel.
    Writes o [B, 1, HV, V] and, if given, a contiguous z [B, HV, V] for the gated norm;
    with ``norm_weight`` it applies the gated RMSNorm itself and writes [B, HV * V].
"""

from __future__ import annotations

import torch

from .kernels.gdn_decode_fused import create_gdn_decode_fused_kernel
from .kernels.tensor_shim import _run_compiled

__all__ = [
    "fused_recurrent_gated_delta_rule_packed_decode_flydsl",
    "fused_qkvzba_conv1d_gated_delta_rule_decode_flydsl",
]

_NUM_WARPS = 4
_WARP_THREADS_K = 8
_COUNTERS: dict = {}


def _arrival_counter(device, n: int) -> torch.Tensor:
    """Zero-initialised per-(token, k-head) counters; the kernel resets what it uses."""
    buf = _COUNTERS.get(str(device))
    if buf is None or buf.numel() < n:
        buf = torch.zeros(max(n, 4096), dtype=torch.int32, device=device)
        _COUNTERS[str(device)] = buf
    return buf


def _dt(t: torch.Tensor) -> str:
    return "bf16" if t.dtype == torch.bfloat16 else "f32"


def _check_state(initial_state: torch.Tensor, scale: float):
    HV, V, K = initial_state.shape[-3:]
    assert K == V == 128 and abs(scale - K**-0.5) < 1e-12
    assert initial_state.dtype == torch.float32 and initial_state.stride(-1) == 1
    assert initial_state.stride(-2) == K and initial_state.stride(-3) == V * K
    return HV, V, K


def fused_recurrent_gated_delta_rule_packed_decode_flydsl(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    out: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert use_qk_l2norm_in_kernel, "the FlyDSL kernel always L2-normalises q and k"
    B = mixed_qkv.shape[0]
    HV, V, K = _check_state(initial_state, scale)
    H = (mixed_qkv.shape[1] - HV * V) // (2 * K)
    assert mixed_qkv.stride(-1) == 1 and a.stride(-1) == 1 and b.stride(-1) == 1
    assert out.is_contiguous() and out.shape == (B, 1, HV, V)
    assert ssm_state_indices.dtype == torch.int32
    exe = create_gdn_decode_fused_kernel(
        H, HV, K, False, False, "f32", _dt(dt_bias), _NUM_WARPS, _WARP_THREADS_K
    )
    _run_compiled(
        exe,
        mixed_qkv, b, a, mixed_qkv, mixed_qkv, ssm_state_indices, A_log, dt_bias,
        initial_state, ssm_state_indices, dt_bias, out, out,
        mixed_qkv.stride(0), b.stride(0), a.stride(0), 0, 0, 0,
        initial_state.stride(0), HV * V, B, torch.cuda.current_stream(),
    )
    return out, initial_state


def fused_qkvzba_conv1d_gated_delta_rule_decode_flydsl(
    projected_qkvz: torch.Tensor,
    projected_ba: torch.Tensor,
    conv_states: torch.Tensor,
    conv_weights: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    out: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    num_k_heads: int,
    z_out: torch.Tensor | None = None,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
    use_qk_l2norm_in_kernel: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``ssm_state_indices`` indexes both the SSM and the conv state pools (-1 = pad)."""
    assert use_qk_l2norm_in_kernel, "the FlyDSL kernel always L2-normalises q and k"
    B = projected_qkvz.shape[0]
    HV, V, K = _check_state(initial_state, scale)
    H = num_k_heads
    assert projected_qkvz.stride(-1) == 1 and projected_ba.stride(-1) == 1
    assert projected_qkvz.shape[1] >= 2 * H * K + 2 * HV * V
    assert projected_ba.shape[1] >= 2 * HV
    assert conv_weights.is_contiguous() and conv_weights.shape[-1] == 4
    assert conv_weights.dtype == torch.bfloat16
    assert conv_states.dim() == 3 and conv_states.shape[-1] == 3
    assert conv_states.dtype == torch.bfloat16
    assert ssm_state_indices.dtype == torch.int32
    fuse_norm = norm_weight is not None
    write_z = (not fuse_norm) and z_out is not None
    assert out.is_contiguous() and out.numel() == B * HV * V
    if write_z:
        assert z_out.is_contiguous() and z_out.numel() == B * HV * V
    exe = create_gdn_decode_fused_kernel(
        H, HV, K, True, fuse_norm, "f32", _dt(dt_bias), _NUM_WARPS, _WARP_THREADS_K,
        float(norm_eps), 20.0, 0, write_z,
    )
    _run_compiled(
        exe,
        projected_qkvz, projected_ba, projected_ba, conv_weights, conv_states,
        _arrival_counter(projected_qkvz.device, B * H), A_log, dt_bias, initial_state,
        ssm_state_indices, norm_weight if fuse_norm else dt_bias, out,
        z_out if write_z else out,
        projected_qkvz.stride(0), projected_ba.stride(0), projected_ba.stride(0),
        conv_states.stride(0), conv_states.stride(1), conv_states.stride(2),
        initial_state.stride(0), HV * V, B, torch.cuda.current_stream(),
    )
    return out, initial_state
