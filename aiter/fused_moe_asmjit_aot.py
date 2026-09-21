# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Precompiled gfx942 PTPC MoE kernels for Qwen3.5 397B/122B/35B.

The shipped COs and launch ABI come from Qwen3.5_dev at a3a280f437c1,
including the prefill fixes and 397B TP8 decode support through 128 tokens.
"""

from dataclasses import dataclass
from typing import Any

import torch

import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import moe_sorting
from aiter.fused_moe_registry import FusedMoeRequest
from aiter.jit.utils.chip_info import get_gfx
from csrc.cpp_itfs.hsaco_tools import hsaco


@dataclass
class Config:
    BLOCK_M: int
    use_down_loopn: bool
    use_prefill: bool
    use_dyn_sched: bool

    def to_string(self):
        return (
            str(self.BLOCK_M)
            + "_"
            + str(self.use_down_loopn)
            + "_"
            + str(self.use_prefill)
            + "_"
            + str(self.use_dyn_sched)
        )

    @classmethod
    def from_string(cls, data: str):
        parts = data.split("_")
        if len(parts) != 4 or any(p not in ("True", "False") for p in parts[1:]):
            raise ValueError(f"Invalid asmjit config: {data!r}")
        config = cls(int(parts[0]), *(p == "True" for p in parts[1:]))
        if config.BLOCK_M not in (16, 64, 128):
            raise ValueError(f"Invalid asmjit block size: {config.BLOCK_M}")
        if (
            (not config.use_prefill and config.BLOCK_M != 16)
            or (config.use_prefill and config.BLOCK_M not in (64, 128))
            or (
                config.use_dyn_sched
                and (not config.use_prefill or config.BLOCK_M != 128)
            )
        ):
            raise ValueError(f"Unsupported asmjit config: {data!r}")
        return config


def _get_decode_max_batch(E: int, TOPK: int, K1: int, K2: int) -> int:
    # Only Qwen3.5-397B TP8 extends the ordinary decode limit to 64/128.
    if E == 513 and TOPK == 11 and K1 == 4096 and K2 == 128:
        return 128
    return 32


def fused_moe_asmjit_aot(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: ActivationType,
    quant_type: QuantType,
    w1_scale: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
    expert_mask: Any,
    num_local_tokens: Any,
    moe_sorting_dispatch_policy: int,
    config_string: str,
) -> torch.Tensor:

    # decode kernel configs from kernel name
    kcfgs = Config.from_string(config_string)

    B = int(hidden_states.shape[0])
    if (
        hidden_states.dtype != torch.bfloat16
        or expert_mask is not None
        or activation != ActivationType.Silu
        or w1.dtype != torch.float8_e4m3fnuz
        or w2.dtype != torch.float8_e4m3fnuz
    ):
        raise ValueError("Unsupported asmjit input")
    if get_gfx() != "gfx942":
        raise ValueError("asmjit requires gfx942")

    if quant_type != QuantType.per_Token:
        raise ValueError(f"Unsupported asmjit quant_type: {quant_type}")

    qtype_str = str(quant_type).split(".")[1]

    E, N1, K1 = w1.shape  # num_experts, 2*moe_intermediate_size//TP, head_size
    N2, K2 = w2.shape[1], w2.shape[2]  # K2 is moe_intermediate_size//TP
    TOPK = topk_ids.shape[1]
    if (E, TOPK, K1, K2) not in (
        (513, 11, 4096, 128),
        (257, 9, 3072, 128),
        (257, 9, 2048, 128),
    ) or N2 != K1:
        raise ValueError(f"Unsupported asmjit shape: {(E, TOPK, K1, K2, N2)}")
    fp8_ptpc = w1.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz) and (
        quant_type == QuantType.per_Token
    )
    num_CU = torch.cuda.get_device_properties(
        hidden_states.device
    ).multi_processor_count
    assert N1 == 2 * K2
    decode_max_B = _get_decode_max_batch(E, TOPK, K1, K2)

    topk_w_f32 = (
        topk_weight if topk_weight.dtype == torch.float32 else topk_weight.float()
    )

    gemm1_out = torch.empty(
        [B, TOPK, N1 // 2],
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    if kcfgs.use_prefill:
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, cur_out = (
            moe_sorting(
                topk_ids,
                topk_weight,
                E,
                N2,  # reduce dim is same with output dim
                hidden_states.dtype,
                kcfgs.BLOCK_M,
                None,
                None,
                0,
            )
        )
        quant_func = aiter.get_hip_quant(aiter.QuantType.per_Token)
        hidden_states_q, hidden_states_scale = quant_func(
            hidden_states,
            scale=None,
            quant_dtype=w1.dtype,
            num_rows=None,
        )
        if kcfgs.use_dyn_sched:
            dyn_buf1 = torch.zeros(64, dtype=torch.int32, device=hidden_states_q.device)
            dyn_buf2 = torch.zeros(64, dtype=torch.int32, device=hidden_states_q.device)
            grid_gate_up = num_CU
            grid_down = num_CU * 2  # occupancy is 2
            GATEUP_BLOCK_TILE_SIZE_N = 256
            DOWN_BLOCK_TILE_SIZE_N = 128
        else:
            GATEUP_BLOCK_TILE_SIZE_N = 128
            DOWN_BLOCK_TILE_SIZE_N = 128
            dyn_buf1 = None
            dyn_buf2 = None
            grid_gate_up = N1 // GATEUP_BLOCK_TILE_SIZE_N * sorted_expert_ids.shape[0]
            grid_down = sorted_expert_ids.shape[0]

        hsaco.fmoe_asmjit.moe_2stage_gateup(
            [grid_gate_up],
            [256],
            dyn_buf1,
            hidden_states_q,
            w1,
            gemm1_out,
            sorted_ids,
            sorted_expert_ids,
            num_valid_ids,
            hidden_states_scale,
            w1_scale,
            B,
            N1 // GATEUP_BLOCK_TILE_SIZE_N * sorted_expert_ids.shape[0],
            weight_dtype=str(w1.dtype),
            TOPK=TOPK,
            K=K1,
            N=N1,
            BLOCK_TILE_SIZE_M=kcfgs.BLOCK_M,
            BLOCK_TILE_SIZE_N=GATEUP_BLOCK_TILE_SIZE_N,
            quant_type_w=f"QuantType.{qtype_str}",
            dyn=kcfgs.use_dyn_sched,
        )
        gemm1_out_q, gemm1_out_scale = quant_func(
            gemm1_out.view(B * TOPK, -1),
            scale=None,
            quant_dtype=w2.dtype,
            num_rows=None,
        )
        gemm2_out = torch.empty(
            B, TOPK, N2, dtype=torch.bfloat16, device=gemm1_out_q.device
        )
        hsaco.fmoe_asmjit.moe_2stage_down(
            [grid_down],
            [256],
            dyn_buf2,
            gemm1_out_q,
            w2,
            gemm2_out,  # cur_out,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            gemm1_out_scale,
            w2_scale,
            B,
            sorted_expert_ids.shape[0],
            weight_dtype=str(w2.dtype),
            TOPK=TOPK,
            K=K2,
            N=N2,
            with_silu=False,
            BLOCK_TILE_SIZE_M=kcfgs.BLOCK_M,
            BLOCK_TILE_SIZE_N=DOWN_BLOCK_TILE_SIZE_N,
            quant_type_w=f"QuantType.{qtype_str}",
            dyn=kcfgs.use_dyn_sched,
        )
        num_WG = num_CU * 4
        num_tokens_wg = B // num_WG
        num_extra_tokens = B % num_WG
        hsaco.fmoe_asmjit.moe_gemm_final_reduce_bf16(
            [num_WG],
            [64],
            gemm2_out,
            cur_out,
            num_tokens_wg,
            num_extra_tokens,
            B,
            TOPK=TOPK,
            OC=N2,
        )
        return cur_out

    if B == 1:
        assert N1 == 2 * K2
        cur_out = torch.zeros(
            [1, N2], dtype=hidden_states.dtype, device=hidden_states.device
        )
        hsaco.fmoe_asmjit.moe_gemm_batch1(
            [N1 // 32, TOPK],
            [256],
            hidden_states,
            w1,
            gemm1_out,
            topk_ids,
            topk_w_f32,
            w1_scale,
            1,
            N1,
            K1,
            weight_dtype=torch.float8_e4m3fnuz,
            with_silu=True,
            quant_type_str=qtype_str,
        )
        hsaco.fmoe_asmjit.moe_gemm_batch1(
            [N2 // 32, TOPK],
            [64],
            gemm1_out,
            w2,
            cur_out,
            topk_ids,
            topk_w_f32,
            w2_scale,
            1,
            N2,
            K2,
            weight_dtype=torch.float8_e4m3fnuz,
            with_silu=False,
            quant_type_str=qtype_str,
        )
    elif 2 <= B <= decode_max_B:
        # Stage 1: Shared ``moe_sorting`` + ``moe_gemm_batch``;
        # stage 2: Choose between ``moe_2stage_down_loopn`` and ``moe_2stage_splitk`` based on ``use_down_loopn`` condition.
        BLOCK_M = kcfgs.BLOCK_M
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, cur_out = (
            moe_sorting(
                topk_ids,
                topk_weight,
                E,
                K1,
                hidden_states.dtype,
                BLOCK_M,
                expert_mask,
                num_local_tokens,
                moe_sorting_dispatch_policy,
            )
        )
        grid = int(sorted_expert_ids.shape[0])
        if B * TOPK <= E:
            grid = B * TOPK

        hsaco.fmoe_asmjit.moe_gemm_batch(
            [N1 // 32, grid],
            [256],
            hidden_states,
            w1,
            gemm1_out,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            w1_scale,
            B,
            N1,
            K1,
            TOPK,
            weight_dtype=torch.float8_e4m3fnuz,
            with_silu=True,
            quant_type_str=qtype_str,
        )

        BLOCK_N = 1024
        if kcfgs.use_down_loopn:
            # extra checks
            use_down_loopn = (
                fp8_ptpc
                and (N2 // BLOCK_N) * grid >= num_CU
                and N2 % BLOCK_N == 0
                and 16 <= B <= decode_max_B
            )
        else:
            use_down_loopn = False

        if use_down_loopn:
            gemm2_out = torch.empty(
                [B, TOPK, N2],
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            hsaco.fmoe_asmjit.moe_2stage_down_loopn(
                [N2 // BLOCK_N, grid],
                [256],
                gemm1_out,
                w2,
                gemm2_out,
                sorted_ids,
                sorted_weights,
                sorted_expert_ids,
                num_valid_ids,
                w2_scale,
                B,
                weight_dtype=torch.float8_e4m3fnuz,
                TOPK=TOPK,
                K=K2,
                N=N2,
                BLOCK_TILE_SIZE_M=16,
                BLOCK_TILE_SIZE_N=16,
                fp8_ptpc=True,
                BLOCK_N=BLOCK_N,
                atomic_write=False,
                STAGES=3,
            )
            cur_out = torch.sum(gemm2_out, dim=1)
        else:
            BLOCK_TILE_SIZE_N = 64
            hsaco.fmoe_asmjit.moe_2stage_splitk(
                [N2 // BLOCK_TILE_SIZE_N, grid],
                [64],
                gemm1_out,
                w2,
                cur_out,
                sorted_ids,
                sorted_weights,
                sorted_expert_ids,
                num_valid_ids,
                w2_scale,
                B,
                weight_dtype=torch.float8_e4m3fnuz,
                TOPK=TOPK,
                K=K2,
                N=N2,
                with_silu=False,
                BLOCK_TILE_SIZE_M=16,
                BLOCK_TILE_SIZE_N=BLOCK_TILE_SIZE_N,
                quant_type_str=qtype_str,
            )
    else:
        raise ValueError(f"Unsupported asmjit batch-size {B}")
    return cur_out


def run_asmjit_moe_gfx942_impl(
    request: FusedMoeRequest,
    config_string: str,
) -> torch.Tensor:
    """Whole-graph adapter for the Qwen3.5 PTPC precompiled kernels."""
    if (
        request.doweight_stage1
        or request.bias1 is not None
        or request.bias2 is not None
        or request.hidden_pad
        or request.intermediate_pad
        or request.expert_mask is not None
        or request.num_local_tokens is not None
        or request.a1_scale is not None
        or request.a2_scale is not None
        or request.dtype not in (None, torch.bfloat16)
        or getattr(request.gate_mode, "value", request.gate_mode)
        not in (None, "separated")
    ):
        raise ValueError(
            "asmjit requires unpadded BF16 PTPC inputs, separated gate/up, "
            "stage2 routing weights, and no bias or expert-parallel masking"
        )
    if not all(getattr(w, "is_shuffled", False) for w in (request.w1, request.w2)):
        raise ValueError(
            "asmjit requires both weights preshuffled with layout (16, 16)"
        )
    if any(
        scale is None or scale.dtype != torch.float32 or not scale.is_contiguous()
        for scale in (request.w1_scale, request.w2_scale)
    ):
        raise ValueError("asmjit requires per-channel FP32 weight scales")
    return fused_moe_asmjit_aot(
        request.hidden_states,
        request.w1,
        request.w2,
        request.topk_weight,
        request.topk_ids,
        request.activation,
        request.quant_type,
        request.w1_scale,
        request.w2_scale,
        request.expert_mask,
        request.num_local_tokens,
        request.moe_sorting_dispatch_policy,
        config_string,
    )
