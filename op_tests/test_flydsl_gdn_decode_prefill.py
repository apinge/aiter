# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL GDN (gated delta rule) kernels for Qwen3.5/3.8-style linear attention on gfx942.

decode : fused_qkvzba_conv1d_gated_delta_rule_decode_flydsl (qkvz/ba unpack + causal conv
         update + gating + l2norm + recurrence, one launch)
         fused_recurrent_gated_delta_rule_packed_decode_flydsl (recurrence only)
prefill: flydsl_gdn_prefill_front (varlen causal conv + SiLU + q/k/v split + q/k l2norm)

Shapes follow the model call: Qwen3.8-Flash-Next has 16 k-heads / 48 v-heads, K = V = 128,
so per rank TP2 -> (H, HV) = (8, 24), TP4 -> (4, 12). in_proj qkvz rows are
[q(H*K) | k(H*K) | v(HV*V) | z(HV*V)], ba rows are [b(HV) | a(HV)]; the conv state pool is
[slot, conv_dim, 3] bf16 and the SSM state pool is [slot, HV, V, K] fp32 (updated in place).
"""

import argparse
import itertools

import aiter
import pandas as pd
import torch
import torch.nn.functional as F
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942"]
D = 128


def _bf(x):
    return x.to(dtypes.bf16).to(dtypes.fp32)


def _conv(window, w):
    """window [..., 4, C] (oldest first, current last), w [C, 4] -> silu(sum bf16(x*w))."""
    acc = torch.zeros_like(window[..., 0, :])
    for j in range(4):
        acc = acc + _bf(window[..., j, :] * w[:, j].float())
    return _bf(acc * torch.sigmoid(acc))


def run_torch_decode(qkvz, ba, conv_w, conv_state, state, idx, A_log, dt_bias, H, HV):
    """Reference of the sglang decode chain. Returns o [B, HV, D] and the conv output
    mixed_qkv [B, qkv_dim], and updates both caches."""
    B = qkvz.shape[0]
    qk = 2 * H * D
    o_all = torch.zeros(B, HV, D, dtype=dtypes.fp32)
    mixed_all = torch.zeros(B, qk + HV * D, dtype=dtypes.fp32)
    for b in range(B):
        s = int(idx[b])
        if s < 0:
            continue
        x = qkvz[b, : qk + HV * D].float()
        cs = conv_state[s].float()  # [C, 3]
        win = torch.cat([cs.t(), x[None]], 0)  # [4, C]
        mixed = _conv(win, conv_w)
        mixed_all[b] = mixed
        conv_state[s] = win[1:].t().to(conv_state.dtype)
        q = mixed[: H * D].view(H, D)
        k = mixed[H * D: qk].view(H, D)
        v = mixed[qk:].view(HV, D)
        q = F.normalize(q, dim=-1, eps=0) * (D ** -0.5)
        k = F.normalize(k, dim=-1, eps=0)
        bb = ba[b, :HV].float()
        aa = ba[b, HV:].float()
        g = -torch.exp(A_log.float()) * F.softplus(aa + dt_bias.float())
        beta = _bf(torch.sigmoid(bb))
        rep = HV // H
        kk, qq = k.repeat_interleave(rep, 0), q.repeat_interleave(rep, 0)
        h = state[s] * torch.exp(g)[:, None, None]  # [HV, V, K]
        v_new = (v - (h * kk[:, None, :]).sum(-1)) * beta[:, None]
        h = h + v_new[:, :, None] * kk[:, None, :]
        state[s] = h
        o = _bf((h * qq[:, None, :]).sum(-1))  # [HV, V]
        o_all[b] = o
    return o_all.to(dtypes.bf16), mixed_all.to(dtypes.bf16)


def run_torch_prefill_front(qkvz, conv_w, conv_state, idx, has_init, cu, H, HV):
    qk = 2 * H * D
    T = qkvz.shape[0]
    q = torch.empty(1, T, H, D, dtype=dtypes.bf16)
    k = torch.empty(1, T, H, D, dtype=dtypes.bf16)
    v = torch.empty(1, T, HV, D, dtype=dtypes.bf16)
    for n in range(len(idx)):
        s0, s1, slot = int(cu[n]), int(cu[n + 1]), int(idx[n])
        x = qkvz[s0:s1, : qk + HV * D].float()
        init = conv_state[slot].float().t() if bool(has_init[n]) else torch.zeros(3, x.shape[1])
        full = torch.cat([init, x], 0)  # [3 + L, C]
        win = torch.stack([full[j: j + x.shape[0]] for j in range(4)], 1)  # [L, 4, C]
        y = _conv(win, conv_w)
        q[0, s0:s1] = _bf(F.normalize(y[:, : H * D].view(-1, H, D), dim=-1, eps=0)).to(dtypes.bf16)
        k[0, s0:s1] = _bf(F.normalize(y[:, H * D: qk].view(-1, H, D), dim=-1, eps=0)).to(dtypes.bf16)
        v[0, s0:s1] = y[:, qk:].view(-1, HV, D).to(dtypes.bf16)
        conv_state[slot] = full[-3:].t().to(conv_state.dtype)
    return q, k, v


def _decode_inputs(B, H, HV, pad):
    C = 2 * H * D + HV * D
    slots = 2 * B + 3
    qkvz = torch.randn(B, C + HV * D, dtype=dtypes.bf16)
    ba = torch.randn(B, 2 * HV, dtype=dtypes.bf16)
    conv_w = (torch.randn(C, 4) * 0.5).to(dtypes.bf16)
    conv_state = torch.randn(slots, C, 3).to(dtypes.bf16)
    state = torch.randn(slots, HV, D, D) * 0.1
    idx = (torch.randperm(slots - 1)[:B] + 1).to(torch.int32)
    if pad:
        idx[-pad:] = -1
    A_log = torch.rand(HV) * 2 - 1
    dt_bias = torch.rand(HV).to(dtypes.bf16)
    return qkvz, ba, conv_w, conv_state, state, idx, A_log, dt_bias


@benchmark()
def test_gdn_decode(tp, batch, pad, mode):
    from aiter.ops.flydsl.gdn_decode import (
        fused_qkvzba_conv1d_gated_delta_rule_decode_flydsl as fused_decode,
        fused_recurrent_gated_delta_rule_packed_decode_flydsl as packed_decode,
    )

    H, HV = 16 // tp, 48 // tp
    qkvz, ba, conv_w, conv_state, state, idx, A_log, dt_bias = _decode_inputs(batch, H, HV, pad)
    cs_ref, st_ref = conv_state.clone(), state.clone()
    o_ref, mixed = run_torch_decode(qkvz, ba, conv_w, cs_ref, st_ref, idx, A_log, dt_bias, H, HV)
    live = idx >= 0

    cs, st = conv_state.clone(), state.clone()
    z = torch.empty(batch, HV, D, dtype=dtypes.bf16)
    out = torch.empty(batch, 1, HV, D, dtype=dtypes.bf16)
    if mode == "fused":  # in_proj outputs in: unpack + conv update + recurrence, z out
        def run():
            fused_decode(qkvz, ba, cs, conv_w, A_log, dt_bias, D**-0.5, st, out, idx, H, z_out=z)
            return out
    else:  # "packed": drop-in for the Triton packed decode, conv output in
        a, b = ba[:, HV:].contiguous(), ba[:, :HV].contiguous()

        def run():
            packed_decode(mixed, a, b, A_log, dt_bias, D**-0.5, st, out, idx, True)
            return out

    run()
    err = checkAllclose(o_ref[live].float(), out[live, 0].float(),
                        rtol=2e-2, atol=2e-2, msg=f"flydsl {mode} decode: out")
    checkAllclose(st_ref, st, rtol=1e-4, atol=1e-4, msg=f"flydsl {mode} decode: ssm state")
    if mode == "fused":
        checkAllclose(cs_ref.float(), cs.float(), rtol=0, atol=0, msg="flydsl fused decode: conv state")
        z_ref = qkvz[:, 2 * H * D + HV * D:].reshape(batch, HV, D)
        checkAllclose(z_ref.float(), z.float(), rtol=0, atol=0, msg="flydsl fused decode: z")

    _, us = run_perftest(run)
    ret = {"gfx": get_gfx()}
    ret["flydsl us"] = us
    ret["flydsl TFLOPS"] = batch * HV * D * D * 6 / us / 1e6
    ret["flydsl TB/s"] = batch * HV * D * D * 4 * 2 / us / 1e6  # fp32 state read + write
    ret["flydsl err"] = err
    return ret


@benchmark()
def test_gdn_prefill_front(tp, seq_lens):
    from aiter.ops.flydsl.gdn_prefill import flydsl_gdn_prefill_front

    H, HV = 16 // tp, 48 // tp
    C = 2 * H * D + HV * D
    T, N = sum(seq_lens), len(seq_lens)
    qkvz = torch.randn(T, C + HV * D, dtype=dtypes.bf16)
    conv_w = (torch.randn(C, 4) * 0.5).to(dtypes.bf16)
    conv_state = torch.randn(N + 3, C, 3).to(dtypes.bf16)
    idx = (torch.arange(N) + 1).to(torch.int32)
    has_init = torch.arange(N) % 2 == 0
    cu = torch.tensor([0] + list(itertools.accumulate(seq_lens)), dtype=torch.int32)
    cs_ref = conv_state.clone()
    rq, rk, rv = run_torch_prefill_front(qkvz, conv_w, cs_ref, idx, has_init, cu, H, HV)

    ret = {"gfx": get_gfx()}
    cs = conv_state.clone()
    q, k, v = flydsl_gdn_prefill_front(qkvz, conv_w, cs, idx, has_init, cu, H, HV, max(seq_lens))
    err = checkAllclose(rq.float(), q.float(), rtol=1e-2, atol=1e-2, msg="flydsl prefill front: q")
    checkAllclose(rk.float(), k.float(), rtol=1e-2, atol=1e-2, msg="flydsl prefill front: k")
    checkAllclose(rv.float(), v.float(), rtol=1e-2, atol=1e-2, msg="flydsl prefill front: v")
    checkAllclose(cs_ref.float(), cs.float(), rtol=0, atol=0, msg="flydsl prefill front: conv state")

    _, us = run_perftest(
        lambda: flydsl_gdn_prefill_front(qkvz, conv_w, cs, idx, has_init, cu, H, HV, max(seq_lens))
    )
    nbytes = T * C * 2 * 2
    ret["flydsl us"] = us
    ret["flydsl TFLOPS"] = T * C * 4 * 2 / us / 1e6
    ret["flydsl TB/s"] = nbytes / us / 1e6
    ret["flydsl err"] = err
    return ret


def summarize(name, rows):
    df = pd.DataFrame(rows)
    aiter.logger.info("%s summary (markdown):\n%s", name, df.to_markdown(index=False))


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("FlyDSL GDN decode/prefill kernels unsupported on %s; skipping", get_gfx())
        return
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description="config input of test"
    )
    parser.add_argument("--tp", type=int, nargs="*", default=[2, 4],
                        help="tensor parallel size of Qwen3.8-Flash-Next (16 k-heads, 48 v-heads)")
    parser.add_argument("-b", "--batch", type=int, nargs="*", default=[1, 8, 32], help="decode batch")
    parser.add_argument("--pad", type=int, nargs="*", default=[0, 1], help="CUDA-graph pad slots (index -1)")
    parser.add_argument("--mode", type=str, nargs="*", default=["fused", "packed"],
                        help="fused: from the in_proj outputs (conv fused, z out); packed: from the conv output")
    parser.add_argument("--seq_lens", type=dtypes.str2tuple, nargs="*", default=[(5,), (1, 2, 3, 130), (300, 7, 1000), (4096,)],
                        help="prefill sequence lengths of one batch")
    args = parser.parse_args()
    summarize("gdn_decode", [
        test_gdn_decode(tp, b, p, m)
        for tp, b, p, m in itertools.product(args.tp, args.batch, args.pad, args.mode) if p < b
    ])
    seq_lens = [tuple(s) if isinstance(s, (tuple, list)) else (int(s),) for s in args.seq_lens]
    summarize("gdn_prefill_front", [test_gdn_prefill_front(tp, s) for tp, s in itertools.product(args.tp, seq_lens)])


if __name__ == "__main__":
    main()
