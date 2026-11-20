# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""GDN prefill front end for gfx942: in_proj qkvz -> varlen causal conv(4) + SiLU ->
q/k/v split -> q/k L2 norm, in one pass (replaces cat + causal_conv1d_fwd + qkv split +
2x l2norm).

Grid (max_chunks, 2H + HV, seq): one workgroup per (128-token chunk, head, sequence).
Each thread owns 8 channels (one 16 B vector) of 8 consecutive tokens and keeps the
3-token causal halo in registers; the 16 lanes of a token sit in one DPP row, so the
per-token L2 norm is a DPP all-reduce. Numerics follow the Triton kernels: products are
rounded to bf16, accumulated in fp32, SiLU, rounded to bf16, then y / sqrt(sum y^2 + eps).
The chunk-0 workgroup of a sequence reads the initial conv state (when has_initial_state)
and writes the final conv state (raw inputs of the last 3 tokens), like the Triton kernel.
"""

import functools
import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf
from flydsl._mlir.dialects import vector as mlir_vector
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.tensor_shim import GTensor, _to_raw

_LOG2E = math.log2(math.e)
_D = 128
_OCT = 8  # channels per thread
_LANES_PER_TOK = _D // _OCT  # 16 (one DPP row)
_NT = 256
_TG = 8  # tokens per thread
_CHUNK = (_NT // _LANES_PER_TOK) * _TG  # 128


def _dpp_f32(src, ctrl):
    i = fx.Float32(src).bitcast(fx.Int32)
    r = llvm.call_intrinsic(
        fx.Int32.ir_type,
        "llvm.amdgcn.update.dpp.i32",
        [
            fx.Int32(i).ir_value(),
            fx.Int32(i).ir_value(),
            fx.Int32(ctrl).ir_value(),
            fx.Int32(0xF).ir_value(),
            fx.Int32(0xF).ir_value(),
            fx.Boolean(False).ir_value(),
        ],
        [],
        [],
    )
    return fx.Int32(r).bitcast(fx.Float32)


def _row16_sum(x):
    for ctrl in (177, 78, 0x141, 0x140):  # xor1, xor2, half_mirror, mirror
        x = fx.Float32(x) + _dpp_f32(x, ctrl)
    return x


def _sload_i32(mem, elem_idx):
    addr = GTensor.get_llvm_ptr(None, mem, fx.Int64(elem_idx) * fx.Int64(4))
    p = llvm.IntToPtrOp(llvm.PointerType.get(address_space=4), addr).result
    return fx.Int32(llvm.LoadOp(T.i32, p, alignment=4, invariant=True).result)


def _elems(vec, n):
    return [fx.Float32(mlir_vector.ExtractOp(_to_raw(vec), [], [i]).result) for i in range(n)]


def _bf16_round(x):
    """Round-to-nearest-even to bf16, kept in f32 (finite inputs): bfe + add3 + and."""
    b = fx.Float32(x).bitcast(fx.Int32)
    lsb = (b >> fx.Int32(16)) & fx.Int32(1)
    r = (b + fx.Int32(0x7FFF) + lsb) & fx.Int32(-65536)
    return r.bitcast(fx.Float32)


def _bf16_round_product(p):
    """RNE to bf16 of an exact bf16 x bf16 product via Veltkamp splitting (pure FP, so
    it packs into v_pk_mul/v_pk_add_f32); bit-identical to RNE for such products."""
    c = fx.Float32(p) * fx.Float32(65537.0)
    return c - (c - fx.Float32(p))


def _exp2(x):
    from flydsl.expr import rocdl

    return fx.Float32(rocdl.exp2(T.f32, _to_raw(fx.Float32(x))))


def _rcp(x):
    from flydsl.expr import rocdl

    return fx.Float32(rocdl.rcp(T.f32, _to_raw(fx.Float32(x))))


@functools.lru_cache(maxsize=None)
def create_gdn_prefill_front_kernel(num_k_heads: int, num_v_heads: int, l2_eps: float = 1e-6):
    H, HV, D = num_k_heads, num_v_heads, _D
    QK_DIM = 2 * H * D
    NG = 2 * H + HV
    name = f"gdn_prefill_front_h{H}x{HV}_d{D}"

    @flyc.kernel(name=name, known_block_size=[_NT, 1, 1])
    def kernel(
        x_mem: fx.Tensor,  # qkvz [T, >= QK_DIM + HV*D] bf16
        conv_w_mem: fx.Tensor,  # [conv_dim, 4] bf16 contiguous
        conv_state_mem: fx.Tensor,  # [slots, conv_dim, 3] bf16 (inner contiguous)
        cache_idx_mem: fx.Tensor,  # int32 [N]
        has_init_mem: fx.Tensor,  # int32 [N]
        cu_mem: fx.Tensor,  # int32 [N + 1]
        q_mem: fx.Tensor,  # bf16 [T, H, D]
        k_mem: fx.Tensor,  # bf16 [T, H, D]
        v_mem: fx.Tensor,  # bf16 [T, HV, D]
        stride_x_tok: fx.Int32,
        stride_conv_slot: fx.Int64,
        max_chunks: fx.Int32,
    ):
        vec8_bf16 = T.vec(_OCT, T.bf16)
        vec8_f32 = T.vec(_OCT, T.f32)
        x = GTensor(x_mem, dtype=T.bf16, shape=(-1,))
        conv_w = GTensor(conv_w_mem, dtype=T.bf16, shape=(-1,))

        tid = fx.thread_idx.x
        bx = fx.block_idx.x
        grp = fx.block_idx.y
        seq = fx.block_idx.z
        chunk = bx
        start = _sload_i32(cu_mem, seq)
        seqlen = _sload_i32(cu_mem, seq + fx.Int32(1)) - start
        slot = _sload_i32(cache_idx_mem, seq)
        has_init = _sload_i32(has_init_mem, seq) != fx.Int32(0)

        oct_ = tid % fx.Int32(_LANES_PER_TOK)
        tgi = tid // fx.Int32(_LANES_PER_TOK)
        tok0 = chunk * fx.Int32(_CHUNK) + tgi * fx.Int32(_TG)

        is_q = grp < fx.Int32(H)
        is_k = (grp >= fx.Int32(H)) & (grp < fx.Int32(2 * H))
        is_qk = grp < fx.Int32(2 * H)
        ch0 = is_qk.select(grp * fx.Int32(D), fx.Int32(QK_DIM) + (grp - fx.Int32(2 * H)) * fx.Int32(D))
        ch = ch0 + oct_ * fx.Int32(_OCT)

        active = (chunk * fx.Int32(_CHUNK) < seqlen) & (slot >= fx.Int32(0))
        act_if = scf.IfOp(_to_raw(active), results_=[], has_else=False)
        with ir.InsertionPoint(act_if.then_block):
            # conv taps for my 8 channels: w[c*4 + j]
            wv = []
            for i in range_constexpr(4):
                wv += _elems(conv_w.vec_load((ch * fx.Int32(4) + fx.Int32(i * 8),), 8).extf(vec8_f32), 8)
            # initial conv state of my 8 channels: s[c*3 + k] (48 B, 16 B aligned)
            cs = GTensor(
                conv_state_mem,
                dtype=T.bf16,
                shape=(-1,),
                static_bytes_offset_i64=fx.Int64(slot) * stride_conv_slot * fx.Int64(2),
            )
            sv = []
            for i in range_constexpr(3):
                sv += _elems(cs.vec_load((ch * fx.Int32(3) + fx.Int32(i * 8),), 8).extf(vec8_f32), 8)

            # rows tok0-3 .. tok0+TG-1 of the sequence (raw conv input)
            rows = []
            last = seqlen - fx.Int32(1)
            for r in range_constexpr(_TG + 3):
                ri = tok0 + fx.Int32(r - 3)
                rc = (ri < fx.Int32(0)).select(fx.Int32(0), (ri > last).select(last, ri))
                vals = _elems(
                    x.vec_load(((start + rc) * stride_x_tok + ch,), 8).extf(vec8_f32), 8
                )
                if fx.const_expr(r < 3):
                    # only the very first rows of a sequence can fall before it
                    before = ri < fx.Int32(0)
                    new = []
                    for c in range_constexpr(_OCT):
                        # ri in {-3,-2,-1} -> state k = 3 + ri
                        s0, s1, s2 = sv[c * 3 + 0], sv[c * 3 + 1], sv[c * 3 + 2]
                        init_v = (ri == fx.Int32(-3)).select(
                            s0, (ri == fx.Int32(-2)).select(s1, s2)
                        )
                        init_v = has_init.select(init_v, fx.Float32(0.0))
                        new.append(before.select(init_v, vals[c]))
                    vals = new
                rows.append(vals)

            out_q = GTensor(q_mem, dtype=T.bf16, shape=(-1,))
            out_k = GTensor(k_mem, dtype=T.bf16, shape=(-1,))
            out_v = GTensor(v_mem, dtype=T.bf16, shape=(-1,))
            for t in range_constexpr(_TG):
                ys = []
                ss = fx.Float32(0.0)
                for c in range_constexpr(_OCT):
                    acc = fx.Float32(0.0)
                    for j in range_constexpr(4):
                        acc = acc + _bf16_round_product(rows[t + j][c] * wv[c * 4 + j])
                    y = _bf16_round(
                        acc * _rcp(fx.Float32(1.0) + _exp2(-acc * fx.Float32(_LOG2E)))
                    )
                    ys.append(y)
                    ss = ss + y * y
                ss = _row16_sum(ss)
                # v heads scale by 1.0 (exact, rounding is then the identity)
                inv = is_qk.select(_rcp(fx.math.sqrt(ss + fx.Float32(l2_eps))), fx.Float32(1.0))
                outv = [_bf16_round(ys[c] * inv).bitcast(fx.Int32) for c in range(_OCT)]
                # values are bf16-exact: pack the high halves pairwise (no NaN handling)
                words = []
                for c in range_constexpr(_OCT // 2):
                    lo = (outv[2 * c] >> fx.Int32(16)) & fx.Int32(0xFFFF)
                    hi = outv[2 * c + 1] & fx.Int32(-65536)
                    words.append(_to_raw(hi | lo))
                ovec = mlir_vector.FromElementsOp(T.vec(4, T.i32), words).result
                tok = tok0 + fx.Int32(t)
                gtok = start + tok
                in_seq = tok < seqlen
                qk_off = gtok * fx.Int32(H * D) + (grp % fx.Int32(H)) * fx.Int32(D) + oct_ * fx.Int32(_OCT)
                v_off = gtok * fx.Int32(HV * D) + (grp - fx.Int32(2 * H)) * fx.Int32(D) + oct_ * fx.Int32(_OCT)
                q_if = scf.IfOp(_to_raw(in_seq & is_q), results_=[], has_else=False)
                with ir.InsertionPoint(q_if.then_block):
                    buffer_ops.buffer_store(ovec, out_q.rsrc, qk_off * fx.Int32(2), offset_is_bytes=True)
                    scf.YieldOp([])
                k_if = scf.IfOp(_to_raw(in_seq & is_k), results_=[], has_else=False)
                with ir.InsertionPoint(k_if.then_block):
                    buffer_ops.buffer_store(ovec, out_k.rsrc, qk_off * fx.Int32(2), offset_is_bytes=True)
                    scf.YieldOp([])
                v_if = scf.IfOp(_to_raw(in_seq & (grp >= fx.Int32(2 * H))), results_=[], has_else=False)
                with ir.InsertionPoint(v_if.then_block):
                    buffer_ops.buffer_store(ovec, out_v.rsrc, v_off * fx.Int32(2), offset_is_bytes=True)
                    scf.YieldOp([])

            # final conv state: chunk-0 / token-group-0 threads (they also read the
            # initial state above, so the in-place update needs no barrier)
            st_if = scf.IfOp(
                _to_raw((chunk == fx.Int32(0)) & (tgi == fx.Int32(0))), results_=[], has_else=False
            )
            with ir.InsertionPoint(st_if.then_block):
                tail = []
                for k3 in range_constexpr(3):
                    ri = seqlen - fx.Int32(3 - k3)
                    rc = (ri < fx.Int32(0)).select(fx.Int32(0), ri)
                    vals = x.vec_load(((start + rc) * stride_x_tok + ch,), 8).extf(vec8_f32)
                    tail.append((ri, _elems(vals, 8)))
                news = []
                for c in range_constexpr(_OCT):
                    for k3 in range_constexpr(3):
                        ri, vals = tail[k3]
                        # ri < 0 -> keep the shifted initial state: state index 3 + ri
                        s0, s1, s2 = sv[c * 3 + 0], sv[c * 3 + 1], sv[c * 3 + 2]
                        init_v = (ri == fx.Int32(-3)).select(
                            s0, (ri == fx.Int32(-2)).select(s1, s2)
                        )
                        init_v = has_init.select(init_v, fx.Float32(0.0))
                        news.append((ri < fx.Int32(0)).select(init_v, vals[c]).to(fx.BFloat16))
                for i in range_constexpr(3):
                    nv = mlir_vector.FromElementsOp(
                        vec8_bf16, [_to_raw(o) for o in news[i * 8:(i + 1) * 8]]
                    ).result
                    buffer_ops.buffer_store(nv, cs.rsrc, ch * fx.Int32(3) + fx.Int32(i * 8))
                scf.YieldOp([])
            scf.YieldOp([])

    @flyc.jit
    def launch(
        x_mem: fx.Tensor,
        conv_w_mem: fx.Tensor,
        conv_state_mem: fx.Tensor,
        cache_idx_mem: fx.Tensor,
        has_init_mem: fx.Tensor,
        cu_mem: fx.Tensor,
        q_mem: fx.Tensor,
        k_mem: fx.Tensor,
        v_mem: fx.Tensor,
        stride_x_tok: fx.Int32,
        stride_conv_slot: fx.Int64,
        max_chunks: fx.Int32,
        num_seqs: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        kernel(
            x_mem, conv_w_mem, conv_state_mem, cache_idx_mem, has_init_mem, cu_mem,
            q_mem, k_mem, v_mem, stride_x_tok, stride_conv_slot, max_chunks,
        ).launch(grid=(max_chunks, NG, num_seqs), block=(_NT, 1, 1), stream=stream)

    launch.compile_hints = {
        "llvm_options": {"amdgpu-kernarg-preload": True, "amdgpu-kernarg-preload-count": 16}
    }
    return launch


CHUNK_TOKENS = _CHUNK
__all__ = ["create_gdn_prefill_front_kernel", "CHUNK_TOKENS"]
