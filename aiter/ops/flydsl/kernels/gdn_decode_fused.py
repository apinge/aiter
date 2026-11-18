# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused GDN (gated delta rule) T=1 decode for gfx942, one workgroup per (token, v-head).

Per workgroup, in issue order:
  1. index load, then every lane issues its whole fp32 state slice (16 x dwordx4) up front,
  2. q/k/v channels: either read from the packed conv output (FUSE_CONV=False, drop-in for
     sglang's fused_recurrent_gated_delta_rule_packed_decode) or computed here from the
     in_proj output with the width-4 causal conv + SiLU and the indexed conv-state shift
     (FUSE_CONV=True). q/k conv channels are shared by RATIO v-heads; the last of the RATIO
     workgroups to finish reading them writes the shifted q/k conv state back (atomic
     arrival counter, reset by that workgroup, so the counter is CUDA-graph safe),
  3. gating (softplus / exp decay, sigmoid beta rounded to the input dtype like the
     Triton kernel), q/k L2 norm,
  4. recurrent update h = h*exp(g); v' = (v - h.k)*beta; h += k v'^T; o = h.q,
  5. output: o (bf16, [B, 1, HV, V]) and, with WRITE_Z, a contiguous z [B, HV, V] for the
     caller's gated norm.
Pad slots (index < 0) write a zero output and leave both caches untouched.
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

from aiter.ops.flydsl.kernels import buffer_ops, vector
from aiter.ops.flydsl.kernels.tensor_shim import GTensor, _to_raw

_LOG2E = math.log2(math.e)
_WARP = 64
# gfx942 buffer aux bits: the state is read and written back in place once per step.
# A store that hits a line the same kernel just read is slow on MI308X; streaming
# (nt) loads and stores of the state avoid most of that penalty (~20% at B=8).
_STATE_CACHE_POLICY = 2  # nt


def _gptr(addr_i64):
    return llvm.IntToPtrOp(
        llvm.PointerType.get(address_space=1), fx.Int64(addr_i64).ir_value()
    ).result


def _atomic_add_i32(addr_i64, val):
    return llvm.AtomicRMWOp(
        llvm.AtomicBinOp.add,
        _gptr(addr_i64),
        fx.Int32(val).ir_value(),
        llvm.AtomicOrdering.monotonic,
        syncscope="agent",
    ).res


def _store_i32(addr_i64, val):
    llvm.StoreOp(
        fx.Int32(val).ir_value(),
        _gptr(addr_i64),
        alignment=4,
        ordering=llvm.AtomicOrdering.monotonic,
        syncscope="agent",
    )


def _dpp_f32(src, ctrl: int):
    """Row-local DPP move (no LDS): quad_perm / row_mirror / row_half_mirror."""
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


# all-reduce sum over aligned groups of 2/4/8/16 lanes
_DPP_STEPS = {1: 177, 2: 78, 4: 0x141, 8: 0x140}  # xor1, xor2, half_mirror, mirror


def _group_allreduce(x, group: int):
    o = 1
    while o < group:
        x = fx.Float32(x) + _dpp_f32(x, _DPP_STEPS[o])
        o *= 2
    return x


def _sload_i32(mem, elem_idx):
    """Uniform int32 element load through the scalar unit (s_load_dword)."""
    addr = GTensor.get_llvm_ptr(None, mem, fx.Int64(elem_idx) * fx.Int64(4))
    p = llvm.IntToPtrOp(llvm.PointerType.get(address_space=4), addr).result
    return fx.Int32(llvm.LoadOp(T.i32, p, alignment=4, invariant=True).result)


def _sload_f32(mem, elem_idx):
    return _sload_i32(mem, elem_idx).bitcast(fx.Float32)


def _sload_bf16_as_f32(mem, elem_idx):
    """Uniform bf16 element: load its aligned dword and pick the half."""
    e = fx.Int32(elem_idx)
    w = _sload_i32(mem, e // fx.Int32(2))
    hi = (e % fx.Int32(2)) == fx.Int32(1)
    bits = hi.select(w & fx.Int32(-65536), w << fx.Int32(16))
    return bits.bitcast(fx.Float32)


def _vsum(v):
    return mlir_vector.ReductionOp(T.f32, vector.CombiningKind.ADD, v).dest


def _fma(a, b, c):
    return mlir_vector.FMAOp(a, b, c).result


def _sigmoid(x):
    return fx.Float32(1.0) / (fx.Float32(1.0) + fx.math.exp2(-x * fx.Float32(_LOG2E)))


@functools.lru_cache(maxsize=None)
def create_gdn_decode_fused_kernel(
    num_k_heads: int,
    num_v_heads: int,
    head_dim: int = 128,
    fuse_conv: bool = True,
    dt_bias_dtype: str = "bf16",
    warp_threads_k: int = 8,
    softplus_threshold: float = 20.0,
    write_z: bool = False,
):
    H, HV, D = num_k_heads, num_v_heads, head_dim
    RATIO = HV // H
    assert HV % H == 0 and D == 128
    assert not write_z or fuse_conv
    VPT = 4  # f32 values per lane per state transaction (16 B)
    num_warps = 4  # one thread per q/v channel plus one per k channel: NT = 2 * D
    NT = num_warps * _WARP
    WTK = warp_threads_k
    WTV = _WARP // WTK
    TILE_K = WTK * VPT
    K_ITERS = D // TILE_K
    V_GROUP = num_warps * WTV
    V_ITERS = D // V_GROUP
    assert D % TILE_K == 0 and D % V_GROUP == 0 and NT == 2 * D
    QK_DIM = 2 * H * D
    SCALE = D ** -0.5
    dt_bias_is_bf16 = dt_bias_dtype == "bf16"

    name = (
        f"gdn_decode_fused_h{H}x{HV}_d{D}_c{int(fuse_conv)}_{num_warps}w{WTV}x{WTK}"
    ) + ("_z" if write_z else "")
    assert WTK <= 16

    @fx.struct
    class SharedStorage:
        qk: fx.Array[fx.Float32, 2 * D, 16]
        v: fx.Array[fx.Float32, 2 * D, 16]
        flag: fx.Array[fx.Int32, 4, 16]

    @flyc.kernel(name=name, known_block_size=[NT, 1, 1])
    def kernel(
        x_mem: fx.Tensor,  # fuse_conv: in_proj qkvz [B, *]; else conv output mixed_qkv [B, *]
        ba_mem: fx.Tensor,  # fuse_conv: in_proj ba [B, *] = [b(HV) | a(HV)]; else b [B, HV]
        a_mem: fx.Tensor,  # unused with fuse_conv; else a [B, HV]
        conv_w_mem: fx.Tensor,  # [conv_dim, 4] bf16 (fuse_conv)
        conv_state_mem: fx.Tensor,  # conv state pool (fuse_conv)
        counter_mem: fx.Tensor,  # int32 [>= B*H] zero-initialised (fuse_conv)
        A_log_mem: fx.Tensor,
        dt_bias_mem: fx.Tensor,
        state_mem: fx.Tensor,  # fp32 ssm state pool, [slot, HV, V, K] inner contiguous
        indices_mem: fx.Tensor,  # int32 [B]
        out_mem: fx.Tensor,  # bf16
        z_out_mem: fx.Tensor,  # bf16 [B, HV, V] contiguous (write_z)
        stride_x_tok: fx.Int32,
        stride_ba_tok: fx.Int32,
        stride_a_tok: fx.Int32,
        stride_conv_slot: fx.Int64,
        stride_conv_ch: fx.Int32,
        stride_conv_w: fx.Int32,
        stride_state_slot: fx.Int64,
        stride_out_tok: fx.Int32,
    ):
        x = GTensor(x_mem, dtype=T.bf16, shape=(-1,))
        conv_w = GTensor(conv_w_mem, dtype=T.bf16, shape=(-1,))
        out = GTensor(out_mem, dtype=T.bf16, shape=(-1,))

        shared = fx.SharedAllocator().allocate(SharedStorage).peek()
        qk_lds = shared.qk.ptr
        q_lds = qk_lds
        k_lds = qk_lds + fx.Int32(D)  # q at [0, D), k at [D, 2D)
        v_lds = shared.v.ptr
        flag_lds = shared.flag.ptr

        tid = fx.thread_idx.x
        block = fx.block_idx.x
        b_i = block // fx.Int32(HV)
        hv = block % fx.Int32(HV)
        hk = hv // fx.Int32(RATIO)
        lane = tid % fx.Int32(_WARP)
        warp = tid // fx.Int32(_WARP)
        lane_k = lane % fx.Int32(WTK)
        k_vec_start = lane_k * fx.Int32(VPT)
        v_start = warp * fx.Int32(WTV) + lane // fx.Int32(WTK)

        # ---- metadata (uniform; scalar unit) ---------------------------------------
        b_off = b_i * stride_ba_tok + hv
        slot_raw = _sload_i32(indices_mem, b_i)
        r_b = _sload_bf16_as_f32(ba_mem, b_off)
        if fx.const_expr(fuse_conv):
            r_a = _sload_bf16_as_f32(ba_mem, b_off + fx.Int32(HV))
        else:
            r_a = _sload_bf16_as_f32(a_mem, b_i * stride_a_tok + hv)
        r_A_log = _sload_f32(A_log_mem, hv)
        if fx.const_expr(dt_bias_is_bf16):
            r_dt = _sload_bf16_as_f32(dt_bias_mem, hv)
        else:
            r_dt = _sload_f32(dt_bias_mem, hv)
        valid = slot_raw >= fx.Int32(0)
        # pad slots read slot 0 (discarded) so every load stays in straight-line code
        slot64 = fx.Int64(valid.select(slot_raw, fx.Int32(0)))

        # ---- per-thread channel loads ----------------------------------------------
        # thread t < D owns q channel t and v channel t; D <= t < 2D owns k channel t-D.
        is_qv = tid < fx.Int32(D)
        ch_local = is_qv.select(tid, tid - fx.Int32(D))
        qk_ch = is_qv.select(
            hk * fx.Int32(D) + ch_local, fx.Int32(H * D) + hk * fx.Int32(D) + ch_local
        )
        v_ch = fx.Int32(QK_DIM) + hv * fx.Int32(D) + ch_local
        row_x = b_i * stride_x_tok
        x_qk = fx.Float32(x[row_x + qk_ch])
        x_v = fx.Float32(x[row_x + v_ch])
        if fx.const_expr(fuse_conv):
            wq = conv_w.vec_load((qk_ch * fx.Int32(4),), 4).extf(T.vec(4, T.f32))
            wv = conv_w.vec_load((v_ch * fx.Int32(4),), 4).extf(T.vec(4, T.f32))
            cs = GTensor(
                conv_state_mem,
                dtype=T.bf16,
                shape=(-1,),
                static_bytes_offset_i64=slot64 * stride_conv_slot * fx.Int64(2),
            )
            cqk = qk_ch * stride_conv_ch
            cv = v_ch * stride_conv_ch
            s_qk = [fx.Float32(cs[cqk + fx.Int32(j) * stride_conv_w]) for j in range(3)]
            s_v = [fx.Float32(cs[cv + fx.Int32(j) * stride_conv_w]) for j in range(3)]
        if fx.const_expr(write_z):
            z_ch = fx.Int32(QK_DIM + HV * D) + hv * fx.Int32(D) + ch_local
            r_z_bf = fx.BFloat16(x[row_x + z_ch])
            # z rides along so the caller's gated norm reads a contiguous [B, HV, V]
            z_out = GTensor(z_out_mem, dtype=T.bf16, shape=(-1,))
            z_if = scf.IfOp(_to_raw(is_qv), results_=[], has_else=False)
            with ir.InsertionPoint(z_if.then_block):
                z_out.store(b_i * fx.Int32(HV * D) + hv * fx.Int32(D) + ch_local, r_z_bf)
                scf.YieldOp([])

        # ---- state slice: all 16 transactions before anything waits ----------------
        st_base = slot64 * stride_state_slot + fx.Int64(hv) * fx.Int64(D * D)
        st = GTensor(
            state_mem, dtype=T.f32, shape=(-1,), static_bytes_offset_i64=st_base * fx.Int64(4)
        )
        state_vecs = []
        for vi in range_constexpr(V_ITERS):
            gv = v_start + fx.Int32(vi * V_GROUP)
            for ki in range_constexpr(K_ITERS):
                kb = k_vec_start + fx.Int32(ki * TILE_K)
                state_vecs.append(
                    buffer_ops.buffer_load(
                        st.rsrc, gv * fx.Int32(D) + kb, vec_width=VPT, dtype=T.f32,
                        cache_modifier=_STATE_CACHE_POLICY,
                    )
                )

        # ---- conv (optional) and q/k/v exchange through LDS -------------------------
        if fx.const_expr(fuse_conv):

            def conv(svals, xv, w):
                acc = fx.Float32(0.0)
                for j in range_constexpr(3):
                    wj = fx.Float32(mlir_vector.ExtractOp(_to_raw(w), [], [j]).result)
                    acc = acc + (svals[j] * wj).to(fx.BFloat16).to(fx.Float32)
                w3 = fx.Float32(mlir_vector.ExtractOp(_to_raw(w), [], [3]).result)
                acc = acc + (xv * w3).to(fx.BFloat16).to(fx.Float32)
                return (acc * _sigmoid(acc)).to(fx.BFloat16).to(fx.Float32)

            q_or_k = conv(s_qk, x_qk, wq)
            vv = conv(s_v, x_v, wv)
        else:
            q_or_k = x_qk
            vv = x_v
        fx.ptr_store(q_or_k, qk_lds + tid)
        # threads >= D store a duplicate into the pad half: no branch around the load
        fx.ptr_store(vv, v_lds + tid)

        if fx.const_expr(fuse_conv):
            # v conv channels are owned by this workgroup: shift now.
            v_if = scf.IfOp(_to_raw(is_qv & valid), results_=[], has_else=False)
            with ir.InsertionPoint(v_if.then_block):
                cs.store(cv, s_v[1].to(fx.BFloat16))
                cs.store(cv + stride_conv_w, s_v[2].to(fx.BFloat16))
                cs.store(cv + fx.Int32(2) * stride_conv_w, x_v.to(fx.BFloat16))
                scf.YieldOp([])
        # LDS exchange barrier. Every thread has consumed its conv-state reads by now,
        # so the arrival counter can be bumped right after it.
        fx.gpu.barrier()
        if fx.const_expr(fuse_conv):
            # q/k conv channels are shared by RATIO workgroups: the last reader shifts
            # them. The atomic's return value is consumed only at the end (before the
            # epilogue barrier), so its latency hides behind the recurrence.
            ctr_addr = fx.Int64(
                GTensor.get_llvm_ptr(
                    None, counter_mem, fx.Int64(b_i * fx.Int32(H) + hk) * fx.Int64(4)
                )
            )
            ctr_if = scf.IfOp(
                _to_raw((tid == fx.Int32(0)) & valid), results_=[T.i32], has_else=True
            )
            with ir.InsertionPoint(ctr_if.then_block):
                seen_t = fx.Int32(_atomic_add_i32(ctr_addr, fx.Int32(1)))
                scf.YieldOp([_to_raw(seen_t)])
            with ir.InsertionPoint(ctr_if.else_block):
                scf.YieldOp([_to_raw(fx.Int32(0))])
            seen = fx.Int32(ctr_if.results[0])

        # ---- gating -----------------------------------------------------------------
        xg = r_a + r_dt
        softplus = (xg <= fx.Float32(softplus_threshold)).select(
            fx.math.log(fx.Float32(1.0) + fx.math.exp2(xg * fx.Float32(_LOG2E))), xg
        )
        g = -fx.math.exp2(r_A_log * fx.Float32(_LOG2E)) * softplus
        decay = fx.math.exp2(g * fx.Float32(_LOG2E))
        beta = _sigmoid(r_b).to(fx.BFloat16).to(fx.Float32)

        # ---- q/k fragments + l2 norm -------------------------------------------------
        vec_f32 = T.vec(VPT, T.f32)
        q_vecs, k_vecs = [], []
        sq = fx.Float32(0.0)
        sk = fx.Float32(0.0)
        for ki in range_constexpr(K_ITERS):
            kb = k_vec_start + fx.Int32(ki * TILE_K)
            qv = fx.ptr_load(q_lds + kb, result_type=vec_f32)
            kv = fx.ptr_load(k_lds + kb, result_type=vec_f32)
            q_vecs.append(qv)
            k_vecs.append(kv)
            sq = sq + _vsum(qv * qv)
            sk = sk + _vsum(kv * kv)
        sq = _group_allreduce(sq, WTK)
        sk = _group_allreduce(sk, WTK)
        inv_q = fx.Float32(1.0) / fx.math.sqrt(fx.Float32(sq) + fx.Float32(1e-6))
        inv_k = fx.Float32(1.0) / fx.math.sqrt(fx.Float32(sk) + fx.Float32(1e-6))
        for ki in range_constexpr(K_ITERS):
            q_vecs[ki] = q_vecs[ki] * fx.Float32(inv_q) * fx.Float32(SCALE)
            k_vecs[ki] = k_vecs[ki] * fx.Float32(inv_k)
        dkq = fx.full(VPT, 0.0, fx.Float32)
        for ki in range_constexpr(K_ITERS):
            dkq = _fma(k_vecs[ki], q_vecs[ki], dkq)
        dot_kq = _group_allreduce(_vsum(dkq), WTK)

        # ---- recurrence ----------------------------------------------------------------
        for vi in range_constexpr(V_ITERS):
            gv = v_start + fx.Int32(vi * V_GROUP)
            shk = fx.full(VPT, 0.0, fx.Float32)
            shq = fx.full(VPT, 0.0, fx.Float32)
            for ki in range_constexpr(K_ITERS):
                p = vi * K_ITERS + ki
                h = state_vecs[p] * fx.Float32(decay)
                state_vecs[p] = h
                shk = _fma(h, k_vecs[ki], shk)
                shq = _fma(h, q_vecs[ki], shq)
            s_hk = _group_allreduce(_vsum(shk), WTK)
            s_hq = _group_allreduce(_vsum(shq), WTK)
            v_val = fx.Float32(fx.ptr_load(v_lds + gv))
            v_new = (v_val - fx.Float32(s_hk)) * beta
            o_val = fx.Float32(s_hq) + v_new * fx.Float32(dot_kq)
            v_new_vec = mlir_vector.BroadcastOp(vec_f32, _to_raw(v_new)).vector
            upd = []
            for ki in range_constexpr(K_ITERS):
                p = vi * K_ITERS + ki
                upd.append(_fma(k_vecs[ki], v_new_vec, state_vecs[p]))
            s_if = scf.IfOp(_to_raw(valid), results_=[], has_else=False)
            with ir.InsertionPoint(s_if.then_block):
                for ki in range_constexpr(K_ITERS):
                    kb = k_vec_start + fx.Int32(ki * TILE_K)
                    buffer_ops.buffer_store(
                        upd[ki], st.rsrc, gv * fx.Int32(D) + kb,
                        cache_modifier=_STATE_CACHE_POLICY,
                    )
                scf.YieldOp([])
            o_bf = valid.select(o_val, fx.Float32(0.0)).to(fx.BFloat16)
            o_if = scf.IfOp(_to_raw(lane_k == fx.Int32(0)), results_=[], has_else=False)
            with ir.InsertionPoint(o_if.then_block):
                out.store(b_i * stride_out_tok + hv * fx.Int32(D) + gv, o_bf)
                scf.YieldOp([])

        # ---- publish the q/k conv-state write-back flag ------------------------------
        if fx.const_expr(fuse_conv):
            last = ((seen == fx.Int32(RATIO - 1)) & (tid == fx.Int32(0))) & valid
            pub_if = scf.IfOp(_to_raw(tid == fx.Int32(0)), results_=[], has_else=False)
            with ir.InsertionPoint(pub_if.then_block):
                fx.ptr_store(last.select(fx.Int32(1), fx.Int32(0)), flag_lds)
                rst_if = scf.IfOp(_to_raw(last), results_=[], has_else=False)
                with ir.InsertionPoint(rst_if.then_block):
                    _store_i32(ctr_addr, fx.Int32(0))
                    scf.YieldOp([])
                scf.YieldOp([])
            fx.gpu.barrier()
        if fx.const_expr(fuse_conv):
            is_last = (fx.Int32(fx.ptr_load(flag_lds)) == fx.Int32(1)) & valid
            wb_if = scf.IfOp(_to_raw(is_last), results_=[], has_else=False)
            with ir.InsertionPoint(wb_if.then_block):
                cs.store(cqk, s_qk[1].to(fx.BFloat16))
                cs.store(cqk + stride_conv_w, s_qk[2].to(fx.BFloat16))
                cs.store(cqk + fx.Int32(2) * stride_conv_w, x_qk.to(fx.BFloat16))
                scf.YieldOp([])


    @flyc.jit
    def launch(
        x_mem: fx.Tensor,
        ba_mem: fx.Tensor,
        a_mem: fx.Tensor,
        conv_w_mem: fx.Tensor,
        conv_state_mem: fx.Tensor,
        counter_mem: fx.Tensor,
        A_log_mem: fx.Tensor,
        dt_bias_mem: fx.Tensor,
        state_mem: fx.Tensor,
        indices_mem: fx.Tensor,
        out_mem: fx.Tensor,
        z_out_mem: fx.Tensor,
        stride_x_tok: fx.Int32,
        stride_ba_tok: fx.Int32,
        stride_a_tok: fx.Int32,
        stride_conv_slot: fx.Int64,
        stride_conv_ch: fx.Int32,
        stride_conv_w: fx.Int32,
        stride_state_slot: fx.Int64,
        stride_out_tok: fx.Int32,
        batch_size: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        kernel(
            x_mem,
            ba_mem,
            a_mem,
            conv_w_mem,
            conv_state_mem,
            counter_mem,
            A_log_mem,
            dt_bias_mem,
            state_mem,
            indices_mem,
            out_mem,
            z_out_mem,
            stride_x_tok,
            stride_ba_tok,
            stride_a_tok,
            stride_conv_slot,
            stride_conv_ch,
            stride_conv_w,
            stride_state_slot,
            stride_out_tok,
        ).launch(
            grid=(batch_size * fx.Int32(HV), 1, 1),
            block=(NT, 1, 1),
            stream=stream,
        )

    launch.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": True,
            "amdgpu-kernarg-preload-count": 16,
        }
    }
    return launch


__all__ = ["create_gdn_decode_fused_kernel"]
