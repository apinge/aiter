# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused GDN (gated delta rule) chain target-verify for gfx942, one workgroup per
(request, v-head), T draft tokens per request (MTP / EAGLE with topk = 1).

The T-step form of gdn_decode_fused: per workgroup, in issue order,
  1. index loads, then the whole fp32 state slice (16 x dwordx4 per lane), read once,
  2. the width-4 causal conv + SiLU of all T tokens from the in_proj output; each step's
     conv window [x(t-2), x(t-1), x(t)] goes to the intermediate conv-window cache and the
     conv state is shifted in place by T (q/k channels are shared by RATIO v-heads: the
     last workgroup to finish reading them writes them back, as in the decode kernel),
  3. per step: gating (fp32 sigmoid beta, like the Triton verify kernel), q/k L2 norm,
     h = h*exp(g); v' = (v - h.k)*beta; h += k v'^T; o = h.q; h is written to the
     intermediate state cache at step t; the SSM state itself is not written back.
Outputs: o [B*T, HV, V] bf16 and, with WRITE_Z, a contiguous z [B*T, HV, V].
Pad slots (index < 0) write a zero output and leave every cache untouched.
"""

import functools
import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf
from flydsl._mlir.dialects import vector as mlir_vector
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.gdn_decode_fused import (
    _STATE_CACHE_POLICY,
    _WARP,
    _atomic_add_i32,
    _fma,
    _group_allreduce,
    _sigmoid,
    _sload_bf16_as_f32,
    _sload_f32,
    _sload_i32,
    _store_i32,
    _vsum,
)
from aiter.ops.flydsl.kernels.tensor_shim import GTensor, _to_raw

_LOG2E = math.log2(math.e)


@functools.lru_cache(maxsize=None)
def create_gdn_verify_fused_kernel(
    num_k_heads: int,
    num_v_heads: int,
    num_steps: int,
    head_dim: int = 128,
    dt_bias_dtype: str = "bf16",
    warp_threads_k: int = 8,
    softplus_threshold: float = 20.0,
    write_z: bool = True,
):
    H, HV, D, TS = num_k_heads, num_v_heads, head_dim, num_steps
    RATIO = HV // H
    assert HV % H == 0 and D == 128 and TS >= 1
    VPT = 4
    num_warps = 4
    NT = num_warps * _WARP
    WTK = warp_threads_k
    WTV = _WARP // WTK
    TILE_K = WTK * VPT
    K_ITERS = D // TILE_K
    V_GROUP = num_warps * WTV
    V_ITERS = D // V_GROUP
    assert D % TILE_K == 0 and D % V_GROUP == 0 and NT == 2 * D and V_ITERS <= WTK <= 16
    QK_DIM = 2 * H * D
    SCALE = D**-0.5
    dt_bias_is_bf16 = dt_bias_dtype == "bf16"

    name = f"gdn_verify_fused_h{H}x{HV}_d{D}_t{TS}_{num_warps}w{WTV}x{WTK}" + (
        "_z" if write_z else ""
    )

    @fx.struct
    class SharedStorage:
        qk: fx.Array[fx.Float32, TS * 2 * D, 16]
        v: fx.Array[fx.Float32, TS * 2 * D, 16]
        flag: fx.Array[fx.Int32, 4, 16]

    @flyc.kernel(name=name, known_block_size=[NT, 1, 1])
    def kernel(
        x_mem: fx.Tensor,  # in_proj qkvz [B*T, *] = [q | k | v | z]
        ba_mem: fx.Tensor,  # in_proj ba [B*T, *] = [b(HV) | a(HV)]
        conv_w_mem: fx.Tensor,  # [conv_dim, 4] bf16
        conv_state_mem: fx.Tensor,  # conv state pool
        conv_win_mem: fx.Tensor,  # intermediate conv windows [row, step, conv_dim, 3]
        counter_mem: fx.Tensor,  # int32 [>= B*H] zero-initialised
        A_log_mem: fx.Tensor,
        dt_bias_mem: fx.Tensor,
        state_mem: fx.Tensor,  # fp32 ssm state pool, [slot, HV, V, K] inner contiguous
        inter_state_mem: fx.Tensor,  # fp32 [row, step, HV, V, K] inner contiguous
        indices_mem: fx.Tensor,  # int32 [B] ssm / conv slot
        inter_indices_mem: fx.Tensor,  # int32 [B] intermediate cache row
        out_mem: fx.Tensor,  # bf16 [B*T, HV, V]
        z_out_mem: fx.Tensor,  # bf16 [B*T, HV, V] (write_z)
        stride_x_tok: fx.Int32,
        stride_ba_tok: fx.Int32,
        stride_conv_slot: fx.Int64,
        stride_conv_ch: fx.Int32,
        stride_conv_w: fx.Int32,
        stride_win_row: fx.Int64,
        stride_win_step: fx.Int32,
        stride_win_ch: fx.Int32,
        stride_win_w: fx.Int32,
        stride_state_slot: fx.Int64,
        stride_inter_row: fx.Int64,
        stride_inter_step: fx.Int32,
    ):
        x = GTensor(x_mem, dtype=T.bf16, shape=(-1,))
        conv_w = GTensor(conv_w_mem, dtype=T.bf16, shape=(-1,))
        out = GTensor(out_mem, dtype=T.bf16, shape=(-1,))

        shared = fx.SharedAllocator().allocate(SharedStorage).peek()
        qk_lds = shared.qk.ptr
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
        slot_raw = _sload_i32(indices_mem, b_i)
        irow_raw = _sload_i32(inter_indices_mem, b_i)
        tok0 = b_i * fx.Int32(TS)
        r_b, r_a = [], []
        for t in range_constexpr(TS):
            b_off = (tok0 + fx.Int32(t)) * stride_ba_tok + hv
            r_b.append(_sload_bf16_as_f32(ba_mem, b_off))
            r_a.append(_sload_bf16_as_f32(ba_mem, b_off + fx.Int32(HV)))
        r_A_log = _sload_f32(A_log_mem, hv)
        if fx.const_expr(dt_bias_is_bf16):
            r_dt = _sload_bf16_as_f32(dt_bias_mem, hv)
        else:
            r_dt = _sload_f32(dt_bias_mem, hv)
        valid = slot_raw >= fx.Int32(0)
        cache_ok = valid & (irow_raw >= fx.Int32(0))
        # pad slots read slot 0 / row 0 (discarded) so every load stays straight-line
        slot64 = fx.Int64(valid.select(slot_raw, fx.Int32(0)))
        irow64 = fx.Int64(cache_ok.select(irow_raw, fx.Int32(0)))

        # ---- per-thread channel loads ----------------------------------------------
        # thread t < D owns q channel t and v channel t; D <= t < 2D owns k channel t-D.
        is_qv = tid < fx.Int32(D)
        ch_local = is_qv.select(tid, tid - fx.Int32(D))
        qk_ch = is_qv.select(
            hk * fx.Int32(D) + ch_local, fx.Int32(H * D) + hk * fx.Int32(D) + ch_local
        )
        v_ch = fx.Int32(QK_DIM) + hv * fx.Int32(D) + ch_local
        x_qk, x_v = [], []
        for t in range_constexpr(TS):
            row_x = (tok0 + fx.Int32(t)) * stride_x_tok
            x_qk.append(fx.Float32(x[row_x + qk_ch]))
            x_v.append(fx.Float32(x[row_x + v_ch]))
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
        # ---- state slice: all 16 transactions before anything waits ----------------
        st_base = slot64 * stride_state_slot + fx.Int64(hv) * fx.Int64(D * D)
        st = GTensor(
            state_mem,
            dtype=T.f32,
            shape=(-1,),
            static_bytes_offset_i64=st_base * fx.Int64(4),
        )
        state_vecs = []
        for vi in range_constexpr(V_ITERS):
            gv = v_start + fx.Int32(vi * V_GROUP)
            for ki in range_constexpr(K_ITERS):
                kb = k_vec_start + fx.Int32(ki * TILE_K)
                state_vecs.append(
                    buffer_ops.buffer_load(
                        st.rsrc,
                        gv * fx.Int32(D) + kb,
                        vec_width=VPT,
                        dtype=T.f32,
                        cache_modifier=_STATE_CACHE_POLICY,
                    )
                )

        # ---- conv of all T tokens, conv windows, LDS exchange ------------------------
        def conv(svals, xv, w):
            acc = fx.Float32(0.0)
            for j in range_constexpr(3):
                wj = fx.Float32(mlir_vector.ExtractOp(_to_raw(w), [], [j]).result)
                acc = acc + (svals[j] * wj).to(fx.BFloat16).to(fx.Float32)
            w3 = fx.Float32(mlir_vector.ExtractOp(_to_raw(w), [], [3]).result)
            acc = acc + (xv * w3).to(fx.BFloat16).to(fx.Float32)
            return (acc * _sigmoid(acc)).to(fx.BFloat16).to(fx.Float32)

        win = GTensor(
            conv_win_mem,
            dtype=T.bf16,
            shape=(-1,),
            static_bytes_offset_i64=irow64 * stride_win_row * fx.Int64(2),
        )
        wqk = qk_ch * stride_win_ch
        wvo = v_ch * stride_win_ch
        # q/k windows are identical in the RATIO workgroups of a k-head: the first writes.
        qk_win_ok = cache_ok & ((hv % fx.Int32(RATIO)) == fx.Int32(0))
        w_qk, w_v = list(s_qk), list(s_v)
        for t in range_constexpr(TS):
            fx.ptr_store(conv(w_qk, x_qk[t], wq), qk_lds + fx.Int32(t * 2 * D) + tid)
            # threads >= D store a duplicate into the pad half: no branch around the load
            fx.ptr_store(conv(w_v, x_v[t], wv), v_lds + fx.Int32(t * 2 * D) + tid)
            w_qk = [w_qk[1], w_qk[2], x_qk[t]]
            w_v = [w_v[1], w_v[2], x_v[t]]
            step = fx.Int32(t) * stride_win_step
            v_win_if = scf.IfOp(_to_raw(is_qv & cache_ok), results_=[], has_else=False)
            with ir.InsertionPoint(v_win_if.then_block):
                for j in range_constexpr(3):
                    win.store(
                        step + wvo + fx.Int32(j) * stride_win_w,
                        w_v[j].to(fx.BFloat16),
                    )
                scf.YieldOp([])
            qk_win_if = scf.IfOp(_to_raw(qk_win_ok), results_=[], has_else=False)
            with ir.InsertionPoint(qk_win_if.then_block):
                for j in range_constexpr(3):
                    win.store(
                        step + wqk + fx.Int32(j) * stride_win_w,
                        w_qk[j].to(fx.BFloat16),
                    )
                scf.YieldOp([])

        # v conv channels are owned by this workgroup: shift them by T now.
        v_if = scf.IfOp(_to_raw(is_qv & valid), results_=[], has_else=False)
        with ir.InsertionPoint(v_if.then_block):
            for j in range_constexpr(3):
                cs.store(cv + fx.Int32(j) * stride_conv_w, w_v[j].to(fx.BFloat16))
            scf.YieldOp([])
        fx.gpu.barrier()
        # q/k conv channels are shared by RATIO workgroups: the last reader shifts them.
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
        if fx.const_expr(write_z):
            z_ch = fx.Int32(QK_DIM + HV * D) + hv * fx.Int32(D) + ch_local
            z_out = GTensor(z_out_mem, dtype=T.bf16, shape=(-1,))
            r_z = [
                fx.BFloat16(x[(tok0 + fx.Int32(t)) * stride_x_tok + z_ch])
                for t in range(TS)
            ]
            z_if = scf.IfOp(_to_raw(is_qv), results_=[], has_else=False)
            with ir.InsertionPoint(z_if.then_block):
                for t in range_constexpr(TS):
                    z_out.store(
                        (tok0 + fx.Int32(t)) * fx.Int32(HV * D)
                        + hv * fx.Int32(D)
                        + ch_local,
                        r_z[t],
                    )
                scf.YieldOp([])


        # ---- per-step gating, l2 norm, recurrence ------------------------------------
        neg_A = -fx.math.exp2(r_A_log * fx.Float32(_LOG2E))
        vec_f32 = T.vec(VPT, T.f32)
        # Branch-free intermediate-state stores: a pad request gets a zero-sized buffer
        # descriptor, so the hardware drops its stores and they schedule with the math.
        ist_addr = GTensor.get_llvm_ptr(
            None,
            inter_state_mem,
            (irow64 * stride_inter_row + fx.Int64(hv) * fx.Int64(D * D)) * fx.Int64(4),
        )
        ist_rsrc = buffer_ops.create_buffer_resource_from_addr(
            ist_addr,
            num_records_bytes=_to_raw(
                cache_ok.select(fx.Int64(0x7FFFFFFF), fx.Int64(0))
            ),
        )
        for t in range_constexpr(TS):
            xg = r_a[t] + r_dt
            softplus = (xg <= fx.Float32(softplus_threshold)).select(
                fx.math.log(fx.Float32(1.0) + fx.math.exp2(xg * fx.Float32(_LOG2E))),
                xg,
            )
            decay = fx.math.exp2(neg_A * softplus * fx.Float32(_LOG2E))
            beta = _sigmoid(r_b[t])

            q_lds = qk_lds + fx.Int32(t * 2 * D)
            k_lds = q_lds + fx.Int32(D)
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

            out_row = (tok0 + fx.Int32(t)) * fx.Int32(HV * D) + hv * fx.Int32(D)
            o_vals = []
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
                v_val = fx.Float32(fx.ptr_load(v_lds + fx.Int32(t * 2 * D) + gv))
                v_new = (v_val - fx.Float32(s_hk)) * beta
                o_val = fx.Float32(s_hq) + v_new * fx.Float32(dot_kq)
                v_new_vec = mlir_vector.BroadcastOp(vec_f32, _to_raw(v_new)).vector
                for ki in range_constexpr(K_ITERS):
                    p = vi * K_ITERS + ki
                    state_vecs[p] = _fma(k_vecs[ki], v_new_vec, state_vecs[p])
                for ki in range_constexpr(K_ITERS):
                    kb = k_vec_start + fx.Int32(ki * TILE_K)
                    buffer_ops.buffer_store(
                        state_vecs[vi * K_ITERS + ki],
                        ist_rsrc,
                        fx.Int32(t) * stride_inter_step + gv * fx.Int32(D) + kb,
                        cache_modifier=_STATE_CACHE_POLICY,
                    )
                o_vals.append(o_val)
            # every lane of a k-group holds the reduced o of all V_ITERS rows: lane_k
            # stores row lane_k % V_ITERS (duplicates write the same value), no branch
            o_vi = lane_k % fx.Int32(V_ITERS)
            o_sel = o_vals[0]
            for vi in range_constexpr(1, V_ITERS):
                o_sel = (o_vi == fx.Int32(vi)).select(o_vals[vi], o_sel)
            o_bf = valid.select(o_sel, fx.Float32(0.0)).to(fx.BFloat16)
            out.store(out_row + v_start + o_vi * fx.Int32(V_GROUP), o_bf)

        # ---- publish the q/k conv-state write-back flag ------------------------------
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
        is_last = (fx.Int32(fx.ptr_load(flag_lds)) == fx.Int32(1)) & valid
        wb_if = scf.IfOp(_to_raw(is_last), results_=[], has_else=False)
        with ir.InsertionPoint(wb_if.then_block):
            for j in range_constexpr(3):
                cs.store(cqk + fx.Int32(j) * stride_conv_w, w_qk[j].to(fx.BFloat16))
            scf.YieldOp([])

    @flyc.jit
    def launch(
        x_mem: fx.Tensor,
        ba_mem: fx.Tensor,
        conv_w_mem: fx.Tensor,
        conv_state_mem: fx.Tensor,
        conv_win_mem: fx.Tensor,
        counter_mem: fx.Tensor,
        A_log_mem: fx.Tensor,
        dt_bias_mem: fx.Tensor,
        state_mem: fx.Tensor,
        inter_state_mem: fx.Tensor,
        indices_mem: fx.Tensor,
        inter_indices_mem: fx.Tensor,
        out_mem: fx.Tensor,
        z_out_mem: fx.Tensor,
        stride_x_tok: fx.Int32,
        stride_ba_tok: fx.Int32,
        stride_conv_slot: fx.Int64,
        stride_conv_ch: fx.Int32,
        stride_conv_w: fx.Int32,
        stride_win_row: fx.Int64,
        stride_win_step: fx.Int32,
        stride_win_ch: fx.Int32,
        stride_win_w: fx.Int32,
        stride_state_slot: fx.Int64,
        stride_inter_row: fx.Int64,
        stride_inter_step: fx.Int32,
        batch_size: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        kernel(
            x_mem,
            ba_mem,
            conv_w_mem,
            conv_state_mem,
            conv_win_mem,
            counter_mem,
            A_log_mem,
            dt_bias_mem,
            state_mem,
            inter_state_mem,
            indices_mem,
            inter_indices_mem,
            out_mem,
            z_out_mem,
            stride_x_tok,
            stride_ba_tok,
            stride_conv_slot,
            stride_conv_ch,
            stride_conv_w,
            stride_win_row,
            stride_win_step,
            stride_win_ch,
            stride_win_w,
            stride_state_slot,
            stride_inter_row,
            stride_inter_step,
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


__all__ = ["create_gdn_verify_fused_kernel"]
