# Copyright (c) 2025 by FlashInfer team.
# Copyright (c) 2026 AquiLLM contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Narrow SM90 CuTe MTP kernel; no released FlashInfer entrypoint is modified.

Derived from FlashInfer v0.6.18 gdn_decode_mtp.py inline ILP=2 recurrence:
https://github.com/flashinfer-ai/flashinfer/blob/v0.6.18/flashinfer/gdn_kernels/gdn_decode_mtp.py
Source SHA256: a091f7fc33e4c5a3209683e3b8c3a8da2a0126a1cb5447c58a243bad2fff5907

Specialization: N1, H16/HV48, K128/V128, tile_v8, vec_size4, two state rows
per warp. FP16 operands feed FP32 recurrence directly by default. A private
compile-time variant retains BF16 operand/output rounding for comparison.
Metadata, output conversion and masked direct state scatter
run in the same kernel. Compilation cache holds code only, never GPU scratch.
"""
import threading

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda

_compiled = {}
_compile_lock = threading.Lock()


@cute.kernel
def _native_mtp(
    State: cute.Tensor, A_log: cute.Tensor, Dt: cute.Tensor,
    Q: cute.Tensor, K: cute.Tensor, V: cute.Tensor,
    A: cute.Tensor, B: cute.Tensor, O: cute.Tensor,
    Offsets: cute.Tensor, Indices: cute.Tensor, Accepted: cute.Tensor,
    T: cutlass.Constexpr[int], SLOTS: cutlass.Constexpr[int],
    COLUMNS: cutlass.Constexpr[int], SCALE: cutlass.Constexpr[float],
    NORM: cutlass.Constexpr[bool], ROUND_BF16: cutlass.Constexpr[bool],
):
    tidx, _, _ = cute.arch.thread_idx()
    lane = tidx % 32
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    block, _, _ = cute.arch.block_idx()
    head = block // 16
    q_head = head // 3
    v0 = (block % 16) * 8 + warp * 2
    v1 = v0 + 1

    start = cutlass.Int64(Offsets[0])
    end = cutlass.Int64(Offsets[1])
    accepted = cutlass.Int64(Accepted[0])
    valid = ((start >= 0) & (end >= start) & (end <= T)
             & (end > start) & (accepted > 0) & (accepted <= COLUMNS))
    # Form the accepted-column address only inside its bounds guard. Every
    # thread in a warp sees the same metadata, including each shuffle guard.
    if valid:
        slot = cutlass.Int64(Indices[0, accepted - 1])
        if (slot > 0) & (slot < SLOTS):
            length = end - start
            read_view = State[(slot, head, None, None)]
            h0 = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), cutlass.Float32)
            h1 = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), cutlass.Float32)
            cute.autovec_copy(cute.local_tile(read_view, (1,4), (v0,lane)), h0)
            cute.autovec_copy(cute.local_tile(read_view, (1,4), (v1,lane)), h1)

            q_source = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), cutlass.Float16)
            k_source = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), cutlass.Float16)
            q_all = cute.make_rmem_tensor(cute.make_layout((T,4), stride=(4,1)), cutlass.Float32)
            k_all = cute.make_rmem_tensor(cute.make_layout((T,4), stride=(4,1)), cutlass.Float32)
            inv_q = cute.make_rmem_tensor(cute.make_layout((T,), stride=(1,)), cutlass.Float32)
            inv_k = cute.make_rmem_tensor(cute.make_layout((T,), stride=(1,)), cutlass.Float32)
            decay = cute.make_rmem_tensor(cute.make_layout((T,), stride=(1,)), cutlass.Float32)
            beta = cute.make_rmem_tensor(cute.make_layout((T,), stride=(1,)), cutlass.Float32)
            log_decay = cutlass.Float32(A_log[head])
            dt_bias = cutlass.Float32(Dt[head])

            # Match the inspected inline kernel: precompute q/k, normalization
            # and gates across T while retaining the state tile in registers.
            for t in cutlass.range_constexpr(T):
                if t < length:
                    cute.autovec_copy(cute.local_tile(Q, (1,1,1,4), (0,start+t,q_head,lane)), q_source)
                    cute.autovec_copy(cute.local_tile(K, (1,1,1,4), (0,start+t,q_head,lane)), k_source)
                    sum_q = cutlass.Float32(0.0)
                    sum_k = cutlass.Float32(0.0)
                    for i in cutlass.range_constexpr(4):
                        q_value = cutlass.Float32(q_source[i])
                        k_value = cutlass.Float32(k_source[i])
                        if cutlass.const_expr(ROUND_BF16):
                            q_value = cutlass.Float32(cutlass.BFloat16(q_source[i]))
                            k_value = cutlass.Float32(cutlass.BFloat16(k_source[i]))
                        q_all[t,i] = q_value
                        k_all[t,i] = k_value
                        if cutlass.const_expr(NORM):
                            sum_q += q_value * q_value
                            sum_k += k_value * k_value
                        else:
                            q_all[t,i] = q_value * SCALE
                    if cutlass.const_expr(NORM):
                        for offset in [16,8,4,2,1]:
                            sum_q += cute.arch.shuffle_sync_bfly(sum_q, offset=offset, mask=-1, mask_and_clamp=31)
                            sum_k += cute.arch.shuffle_sync_bfly(sum_k, offset=offset, mask=-1, mask_and_clamp=31)
                        inv_q[t] = cute.rsqrt(sum_q + 1e-6, fastmath=True) * SCALE
                        inv_k[t] = cute.rsqrt(sum_k + 1e-6, fastmath=True)

                    a_value = cutlass.Float32(A[0,start+t,head])
                    b_value = cutlass.Float32(B[0,start+t,head])
                    if cutlass.const_expr(ROUND_BF16):
                        a_value = cutlass.Float32(cutlass.BFloat16(A[0,start+t,head]))
                        b_value = cutlass.Float32(cutlass.BFloat16(B[0,start+t,head]))
                    x = a_value + dt_bias
                    # Only guard the inactive exponential. Ordinary softplus
                    # operands and threshold20 are preserved exactly; no x clamp.
                    exp_argument = x if x <= 20.0 else cutlass.Float32(0.0)
                    sp_value = cute.log(cutlass.Float32(1.0) + cute.exp(exp_argument, fastmath=True), fastmath=True)
                    use_sp = cutlass.Float32(1.0) if x <= 20.0 else cutlass.Float32(0.0)
                    sp_x = use_sp * sp_value + (cutlass.Float32(1.0) - use_sp) * x
                    decay[t] = cute.exp(-cute.exp(log_decay, fastmath=True) * sp_x, fastmath=True)
                    beta[t] = cutlass.Float32(1.0) / (cutlass.Float32(1.0) + cute.exp(-b_value, fastmath=True))

            for t in cutlass.range_constexpr(T):
                if t < length:
                    sum_hk0 = cutlass.Float32(0.0)
                    sum_hk1 = cutlass.Float32(0.0)
                    for i in cutlass.range_constexpr(4):
                        h0[i] *= decay[t]
                        h1[i] *= decay[t]
                        sum_hk0 += h0[i] * k_all[t,i]
                        sum_hk1 += h1[i] * k_all[t,i]
                    for offset in [16,8,4,2,1]:
                        sum_hk0 += cute.arch.shuffle_sync_bfly(sum_hk0, offset=offset, mask=-1, mask_and_clamp=31)
                        sum_hk1 += cute.arch.shuffle_sync_bfly(sum_hk1, offset=offset, mask=-1, mask_and_clamp=31)
                    if cutlass.const_expr(NORM):
                        sum_hk0 *= inv_k[t]
                        sum_hk1 *= inv_k[t]
                    value0 = cutlass.Float32(V[0,start+t,head,v0])
                    value1 = cutlass.Float32(V[0,start+t,head,v1])
                    if cutlass.const_expr(ROUND_BF16):
                        value0 = cutlass.Float32(cutlass.BFloat16(V[0,start+t,head,v0]))
                        value1 = cutlass.Float32(cutlass.BFloat16(V[0,start+t,head,v1]))
                    update0 = (value0 - sum_hk0) * beta[t]
                    update1 = (value1 - sum_hk1) * beta[t]
                    if cutlass.const_expr(NORM):
                        update0 *= inv_k[t]
                        update1 *= inv_k[t]
                    sum_hq0 = cutlass.Float32(0.0)
                    sum_hq1 = cutlass.Float32(0.0)
                    for i in cutlass.range_constexpr(4):
                        h0[i] += k_all[t,i] * update0
                        h1[i] += k_all[t,i] * update1
                        sum_hq0 += h0[i] * q_all[t,i]
                        sum_hq1 += h1[i] * q_all[t,i]

                    # Each CTA owns a disjoint state tile for all timesteps;
                    # repeated destinations are overwritten sequentially.
                    destination = cutlass.Int64(Indices[0,t])
                    if (destination > 0) & (destination < SLOTS):
                        write_view = State[(destination,head,None,None)]
                        cute.autovec_copy(h0, cute.local_tile(write_view, (1,4), (v0,lane)))
                        cute.autovec_copy(h1, cute.local_tile(write_view, (1,4), (v1,lane)))

                    for offset in [16,8,4,2,1]:
                        sum_hq0 += cute.arch.shuffle_sync_bfly(sum_hq0, offset=offset, mask=-1, mask_and_clamp=31)
                        sum_hq1 += cute.arch.shuffle_sync_bfly(sum_hq1, offset=offset, mask=-1, mask_and_clamp=31)
                    if cutlass.const_expr(NORM):
                        sum_hq0 *= inv_q[t]
                        sum_hq1 *= inv_q[t]
                    if lane == 0:
                        if cutlass.const_expr(ROUND_BF16):
                            O[0,start+t,head,v0] = cutlass.Float16(cutlass.BFloat16(sum_hq0))
                            O[0,start+t,head,v1] = cutlass.Float16(cutlass.BFloat16(sum_hq1))
                        else:
                            O[0,start+t,head,v0] = cutlass.Float16(sum_hq0)
                            O[0,start+t,head,v1] = cutlass.Float16(sum_hq1)


@cute.jit
def _entry(State, A_log, Dt, Q, K, V, A, B, O, Offsets, Indices, Accepted,
           T: cutlass.Constexpr[int], SLOTS: cutlass.Constexpr[int],
           COLUMNS: cutlass.Constexpr[int], SCALE: cutlass.Constexpr[float],
           NORM: cutlass.Constexpr[bool], ROUND_BF16: cutlass.Constexpr[bool],
           stream: cuda.CUstream):
    _native_mtp(State,A_log,Dt,Q,K,V,A,B,O,Offsets,Indices,Accepted,
                T,SLOTS,COLUMNS,SCALE,NORM,ROUND_BF16).launch(
        grid=(48*16,1,1), block=(128,1,1), smem=0, stream=stream)


def launch(A_log,a,b,dt_bias,q,k,v,initial_state,cu_seqlens,ssm_state_indices,
           num_accepted_tokens,scale=None,use_qk_l2norm_in_kernel=False,
           _operand_precision="fp16",**unused):
    """One CuTe launch; output allocations belong to this call/captured graph."""
    if _operand_precision not in ("fp16", "bf16"):
        raise ValueError("unsupported private operand precision")
    round_bf16 = _operand_precision == "bf16"
    if a.ndim == 2:
        a = a.unsqueeze(0)
    if b.ndim == 2:
        b = b.unsqueeze(0)
    output = torch.empty(v.shape,device=v.device,dtype=torch.float16)
    values = (initial_state,A_log,dt_bias,q,k,v,a,b,output,cu_seqlens,
              ssm_state_indices,num_accepted_tokens)
    # Real model Parameters retain requires_grad even in inference_mode.
    # DLPack/FFI need detached views on cold and cached calls; detach shares
    # storage and leaves original Parameter flags, pointers and dtype intact.
    values = tuple(tensor.detach() if tensor.requires_grad else tensor for tensor in values)
    t = q.shape[1]
    scale_value = 128**-0.5 if scale is None else float(scale)
    # Every static shape/stride and polymorphic dtype is represented. Layouts
    # for operands/metadata are dynamic; the packed-inner pool is static.
    key = (q.device,t,initial_state.shape[0],tuple(initial_state.stride()),
           ssm_state_indices.shape[1],scale_value,use_qk_l2norm_in_kernel,
           tuple(tensor.dtype for tensor in values),round_bf16)
    stream = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
    compiled = _compiled.get(key)
    if compiled is None:
        with _compile_lock:
            compiled = _compiled.get(key)
            if compiled is None:
                arguments = []
                for index,tensor in enumerate(values):
                    alignment = 16 if index in (0,3,4,5,8) else 1
                    argument = from_dlpack(tensor,assumed_align=alignment)
                    if index != 0:
                        argument = argument.mark_layout_dynamic()
                    arguments.append(argument)
                compiled = cute.compile(
                    _entry,*arguments,T=t,SLOTS=initial_state.shape[0],
                    COLUMNS=ssm_state_indices.shape[1],SCALE=scale_value,
                    NORM=use_qk_l2norm_in_kernel,ROUND_BF16=round_bf16,stream=stream,
                    options="--enable-tvm-ffi --generate-line-info")
                _compiled[key] = compiled
    compiled(*values,stream)
    return output
