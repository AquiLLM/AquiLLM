# SPDX-License-Identifier: Apache-2.0
"""Genesis 34e2693 arithmetic with fixed-grid device split selection.

Derived from sndr/engines/vllm/kernels_legacy/p67_multi_query_kernel.py.
Committed QK remains tf32, PV remains tf32x3, raw tail remains scalar FP32,
and scratch stores normalized output with log2 LSE. This module enables no
tuning: the coordinator supplies an already validated frozen SplitPlan.
"""
from __future__ import annotations

from ..contracts import SplitPlan, VerifyBatch

_KERNEL = None


def _build_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def reference_stage1(
        Q_ptr,
        KV_cache_ptr,
        Block_table_ptr,
        Seq_lens_ptr,
        K_chunk_ptr,
        V_chunk_ptr,
        Mid_o_ptr,
        stride_qb, stride_qt, stride_qh, stride_qd,
        stride_cache_block, stride_cache_pos, stride_cache_head,
        stride_bt_b,
        stride_kkb, stride_kkt, stride_kkh, stride_kkd,
        stride_vkb, stride_vkt, stride_vkh, stride_vkd,
        stride_mb, stride_mh, stride_ms, stride_mq, stride_moh, stride_md,
        SCALE: tl.constexpr,
        K_PLUS_1: tl.constexpr,
        BLOCK_D: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        BLOCK_KV: tl.constexpr,
        HEADS_PER_KV: tl.constexpr,
        BLOCK_QH: tl.constexpr,
        Hq_TOTAL: tl.constexpr,
        KPS: tl.constexpr,
        VAL_DATA_BYTES: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
        RAW_SID: tl.constexpr,
        BUCKETS: tl.constexpr,
        KP1_PAD: tl.constexpr,
        FP8_E4B15: tl.constexpr = 0,
        DOT_FP16: tl.constexpr = 0,
    ):
        bid = tl.program_id(0)
        kv_head = tl.program_id(1)
        sid = tl.program_id(2)
        is_raw = sid == RAW_SID

        offs_h = tl.arange(0, BLOCK_QH)
        abs_head = kv_head * HEADS_PER_KV + offs_h
        lane_valid = offs_h < HEADS_PER_KV
        head_mask = lane_valid & (abs_head < Hq_TOTAL)

        total_seq_len = tl.load(Seq_lens_ptr + bid)
        prior_seq_len = total_seq_len - K_PLUS_1

        offs_d = tl.arange(0, BLOCK_D)
        d_mask = offs_d < HEAD_DIM
        offs_kv = tl.arange(0, BLOCK_KV)
        vb_idx = offs_d // 2
        vb_shift = (offs_d % 2) * 4

        KP1_EFF: tl.constexpr = KP1_PAD if KP1_PAD > 0 else K_PLUS_1
        M_state = tl.zeros([KP1_EFF, BLOCK_QH], dtype=tl.float32) - float("inf")
        L_state = tl.zeros([KP1_EFF, BLOCK_QH], dtype=tl.float32)
        acc = tl.zeros([KP1_EFF, BLOCK_QH, BLOCK_D], dtype=tl.float32)
        q_t_range = tl.arange(0, KP1_EFF)
        qt_valid = q_t_range < K_PLUS_1
        q_base = bid * stride_qb
        bt_base = bid * stride_bt_b
        _kv_head_byte_offset = tl.cast(kv_head, tl.int64) * stride_cache_head
        SCALE_LOG2E_split = SCALE * 1.4426950408889634

        # This split's committed sub-range of [0, prior_seq_len). The raw slot
        # does NO committed work (empty range). split_end CLAMPED to prior — a
        # split reading into [prior,total) would re-read the 4-bit-V spec tokens.
        active_splits = NUM_SPLITS
        if len(BUCKETS) > 0:
            active_splits = BUCKETS[-1][1]
            for bucket_index in tl.static_range(len(BUCKETS) - 1, -1, -1):
                active_splits = tl.where(prior_seq_len <= BUCKETS[bucket_index][0],
                                         BUCKETS[bucket_index][1], active_splits)
        split_len = tl.cdiv(prior_seq_len, active_splits)
        c_start = tl.minimum(sid * split_len, prior_seq_len)
        c_end = tl.minimum((sid + 1) * split_len, prior_seq_len)
        inactive = is_raw | (sid >= active_splits)
        c_start = tl.where(inactive, 0, c_start)
        c_end = tl.where(inactive, 0, c_end)

        for start_n in tl.range(c_start, c_end, BLOCK_KV):
            seq_offset = start_n + offs_kv
            tile_mask = seq_offset < c_end
            page_idx = seq_offset // BLOCK_SIZE
            page_off = seq_offset % BLOCK_SIZE
            physical_block = tl.load(
                Block_table_ptr + bt_base + page_idx,
                mask=tile_mask, other=0, cache_modifier=".ca",
            ).to(tl.int64)
            slot_bases = (
                physical_block * stride_cache_block
                + page_off.to(tl.int64) * stride_cache_pos
                + _kv_head_byte_offset
            )
            k_addrs = slot_bases[None, :] + offs_d[:, None]
            k_raw = tl.load(
                KV_cache_ptr + k_addrs,
                mask=d_mask[:, None] & tile_mask[None, :],
                other=0, cache_modifier=".cg",
            )
            if FP8_E4B15:
                k_float = k_raw.to(tl.float8e4b15, bitcast=True).to(tl.float32)
            else:
                k_float = k_raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            K_tile = k_float

            val_bases = slot_bases + KPS
            val_addrs = val_bases[:, None] + vb_idx[None, :]
            val_raw = tl.load(
                KV_cache_ptr + val_addrs,
                mask=tile_mask[:, None] & d_mask[None, :],
                other=0, cache_modifier=".cg",
            ).to(tl.int32)
            v_idx = ((val_raw >> vb_shift[None, :]) & 0xF).to(tl.float32)
            sc_bases = val_bases + VAL_DATA_BYTES
            sc_lo = tl.load(KV_cache_ptr + sc_bases, mask=tile_mask, other=0,
                            cache_modifier=".ca").to(tl.uint16)
            sc_hi = tl.load(KV_cache_ptr + sc_bases + 1, mask=tile_mask, other=0,
                            cache_modifier=".ca").to(tl.uint16)
            v_scales = (sc_lo | (sc_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            zr_lo = tl.load(KV_cache_ptr + sc_bases + 2, mask=tile_mask, other=0,
                            cache_modifier=".ca").to(tl.uint16)
            zr_hi = tl.load(KV_cache_ptr + sc_bases + 3, mask=tile_mask, other=0,
                            cache_modifier=".ca").to(tl.uint16)
            v_zeros = (zr_lo | (zr_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            V_tile = v_idx * v_scales[:, None] + v_zeros[:, None]

            for t in tl.static_range(0, K_PLUS_1):
                q_abs_pos_t = prior_seq_len + t
                q_addrs_t = (
                    q_base + t * stride_qt
                    + abs_head[:, None] * stride_qh + offs_d[None, :] * stride_qd
                )
                Q_t = tl.load(
                    Q_ptr + q_addrs_t,
                    mask=head_mask[:, None] & d_mask[None, :],
                    other=0.0, cache_modifier=".ca",
                ).to(tl.float32)
                if DOT_FP16:
                    S_t = SCALE_LOG2E_split * tl.dot(
                        Q_t.to(tl.float16), K_tile.to(tl.float16), out_dtype=tl.float32)
                else:
                    # tf32 single-pass (split-K committed QK). ieee A/B ruled out
                    # precision for the 27B tool-call mangle (see fused-M comment).
                    S_t = SCALE_LOG2E_split * tl.dot(
                        Q_t, K_tile, out_dtype=tl.float32, input_precision='tf32')
                # committed causal is always true (seq_offset<prior<=q_abs_pos_t);
                # keep the mask for tail-tile padding correctness.
                causal = q_abs_pos_t >= seq_offset
                valid = head_mask[:, None] & tile_mask[None, :] & causal[None, :]
                S_t = tl.where(valid, S_t, -float("inf"))
                t_mask = q_t_range == t
                M_old_t = tl.sum(tl.where(t_mask[:, None], M_state, 0.0), axis=0)
                L_old_t = tl.sum(tl.where(t_mask[:, None], L_state, 0.0), axis=0)
                acc_old_t = tl.sum(tl.where(t_mask[:, None, None], acc, 0.0), axis=0)
                M_new_t = tl.maximum(tl.max(S_t, axis=1), M_old_t)
                alpha_t = tl.exp2(M_old_t - M_new_t)
                P_t = tl.exp2(S_t - M_new_t[:, None])
                L_new_t = L_old_t * alpha_t + tl.sum(P_t, axis=1)
                if DOT_FP16:
                    acc_new_t = acc_old_t * alpha_t[:, None] + tl.dot(
                        P_t.to(tl.float16), V_tile.to(tl.float16), out_dtype=tl.float32)
                else:
                    acc_new_t = acc_old_t * alpha_t[:, None] + tl.dot(
                        P_t, V_tile, out_dtype=tl.float32, input_precision='tf32x3')
                M_state = tl.where(t_mask[:, None], M_new_t[None, :], M_state)
                L_state = tl.where(t_mask[:, None], L_new_t[None, :], L_state)
                acc = tl.where(t_mask[:, None, None], acc_new_t[None, :, :], acc)

        # Raw slot ONLY: attend the K+1 raw bf16 chunk (fresh state), causal j<=t.
        if is_raw:
            kchunk_base = bid * stride_kkb + kv_head * stride_kkh
            vchunk_base = bid * stride_vkb + kv_head * stride_vkh
            for t in tl.static_range(0, K_PLUS_1):
                q_addrs_rt = (
                    q_base + t * stride_qt
                    + abs_head[:, None] * stride_qh + offs_d[None, :] * stride_qd
                )
                Q_rt = tl.load(
                    Q_ptr + q_addrs_rt,
                    mask=head_mask[:, None] & d_mask[None, :],
                    other=0.0, cache_modifier=".ca",
                ).to(tl.float32)
                t_mask_rt = q_t_range == t
                M_t = tl.sum(tl.where(t_mask_rt[:, None], M_state, 0.0), axis=0)
                L_t = tl.sum(tl.where(t_mask_rt[:, None], L_state, 0.0), axis=0)
                acc_t = tl.sum(tl.where(t_mask_rt[:, None, None], acc, 0.0), axis=0)
                for j in tl.static_range(0, K_PLUS_1):
                    if j <= t:
                        k_addr_j = kchunk_base + j * stride_kkt + offs_d * stride_kkd
                        K_j = tl.load(K_chunk_ptr + k_addr_j, mask=d_mask, other=0.0).to(tl.float32)
                        v_addr_j = vchunk_base + j * stride_vkt + offs_d * stride_vkd
                        V_j = tl.load(V_chunk_ptr + v_addr_j, mask=d_mask, other=0.0).to(tl.float32)
                        s_j = SCALE_LOG2E_split * tl.sum(Q_rt * K_j[None, :], axis=1)
                        s_j = tl.where(head_mask, s_j, -float("inf"))
                        M_new = tl.maximum(M_t, s_j)
                        alpha = tl.exp2(M_t - M_new)
                        p_j = tl.exp2(s_j - M_new)
                        L_t = L_t * alpha + p_j
                        acc_t = acc_t * alpha[:, None] + p_j[:, None] * V_j[None, :]
                        M_t = M_new
                M_state = tl.where(t_mask_rt[:, None], M_t[None, :], M_state)
                L_state = tl.where(t_mask_rt[:, None], L_t[None, :], L_state)
                acc = tl.where(t_mask_rt[:, None, None], acc_t[None, :, :], acc)

        # Partial-store epilogue: tv=acc/L, lse2=M+log2(L) (log2 domain).
        # Empty split -> L=0 -> tv=0, lse2=-inf sentinel (ALWAYS written for
        # consumed valid lanes so stage-2 never reads stale cudagraph scratch).
        safe_L = tl.where(L_state > 0.0, L_state, 1.0)
        tv = acc / safe_L[:, :, None]
        lse2 = tl.where((L_state > 0.0) & head_mask, M_state + tl.log2(safe_L),
                        -float("inf"))
        m_qbase = bid * stride_mb + kv_head * stride_mh + sid * stride_ms
        tv_addr = (
            m_qbase
            + q_t_range[:, None, None] * stride_mq
            + offs_h[None, :, None] * stride_moh
            + offs_d[None, None, :] * stride_md
        )
        tl.store(
            Mid_o_ptr + tv_addr,
            tl.where(qt_valid[:, None, None] & head_mask[None, :, None], tv, 0.0),
        )
        lse_addr = (
            m_qbase
            + q_t_range[:, None] * stride_mq
            + offs_h[None, :] * stride_moh
            + BLOCK_D * stride_md
        )
        tl.store(
            Mid_o_ptr + lse_addr,
            tl.where(qt_valid[:, None] & head_mask[None, :], lse2, -float("inf")),
        )

    return reference_stage1


def launch_reference_stage1(batch: VerifyBatch, plan: SplitPlan, mid) -> None:
    """Enqueue into caller-owned FP32 scratch, with no device-to-host reads.

    Prewarm each frozen table/L/dtype/batch geometry before capture. Device sequence
    lengths affect partition bounds only, and never the grid or JIT specialization.
    Runtime eligibility/profile validation belongs to the coordinator adapter.
    """
    import torch

    b, length, hq, dim = batch.q.shape
    hkv = batch.spec.num_kv_heads
    if length < 2 or dim != 256 or dim != batch.spec.head_dim or hq != batch.spec.num_q_heads:
        raise ValueError("reference split-K requires L>=2 and the validated D256 geometry")
    gqa = hq // hkv
    qpad, gpad = 1 << (length - 1).bit_length(), 1 << (gqa - 1).bit_length()
    expected = (b, hkv, plan.max_splits + 1, qpad, gpad, dim + 1)
    if tuple(mid.shape) != expected or mid.dtype != torch.float32:
        raise ValueError("caller scratch must be FP32 [B,Hkv,Smax+1,Qpad,Gpad,D+1]")
    if batch.q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("reference split-K supports FP16/BF16 activations")
    if any(tuple(t.shape) != (b, length, hkv, dim) or t.dtype != batch.q.dtype
           for t in (batch.raw_k, batch.raw_v)):
        raise ValueError("raw tail must match query dtype and [B,L,Hkv,D]")
    if tuple(batch.seq_lens.shape) != (b,) or batch.seq_lens.dtype not in (torch.int32, torch.int64) or batch.seq_lens.stride(0) != 1:
        raise ValueError("sequence lengths must be contiguous device int32/int64 [B]")
    if batch.kv_cache.dtype != torch.uint8 or batch.kv_cache.stride(-1) != 1:
        raise ValueError("packed cache must contain contiguous bytes per head")
    if batch.kv_cache.ndim != 4 or batch.kv_cache.shape[1:3] != (batch.spec.block_size, hkv):
        raise ValueError("packed cache must have [pages,block_size,Hkv,bytes] geometry")
    if batch.kv_cache.shape[-1] < batch.spec.key_packed_size + batch.spec.value_data_bytes + 4:
        raise ValueError("packed cache slot is missing k8v4 scale/zero bytes")
    if batch.block_table.ndim != 2 or batch.block_table.shape[0] != b or batch.block_table.stride(1) != 1 or batch.block_table.dtype not in (torch.int32, torch.int64):
        raise ValueError("block table must have contiguous integer rows [B,max_blocks]")
    tensors = (batch.q, batch.kv_cache, batch.block_table, batch.seq_lens, batch.raw_k, batch.raw_v, mid)
    if any(t.device != batch.q.device or not t.is_cuda for t in tensors):
        raise ValueError("all verifier tensors must share one CUDA device")
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = _build_kernel()
    _KERNEL[(b, hkv, plan.max_splits + 1)](
        batch.q, batch.kv_cache, batch.block_table, batch.seq_lens, batch.raw_k, batch.raw_v, mid,
        *batch.q.stride(), *batch.kv_cache.stride()[:3], batch.block_table.stride(0),
        *batch.raw_k.stride(), *batch.raw_v.stride(), *mid.stride(),
        SCALE=batch.scale, K_PLUS_1=length, BLOCK_D=dim, HEAD_DIM=dim,
        BLOCK_SIZE=batch.spec.block_size, BLOCK_KV=32, HEADS_PER_KV=gqa,
        BLOCK_QH=gpad, Hq_TOTAL=hq, KPS=batch.spec.key_packed_size,
        VAL_DATA_BYTES=batch.spec.value_data_bytes, NUM_SPLITS=plan.max_splits,
        RAW_SID=plan.max_splits, BUCKETS=plan.buckets, KP1_PAD=qpad,
        FP8_E4B15=0, DOT_FP16=0, num_warps=8, num_stages=3,
    )
