# SPDX-License-Identifier: Apache-2.0
"""Compact query/head fusion for k8v4 verification, with a separate raw slot.

The packed-byte decoder and raw-tail online-softmax arithmetic are adapted
from Genesis 34e2693, p67_multi_query_kernel.py (Apache-2.0), authored by
Sandermage (Sander) Barzov Aleksandr. Changes: compact padded row mapping,
caller-owned scratch ABI, device bucket selection, fixed 32-token committed
tiles, and neutralization of every empty/padded lane. This does not dispatch
the old GENESIS_P67_USE_FUSED kernel.
"""
from functools import lru_cache

from ..contracts import SplitPlan, VerifyBatch


@lru_cache(maxsize=1)
def _kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mtp_fused_stage1(
        Q, Cache, Table, Lengths, RawK, RawV, Mid,
        qb: tl.constexpr, qt: tl.constexpr, qh: tl.constexpr, qd: tl.constexpr,
        cb: tl.constexpr, cp: tl.constexpr, ch: tl.constexpr, cd: tl.constexpr,
        tb: tl.constexpr, tp: tl.constexpr, ls: tl.constexpr,
        kb: tl.constexpr, kt: tl.constexpr, kh: tl.constexpr, kd: tl.constexpr,
        vb: tl.constexpr, vt: tl.constexpr, vh: tl.constexpr, vd: tl.constexpr,
        mb: tl.constexpr, mh: tl.constexpr, ms: tl.constexpr,
        mq: tl.constexpr, mg: tl.constexpr, md: tl.constexpr,
        L: tl.constexpr, G: tl.constexpr, D: tl.constexpr,
        QPAD: tl.constexpr, GPAD: tl.constexpr, ROWS: tl.constexpr,
        DPAD: tl.constexpr, PAGE: tl.constexpr, KPS: tl.constexpr,
        VBYTES: tl.constexpr, MAX_SPLITS: tl.constexpr,
        BUCKETS: tl.constexpr, SCALE: tl.constexpr,
    ):
        b, h, sid = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        row = tl.arange(0, ROWS)
        t, g = row // G, row % G
        live = row < L * G
        d = tl.arange(0, DPAD)
        dmask = d < D
        q = tl.load(Q + b * qb + t[:, None] * qt
                    + (h * G + g[:, None]) * qh + d[None, :] * qd,
                    mask=live[:, None] & dmask[None, :], other=0,
                    cache_modifier=".ca").to(tl.float32)
        prior = tl.maximum(tl.load(Lengths + b * ls) - L, 0)
        active = tl.full((), MAX_SPLITS, tl.int32)
        for i in tl.static_range(len(BUCKETS) - 1, -1, -1):
            active = tl.where(prior <= BUCKETS[i][0], BUCKETS[i][1], active)
        m = tl.full((ROWS,), -float("inf"), tl.float32)
        norm = tl.zeros((ROWS,), tl.float32)
        acc = tl.zeros((ROWS, DPAD), tl.float32)
        scale_log2 = SCALE * 1.4426950408889634
        if sid == MAX_SPLITS:
            # Retain baseline scalar-FP32 reductions; raw tokens never use dot.
            for j in tl.static_range(L):
                k = tl.load(RawK + b * kb + j * kt + h * kh + d * kd,
                            mask=dmask, other=0).to(tl.float32)
                v = tl.load(RawV + b * vb + j * vt + h * vh + d * vd,
                            mask=dmask, other=0).to(tl.float32)
                valid = live & (j <= t)
                score = scale_log2 * tl.sum(q * k[None, :], axis=1)
                score = tl.where(valid, score, -float("inf"))
                new_m = tl.maximum(m, score)
                safe_m = tl.where(new_m == -float("inf"), 0.0, new_m)
                alpha = tl.exp2(m - safe_m)
                p = tl.exp2(score - safe_m)
                norm = norm * alpha + p
                acc = acc * alpha[:, None] + p[:, None] * v[None, :]
                m = new_m
        elif sid < active:
            split_len = tl.cdiv(prior, active)
            start, end = sid * split_len, tl.minimum((sid + 1) * split_len, prior)
            n = tl.arange(0, 32)
            for tile in tl.range(start, end, 32):
                pos = tile + n
                mask = pos < end
                page = tl.load(Table + b * tb + (pos // PAGE) * tp,
                               mask=mask, other=0, cache_modifier=".ca").to(tl.int64)
                base = page * cb + (pos % PAGE).to(tl.int64) * cp + h * ch
                packed_k = tl.load(Cache + base[None, :] + d[:, None] * cd,
                                   mask=dmask[:, None] & mask[None, :], other=0,
                                   cache_modifier=".cg")
                k = packed_k.to(tl.float8e4nv, bitcast=True).to(tl.float32)
                valbase = base + KPS * cd
                packed_v = tl.load(Cache + valbase[:, None] + (d[None, :] // 2) * cd,
                                   mask=mask[:, None] & dmask[None, :], other=0,
                                   cache_modifier=".cg").to(tl.int32)
                nibble = ((packed_v >> ((d[None, :] % 2) * 4)) & 15).to(tl.float32)
                meta = valbase + VBYTES * cd
                slo = tl.load(Cache + meta, mask=mask, other=0).to(tl.uint16)
                shi = tl.load(Cache + meta + cd, mask=mask, other=0).to(tl.uint16)
                zlo = tl.load(Cache + meta + 2 * cd, mask=mask, other=0).to(tl.uint16)
                zhi = tl.load(Cache + meta + 3 * cd, mask=mask, other=0).to(tl.uint16)
                scale = (slo | (shi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
                zero = (zlo | (zhi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
                v = nibble * scale[:, None] + zero[:, None]
                score = scale_log2 * tl.dot(q, k, input_precision="tf32")
                score = tl.where(live[:, None] & mask[None, :], score, -float("inf"))
                new_m = tl.maximum(m, tl.max(score, axis=1))
                safe_m = tl.where(new_m == -float("inf"), 0.0, new_m)
                alpha = tl.exp2(m - safe_m)
                p = tl.exp2(score - safe_m[:, None])
                norm = norm * alpha + tl.sum(p, axis=1)
                acc = acc * alpha[:, None] + tl.dot(p, v, input_precision="tf32x3")
                m = new_m
        safe_norm = tl.where(norm > 0, norm, 1.0)
        out = acc / safe_norm[:, None]
        lse = tl.where(norm > 0, m + tl.log2(safe_norm), -float("inf"))
        base = b * mb + h * mh + sid * ms
        offset = base + t * mq + g * mg
        tl.store(Mid + offset[:, None] + d[None, :] * md, out,
                 mask=live[:, None] & dmask[None, :])
        tl.store(Mid + offset + D * md, lse, mask=live)
        # Compact dot rows do not cover the ABI's padded rectangular lanes.
        padrow = tl.arange(0, QPAD * GPAD)
        pt, pg = padrow // GPAD, padrow % GPAD
        padding = (pt >= L) | (pg >= G)
        padbase = base + pt * mq + pg * mg
        tl.store(Mid + padbase[:, None] + d[None, :] * md, 0.0,
                 mask=padding[:, None] & dmask[None, :])
        tl.store(Mid + padbase + D * md, -float("inf"), mask=padding)

    return mtp_fused_stage1


def launch_fused_stage1(batch: VerifyBatch, plan: SplitPlan, mid) -> None:
    """Asynchronously write all split slots without allocation or device reads.

    H100 uses FP8 E4M3NV K bytes. QK stays TF32, PV stays TF32x3; destination
    precision and the baseline FP16 reduction cast belong to stage two.
    """
    import triton

    b, length, hq, dim = batch.q.shape
    spec = batch.spec
    group = hq // spec.num_kv_heads
    qpad, gpad = triton.next_power_of_2(length), triton.next_power_of_2(group)
    expected = (b, spec.num_kv_heads, plan.max_splits + 1, qpad, gpad, dim + 1)
    if tuple(mid.shape) != expected or str(mid.dtype) != "torch.float32":
        raise ValueError(f"mid must be FP32 scratch with shape {expected}")
    if dim != spec.head_dim or hq != spec.num_q_heads or length < 2:
        raise ValueError("query geometry must match KVSpec and L must be >=2")
    if batch.kv_cache.ndim != 4 or str(batch.kv_cache.dtype) != "torch.uint8":
        raise ValueError("packed cache must be a four-dimensional uint8 tensor")
    _kernel()[(b, spec.num_kv_heads, plan.max_splits + 1)](
        batch.q, batch.kv_cache, batch.block_table, batch.seq_lens,
        batch.raw_k, batch.raw_v, mid,
        *batch.q.stride(), *batch.kv_cache.stride(), *batch.block_table.stride(),
        batch.seq_lens.stride(0), *batch.raw_k.stride(), *batch.raw_v.stride(),
        *mid.stride(), L=length, G=group, D=dim, QPAD=qpad, GPAD=gpad,
        ROWS=max(16, triton.next_power_of_2(length * group)),
        DPAD=triton.next_power_of_2(dim), PAGE=spec.block_size,
        KPS=spec.key_packed_size, VBYTES=spec.value_data_bytes,
        MAX_SPLITS=plan.max_splits, BUCKETS=plan.buckets, SCALE=batch.scale,
        num_warps=8, num_stages=2,
    )
