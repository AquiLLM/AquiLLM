"""Tiled k8v4 committed-prefix attention, with no historical KV scratch."""
from aquillm_vllm_h100.contracts import AttentionState, KVSpec


def validate_prefix_metadata(q, kv_cache, block_table, cached_len, spec, state):
    """Metadata only: no device data is read or synchronized."""
    import torch
    if type(cached_len) is not int or cached_len < 0:
        raise ValueError("cached_len must be a nonnegative CPU integer")
    if spec.head_dim != 256 or spec.num_q_heads != 6 * spec.num_kv_heads:
        raise ValueError("prefix kernel requires D256/GQA6")
    if q.ndim != 3 or q.shape[1:] != (spec.num_q_heads, spec.head_dim):
        raise ValueError("query must be [Q,Hq,D]")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("query dtype must be FP16/BF16")
    if kv_cache.ndim != 4 or kv_cache.dtype != torch.uint8 or kv_cache.shape[1:3] != (spec.block_size, spec.num_kv_heads):
        raise ValueError("packed cache must be uint8 [pages,block_size,Hkv,slot]")
    if kv_cache.shape[3] < spec.key_packed_size + spec.value_data_bytes + 4:
        raise ValueError("packed cache slot is too small for affine V")
    if block_table.ndim != 1 or block_table.dtype not in (torch.int32, torch.int64):
        raise ValueError("block table must be a request's int32/int64 row")
    if block_table.numel() < (cached_len + spec.block_size - 1) // spec.block_size:
        raise ValueError("block table does not cover the committed prefix")
    if state.output.shape != q.shape or state.output.dtype != torch.float32:
        raise ValueError("prefix output must be FP32 [Q,Hq,D]")
    if state.lse.shape != q.shape[:2] or state.lse.dtype != torch.float32:
        raise ValueError("prefix LSE must be FP32 [Q,Hq]")
    if any(x.device != q.device for x in (kv_cache, block_table, state.output, state.lse)):
        raise ValueError("prefix buffers must share a device")


def prefix_attention(q, kv_cache, block_table, cached_len: int, scale: float,
                     spec: KVSpec, state: AttentionState, *, block_q=32, num_warps=4) -> None:
    """Write FP32 normalized output and natural LSE for the entire prefix.

    Keys are E4M3FN bytes. V stores low/even and high/odd uint4 codes followed
    by little-endian FP16 scale and zero. Paged addressing uses actual strides
    and int64 offsets. Caller provides buffers proportional only to Q.
    """
    validate_prefix_metadata(q, kv_cache, block_table, cached_len, spec, state)
    import torch
    if not q.is_cuda or torch.cuda.get_device_capability(q.device) != (9, 0):
        raise ValueError("prefix kernel requires validated SM90 CUDA runtime")
    if block_q not in (32, 64) or num_warps not in (4, 8):
        raise ValueError("supported prefix launch tiles are Q32/Q64 and 4/8 warps")
    if q.shape[0] == 0:
        return
    import triton
    _kernel()[(triton.cdiv(q.shape[0], block_q), spec.num_q_heads)](
        q, kv_cache, block_table, state.output, state.lse, cached_len, float(scale),
        q.shape[0], *q.stride(), *kv_cache.stride(), block_table.stride(0),
        *state.output.stride(), *state.lse.stride(),
        PAGE_SIZE=spec.block_size, D=spec.head_dim, GROUPS=6,
        KPS=spec.key_packed_size, VBYTES=spec.value_data_bytes,
        BLOCK_Q=block_q, BLOCK_KV=32, num_warps=num_warps)


_prefix_kernel = None


def _kernel():
    global _prefix_kernel
    if _prefix_kernel is not None:
        return _prefix_kernel
    import triton
    import triton.language as tl

    @triton.jit
    def prefix_kernel(Q, CACHE, BT, OUT, LSE, N, SCALE, QN,
                      sq: tl.constexpr, sh: tl.constexpr, sd: tl.constexpr,
                      cb: tl.constexpr, cp: tl.constexpr, ch: tl.constexpr, cs: tl.constexpr,
                      bs: tl.constexpr, oq: tl.constexpr, oh: tl.constexpr, od: tl.constexpr,
                      lq: tl.constexpr, lh: tl.constexpr,
                      PAGE_SIZE: tl.constexpr, D: tl.constexpr, GROUPS: tl.constexpr,
                      KPS: tl.constexpr, VBYTES: tl.constexpr,
                      BLOCK_Q: tl.constexpr, BLOCK_KV: tl.constexpr):
        rows = (tl.program_id(0) * BLOCK_Q + tl.arange(0, BLOCK_Q)).to(tl.int64)
        head = tl.program_id(1).to(tl.int64)
        kvhead = head // GROUPS
        dims = tl.arange(0, D).to(tl.int64)
        q = tl.load(Q + rows[:, None] * sq + head * sh + dims[None, :] * sd,
                    mask=rows[:, None] < QN, other=0)
        maximum = tl.full((BLOCK_Q,), -float("inf"), tl.float32)
        denominator = tl.full((BLOCK_Q,), 0., tl.float32)
        accumulator = tl.full((BLOCK_Q, D), 0., tl.float32)
        for start in range(0, tl.cdiv(N, BLOCK_KV)):
            tokens = (start * BLOCK_KV + tl.arange(0, BLOCK_KV)).to(tl.int64)
            valid = tokens < N
            pages = tl.load(BT + (tokens // PAGE_SIZE) * bs, mask=valid, other=0).to(tl.int64)
            base = pages * cb + (tokens % PAGE_SIZE) * cp + kvhead * ch
            kbytes = tl.load(CACHE + base[:, None] + dims[None, :] * cs,
                             mask=valid[:, None], other=0)
            k = kbytes.to(tl.float8e4nv, bitcast=True).to(q.dtype)
            scores = tl.dot(q, tl.trans(k)) * SCALE
            scores = tl.where(valid[None, :], scores, -float("inf"))
            next_maximum = tl.maximum(maximum, tl.max(scores, 1))
            correction = tl.exp(maximum - next_maximum)
            probability = tl.exp(scores - next_maximum[:, None])
            packed = tl.load(CACHE + base[:, None] + (KPS + dims[None, :] // 2) * cs,
                             mask=valid[:, None], other=0).to(tl.int32)
            codes = ((packed >> ((dims[None, :] % 2) * 4)) & 15).to(tl.float32)
            low = tl.load(CACHE + base + (KPS + VBYTES) * cs, mask=valid, other=0).to(tl.uint16)
            high = tl.load(CACHE + base + (KPS + VBYTES + 1) * cs, mask=valid, other=0).to(tl.uint16)
            vscale = (low | (high << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            low = tl.load(CACHE + base + (KPS + VBYTES + 2) * cs, mask=valid, other=0).to(tl.uint16)
            high = tl.load(CACHE + base + (KPS + VBYTES + 3) * cs, mask=valid, other=0).to(tl.uint16)
            vzero = (low | (high << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            values = (codes * vscale[:, None] + vzero[:, None]).to(q.dtype)
            accumulator = accumulator * correction[:, None] + tl.dot(probability.to(q.dtype), values)
            denominator = denominator * correction + tl.sum(probability, 1)
            maximum = next_maximum
        nonempty = denominator > 0.
        normalized = accumulator / tl.where(nonempty, denominator, 1.)[:, None]
        lse = tl.where(nonempty, maximum + tl.log(tl.where(nonempty, denominator, 1.)), -float("inf"))
        tl.store(OUT + rows[:, None] * oq + head * oh + dims[None, :] * od,
                 normalized, mask=rows[:, None] < QN)
        tl.store(LSE + rows * lq + head * lh, lse, mask=rows < QN)

    _prefix_kernel = prefix_kernel
    return prefix_kernel
