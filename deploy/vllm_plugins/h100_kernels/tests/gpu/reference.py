"""Independent test-only k8v4 decoding and FP32 attention; no candidate imports."""
import math

import torch

from aquillm_vllm_h100.contracts import AttentionState, KVSpec, VerifyBatch


def make_verify_batch(prior_lengths, length=5, dtype=torch.float16, block_size=32,
                      seed=17, device="cuda", strided=False):
    if not prior_lengths or min(prior_lengths) < 0 or length < 1:
        raise ValueError("nonnegative prefix lengths and a positive tail are required")
    rng = torch.Generator(device=device).manual_seed(seed)
    b, hq, hk, d = len(prior_lengths), 24, 4, 256
    spec = KVSpec("turboquant_k8v4", d, hq, hk, block_size, d, d // 2)
    pages = math.ceil((max(prior_lengths) + length) / block_size)
    total_pages = b * pages
    step = 2 if strided else 1
    backing = torch.empty((total_pages, block_size * step, hk * step, 400),
                          dtype=torch.uint8, device=device)
    cache = backing[:, ::step, ::step, :]
    cache.fill_(0xA5)  # poison unused trailing bytes/slots before packing
    keys = torch.randn((total_pages, block_size, hk, d), device=device, generator=rng)
    key_bytes = keys.to(torch.float8_e4m3fn).view(torch.uint8)
    codes = torch.randint(0, 16, (total_pages, block_size, hk, d),
                          device=device, generator=rng, dtype=torch.uint8)
    scales = (torch.rand((total_pages, block_size, hk, 1), device=device,
                         generator=rng) * 0.2 + 0.01).half()
    zeros = (-7.5 * scales).half()
    cache[..., :d] = key_bytes
    cache[..., d:d + d // 2] = codes[..., 0::2] | (codes[..., 1::2] << 4)
    cache[..., d + d // 2:d + d // 2 + 2] = scales.view(torch.uint8)
    cache[..., d + d // 2 + 2:d + d // 2 + 4] = zeros.view(torch.uint8)
    block_table = torch.randperm(total_pages, device=device, generator=rng).reshape(b, pages).int()
    q = torch.randn((b, length, hq, d), device=device, dtype=dtype, generator=rng)
    raw_k = torch.randn((b, length, hk, d), device=device, dtype=dtype, generator=rng)
    raw_v = torch.randn((b, length, hk, d), device=device, dtype=dtype, generator=rng)
    seq_lens = torch.tensor([n + length for n in prior_lengths], device=device, dtype=torch.int32)
    return VerifyBatch(q, cache, block_table, seq_lens, raw_k, raw_v, d ** -0.5, spec)


def unpack_prefix(batch, batch_index):
    spec = batch.spec
    n = int(batch.seq_lens[batch_index].item()) - batch.q.shape[1]
    slots = torch.arange(n, device=batch.kv_cache.device)
    physical = batch.block_table[batch_index, slots // spec.block_size].long()
    packed = batch.kv_cache[physical, slots % spec.block_size]
    d = spec.head_dim
    keys = packed[..., :d].contiguous().view(torch.float8_e4m3fn).float()
    values = packed[..., spec.key_packed_size:spec.key_packed_size + spec.value_data_bytes]
    codes = torch.stack((values & 15, values >> 4), dim=-1).flatten(-2).float()
    offset = spec.key_packed_size + spec.value_data_bytes
    scale = packed[..., offset:offset + 2].contiguous().view(torch.float16).float()
    zero = packed[..., offset + 2:offset + 4].contiguous().view(torch.float16).float()
    return keys, codes * scale + zero


def reference_attention(q, k, v, causal_prefix=0, scale=None):
    """Queries [Q,Hq,D], KV [K,Hkv,D]; None prefix means noncausal."""
    qn, hq, d = q.shape
    if k.shape[0] == 0:
        return AttentionState(torch.zeros_like(q, dtype=torch.float32),
                              torch.full((qn, hq), -torch.inf, device=q.device))
    groups = hq // k.shape[1]
    k = k.float().repeat_interleave(groups, dim=1)
    v = v.float().repeat_interleave(groups, dim=1)
    # IEEE FP32 reference even when the test process has enabled TF32 globally.
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        scores = torch.einsum("qhd,khd->qhk", q.float(), k) * (scale or d ** -0.5)
        if causal_prefix is not None:
            keys = torch.arange(k.shape[0], device=q.device)
            queries = torch.arange(qn, device=q.device) + causal_prefix
            scores = scores.masked_fill(keys[None, None, :] > queries[:, None, None], -torch.inf)
        lse = torch.logsumexp(scores, dim=-1)
        probabilities = torch.softmax(scores, dim=-1)
        output = torch.einsum("qhk,khd->qhd", probabilities, v)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    return AttentionState(output, lse)


def reference_verify(batch):
    result = []
    for i in range(batch.q.shape[0]):
        k, v = unpack_prefix(batch, i)
        n = k.shape[0]
        result.append(reference_attention(batch.q[i], torch.cat((k, batch.raw_k[i].float())),
                                          torch.cat((v, batch.raw_v[i].float())), n,
                                          batch.scale).output)
    return torch.stack(result)


def assert_close(actual, expected, dtype):
    assert torch.isfinite(actual).all(), "nonfinite attention output"
    error = (actual.float() - expected.float()) / expected.abs().max().clamp_min(1.0)
    max_tol, rms_tol = (0.02, 0.01) if dtype == torch.bfloat16 else (0.01, 0.005)
    assert error.abs().max().item() <= max_tol, f"max error {error.abs().max().item()}"
    assert error.square().mean().sqrt().item() <= rms_tol, "RMS error exceeds gate"

