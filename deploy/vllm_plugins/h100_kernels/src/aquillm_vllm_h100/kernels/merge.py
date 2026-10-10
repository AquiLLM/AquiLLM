"""Merge normalized partial attention states with FP32 natural-log weights."""
from aquillm_vllm_h100.contracts import AttentionState


def _validate(prefix, chunk, output):
    import torch
    if output.ndim != 3 or output.shape != prefix.output.shape or output.shape != chunk.output.shape:
        raise ValueError("attention outputs must have matching [Q,Hq,D] shapes")
    for state in (prefix, chunk):
        if state.lse.shape != output.shape[:2] or state.lse.dtype != torch.float32:
            raise ValueError("attention LSE must be FP32 [Q,Hq] natural logarithm")
        if state.output.device != output.device or state.lse.device != output.device:
            raise ValueError("attention states and output must share a device")


def merge_attention_states(prefix: AttentionState, chunk: AttentionState, output) -> None:
    """Write the merged output, casting once at the final store.

    -inf LSE marks an empty component, whose output is not consumed. CPU support
    is for analytic contract tests; serving launches a single same-stream kernel.
    """
    _validate(prefix, chunk, output)
    if output.is_cuda:
        import triton
        kernel = _kernel()
        kernel[(output.shape[0] * output.shape[1],)](
            prefix.output, prefix.lse, chunk.output, chunk.lse, output,
            output.shape[1], output.shape[2], *prefix.output.stride(),
            *prefix.lse.stride(), *chunk.output.stride(), *chunk.lse.stride(),
            *output.stride(), BLOCK_D=triton.next_power_of_2(output.shape[2]), num_warps=4)
        return
    import torch
    lp, lc = prefix.lse, chunk.lse
    vp, vc = lp != -torch.inf, lc != -torch.inf
    m = torch.where(vp | vc, torch.maximum(lp, lc), 0)
    a = torch.where(vp, torch.exp(lp - m), 0)
    b = torch.where(vc, torch.exp(lc - m), 0)
    op = torch.where(vp[..., None], prefix.output.float(), 0)
    oc = torch.where(vc[..., None], chunk.output.float(), 0)
    den = torch.where(vp | vc, a + b, 1)
    output.copy_((a[..., None] * op + b[..., None] * oc) / den[..., None])


_merge_kernel = None


def _kernel():
    global _merge_kernel
    if _merge_kernel is not None:
        return _merge_kernel
    import triton
    import triton.language as tl

    @triton.jit
    def merge_kernel(OP, LP, OC, LC, OUT, H: tl.constexpr, D: tl.constexpr,
                     pq: tl.constexpr, ph: tl.constexpr, pd: tl.constexpr,
                     lq: tl.constexpr, lh: tl.constexpr,
                     cq: tl.constexpr, ch: tl.constexpr, cd: tl.constexpr,
                     mq: tl.constexpr, mh: tl.constexpr,
                     oq: tl.constexpr, oh: tl.constexpr, od: tl.constexpr,
                     BLOCK_D: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        q, h = row // H, row % H
        d = tl.arange(0, BLOCK_D).to(tl.int64)
        lp, lc = tl.load(LP + q * lq + h * lh), tl.load(LC + q * mq + h * mh)
        vp, vc = lp != -float("inf"), lc != -float("inf")
        m = tl.where(vp | vc, tl.maximum(lp, lc), 0.)
        a, b = tl.where(vp, tl.exp(lp - m), 0.), tl.where(vc, tl.exp(lc - m), 0.)
        op = tl.load(OP + q * pq + h * ph + d * pd, mask=(d < D) & vp, other=0).to(tl.float32)
        oc = tl.load(OC + q * cq + h * ch + d * cd, mask=(d < D) & vc, other=0).to(tl.float32)
        merged = (a * op + b * oc) / tl.where(vp | vc, a + b, 1.)
        tl.store(OUT + q * oq + h * oh + d * od, merged, mask=d < D)

    _merge_kernel = merge_kernel
    return merge_kernel
