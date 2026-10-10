"""Retain the pinned Genesis log2-LSE reducer and its FP16 intermediate cast."""


def reduce_verify_partials(batch, mid, output):
    import triton
    from sndr.engines.vllm.kernels_legacy.p67_multi_query_kernel import _get_stage2_kernel

    b, length, hq, d = batch.q.shape
    hk = batch.spec.num_kv_heads
    groups = hq // hk
    expected = (b, hk, mid.shape[2], triton.next_power_of_2(length),
                triton.next_power_of_2(groups), d + 1)
    if tuple(mid.shape) != expected or tuple(output.shape) != tuple(batch.q.shape):
        raise ValueError("verifier output or scratch shape violates the shared ABI")
    kernel = _get_stage2_kernel()
    kernel[(b, hk)](
        mid, output, *mid.stride(), *output.stride(),
        NUM_SPLITS_P1=mid.shape[2], K_PLUS_1=length, KP1_PAD=expected[3],
        BLOCK_D=d, HEAD_DIM=d, HEADS_PER_KV=groups, BLOCK_QH=expected[4],
        Hq_TOTAL=hq, num_warps=4,
    )
