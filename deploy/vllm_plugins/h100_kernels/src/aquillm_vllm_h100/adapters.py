"""Narrow adapters retain the captured, post-Genesis baseline for unsupported calls."""
import functools
import logging
import math

from .contracts import KVSpec, SplitPlan, VerifyBatch

log = logging.getLogger("aquillm.h100")


@functools.lru_cache(maxsize=8)
def _h100(device):
    import torch
    props = torch.cuda.get_device_properties(device)
    return props.major == 9 and props.minor == 0 and props.multi_processor_count == 132 and "H100" in props.name


def _supported_metadata(q, cache, table, lengths, raw_k, raw_v, output, mid, scale, block_size):
    import torch
    tensors = (q, cache, table, lengths, raw_k, raw_v, output, mid)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        return False
    b, length, _, _ = q.shape
    qpad = 1 << (length - 1).bit_length()
    if (raw_k.shape != (b, length, 4, 256) or raw_v.shape != raw_k.shape
            or cache.ndim != 4 or cache.shape[1] != block_size or cache.shape[2] != 4
            or cache.shape[3] < 388 or table.ndim != 2 or table.shape[0] < b
            or table.shape[1] < 1 or lengths.ndim != 1 or lengths.shape[0] < b
            or output.shape != q.shape or mid.ndim != 6 or mid.shape[2] < 2
            or mid.shape != (b, 4, mid.shape[2], qpad, 8, 257)):
        return False
    if (q.dtype not in (torch.float16, torch.bfloat16) or raw_k.dtype != q.dtype
            or raw_v.dtype != q.dtype or output.dtype != q.dtype
            or cache.dtype != torch.uint8 or mid.dtype != torch.float32
            or table.dtype not in (torch.int32, torch.int64)
            or lengths.dtype not in (torch.int32, torch.int64)):
        return False
    if not all(t.device == q.device and all(s > 0 for s in t.stride()) for t in tensors):
        return False
    return q.device.type == "cuda" and block_size > 0 and math.isfinite(scale) and scale > 0


def make_verifier_adapter(original, config, plan=None):
    exercised = False

    @functools.wraps(original)
    def call(q, kv_cache, block_table, seq_lens, k_chunk, v_chunk, scale,
             block_size, kps, val_data_bytes, output=None, num_splits=None, mid_o=None):
        nonlocal exercised
        args = dict(q=q, kv_cache=kv_cache, block_table=block_table, seq_lens=seq_lens,
                    k_chunk=k_chunk, v_chunk=v_chunk, scale=scale, block_size=block_size,
                    kps=kps, val_data_bytes=val_data_bytes, output=output,
                    num_splits=num_splits, mid_o=mid_o)
        if (len(q.shape) != 4 or q.shape[2:] != (24, 256) or not 2 <= q.shape[1] <= 16
                or kps != 256 or val_data_bytes != 128 or output is None or mid_o is None):
            return original(**args)
        if not _supported_metadata(q, kv_cache, block_table, seq_lens, k_chunk, v_chunk,
                                   output, mid_o, scale, block_size) or not _h100(q.device):
            return original(**args)
        selected = plan or SplitPlan(mid_o.shape[2] - 1, ())
        if mid_o.shape[2] != selected.max_splits + 1:
            return original(**args)
        spec = KVSpec("turboquant_k8v4", 256, 24, 4, block_size, kps, val_data_bytes)
        batch = VerifyBatch(q, kv_cache, block_table, seq_lens, k_chunk, v_chunk, scale, spec)
        if config["mtp"] == "fused":
            from .kernels.mtp_fused import launch_fused_stage1 as launch
        else:
            from .kernels.splitk_reference import launch_reference_stage1 as launch
        from .kernels.reduce import reduce_verify_partials
        try:
            launch(batch, selected, mid_o)
            reduce_verify_partials(batch, mid_o, output)
        except Exception as error:
            # Genesis forward catches Exception and falls back. A failed new GPU
            # launch must terminate this worker instead of reusing a bad context.
            raise SystemExit(f"AquiLLM H100 verifier launch failed: {error}") from error
        if not exercised:
            exercised = True
            log.warning("AQUILLM_H100 route_exercised mtp=%s splits=%s shape=%s",
                        config["mtp"], selected.max_splits, tuple(q.shape))
        return output

    return call


def install_adapters(config):
    # Do not import backend attention before Genesis's source patches finish.
    from sndr.engines.vllm.kernels_legacy import p67_multi_query_kernel as baseline
    if config["split"] != "baseline":
        raise ValueError("adaptive policy requires a qualified runtime profile")
    if config["prefill"] != "0":
        raise ValueError("prefill requires a qualified runtime profile")
    if config["mtp"] == "fused":
        baseline.call_p67_splitk = make_verifier_adapter(baseline.call_p67_splitk, config)
    return {"status": "installed", "mtp": config["mtp"], "split": config["split"], "prefill": False}
