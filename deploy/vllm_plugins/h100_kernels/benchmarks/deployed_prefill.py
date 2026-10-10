"""Compare complete candidate continuation against the installed Genesis P101 caller.

Run only in a fresh, coordinator-serialized serving container process. Genesis
is registered before importing backend symbols. Both paths share the identical
committed packed prefix, current raw FP16 KV, installed current-KV store, caller
metadata and P26 output pool. No crossover profile is created or enabled.
"""
import argparse
import hashlib
import inspect
import json
import os
import random
import re
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace


def current_chunk_slots(table_row, block_size, prior, length):
    import torch
    logical = torch.arange(prior, prior + length, device=table_row.device, dtype=torch.int64)
    return table_row[logical // block_size].long() * block_size + logical % block_size


def bounded_reference(q, load_kv, num_keys, *, causal_prefix, scale, query_tile=32, key_tile=1024):
    """Independent IEEE FP32 attention, with bounded Q/H/K score storage.

    load_kv(start,end) returns independently decoded KV for only that key tile.
    Neither historical expanded GQA KV nor full [Q,Hq,K] scores are allocated.
    """
    import torch
    if query_tile < 1 or key_tile < 1:
        raise ValueError("reference tiles must be positive")
    output = torch.empty_like(q, dtype=torch.float32)
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for qstart in range(0, q.shape[0], query_tile):
            qend = min(qstart + query_tile, q.shape[0])
            qt = q[qstart:qend].float().permute(1, 0, 2)
            maximum = torch.full(qt.shape[:2], -torch.inf, device=q.device)
            denominator = torch.zeros_like(maximum)
            accumulator = torch.zeros(qt.shape, device=q.device, dtype=torch.float32)
            queries = torch.arange(qstart, qend, device=q.device) + causal_prefix
            for kstart in range(0, min(num_keys, causal_prefix + qend), key_tile):
                kend = min(kstart + key_tile, num_keys)
                k, v = load_kv(kstart, kend)
                groups = q.shape[1] // k.shape[1]
                kt = k.float().repeat_interleave(groups, 1).permute(1, 2, 0)
                vt = v.float().repeat_interleave(groups, 1).permute(1, 0, 2)
                scores = torch.bmm(qt, kt) * scale
                keys = torch.arange(kstart, kend, device=q.device)
                scores.masked_fill_(keys[None, None, :] > queries[None, :, None], -torch.inf)
                next_maximum = torch.maximum(maximum, scores.amax(-1))
                safe_maximum = torch.where(torch.isfinite(next_maximum), next_maximum, 0.)
                correction = torch.exp(maximum - safe_maximum)
                probabilities = torch.exp(scores - safe_maximum[..., None])
                accumulator = accumulator * correction[..., None] + torch.bmm(probabilities, vt)
                denominator = denominator * correction + probabilities.sum(-1)
                maximum = next_maximum
            output[qstart:qend] = (accumulator / denominator.clamp_min(1e-30)[..., None]).permute(1, 0, 2)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    return output


def unpack_range(batch, start, end):
    """Independent packed-byte decoder, separate from candidate/deployed kernels."""
    import torch
    spec = batch.spec
    logical = torch.arange(start, end, device=batch.kv_cache.device)
    pages = batch.block_table[0, logical // spec.block_size].long()
    packed = batch.kv_cache[pages, logical % spec.block_size]
    k = packed[..., :spec.head_dim].contiguous().view(torch.float8_e4m3fn).float()
    values = packed[..., spec.key_packed_size:spec.key_packed_size + spec.value_data_bytes]
    codes = torch.stack((values & 15, values >> 4), dim=-1).flatten(-2).float()
    offset = spec.key_packed_size + spec.value_data_bytes
    scale = packed[..., offset:offset + 2].contiguous().view(torch.float16).float()
    zero = packed[..., offset + 2:offset + 4].contiguous().view(torch.float16).float()
    return k, codes * scale + zero


def reference_loader(batch, prior, *, compressed_tail, round_prefix=False):
    def load(start, end):
        import torch
        if compressed_tail:
            return unpack_range(batch, start, end)
        if start >= prior:
            return batch.raw_k[0, start-prior:end-prior].float(), batch.raw_v[0, start-prior:end-prior].float()
        k, v = unpack_range(batch, start, min(end, prior))
        if round_prefix:
            k, v = k.to(batch.q.dtype).float(), v.to(batch.q.dtype).float()
        if end > prior:
            k = torch.cat((k, batch.raw_k[0, :end-prior].float()))
            v = torch.cat((v, batch.raw_v[0, :end-prior].float()))
        return k, v
    return load


def error_metrics(actual, expected):
    import torch
    if not torch.isfinite(actual).all():
        raise AssertionError("nonfinite attention output")
    normalized = (actual.float() - expected.float()) / expected.abs().max().clamp_min(1.)
    return {"max_normalized_error": normalized.abs().max().item(),
            "rms_normalized_error": normalized.square().mean().sqrt().item()}


def numerical_gate(actual, expected, dtype):
    import torch
    result = error_metrics(actual, expected)
    max_tol, rms_tol = (0.02, 0.01) if dtype == torch.bfloat16 else (0.01, 0.005)
    if result["max_normalized_error"] > max_tol or result["rms_normalized_error"] > rms_tol:
        raise AssertionError(f"attention numerical gate failed: {result}")
    return result


def paired_measurements(operations, sample, *, repetitions, seed):
    """Randomized, serial AB/BA rounds, with equal samples for each path."""
    rng = random.Random(seed)
    samples = {name: [] for name in operations}
    orders = []
    for _ in range(repetitions):
        order = list(operations)
        rng.shuffle(order)
        orders.append(order)
        for name in order:
            samples[name].append(sample(operations[name]))
    result = {name: {"samples": values} for name, values in samples.items()}
    result["round_order"] = orders
    return result


def source_identity(function):
    original = inspect.unwrap(function)
    source = inspect.getsource(original).lstrip().rstrip()
    return {"module": original.__module__, "qualname": original.__qualname__,
            "file": inspect.getsourcefile(original), "sha256": hashlib.sha256(source.encode()).hexdigest()}


def configure_runtime(args):
    """Preserve inherited Genesis flags; disable only experimental H100 routes."""
    for key, value in {"AQUILLM_H100_MTP_KERNEL": "baseline", "AQUILLM_H100_PREFILL": "0",
                       "AQUILLM_H100_SPLIT_POLICY": "baseline", "AQUILLM_H100_GDN": "baseline"}.items():
        os.environ[key] = value
    from sndr.plugin import register
    register()
    # Import after Genesis registration so the class and P38 symbols are deployed.
    from vllm.config import VllmConfig
    from vllm.v1.attention.backends.turboquant_attn import TurboQuantAttentionImpl
    config = VllmConfig()
    config.attention_config.flash_attn_version = 2
    config.attention_config.tq_max_kv_splits_for_cuda_graph = args.kv_splits
    # The serving TurboQuant configuration overrides H100's default FA3 to FA2.
    config.attention_config.flash_attn_version = 2
    config.scheduler_config.max_num_batched_tokens = 4096
    if getattr(TurboQuantAttentionImpl, "_aquillm_h100_prefill_install", None) is not None:
        raise RuntimeError("experimental prefill already installed; benchmark requires a fresh baseline process")
    return config, TurboQuantAttentionImpl


def prepare_case(impl, backend, prior, length, block_size, seed):
    import torch
    from reference import make_verify_batch
    batch = make_verify_batch([prior], length=length, block_size=block_size, seed=seed)
    layer = torch.nn.Module()
    layer._tq_PiT = torch.eye(256, device="cuda")
    layer._tq_midpoints = torch.empty(0, device="cuda")
    centroids = torch.zeros(1, device="cuda")
    slots = current_chunk_slots(batch.block_table[0], block_size, prior, length)
    # Exact runtime store, before either attention path or oracle is called.
    impl._store_kv(batch.raw_k[0], batch.raw_v[0], batch.kv_cache, slots, layer)
    metadata = backend.TurboQuantMetadata(
        query_start_loc=torch.tensor([0, length], dtype=torch.int32, device="cuda"),
        seq_lens=batch.seq_lens, slot_mapping=slots, block_table=batch.block_table,
        query_start_loc_cpu=torch.tensor([0, length], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([prior + length], dtype=torch.int32),
        num_actual_tokens=length, max_query_len=length, max_seq_len=prior + length,
        is_prefill=True, num_decodes=0, num_decode_tokens=0)
    return batch, layer, metadata, layer._tq_PiT, centroids


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefixes", default="8192,32768,65536", help="committed prefix lengths, not total sequence lengths")
    parser.add_argument("--chunks", default="128,1024,4096")
    parser.add_argument("--block-size", type=int, default=16, help="actual TurboQuant cache page size")
    parser.add_argument("--kv-splits", type=int, default=32, help="deployed attention_config.tq_max_kv_splits_for_cuda_graph")
    parser.add_argument("--block-q", type=int, choices=(32,64), default=32)
    parser.add_argument("--warps", type=int, choices=(4,8), default=4)
    parser.add_argument("--reference-query-tile", type=int, default=32)
    parser.add_argument("--reference-key-tile", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    args.prefixes = [int(value) for value in args.prefixes.split(",")]
    args.chunks = [int(value) for value in args.chunks.split(",")]
    if (not args.prefixes or not args.chunks or min(args.prefixes + args.chunks) < 1 or
        min(args.block_size, args.kv_splits, args.repetitions, args.reference_query_tile, args.reference_key_tile) < 1 or args.warmup < 0):
        parser.error("lengths, splits, tiles and repetitions must be positive; warmup nonnegative")
    return args


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root / "src"), str(root / "tests" / "gpu")]
    config, cls = configure_runtime(args)
    import torch
    import triton
    import vllm
    from vllm.config import set_current_vllm_config
    from vllm.v1.attention.backends import turboquant_attn as backend
    from aquillm_vllm_h100.contracts import AttentionState
    from aquillm_vllm_h100.prefill_adapter import PREFILL_FINGERPRINT, rewrite_prefill_method
    from aquillm_vllm_h100.kernels.prefix import prefix_attention
    from aquillm_vllm_h100.kernels.merge import merge_attention_states
    from aquillm_vllm_h100.prefill import raw_chunk_attention
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9,0):
        raise RuntimeError("serialized H100 SM90 runtime required")
    baseline = cls._prefill_attention
    provenance = source_identity(baseline)
    if provenance["sha256"] != PREFILL_FINGERPRINT:
        raise RuntimeError(f"installed baseline caller changed: {provenance}")
    if (backend._CONTINUATION_DECODE_THRESHOLD, backend._CONTINUATION_DECODE_MAX_CACHED_LEN) != (64,32768):
        raise RuntimeError("installed P101 thresholds differ from the reference semantic classification")
    p38 = cls.__dict__.get("_genesis_p38_dispatch")
    if p38 is None:
        raise RuntimeError("Genesis P38 dispatcher absent after register; refusing to mislabel upstream timing as deployed")
    preflight = {"record": "preflight", "caller": provenance, "p38": source_identity(p38),
                 "continuation": source_identity(cls._continuation_prefill),
                 "execution": "eager_outside_capture", "graph_reason": "complete installed FA helper reads cu_seqlens on host",
                 "torch": torch.__version__, "triton": triton.__version__, "vllm": vllm.__version__,
                 "gpu": torch.cuda.get_device_name(), "sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
                 "geometry": {"q_heads":24,"kv_heads":4,"head_dim":256,"dtype":"float16","cache":"turboquant_k8v4",
                              "page_size":args.block_size,"kv_splits":args.kv_splits,"serving_chunk_budget":4096,
                              "flash_attn_version":2,"prefix_caching":False},
                 "genesis_environment": {key:value for key,value in os.environ.items()
                                         if re.match(r"GENESIS_PN?\d+_",key) or key in
                                         ("GENESIS_BUFFER_MODE","GENESIS_BUFFER_MODE_P38","GENESIS_TQ_MAX_MODEL_LEN","GENESIS_TQ_MAX_BATCHED_TOKENS")},
                 "model": os.environ.get("VLLM_MODEL"), "profile_enabled":False}
    # Values of inherited Genesis flags are benchmark configuration only. Never
    # print secrets/unrelated env; model records identity, not model contents.
    print(json.dumps(preflight), flush=True)
    with set_current_vllm_config(config):
        impl = cls(num_heads=24, head_size=256, scale=0.0625, num_kv_heads=4, kv_cache_dtype="turboquant_k8v4")
        impl._max_num_batched_tokens = 4096
        if not impl.tq_config.key_fp8 or impl.tq_config.effective_value_quant_bits != 4:
            raise RuntimeError("installed TurboQuant config is not k8v4")
        for case_index, (prior,length) in enumerate((p,q) for p in args.prefixes for q in args.chunks):
            batch, layer, metadata, pi, centroids = prepare_case(impl, backend, prior, length, args.block_size, args.seed + case_index)
            q,k,v = batch.q[0],batch.raw_k[0],batch.raw_v[0]

            def candidate_dispatch(self, current_layer, query, raw_k, raw_v, cache, md, index, total_tokens, destination):
                if index != 0 or total_tokens != length:
                    raise RuntimeError("benchmark supports exactly one continuation request")
                state = AttentionState(torch.empty_like(query, dtype=torch.float32),
                                       torch.empty(query.shape[:2], device=query.device, dtype=torch.float32))
                prefix_attention(query,cache,md.block_table[index],prior,self.scale,batch.spec,state,
                                 block_q=args.block_q,num_warps=args.warps)
                chunk = raw_chunk_attention(query,raw_k,raw_v,self.scale,fa_version=2)
                merge_attention_states(state,chunk,destination)
                return destination

            candidate_method = rewrite_prefill_method(baseline,inspect.getsource(baseline),candidate_dispatch)
            operations = {
                "baseline": lambda: baseline(impl,q,k,v,batch.kv_cache,metadata,pi,centroids,pi,layer),
                "candidate": lambda: candidate_method(impl,q,k,v,batch.kv_cache,metadata,pi,centroids,pi,layer),
            }
            # Clone because P26 reuses the same output pool across both callers.
            outputs = {name:operation().clone() for name,operation in operations.items()}
            compressed_tail = length <= 64 or prior >= 32768
            oracle_options = dict(causal_prefix=prior,scale=batch.scale,
                                  query_tile=args.reference_query_tile,key_tile=args.reference_key_tile)
            candidate_expected = bounded_reference(q,reference_loader(batch,prior,compressed_tail=False),prior+length,**oracle_options)
            baseline_expected = bounded_reference(q,reference_loader(batch,prior,compressed_tail=compressed_tail,
                                                                      round_prefix=not compressed_tail),prior+length,**oracle_options)
            checks = {"candidate_vs_raw_tail_oracle": numerical_gate(outputs["candidate"],candidate_expected,q.dtype),
                      "baseline_vs_deployed_semantic_oracle": numerical_gate(outputs["baseline"],baseline_expected,q.dtype),
                      "candidate_vs_deployed_baseline": error_metrics(outputs["candidate"],outputs["baseline"])}
            del outputs,candidate_expected,baseline_expected
            for _ in range(args.warmup):
                for operation in operations.values():
                    operation()
            torch.cuda.synchronize()

            def sample(operation):
                begin,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                wall_start = time.perf_counter()
                begin.record()
                operation()
                end.record()
                end.synchronize()
                return {"cuda_ms":begin.elapsed_time(end),"wall_ms":(time.perf_counter()-wall_start)*1000}

            measured = paired_measurements(operations,sample,repetitions=args.repetitions,seed=args.seed+case_index)
            for name in operations:
                for metric in ("cuda_ms","wall_ms"):
                    values = [item[metric] for item in measured[name]["samples"]]
                    measured[name]["median_"+metric] = statistics.median(values)
                    measured[name]["min_"+metric] = min(values)
            print(json.dumps({"record":"paired_complete_prefill","prefix":prior,"chunk":length,"total_context":prior+length,
                              "page_size":args.block_size,"block_q":args.block_q,"warps":args.warps,
                              "baseline_branch":"P101_compressed_current_decode" if compressed_tail else "P38_raw_current_continuation",
                              "checks":checks,"timings":measured,"candidate_policy_eligible":length>=129,
                              "classification":"paired_kernel_and_caller_microbenchmark_requires_serving_ttft",
                              "prefix_state_bytes":length*24*(256+1)*4,"profile_enabled":False}),flush=True)


if __name__ == "__main__":
    main()
