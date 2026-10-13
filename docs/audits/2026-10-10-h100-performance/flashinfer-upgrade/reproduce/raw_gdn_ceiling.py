"""Synthetic dense/all-positive FI upper bound; never adapter qualification.

Genesis is registered before backend imports. Lower-level BF16 dispatch
avoids the public wrapper's dummy-zero launch. Timing includes CUDA events,
warmup, five repeated rounds, and no preparation in the raw ceiling route.
"""
import argparse
import importlib.metadata
import json
import os
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=300)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--include-fp16-api", action="store_true")
    parser.add_argument("--include-pack-overhead", action="store_true")
    opts = parser.parse_args()
    if opts.repeats < 200 or opts.rounds < 1:
        raise ValueError("require at least 200 repeats and one round")
    os.environ["AQUILLM_H100_GDN"] = "baseline"
    import sndr.plugin
    sndr.plugin.register()
    import torch
    from aquillm_vllm_h100.gdn.adapter import capture_original
    from flashinfer.gdn_decode import gated_delta_rule_mtp
    from flashinfer.gdn_kernels.gdn_decode_mtp import (
        run_mtp_decode, get_tile_v_mtp, get_vec_size_mtp)
    original = capture_original()
    device_name = torch.cuda.get_device_name()
    if "H100" not in device_name or torch.cuda.get_device_capability() != (9, 0):
        raise ValueError("serialized H100 only")
    torch.manual_seed(2026)
    t, h, hv, dim, slots = 5, 16, 48, 128, 9
    size = hv * dim * dim
    dense = torch.randn(slots, hv, dim, dim, device="cuda") * 0.03
    dense_seed = dense.clone()
    padded_backing = torch.full((128 + slots * (size + 128),), 987.25, device="cuda")
    padded = padded_backing.as_strided(dense.shape, (size + 128, dim * dim, dim, 1), storage_offset=128)
    padded.copy_(dense)
    def activation(shape):
        return (torch.randn(shape, device="cuda") * 0.2).half()
    q, k, v = activation((1,t,h,dim)), activation((1,t,h,dim)), activation((1,t,hv,dim))
    a_contig, b_contig = activation((t,hv)), activation((t,hv))
    ba = torch.empty((t, 2*hv), device="cuda", dtype=torch.float16)
    b_strided, a_strided = ba.chunk(2, dim=-1)
    a_strided.copy_(a_contig)
    b_strided.copy_(b_contig)
    A_log = torch.full((hv,), -1.0, device="cuda")
    dt_bias = torch.full((hv,), -0.2, device="cuda", dtype=torch.float16)
    dt_fp32 = dt_bias.float()
    offsets = torch.tensor([0,t], device="cuda", dtype=torch.int32)
    indices = torch.tensor([[1,2,3,4,5]], device="cuda", dtype=torch.int32)
    accepted = torch.tensor([3], device="cuda", dtype=torch.int32)
    read = torch.tensor([3], device="cuda", dtype=torch.int32)
    q_bf, k_bf, v_bf = (x.bfloat16() for x in (q,k,v))
    a_bf, b_bf = a_contig.bfloat16().unsqueeze(0), b_contig.bfloat16().unsqueeze(0)
    out_bf = torch.empty_like(v_bf)
    out_fp16 = torch.empty_like(v)
    dummy = torch.empty((1,dim,dim), device="cuda")
    tile_v = get_tile_v_mtp(1,t,num_v_heads=hv,v_dim=dim)
    vec_size = get_vec_size_mtp(1,t)
    def lower(pq=q_bf,pk=k_bf,pv=v_bf,pa=a_bf,pb=b_bf,po=out_bf,pr=read):
        run_mtp_decode(dense.view(slots*hv,dim,dim), dummy, A_log, pa, dt_fp32,
                       pq,pk,pv,pb,po,pr,1,t,h,hv,dim,dim,slots,t,tile_v,vec_size,
                       dim**-0.5,True,False,False,ssm_state_indices=indices,
                       output_state_indices=None,use_pool_indexing=False)
        return po
    def public_bf16():
        return gated_delta_rule_mtp(q_bf,k_bf,v_bf,dense,read,A_log,a_bf,dt_fp32,b_bf,
                                   output=out_bf,ssm_state_indices=indices,
                                   intermediate_states_buffer=None,disable_state_update=False,
                                   use_qk_l2norm=True)[0]
    kwargs = dict(A_log=A_log,dt_bias=dt_bias,q=q,k=k,v=v,a=a_contig,b=b_contig,
                  initial_state=dense,cu_seqlens=offsets,ssm_state_indices=indices,
                  num_accepted_tokens=accepted,use_qk_l2norm_in_kernel=True)
    rounded = dict(kwargs, **{name:kwargs[name].bfloat16().half() for name in ("q","k","v","a","b")})
    dense.copy_(dense_seed)
    expected_o, _ = original(**rounded)
    expected_state = dense.clone()
    dense.copy_(dense_seed)
    actual = lower()
    torch.cuda.synchronize()
    def error(x,y):
        delta = x.float()-y.float()
        return {"max":delta.abs().max().item(),"rms":delta.square().mean().sqrt().item()}
    numerical = {"rounded_output":error(actual,expected_o),"rounded_state":error(dense,expected_state)}
    torch.testing.assert_close(actual.float(),expected_o.float(),atol=0.005,rtol=0.025)
    torch.testing.assert_close(dense,expected_state,atol=0.0003,rtol=0.02)
    result = {"kind":"synthetic_raw_FI_ceiling_not_adapter_qualification",
              "gpu":device_name,"T":t,"repeats":opts.repeats,"rounds":opts.rounds,
              "numerical_check":numerical,"packages":{},"timings_us":{},
              "notes":["lower BF16 route has no external metadata/preparation or checkpoint traffic",
                       "public BF16 wrapper includes its dummy-zero launch",
                       "all destinations positive and dense pool; unsafe strided/null direct scatter is never exercised",
                       "gate stride96 is inferred from Qwen ba.chunk; not an observed serving trace"]}
    for name in ("vllm","torch","triton","flashinfer-python","nvidia-cutlass-dsl"):
        result["packages"][name] = importlib.metadata.version(name)
    routes = [("original_fp16_dense_gates48",lambda:original(**kwargs)),
              ("original_fp16_strided_pool_gates48",lambda:original(**dict(kwargs,initial_state=padded))),
              ("original_fp16_strided_pool_gates96",lambda:original(**dict(kwargs,initial_state=padded,a=a_strided,b=b_strided))),
              ("raw_lower_bf16_dense_one_launch",lower),
              ("raw_public_bf16_dense",public_bf16)]
    if opts.include_fp16_api:
        def public_fp16():
            return gated_delta_rule_mtp(q,k,v,dense,read,A_log,a_strided.unsqueeze(0),dt_fp32,b_strided.unsqueeze(0),
                                       output=out_fp16,ssm_state_indices=indices,
                                       intermediate_states_buffer=None,disable_state_update=False,
                                       use_qk_l2norm=True)[0]
        routes.append(("raw_public_fp16_dense_including_staging",public_fp16))
    if opts.include_pack_overhead:
        from aquillm_vllm_h100.gdn.kernels import prepare, _unpack
        def packed():
            operands,prepared_read,metadata,unused_checkpoints,packed_out,output = prepare(**dict(kwargs,a=a_strided,b=b_strided))
            lower(*operands[:3],*operands[3:],packed_out,prepared_read)
            _unpack[((t*hv*dim+255)//256,)](packed_out,output,prepared_read,metadata,t,256)
            return output
        routes.append(("raw_lower_plus_existing_meta_pack_unpack",packed))
        result["notes"].append("existing prepare allocates unused checkpoint scratch here, but no checkpoint traffic")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for name,function in routes:
            for _ in range(30):
                output = function()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):
                output = function()
            graph.replay()
            start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            samples = []
            for _ in range(opts.rounds):
                dense.copy_(dense_seed)
                padded.copy_(dense_seed)
                start.record()
                for _ in range(opts.repeats):
                    graph.replay()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end)*1000/opts.repeats)
            result["timings_us"][name] = {"median":statistics.median(samples),"min":min(samples),"max":max(samples),"samples":samples}
    torch.cuda.current_stream().wait_stream(stream)
    print(json.dumps(result,indent=2),flush=True)


if __name__ == "__main__":
    main()
