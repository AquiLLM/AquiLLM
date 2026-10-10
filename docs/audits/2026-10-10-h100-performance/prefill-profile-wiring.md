# Experimental continuation prefill wiring

`AQUILLM_H100_PREFILL=1` selects the bundled `h100-long-prefill-dev-v1` profile
when no explicit profile is supplied. The default remains off. Unknown explicit
profile names raise before either prefill or MTP adapter mutation. MTP baseline
and fused controls remain independent; adaptive split policy remains blocked.

The bounded region is a cached prefix of 32768 through 65536 tokens and a current
query of 1024 through 4096 tokens. Short prefixes, fresh prefill, verification,
and unsupported semantics retain the existing post-Genesis paths. This is an
experimental development profile pending serving qualification. The coordinator
owns the evidence in `prefill-microbenchmark.json`, including the corrected FA2
rerun; selecting the profile does not assert that serving thresholds have passed.

The worker checks actual H100 80GB SM90/132-SM device metadata, FP16 tensors,
k8v4 cache, 24 query heads, 4 KV heads, dimension 256, and page size 2128 before
scratch allocation. Destination, raw KV, cache, and page-table metadata must
match the supported geometry, device, dtype, and strides. CUDA graph capture
falls back because the installed raw FlashAttention helper reads host metadata.
Startup registration does not query CUDA properties.

The strict constructor ABI guard still captures semantic options and rejects
unknown overlays. It also snapshots the resolved Hugging Face commit hash from
the current vLLM model configuration; requested `revision=None` does not qualify
a missing hash. Only `2d783431e303148fc6e16622fac5edac83a6b5c4` and an explicit
`attention_config.flash_attn_version=2` qualify. Missing or mismatched values
fall back before tensor or CUDA access. Existing prefill source fingerprint and
P101 insertion remain unchanged.

The successful worker route emits `AQUILLM_H100 route_exercised prefill` once,
after prefix/raw-chunk/merge execution and outside capture. An enabled but
inactive canary must not qualify; require that marker in external monitoring.
There is no per-layer fallback log flood.

CPU verification: `rtk python -m pytest -q
deploy/vllm_plugins/h100_kernels/tests/cpu` passed 258 tests. New tests exercise
opt-in/default-off behavior, independent controls, model and FA2 snapshots,
geometry rejection before allocation, untouched fallback destinations, capture
fallback, and the successful route marker. GPU allocation/launch boundaries are
stubbed only in CPU routing tests; these do not prove numerical GPU correctness.
The existing GPU caller fixtures now use page size 2128 and FA2/revision
semantics. This worker did not run GPU tests; the coordinator must rerun them
against the committed wiring before a development canary.
