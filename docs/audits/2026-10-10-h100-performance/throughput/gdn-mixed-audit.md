# Mixed GDN gate alignment and batching audit — 2026-10-10

The captured deployed GDN method pairs gathered speculative QKV with gates
from the original full token order. This is a correctness blocker for promoting
`max-num-seqs=2` or `4` on the unchanged pinned stack. The local plugin candidate
backports the upstream gate gather without changing recurrence kernels,
dependencies, TurboQuant, prefill dispatch, MTP configuration, or deployment
defaults. GPU recurrence/state isolation and serving qualification are pending.

This audit targets Qwen3.6-27B with FP16 activations, TurboQuant k8v4, four MTP
drafts (five verification tokens), H100 80GB, vLLM
`0.23.1rc1.dev748+g2dfaae752`, and Genesis
`34e269301cc3df71ae4b0da00a0a159b16b4e5d8`. It does not transfer results from
the INT4/TP2/A5000 multiconcurrency preset to this configuration.

## Source and attribution

Root captured installed source files as data, without importing worker/model
instances, into `.superpowers/sdd/2026-10-10-throughput/sources/`. The three
complete source files used by the standalone reproducer are preserved under
`reproduction/sources/`, retaining their original Apache-2.0 attribution.

| Captured file | Original complete-file SHA256 |
| --- | --- |
| `utils.py` | `192e6b5df24ff52ae57a6a23d6cf58d1b4dd34f14c9283ce268e5f88b1123b96` |
| `gdn_attn.py` | `4f4293919acca150061ee6fa41b05b5c6f5f1f9989dfbc48188bb9b1b5b18296` |
| `qwen_gdn_linear_attn.py` | `55d2f57d434c49dc06472499deaac42392ae889285c760be7a357f966091d14c` |

The candidate delta matches the immutable upstream source at
[`276fbcff2717bd934cfa37c8a2e4c391f3e7237b`](https://github.com/vllm-project/vllm/blob/276fbcff2717bd934cfa37c8a2e4c391f3e7237b/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py#L1309-L1319),
checked on 2026-10-10: pure-spec aliases at lines 1312–1313, mixed gathers at
1317–1318, and recurrence kwargs at 1435–1436. The related upstream
[restoration commit](https://github.com/vllm-project/vllm/commit/38a1c179)
also describes why using full-batch gates undoes the mixed-batch fix. That
broader branch is not adopted; the candidate preserves the existing runtime.

The plugin guard binds the complete captured `_forward_core` method to a
canonical AST SHA256:
`8af7d81452b2f73fac77821e865714baa5c3e9a704d2d42cae562b7304642da1`.
Canonical JSON explicitly includes all semantic AST fields, represents
`Ellipsis`, and omits empty `type_params`. This avoids the changed empty-field
defaults of `ast.dump` between local Python 3.13 and serving Python 3.12.
Whitespace/comments can differ; semantic source drift is rejected.

## Proven ordering defect

The exact captured sources show this path:

1. TurboQuant requests reorder threshold 1 (`turboquant_attn.py:276`), while
   GDN requests speculative-decode support (`gdn_attn.py:119`). The hybrid
   runner chooses the minimum across attention groups
   (`gpu_model_runner.py:7088–7106`), leaving threshold 1.
2. Actual reorder logic (`utils.py:663–740`) orders ordinary decode, short
   continuation, long continuation, then fresh prefill. At threshold 1, a
   context-bearing one-token chunked-prefill continuation sorts before a
   five-token MTP decode. Long continuations and five-token MTP requests can
   also share a region; within-region order is not a spec-prefix guarantee.
3. Runner draft counts initialize to -1 and update only scheduled draft
   entries whose prompt is complete (`gpu_model_runner.py:2264–2279`). The
   remaining chunked-prefill token stays non-speculative.
4. GDN mixed metadata gathers speculative token indices, state rows, query
   offsets, and accepted counts consistently (`gdn_attn.py:315–364`). It
   reclassifies a one-token non-spec decode as prefill when another request
   is speculative (`284–292`); that does not remove the mixed batch.
5. Qwen gathers speculative QKV (`qwen_gdn_linear_attn.py:1397–1403`) but the
   speculative recurrence still receives full-order `a` and `b`
   (`1533–1548`). Non-spec gates already gather correctly (`1482–1484`).
   The recurrence indexes gates from its compacted query offset
   (`fused_sigmoid_gating.py:65–95`), so those offsets address the wrong rows.

For the two-request tail-prefill/MTP case, the executed extracted source gives:

| Boundary | Values |
| --- | --- |
| Post-reorder requests | `chunk_tail, mtp` |
| Original query offsets | `[0, 1, 6]` |
| Draft counts / spec mask | `[-1, 4]` / `[false, true]` |
| Speculative token indices | `[1, 2, 3, 4, 5]` |
| Recurrence QKV token IDs | `[200, 201, 202, 203, 204]` |
| Actual recurrence alpha gates | `[1100, 1200, 1201, 1202, 1203]` |
| Required alpha gates | `[1200, 1201, 1202, 1203, 1204]` |

Beta gates have the same mismatch. State rows remain the correct MTP rows,
so the proven issue is another token's gate influencing this request's
recurrence, not evidence of an allocator pointer alias. This audit does not
claim measured production output corruption or disclose private prompts.

With one real request, this mixed-request ordering cannot arise: speculative
decode uses the pure-spec branch, while prefill has no active speculative
request. Zero-length padded rows add no tokens ahead of the real sequence.
Thus the current max-seqs=1 configuration is unaffected by this specific
defect. That statement is not a general proof of every single-request path.

## Candidate backport and CPU evidence

`gdn_mixed.py` validates the exact known method signature, complete AST, and
unique speculative partition/call. It then applies only these assignments:

```python
# Pure-spec branch: preserve original views.
a_spec = a
b_spec = b
# Mixed branch: gather with the existing speculative QKV indices.
a_spec = a.index_select(0, spec_token_indx)
b_spec = b.index_select(0, spec_token_indx)
# Speculative recurrence receives a_spec and b_spec.
```

Pure-nonspec branches, accepted counts, state indices, convolutions, chunk
prefill, and recurrence implementation remain unchanged. Registration occurs
after Genesis and the existing runtime gate, with configuration/profile
validation preceding mutation. Installation is idempotent. An unknown method
raises `SystemExit` before class mutation, crossing bootstrap's existing
`Exception` handler so startup cannot silently skip this correction. The
candidate image supplies the correction alongside existing enabled H100
adapters; no extra environment flag or dependency installation is introduced.

The standalone `reproduction/gdn_gate_order_cpu.py` extracts and executes the
real reorder function, the real mixed GDN metadata branch, and the real Qwen
method up to a stubbed recurrence boundary. It never imports vLLM or launches
a kernel. It sets `CUDA_VISIBLE_DEVICES=-1` before importing torch. Original
source execution mismatches four mixed cases; pure-spec and new-prefill-last
controls align. Applying the proposed change only in memory aligns all six
cases and retains pure-spec gate pointers. Original/proposed complete JSON
evidence is preserved beside the script.

Run from this `throughput` directory:

```powershell
rtk proxy python reproduction/gdn_gate_order_cpu.py
rtk proxy python reproduction/gdn_gate_order_cpu.py --verify-proposed-fix
```

Automated regression uses the complete 335-line captured method fixture with
CPU tensors and stubbed GPU boundaries. The initial test-first run had 20
expected missing-backport failures and one original-defect reproducer pass.
An additional in-process replay replaced the rewriter with the original method
without changing files: all ten mixed gate-array contract cases failed, then
the normal focused run passed 31/31. Those failures include full gate-array
shape mismatch in spec-prefix controls; the standalone evidence distinguishes
the four cases whose consumed gate values are actually misordered.
The final complete plugin CPU suite passed **305/305** with CUDA hidden before
pytest/torch imports. Its 31 GDN cases cover mixed B2/B4 token partitions,
contiguous/interleaved gates, pure-spec pointer/stride retention, pure-nonspec
execution, source drift, installation order and idempotence. An intermediate
full-suite run had three expected fixture failures because prefill-install
tests lacked the new independent startup boundary; their fixture now stubs
that boundary while the dedicated GDN tests exercise it.

These are source-boundary correctness tests. Stubbed recurrence/conv/chunk
functions do not establish numerical tolerances, GPU state isolation, CUDA
graph behavior, model quality, or throughput.

## Other batch compatibility findings and bounded qualification

No fixed B=1 restriction was found in the actual P67b/PN521 request grids or
the H100 MTP adapter. P67b checks uniform query geometry at
`turboquant_attn.py:622–633`; ordinary ragged/mixed batches normally fall
through. Its output and partial caches include batch size in their keys
(`754–788`), and the raw-tail stage uses request-specific lengths, raw K/V
strides, and block-table rows (`p67_multi_query_kernel.py:962–1008,
1092–1152`). Empty partials are overwritten with zero/-infinity sentinels.
This is source evidence, not a serving concurrency result.

PN399's fixed workspace is for ordinary decode
(`turboquant_attn.py:106–136,1371–1388`), not every P67 holder. It assumes
serialized worker forwards. GDN explicitly rejects UBatching
(`gpu_model_runner.py:2518`); increasing max-seqs must not enable that path.
Keep current stream/DBO settings unchanged.

PN522 warms the configured maximum batch, uses a 3D dummy cache, and invokes
its warmup after the original compile/capture routine
(`pn522_tq_raw_tail_kernel_warmup.py:125–154,168,241–244`). Failures warn and
continue. A healthy startup alone therefore does not prove every real B1/B2/B4
layout has warmed successfully. Measure first real dispatches and reject
P67 fallback warnings (`turboquant_attn.py:819–824`).

The bundled multiconcurrency preset describes INT4, TP2 on A5000, and three
drafts; its older measurements do not validate this FP16/H100/four-draft
stack. Copying it would also change unrelated scheduling/stream settings.

Promotion remains gated on root-operated tests, preserving all existing
numerical limits and every dependency/plugin/configuration except the tested
batch ceiling:

| Gate | Bounded coverage |
| --- | --- |
| Actual GPU forward | Corrected mixed B2/B4 versus isolated original per-request forwards, real conv/recurrence/chunk, two successive calls, differing accepted counts, noncontiguous state slots, untouched-padding sentinels. |
| Startup and memory | Fresh max-seqs=2, then 4; package/plugin identity, capture/warmup diagnostics, cache capacity, actual B1..max routes. |
| Functional serving | Existing strict32 and long6 checks with independent per-request nonces, concurrent clients, shared prefixes with private suffixes. |
| Lifecycle | Arrival during decode, chunk-prefill tail plus MTP, short finish/long continuation, cancellation/replacement, reordered request slots, zero/fewer draft tokens, B1→B2→B4→B2→B1 transitions. |
| Performance | Identical warmed C1/C2/C4 inputs; aggregate/per-request throughput, TTFT/TPOT, queue delay, acceptance, errors and completion flags. Single-client throughput may regress with a larger batch ceiling. |

Root alone owns SSH, GPU execution, immutable candidate builds, serving changes,
and rollback. At this report's completion no GPU or serving numerical result
has been claimed for the gate-alignment candidate, and production defaults
have not been changed by this task.
