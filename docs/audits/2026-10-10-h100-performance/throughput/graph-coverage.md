# MTP graph coverage audit — 2026-10-10

The captured installed dispatcher predicts incomplete FULL graph coverage for
two to four concurrent requests under the default sequence-limit-four policy.
An explicit capture list adds matching FULL keys for those batches. This is a
source and CPU routing result; it makes no performance claim.

Scope: H100 80GB, TP1, Qwen3.6-27B FP16, TurboQuant k8v4, four MTP drafts,
`max-num-batched-tokens=4096`, pinned vLLM `0.23.1rc1.dev748+g2dfaae752` and
Genesis `34e269301cc3df71ae4b0da00a0a159b16b4e5d8`. The derivation assumes
throughput performance mode, no other capture-size override, no LoRA, and
resolved `FULL_AND_PIECEWISE` target graph mode.

## Captured source provenance

Root captured installed source as data without creating a worker or model.
Paths below are relative to `.superpowers/sdd/2026-10-10-throughput/` in the
audit worktree. SHA256 identifies complete captured files, including Genesis
rewrites; upstream source alone does not reproduce those rewrites.

| Captured source | Complete-file SHA256 |
| --- | --- |
| `logits-sources/vllm_config.py` | `8004a7fb61a0db3f03031127532ffa282945bf11aa516e56f3bc0aa0950bf005` |
| `logits-sources/compilation.py` | `163b3f78fea767c65a28f9319201dc146e6d3faca9fe4cf87437bd96feb005be` |
| `logits-sources/cudagraph_dispatcher.py` | `51f632753af729aeee6ce4a802d9d74b64f22eef02a524f36e2a24b07ef0cf63` |
| `sources/gpu_model_runner.py` | `8db8e5bbd05cbbb38dbf0a4e50a48cee301cb382ffeb00998a318ba7a6df236a` |
| `sources/llm_base_proposer.py` | `54d9f45fc173b5370d13a04cb992c252bad1bd94fc30b8157e849c7a3cb65510` |

Line numbers below refer to these captured files. The pinned upstream reference
is [vLLM commit 2dfaae752b4db0d43cfc0715c780e33be030d0f1](https://github.com/vllm-project/vllm/tree/2dfaae752b4db0d43cfc0715c780e33be030d0f1).
All five captured files are preserved by basename in
`reproduction/graph-sources.tar.gz`. The CPU reproduction reads the matching
dispatcher directly from that archive without extraction and rejects a different
hash. An explicit local source-file argument is also supported.

## Default and explicit coverage

`vllm_config.py:1725–1731` computes the default maximum capture size as
`min(max_num_seqs * (1 + num_speculative_tokens) * 2, 512)`, capped by the token
budget. With four drafts, the uniform verification length is five.
The throughput default sequence at `1777–1804` gives `[1,2,4,8]` for sequence
limit one and `[1,2,4,8,16,24,32,40]` for limit four. Genesis P66 at `1814–1837`
retains only multiples of five and adds five itself. Thus the inferred resolved
lists are `[5]` and `[5,40]`, respectively. The subsequent rounding in
`compilation.py:1424–1436,1465–1510` leaves these lists unchanged.

The dispatcher adds PIECEWISE keys for every capture size
(`cudagraph_dispatcher.py:189–203`), but FULL decode keys only through
`uniform_decode_query_len * max_num_seqs` (`207–232`). At limit four that bound
is 20, so size 40 contributes no FULL decode key.

| Policy | Resolved capture list | FULL token keys | PIECEWISE token keys |
| --- | --- | --- | --- |
| Sequence limit one, default | `[5]` | `5` | `5` |
| Sequence limit four, default | `[5,40]` | `5` | `5,40` |
| Sequence limit four, explicit candidate | `[5,10,15,20,40]` | `5,10,15,20` | `5,10,15,20,40` |

Root observed **one FULL and two PIECEWISE capture entries** during the default
sequence-limit-four startup, consistent with the inferred list. The counts do
not independently identify the sizes or prove which graphs a request replays.
The explicit candidate predicts four FULL entries, adding **10, 15, and 20**,
and five PIECEWISE entries. Root's curated read-only startup capture confirms
the explicit configured list `[5,10,15,20,40]` and completed counts of **four
FULL / five PIECEWISE**. It records candidate image
`sha256:6c6005ebae4f46c98ace93c78df10420eb3a39370bc9da7173c501017595eee0`,
started `2026-10-10T21:54:20.652919903Z`.

The captured artifact is
`.superpowers/sdd/2026-10-10-throughput/throughput-graphs-1-graph-settings.json`,
SHA256 `0e18573937350211b1b7872d10b64d86e9831ab5ccb7c09a93d8eaa48ce3a937`.
It is retained with the final graph-comparison archive. This confirms
configured sizes and completed capture counts; the attribution of individual
FULL keys to sizes remains the source derivation above, without a per-request
replay measurement.

## CPU routing evidence and single-request implication

The [CPU reproduction](reproduction/graph_dispatch_cpu.py) extracts six actual
dispatcher methods by AST and supplies metadata-only enum/config/descriptor
objects. It imports neither torch nor vLLM, performs no CUDA operation, and
asserts the following routing after initializing the stated keys:

| Concurrent uniform verification requests | Actual tokens | Default limit-four route | Explicit candidate route |
| --- | --- | --- | --- |
| 1 | 5 | FULL, 5 tokens / 1 request | FULL, 5 tokens / 1 request |
| 2 | 10 | PIECEWISE, 40 tokens | FULL, 10 tokens / 2 requests |
| 3 | 15 | PIECEWISE, 40 tokens | FULL, 15 tokens / 3 requests |
| 4 | 20 | PIECEWISE, 40 tokens | FULL, 20 tokens / 4 requests |

Padding uses the next capture size (`cudagraph_dispatcher.py:72–94`); dispatch
tries the exact FULL key before a relaxed PIECEWISE key (`302–324`). The relaxed
key has `num_reqs=None` and `uniform=False`: a 40-token PIECEWISE route does not
mean 40 live requests. Actual target uniformity is checked in
`gpu_model_runner.py:3888–3906`, then dispatched at `3953–3966`.

A single target request keeps the five-token FULL descriptor under both limits.
The source therefore does not support blanket padding of a single user to four
requests as an explanation for a single-user speed change. The MTP proposer
shares the compilation configuration (`llm_base_proposer.py:77,159–165`), uses
PIECEWISE keys (`418–433`), and consumes the dispatcher's padded token count
(`1759–1763`). Its logical one-token step pads to five in both policies; the
reproducer checks that unchanged routing too.

Run from the repository root using the bundled source archive:

```powershell
rtk proxy python -B docs/audits/2026-10-10-h100-performance/throughput/reproduction/graph_dispatch_cpu.py
```

The CPU command completed successfully, with all embedded routing assertions
passing against the hash-verified captured dispatcher.

This proves conditional metadata routing, not actual graph replay, GPU numerical
correctness, memory cost, startup cost, or throughput. Qualification must verify
runtime graph reuse, preserve both installed
prefill and mixed-GDN hooks, pass serving correctness/lifecycle checks, and
compare paired unprofiled workloads. The experiment retains sizes five and 40
and adds 10/15/20; it changes neither four-draft MTP nor the scheduler limit.
