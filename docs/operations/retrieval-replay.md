# Development retrieval replay

Run `scripts/replay_retrieval.py` in the application environment with database access and an explicitly selected active principal. This invokes the production direct RAG pipeline with authorized selected collections, records its evidence packet instead of synthesizing, and never creates or saves a chat. Embedding and reranking services remain necessary; generation and query rewrite are denied.

```sh
python scripts/replay_retrieval.py --manifest /private/questions.json --principal-id 12 --collection-ids 226 --output /private/parent-baseline.json --repetitions 1 --experiment baseline
```

Use an explicit private manifest; no internal questions, identifiers, or corpus fixtures ship with this tool:

```json
{"version":1,"questions":[{"id":"Q1","question":"Exact original question","targets":[{"source_sha256":"<64 lowercase hex characters>","answer_sha256":"<64 lowercase hex characters>","start":100,"end":150}]}]}
```

Question strings are read exactly as decoded JSON, without normalization. IDs contain 1–64 ASCII letters, digits, underscores, periods, or hyphens and must be unique. Supply 1–100 questions, 1–10 repetitions, at most 1000 total turns, and 1–64 distinct positive collection IDs. An output file is required and must differ from the manifest.

By default, revision comes from `git rev-parse HEAD` in the checkout and is labeled `git_verified`. For an operator-verified archive exported into a container without `.git`, supply `--revision` with the full 40-character hexadecimal commit SHA. The report explicitly labels this revision `operator_supplied`; the operator must verify which archive was copied. An override is never represented as git verification.

Targets are optional. `source_sha256` hashes the authorized document's complete `full_text`, encoded as UTF-8 with no normalization. Duplicate source bodies are ambiguous and rejected. `text_sha256` optionally selects exactly one chunk by its complete `TextChunk.content` fingerprint; duplicate matching chunks are rejected. For answer coverage, use `answer_sha256` with half-open Unicode character offsets `start` and `end` into `full_text`. The sliced source must match the answer hash, and at least one authorized selected chunk must contain that complete span within its stored positions and content. Missing sources, missing chunks, or mismatched spans fail mapping explicitly before any turn. A hash of a chunk is not a full-source hash. Targets spanning chunk boundaries are rejected unless one chunk actually contains the full span.

The report separates supporting-passage pool ranks, rerank ranks, packet ranks, and exact answer-span survival in packet text. A packet may contain the supporting chunk but omit the answer after truncation; its coverage is then false. Chunk-only targets report null coverage. Ranks from different search queries remain separate observations rather than an invented global rank.

Experiments are process-local and restored on exit: `baseline`, `graph-off`, `direct-only`, `extended-only`, `depth30`, `depth60`, `full-text`, `depth60-full`, and `current`. The original experiments explicitly select legacy rerank/evidence/capacity modes and disable both vocabulary proposals and model rewrites, even when deployment flags enable them. Baseline preserves configured graph and selection settings. This is a reproducible current-code legacy-mode baseline; recreating old code also requires an isolated historical revision and matched corpus/embedding state.

Depth experiments set vector/trigram caps to 30 or 60 and candidate multiplier to 4.0; observed effective limits in each retrieval stage are authoritative and can be smaller. Full-text experiments set the rerank character cap to 2048; token caps can still truncate text. This remains legacy text mode. Graph experiments require the deployment's graph prerequisites to be available.

Use `--experiment current` for independently selected development treatments. Every switch defaults to the legacy baseline choice, so specify the axes being tested. Treatment switches with another experiment are rejected to prevent labeling source/windowed retrieval as the legacy baseline.

| Option | Supported values | Default |
| --- | --- | --- |
| `--rerank-text-mode` | `legacy`, `windowed` | `legacy` |
| `--evidence-text-mode` | `legacy`, `source` | `legacy` |
| `--document-capacity-mode` | `legacy`, `budgeted` | `legacy` |
| `--query-mode` | `off`, `vocabulary` | `off` |
| `--graph-mode` | `configured`, `off`, `direct`, `extended`, `both` | `configured` |

For example, compare the same manifest, principal, collection scope and revision against source/windowed preservation, with graph explicitly disabled:

```sh
python scripts/replay_retrieval.py --manifest /private/questions.json --principal-id 12 --collection-ids 226 --output /private/source-window.json --experiment current --rerank-text-mode windowed --evidence-text-mode source --document-capacity-mode budgeted --graph-mode off
```

Add `--query-mode vocabulary` in a separate matched run to exercise deterministic authorized vocabulary proposals. Model rewrite, iterative planner generation and follow-up evidence remain disabled for every retrieval-only replay. The current rewrite entry point is guarded in addition to the generic provider interface guard. Cache use/publication is disabled in this process; the replay creates only an in-memory conversation and never saves chat records. Source preparation and permission/publication fences still execute. Rerank/embedding services and authorized representation/vocabulary generations remain prerequisites.

The JSON report includes revision, manifest digest, requested counts, selected modes, effective configuration including proposal controls and configured turn limits, observed effective retrieval limits, event-derived retrieval/selection timings, branch status, and novel graph candidate count/provenance. Each turn reports observed ledger limits/state/completion counters and the packet retrieval status. Proposal observations report counts and whether vocabulary was observed; model names, ledger identities and raw proposal strings are omitted. Source/windowed acquisition gets replay-only search/packet observations at the same real boundaries; unavailable timings remain null. Graph readiness remains unknown when observers cannot prove it. Graph unique count means observed novel fused/materialized candidates and does not establish answer quality. Fixed-label graph stage timing capture is unavailable; the report states this explicitly. Each turn owns a recorder that closes on completion so late background events cannot enter another question.

For `graph-off`, verify each observed retrieval stage reports graph `unique_count: 0`. The override restores the original graph-on setting on both normal exit and exceptions; it changes only the replay process. Current-release graph-off does **not** recreate the historical Study 2 baseline at `b70f58a`: code, corpus, embeddings, and other retrieval settings can differ. Compare the recorded revision and effective settings explicitly.

Topology rejection logs use the fixed `obs.rag.topology_rejected` event with a closed phase (`source_family_cap`, `family_schema`, `empty_frontier`, `snapshot_build`, or `snapshot_compose`), family/branch labels, bounded count/maximum, and elapsed milliseconds. These distinguish source cap and validation boundaries without source text, identifiers, Cypher, or exception messages. Public branch failure reasons remain unchanged. The reported production-corpus topology-invalid incident has not been reproduced on the development corpus; these diagnostics help locate a future occurrence and do not establish that it is fixed.

Default output includes no question/query/source text, raw document/chunk/entity IDs, or exception messages. It contains question IDs, fingerprints, counts, ranks, safe statuses, and exception class names. Keep manifests and reports private. Existing application service logs still follow deployment logging policy; this CLI adds no prompt-bearing logging.

Exit 0 means every turn produced a recorded packet and complete observations with no generation/SDK attempt, observer failure, unexpected outcome, search failure, or partial/context-limited packet. A handled limited answer is a failed quality outcome. No-result packets are valid retrieval observations on the legacy route; source acquisition that finishes limited without a packet is a failure. A graph branch timeout is also a valid observation when the turn's search succeeds. Exit 1 preserves failed runs in the report instead of treating them as zero-recall success. Invocation errors use argparse exit 2.

For the ten internal questions, run the unchanged private manifest for parent collections, then the same manifest for parent plus figures. Repeat each scope with the selected experiment and distinct output filenames; compare supporting-target membership and span coverage, not graph count alone. Deployment and live validation are separate operator actions.

For a coordinated interactive study on the current release, `KG_OVERLAY_ENABLED=0` in the web application's environment disables the Graph RAG overlay globally after the web process is recreated. Record the release and other retrieval settings, verify the disabled runtime value, and restore the prior value after the agreed study window. The replay `graph-off` experiment changes only its own process and does not disable graph retrieval for other users. Neither method recreates the historical Study 2 revision.
