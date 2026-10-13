# Development retrieval recovery verification

Use this with the [retrieval replay guide](../operations/retrieval-replay.md) and
[source/index generation runbook](rag-source-index-generations.md).
The development rollout is authorized in AquiLLM-internal issues
[#4](https://github.com/AquiLLM/AquiLLM-internal/issues/4) and
[#6](https://github.com/AquiLLM/AquiLLM-internal/issues/6). Keep private corpus,
principal identities, full reports, environment snapshots and backups off Git.

## Release gates

1. Preserve the current application image/configuration and a verified database
   backup. Record the exact development commit and every replaced image.
2. Run the timeout, replay and source/authorization regression suites plus release
   CI. Health checks alone do not verify retrieval.
3. Independently verify embedding checkpoint and tokenizer file digests, running
   engine image and effective arguments, precision, quantization, pooling,
   dimensions and input templates. A model-name echo is insufficient.
4. Register the immutable verified profile, then rebuild eligible owners using
   the existing bounded worker path. Keep historical vectors and generations.
   Count terminal active, failed, excluded and pending outcomes separately.
5. Set DOCUMENT_EMBEDDING_SPACE_KEY consistently in the web and document worker
   environments. Validate an authorized dense query's response-bound observation
   and compatible current generation. Verify a real chat answer and its citations.
6. Freeze the comparison questions, source hashes, selected collections, principal
   and effective modes. Run retrieval-only comparisons separately from answer
   quality evaluation; retain timeouts, exclusions and empty/limited outcomes.

## Embedding input capacity and modality coverage

The current builder uses UTF-8 bytes as a conservative text-token upper bound.
A 2,048-token endpoint paired with a 2,048-byte profile can reject valid chunks.
Never declare a profile limit above the verified endpoint limit to bypass this
check. If changing the development endpoint context, preserve the exact model,
checkpoint, precision, quantization, GPU allocation and other service settings;
keep the original container/configuration for rollback and probe the new service
before registering its immutable profile. Record this context change in comparisons.

Use only modalities verified against the exact serving adapter and retained
artifacts. An explicitly text/caption-only profile excludes image chunks;
successful text indexing does not establish image indexing. Missing retained
image artifacts or an unsupported request format require their own repair.
Do not turn image inputs into text while calling the result an image embedding.

For Qwen3-VL-Embedding, verify template construction against the pinned tokenizer
and the [publisher's usage](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B).
Record server-side pooling/normalization in the engine policy separately from
client-side vector adaptation. Request 1,024 dimensions explicitly when using
the verified vendor-dimensions policy.

## Matched comparisons

Use the pre-remediation development commit
`dae54507c6e55f3cf1fb0a0974d4f22ebe2cdb8f` as the old-code baseline where an
isolated, read-only replay is safe. The internal October 11 graph study used
`759ef8d`; its existing measurements are historical, not current results.
Do not let an older binary mutate representation history or restore legacy
vectors as verified current vectors.

A current-code baseline and a source/window/capacity treatment can share a frozen
manifest and the verified text index. Use `--experiment current` and the explicit
mode switches documented in the replay guide. Set graph implementation mode
(`KG_GRAPH_REMEDIATION_MODE=legacy` or `v2`) explicitly in each isolated process
and retain the reported effective settings. Graph off/direct/extended/both are
branch ablations, independent of graph implementation mode.

Report candidate/final passage overlap, known support-passage reach, limited or
failed turns and full retrieval latency. Distinguish changes from new embeddings,
application code, source preservation and graph mode; a bundled before/after result
cannot attribute gains to one component. Known-question regression and operational
smoke tests do not meet independent held-out quality gates.

## Development reranker readiness

Windowed scoring needs a verified pair counter in each serving worker. Install a
read-only attestation directory, bind it into the web container, set
`RAG_PAIR_CAPABILITY_ATTESTATION` to its attestation file, and enable
`RAG_PAIR_CAPABILITY_WARM_ENABLED=1` with windowed mode. Use the ASGI lifespan
wrapper so each worker performs the bounded real protocol/tokenizer warm-up.
A separate management-command process does not initialize serving workers.

The deployment verifier must compare the running container identity, image,
start time, model/tokenizer/code revisions, effective engine arguments, template,
and retained artifact hashes before atomically renewing the short-lived proof.
Recheck the source verification record immediately before publishing; serialize
renewal attempts. Withdraw proof on a mismatch. A GPU-container restart requires
fresh verification even if its model name is unchanged. Keep the renewal service,
timer, proof-directory bind, and their rollback state with the private deployment
receipt. Do not install credentials in the proof or print runtime environment maps.

Warm-up consumes its own bounded startup allowance. Request retrieval keeps its
original 15-second ledger; diagnostic runs with a longer deadline are not release
acceptance results.

Exercise the real chat consumer as well as the retrieval replay. Chat preparation
uses the same request allowance, so a successful standalone replay does not prove
that acquisition, final evidence selection, model dispatch and answer persistence
fit the live path. Require a saved, validated answer with citations; a persisted
limited answer is a failed release smoke test.

If current graph membership assignments disagree with a ready projection's
registry checksum, leave graph retrieval disabled in the serving configuration
and retain that failure in graph-on comparisons. Do not substitute absent-graph
semantics, overwrite checksums, or replay projections blindly: projection replay
copies existing registry authority. Diagnose the current membership state and
measure readiness cost before using the locked membership advancement/build
workflow for a scoped repair. Repairing a checksum does not establish that graph
readiness and traversal fit the request budget.

## Comparison scope and failure accounting

Confirm that the historical principal, collection, and source hashes still exist
before calling a replay a replication. If only the questions are available, label
the run as transferred questions on a different corpus. The chat reproduction
question can duplicate a study question and must not count as another independent
sample. If historical embedding provenance is unknown, do not compare new query
vectors against those old stored vectors as if their spaces were compatible.

A disposable comparison role needs read-only transactions and no table-write or
privileged-role grants on every configured database alias. Preserve the graph
role configuration while overriding actual connection credentials. If a graph
read API requires explicit EXECUTE, verify the exact installed read-only function
body and grant only those functions to the temporary role.

Freeze ordered question manifests and scoped owner, source, representation,
embedding, permission, and graph fingerprints before each arm. Recheck them after
the run. Require a positive final guard receipt: a native success file left before
a wrapper failure, process timeout, or incomplete cleanup is a failed comparison,
with its raw report retained separately. Do not aggregate it as a successful turn.
