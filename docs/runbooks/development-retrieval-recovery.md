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
