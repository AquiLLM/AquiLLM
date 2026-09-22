# Production reliability backport rollout

This is an operator checklist for a separately scheduled deployment. Merging the
backport does not reload running Python processes, rebuild images, or reconcile
the server's existing `.env`. No database migration is introduced by this backport.

## Record and reconcile

Record the deployed source SHA, image IDs, Compose file/profile selection, and a
protected backup of the existing `.env` before changing the deployment. Keep the
previous images available for rollback. Do not publish resolved Compose output
or environment dumps: they contain credentials.

Use the server's existing Compose project name and files. Supply `--env-file .env`
for every Compose invocation so interpolation and container `env_file` agree.
The base chat service deliberately leaves context configuration to `env_file`;
development and production Compose provide overridable defaults.

Reconcile these settings explicitly rather than replacing the production `.env`:

| Setting | Backport profile |
| --- | --- |
| `VLLM_MAX_MODEL_LEN` | `131072` |
| `OPENAI_CONTEXT_LIMIT`, `PROMPT_BUDGET_CONTEXT_LIMIT` | If set, match the served chat context |
| `APP_RERANK_MAX_MODEL_LEN`, `APP_RERANK_PAIR_TOKEN_LIMIT` | `1024` |
| `APP_RERANK_DOC_CHAR_LIMIT` | `900` |
| `APP_RERANK_TEMPLATE_RESERVE_TOKENS` | `96` |
| `APP_RERANK_SCORE_CONCURRENCY` | `6` (reduce for a constrained executor) |
| `MEM0_EMBED_MAX_MODEL_LEN` | `2048` |

Keep the existing Genesis image/plugin pins, TurboQuant/MTP correctness settings,
and prewarm readiness check. They were already deployed during diagnosis. Do not
substitute a floating Genesis branch or discard the known working configuration.
`OPENAI_COMPAT_PROMPT_SLACK_TOKENS` is the local provider's context reserve;
`PROMPT_BUDGET_SLACK_TOKENS` is a separate cross-provider preflight setting.
Existing overrides for either remain operator choices.

## Shared GPU capacity

The aligned sample allocations are chat `.45`, Whisper `.08`, embedding `.12`,
and reranker `.15`: total `.80` of one GPU. The optional OCR profile adds `.15`,
bringing the total to `.95`; enable it only after measuring sufficient workspace
headroom. Native chat OCR does not require that sidecar. These fractions are
starting budgets, not guarantees for another GPU or model checkpoint. The `.15`
reranker budget reflects the inspected healthy H100 deployment; it is not a
universal minimum.

Do not copy development's larger `.20` embedding and `.30` reranker allocations
independently. Account for every process on the device, including transcription,
other applications, CUDA workspaces, and optional profiles. An aggregate below
one is necessary but does not guarantee enough cache blocks within each service's
allocation. Adjust the profile together if startup reports unavailable cache
blocks or insufficient free GPU memory.

The embedding startup fix preserves the requested bitsandbytes 4-bit weights.
Verify the effective launch arguments after rebuilding: pooling embeddings must
retain quantization, while the Qwen3-VL classifier reranker remains fp16. Retain
the embedding model and vector width used by the existing database; this rollout
does not migrate or reindex stored embeddings.

## Deploy and verify

1. Validate the chosen Compose configuration with `config --quiet`, using the
   actual `.env`, project name, file combination, and enabled profiles.
2. Build the application/frontend image from the reviewed SHA and recreate the
   web process and affected workers. A source bind mount alone does not reload
   imported Python modules. Ensure the newly built frontend assets are served.
3. Rebuild and recreate the embedding service from the updated vLLM startup
   script to apply the quantization fix. Recreate any other service whose image
   or environment changed. Retain the main Genesis runtime if unchanged.
4. Wait for dependency health and chat prewarm readiness. Check GPU memory and
   logs for cache-block failures, CUDA errors, and container restarts. Query
   `/v1/models` with the configured authentication and compare the served chat
   context with the application's advertised limit.
5. Exercise a collection query that produces empty and nonempty search results,
   then a longer collection query. Confirm tool results validate, the final
   answer appears, and reranking does not repeatedly probe unsupported shapes or
   overflow its 1024-token limit. In staging, force a tool failure and verify the
   UI shows a terminal error and allows a fresh retry.

The knowledge-graph DNS changes in development are not included: those services
are outside this backport. Direct RAG remains controlled by its existing flag;
the tool-loop regression must be verified with direct RAG disabled.

## Rollback

Restore the recorded source revision, images, and protected environment backup,
then recreate only the affected services with the same Compose project/files.
Check health and the previous baseline query again. Do not remove volumes or
run `down -v`; the database, model caches, and stored documents are retained.
