# mimalloc experiment protocol

The user requested model-API latency testing followed by full chat/RAG replay. This experiment uses the allocator implementation from [PR #240](https://github.com/AquiLLM/AquiLLM/pull/240), initially pinned at `fedc29373c634fd47cf1f4429d133edaa2bc7e91`. The subsequent Compose-test correction at `a1bc3f6b2b4ba8b9541da470bd3e3b38339bf747` leaves the allocator build, verifier and launcher identical.

## Model API comparison

Build one thin image on the exact deployed prefill image, copy the PR's pinned allocator build and launcher, and wrap its existing Genesis entrypoint. Both arms use this same image: `sha256:dc14ba6ec72907fdcbc097a08eb99d69f104d9817a6e5758d29b5c819694926c`.

- Control: `AQUILLM_ALLOCATOR=system`, `PYTHONMALLOC=default`.
- Candidate: `AQUILLM_ALLOCATOR=mimalloc`, `PYTHONMALLOC=default`.
- Both: bounded prefill enabled, original MTP/verifier/split policy, identical model, scheduler settings, cache, commands, mounts and sidecars.
- Verify allocator startup checks and loaded mappings in both the API process and EngineCore. A new `docker exec` interpreter does not prove serving-process activation.
- Use a separate pinned experiment state and guarded switch. Preserve the original H100 rollback state. Restore the deployed prefill image if correctness, configuration or activation verification fails.

Measure three alternating control/candidate pairs, each with frozen 512/8192/32768/36864-token inputs, 256 output tokens, one warmup and ten measured repeats per shape. Record full output text/hash, completion and usage, TTFT, decode duration, complete response time, per-shape MTP counters, and process memory before/after each block. Keep benchmark client allocation unchanged. Startup and compilation are separate from warmed request latency.

Report each shape and the equal-request mixed workload. A meaningful allocator performance claim requires at least 5% improvement with paired uncertainty excluding zero and no unexplained protected p95/error/quality regression. Smaller effects or wide intervals are inconclusive. The already accepted H100 prefill tradeoff does not automatically waive allocator qualification.

## Full chat/RAG comparison

After model-API measurement, compare the system allocator with mimalloc in both the main inference service and web service through the real authenticated application. This application-stage extension was frozen before the first mimalloc model-API arm produced results. Use one thin web image derived from its exact existing image, with PR #240's same pinned library and launcher, and preserve the original command. Both application arms use the same two thin images; only the allocator selection changes. Keep embedding, reranker, Celery and other service images unchanged.

Use two system/mimalloc pairs, each with one warmup and five measured requests for each of plain chat and RAG. This small synthetic replay is exploratory, not a production workload or concurrency-throughput qualification. Use a disposable principal and private synthetic collection, ordinary VTT ingestion and embedding, normal chat creation, and the production WebSocket append protocol. Reuse the same fixture and prompt across arms; clear only created conversations and the principal's live memory namespace between arms. Check that no profile facts have accumulated. Delayed background jobs are observed separately from answer timing and are not included in a claim of complete background-job latency.

Measure user-append to first visible answer, final stream completion and persisted assistant message separately. Verify exact synthetic answers, retrieved/cited fixture chunks, actual local-model routing and successful synthesis; mocked evaluation and extractive fallback are not live model evidence. Record background work and memory activity as potential contention. Clean up only the created chats, collection, principal and its memory namespace.

This tests main inference alone first, then main inference plus web through the application. It does not establish an all-service mimalloc speedup or authorize production rollout. PR #240's broader image/default changes remain a separate deployment decision.

## Completed development experiment

Both stages completed on `aquillm-dev2` (`149.165.150.254`). The original prefill
model image and original web image were restored, with their original allocator
environment, and independently verified healthy at `2026-10-10T18:16:15Z`.
The model answered the final exact smoke prompt; the web login returned HTTP 200.
The disposable principal, source, chunks and 48 conversations were independently
confirmed absent. Unrelated service container IDs match the earlier H100 rollout.

- [Model API results](mimalloc/api/results.md): 264/264 valid requests, 240 measured.
  Paired mixed throughput difference +0.1024%, t 95% interval [-0.8923%, +1.1072%].
  No meaningful latency/throughput improvement qualified. Both arms passed the
  32 strict and six long quality cases; timed free-form output pairing varied.
- [Application results](mimalloc/application/results.md): 48/48 valid requests,
  40 measured. Pooled median visible/final latency was 2.840 to 2.740 seconds for
  chat and 2.529 to 2.415 seconds for RAG. Only two fixed-order pairs were run,
  and provider completion-token work differed. RAG used a failing graph path and
  final-selection scoring fallback. These descriptive reductions do not qualify
  an allocator speedup or healthy graph/scoring behavior.

Keep the system allocator selected. The bounded H100 prefill upgrade remains
enabled on development. PR #240 remains unmerged; its dedicated mimalloc CI
passed, while the existing broader repository checks remain failing. The replay
changed only main inference and web allocation, so it does not qualify the PR's
other service defaults. This is not a baseline-versus-prefill application A/B:
both allocator arms had prefill enabled, and no saturation throughput was tested.

The [reproduction archive](mimalloc/reproduction/README.md) preserves executed
coordinators, builders and recovery scripts. The aborted first application run
is [archived separately](mimalloc/application/attempt1/README.md) and excluded
from all paired results. Final request, cleanup and restoration evidence is
bound by the [application manifest](mimalloc/application/manifest.json).
