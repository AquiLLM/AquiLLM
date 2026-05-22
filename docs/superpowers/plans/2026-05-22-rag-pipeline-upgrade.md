# RAG Pipeline Upgrade Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make document and collection questions fast, reliable, cited, image-aware, and strong enough for academic paper review without depending on fragile LLM tool-selection loops.

**Architecture:** Move obvious document RAG requests from model-orchestrated tool choice into a backend-orchestrated RAG path: classify intent, rewrite follow-up queries, retrieve with hybrid search, rerank, package evidence with citations and figure URLs, then run one synthesis call. Keep the existing agentic tool loop only for ambiguous or non-document work.

**Tech Stack:** Django/Channels chat backend, existing `Conversation`/`ToolMessage` LLM types, pgvector + trigram hybrid retrieval, local vLLM reranker with Cohere fallback, existing citation/image validators, pytest/Django tests, structlog telemetry.

---

## Research Baseline

Enterprise RAG systems are converging on predictable pipelines for normal document QA, not free-form agent loops for every query.

- Microsoft Azure RAG guidance recommends choosing search strategy up front, using vector + keyword/hybrid search, query translation/decomposition for complex questions, reranking, source/title/raw-content returns, and retrieval evaluation with Precision@K, Recall@K, and MRR. Source: https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-information-retrieval
- Google Vertex AI treats grounding as a lifecycle with retrieval providers, layout parsing, vector search, a grounding checker, and RAG reranking. Its RAG Engine exposes `top_k` plus ranker configuration and distinguishes low-latency rank APIs from slower LLM rerankers. Sources: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/grounding/ground-responses-using-rag and https://docs.cloud.google.com/vertex-ai/generative-ai/docs/rag-engine/retrieval-and-ranking
- AWS Bedrock Knowledge Bases emphasizes advanced parsing, document-aware chunking, query decomposition/reformulation, and rerankers for knowledge-base queries. Sources: https://aws.amazon.com/blogs/machine-learning/amazon-bedrock-knowledge-bases-now-supports-advanced-parsing-chunking-and-query-reformulation-giving-greater-control-of-accuracy-in-rag-based-applications/ and https://docs.aws.amazon.com/bedrock/latest/userguide/rerank-use.html
- LangChain documents the tradeoff directly: 2-step RAG is predictable and fast when retrieval is clearly required; agentic RAG is flexible but variable; hybrid RAG adds query enhancement, retrieval validation, and answer validation. Source: https://docs.langchain.com/oss/python/langchain/retrieval
- NVIDIA's RAG blueprint names the production query path as query rewriter, retriever, optional reranker, LLM generation, with per-stage latency metrics (`retrieval_time_ms`, `context_reranker_time_ms`, `llm_ttft_ms`, `rag_ttft_ms`). Source: https://docs.nvidia.com/rag/latest/query-to-answer-pipeline.html
- Anthropic's contextual retrieval work recommends combining contextual embeddings, BM25, rank fusion, reranking, and larger top-K context. It also notes that small corpora can sometimes be included whole with prompt caching, while larger corpora need scalable retrieval. Source: https://www.anthropic.com/engineering/contextual-retrieval
- Ragas-style evaluation covers retrieval and answer quality with Context Precision, Context Recall, Response Relevancy, Faithfulness, and multimodal metrics. Source: https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/

Implication for AquiLLM: selected-document questions like "tell me about the paper and show figures" should not spend one or two LLM calls deciding whether to call `vector_search`. The backend already knows this is RAG. Use the model for query rewriting and final synthesis, not for obvious tool routing.

## Current Code Findings

- `aquillm/apps/chat/consumers/chat_receive.py` detects document intent and attaches `document_tools` with `ToolChoice(type="any")`, but the LLM still gets first chance to produce the tool call.
- `aquillm/lib/llm/providers/complete_turn.py` now has `_deterministic_required_tool_call()` as a safety fallback, but it runs after hidden/interim model attempts. This fixed blank answers but preserved slow starts.
- `aquillm/lib/llm/providers/base.py` still has loop guards: `CHAT_MAX_FUNC_CALLS`, per-tool call limits, repeated signature breaks, no-progress breaks, and optional weighted tool budgets. We did not give those up.
- `aquillm/apps/documents/services/chunk_search.py` already does hybrid vector + trigram + exact-term retrieval, dedupes candidates, reranks, and logs per-stage latency.
- `aquillm/apps/chat/services/tool_wiring/documents.py` already returns image-aware `vector_search`, `whole_document`, `search_single_document`, and `more_context` results. The missing layer is orchestration and evidence packaging.

## Local Token-Efficiency Audit Inputs

Use the relevant findings from `docs/documents/2026-05-20-token-efficiency-context-window-audit.md` as local constraints for this work:

- F1: Gemini prompt trimming can leave dropped messages in `messages_pydantic`; fix provider-history alignment before relying on packer-driven drops in long RAG chats.
- F2: `CONTEXT_BUDGET_RETRIEVAL_TOKENS` and `CONTEXT_MAX_SNIPPETS_PER_DOC` exist but are not fully enforced; direct RAG evidence packing should make those knobs real.
- F3: fallback overflow trimming deletes by position; direct RAG should avoid unpaired user/assistant/tool drops and preserve active tool evidence.
- F4/F5: `whole_document` currently uses a coarse 150k-token gate and weak projected prompt estimate; whole-document mode must be budget-aware.
- F6: broad tool-schema attachment is recurring prompt overhead; the direct RAG router should send no tools for normal chat and skip tool schemas entirely for backend-orchestrated document turns.
- F7: compact vector-search payloads exist but are off in example config; use compact payloads where validation shows citations/images still work.
- F8: `more_context` can truncate away central evidence; adjacent-context expansion must preserve the central chunk first.
- F9/F10: profile memory and prompt-only skills can grow the system prompt; RAG synthesis should reserve context for evidence before optional memories/skills.
- F12/F14: inline image and multimodal rerank payloads need separate byte/count controls; figure retrieval should prefer URLs/captions unless image pixels are truly required.
- F13: search-result row truncation is character-based; evidence rows should use query-aware sentence snippets.
- F15: citation enforcement is quality-positive but retries can bloat prompts; allowlists and retry prompts should stay compact.
- F16/F17: raw tool results and frontend token gauges are useful but not equivalent to projected next-turn prompt pressure; add projected prompt/evidence metrics.

## Desired Behavior

- A selected-collection paper question starts retrieval immediately, without waiting for an LLM tool-call attempt.
- Follow-up questions rewrite into a standalone query using prior conversation context before retrieval.
- "Show figures/images" retrieves text evidence plus relevant figure rows and preserves markdown image URLs.
- Single small paper questions can use whole-document context when it is cheaper and better than chunk search.
- Multi-document or broad questions use hybrid search, reranking, and diversified evidence across documents.
- Final answers cite only retrieved chunks/documents and include a source block when citations are enabled.
- The UI never receives the generic clean-response fallback when tool evidence exists; it gets either a synthesized answer or a transparent evidence-based fallback.
- Latency is measurable by stage: intent, rewrite, retrieval, rerank, evidence packing, synthesis, validation.

## Phase 0: Token-Budget Correctness Prerequisites

- [ ] Fix provider history alignment before enabling aggressive packing.
  - Update `aquillm/lib/llm/utils/prompt_budget.py` so trimmed dict messages can be synced back to pydantic messages by kept indices or stable message identity.
  - Update `aquillm/lib/llm/providers/gemini.py` so dropped dict messages are also dropped from `messages_pydantic`.
  - Preserve tool-call/function-response pairs when trimming.

- [ ] Replace positional overflow deletion in the RAG path.
  - Ensure direct RAG prompt packing preserves system prompt, latest user turn, active tool evidence, retained citation allowlist, and final-answer reserve.
  - Add trim reports with reason codes instead of silent oldest-message deletion.

- [ ] Add projected prompt estimation.
  - Add or reuse a provider-agnostic helper for `system + history + proposed evidence + optional tool schemas + completion reserve`.
  - Use it for direct RAG whole-document decisions and observability.

- [ ] Tests.
  - Gemini long-history trimming sends exactly the kept pydantic messages.
  - Tool-call/tool-result pairs remain paired under trimming.
  - Projected prompt usage differs from, and does not overwrite, last-turn token usage.

Commit:

```bash
git add aquillm/lib/llm/utils/prompt_budget.py aquillm/lib/llm/providers/gemini.py aquillm/lib/llm/tests/test_prompt_budget.py aquillm/lib/llm/tests/test_gemini_prompt_budget.py
git commit -m "fix(context): align provider history after prompt packing"
```

## Phase 1: Route Obvious RAG Outside The Tool Loop

- [ ] Add `aquillm/apps/chat/services/rag_intent.py`.
  - Move the document/figure/retry regex logic out of `chat_receive.py`.
  - Return a small dataclass:
    - `requires_rag: bool`
    - `wants_figures: bool`
    - `wants_whole_document: bool`
    - `is_retry: bool`
    - `reason: str`
  - Keep the current regex behavior intact for existing tests.

- [ ] Update `aquillm/apps/chat/consumers/chat_receive.py`.
  - Replace local `_looks_like_*` helpers with `rag_intent.classify_chat_message()`.
  - Keep existing `_configure_append_tools()` behavior as the fallback path.
  - Do not attach any tool schemas for ordinary chat.
  - For direct RAG turns, avoid attaching document tool schemas because retrieval will be backend-orchestrated.
  - Set a private runtime marker on the latest `UserMessage`, for example `metadata["rag_intent"]` if message metadata exists, or a sidecar on the consumer if not.

- [ ] Add tests in `aquillm/apps/chat/tests/test_document_search_intent.py`.
  - Existing "paper + figures" prompts classify as direct RAG.
  - Follow-up "try again" inherits prior RAG intent.
  - Plain physics/general chat has no tools.
  - Astronomy/FITS local tool requests still use normal tool routing.
  - Direct document RAG does not serialize document tool schemas into the first provider call.

Commit:

```bash
git add aquillm/apps/chat/consumers/chat_receive.py aquillm/apps/chat/services/rag_intent.py aquillm/apps/chat/tests/test_document_search_intent.py
git commit -m "feat(rag): classify direct document retrieval turns"
```

## Phase 2: Add A Backend RAG Orchestrator

- [ ] Add `aquillm/apps/chat/services/rag_pipeline.py`.
  - Public entry point:
    - `run_direct_rag_turn(conversation, user, col_ref, llm_if, *, stream_func=None) -> Conversation | None`
  - It should return `None` for non-RAG turns so existing `run_llm_spin()` handles them.
  - It should create assistant-visible `ToolMessage` evidence using existing tool result serialization conventions, then call the existing post-tool synthesis path.

- [ ] Start with a conservative retrieval policy.
  - If no collections are selected, do not call retrieval; let normal chat answer or ask for collection selection.
  - If exactly one accessible document is selected and full text fits the projected prompt budget, call the same logic as `whole_document`.
  - If the projected prompt budget is tight, use section/search retrieval instead of opening the whole document.
  - Otherwise call the same logic as `vector_search` immediately with `top_k=10`.
  - If the user requests figures, preserve figure/image rows and set figure-aware synthesis instructions.

- [ ] Avoid double retrieval.
  - Direct RAG turns should not attach `document_tools` to the latest user message for the first synthesis call.
  - Existing tools can still be re-enabled only during a later validation retry if retrieved evidence is thin.

- [ ] Integrate in `aquillm/apps/chat/consumers/chat_receive.py` before `run_llm_spin()`.
  - After memory augmentation, call `run_direct_rag_turn()` when intent says direct RAG.
  - If it handles the turn, save/send conversation and skip `run_llm_spin()`.
  - If it returns `None`, use current behavior.

- [ ] Tests.
  - New `aquillm/apps/chat/tests/test_direct_rag_pipeline.py`.
  - Assert direct RAG calls retrieval without first calling `llm_if.get_message()` for tool selection.
  - Assert no generic fallback is emitted when retrieval returns evidence.
  - Assert normal chat still bypasses document retrieval.
  - Assert whole-document mode declines full text when projected prompt budget is insufficient.

Commit:

```bash
git add aquillm/apps/chat/services/rag_pipeline.py aquillm/apps/chat/consumers/chat_receive.py aquillm/apps/chat/tests/test_direct_rag_pipeline.py
git commit -m "feat(rag): bypass tool selection for document questions"
```

## Phase 3: Query Rewriting For Follow-Ups

- [ ] Add `aquillm/apps/chat/services/rag_query.py`.
  - Function: `build_retrieval_query(conversation, latest_user_text, *, llm_if=None) -> str`.
  - Start deterministic: combine latest user turn with the most recent assistant answer title/source context.
  - Add optional LLM rewrite behind `RAG_QUERY_REWRITE_ENABLED=1`.
  - The rewrite prompt must output only the standalone search query, not an answer.

- [ ] Use rewriting only where it helps.
  - Follow-ups with pronouns ("it", "they", "this paper", "those figures") use rewrite.
  - First-turn explicit paper questions can use the raw user text.
  - Retry requests reuse the previous failed RAG query.

- [ ] Tests.
  - `can you explain the math in figure 2` after a paper answer becomes a standalone query containing the paper title or prior source.
  - `try again` reuses prior RAG query instead of searching for "try again".

Commit:

```bash
git add aquillm/apps/chat/services/rag_query.py aquillm/apps/chat/services/rag_pipeline.py aquillm/apps/chat/tests/test_direct_rag_pipeline.py
git commit -m "feat(rag): rewrite follow-up retrieval queries"
```

## Phase 4: Evidence Packet And Figure Policy

- [ ] Add `aquillm/apps/chat/services/rag_evidence.py`.
  - Normalize results from `vector_search`, `whole_document`, `search_single_document`, and `more_context`.
  - Produce a compact evidence packet:
    - query
    - search scope
    - chunks with `doc_id`, `chunk_id`, title, snippet, citation token
    - figures with title, caption, `image_url`, parent document
    - source list
    - retrieval diagnostics

- [ ] Add diversification.
  - Group candidate chunks by document.
  - Keep the top chunks, but prevent one document from consuming the entire context when multiple documents are selected.
  - For single-paper requests, prioritize coherence over diversity.
  - Enforce `CONTEXT_BUDGET_RETRIEVAL_TOKENS` and `CONTEXT_MAX_SNIPPETS_PER_DOC` against retained evidence rows, not serialized JSON after the fact.

- [ ] Add figure retrieval policy.
  - If `wants_figures`, include up to `RAG_MAX_FIGURES_PER_TURN` relevant figures.
  - Prefer figures whose captions or OCR text match the query.
  - If text chunks come from a parent paper that has related figures, include the most relevant related figures even if the vector search did not rank image chunks high enough.
  - Prefer figure URLs and captions for synthesis; reserve inline image payloads for visual reasoning requests.

- [ ] Add query-aware snippets.
  - Extract sentence windows around query terms, entities, figure labels, equation labels, and model names.
  - Preserve citation tokens and source titles even under tight row budgets.
  - Avoid cutting snippets mid-sentence where possible.

- [ ] Tests.
  - Figure/image requests produce an evidence packet with markdown-safe `/aquillm/document_image/<uuid>/` URLs.
  - Chunk citations survive compaction.
  - Evidence packet never includes raw base64 image payloads in model text.
  - Per-document snippet caps and retrieval-token budgets drop duplicate rows while retaining citation-bearing evidence.

Commit:

```bash
git add aquillm/apps/chat/services/rag_evidence.py aquillm/apps/chat/services/rag_pipeline.py aquillm/apps/chat/tests/test_direct_rag_pipeline.py
git commit -m "feat(rag): package cited text and figures for synthesis"
```

## Phase 5: Synthesis Prompt And Validation

- [ ] Add `aquillm/apps/chat/services/rag_synthesis.py`.
  - Build the final synthesis prompt from the evidence packet.
  - Use a structured prompt for academic paper questions:
    - thesis/contribution
    - methods/data
    - math or mechanisms
    - figures
    - limitations
    - source-backed citations
  - Keep final `max_tokens` uncapped by tool-step settings.

- [ ] Reuse existing validators.
  - `aquillm/lib/llm/providers/rag_citations.py` for allowed citation enforcement.
  - `aquillm/lib/llm/providers/image_context.py` for image markdown recovery.
  - `aquillm/lib/llm/providers/visibility.py` for hidden/interim text filtering.

- [ ] Add one validation retry.
  - If the answer lacks citations while citations are required, retry synthesis with the allow-list.
  - If the user requested figures and no image URL appears but evidence contains images, append or retry with figure instructions.
  - If synthesis still fails, return an extractive evidence answer instead of the generic clean-response fallback.
  - Keep citation retry prompts compact by grouping refs by document and only listing retained evidence rows.

- [ ] Tests.
  - Blank synthesis over evidence returns an evidence fallback, not generic failure.
  - Invalid citations are rejected and retried.
  - Figure requests include at least one markdown image when evidence contains figures.
  - Citation retry does not re-add dropped retrieval rows or obsolete allowlist entries.

Commit:

```bash
git add aquillm/apps/chat/services/rag_synthesis.py aquillm/apps/chat/services/rag_pipeline.py aquillm/apps/chat/tests/test_direct_rag_pipeline.py
git commit -m "feat(rag): validate cited figure-aware synthesis"
```

## Phase 6: Latency Observability And Controls

- [ ] Add per-stage timing logs.
  - `rag_intent_ms`
  - `rag_query_rewrite_ms`
  - `rag_retrieval_ms`
  - `rag_rerank_ms` if available from retrieval logs
  - `rag_evidence_pack_ms`
  - `rag_synthesis_ttft_ms` when stream timing is available
  - `rag_total_ms`
  - `estimated_next_prompt_tokens`
  - `retrieval_evidence_tokens`
  - `tool_schema_tokens`

- [ ] Make slow knobs explicit in `.env.example`.
  - `RAG_DIRECT_ENABLED=1`
  - `RAG_QUERY_REWRITE_ENABLED=0`
  - `RAG_DIRECT_TOP_K=10`
  - `RAG_MAX_FIGURES_PER_TURN=3`
  - `RAG_DIRECT_WHOLE_DOC_TOKEN_LIMIT=80000`
  - `TOOL_SEARCH_COMPACT_PAYLOAD=1` for staging once citation/image tests pass.
  - `LM_LINGUA2_ENABLED=0` unless explicitly validating compression quality.
  - Keep `APP_EMBED_DIMS=1024` because the Mem0/vector compatibility truncation is intentional.

- [ ] Add timeout behavior.
  - Retrieval timeout should produce a useful "I could not retrieve the selected documents" message.
  - Rerank timeout should fall back to hybrid order, not fail the turn.
  - Synthesis timeout should preserve retrieved evidence in the conversation for retry.

- [ ] Tests.
  - Rerank failure falls back to candidate order.
  - Direct RAG logs stage names without logging full user document text.

Commit:

```bash
git add .env.example aquillm/apps/chat/services/rag_pipeline.py aquillm/apps/chat/tests/test_direct_rag_pipeline.py
git commit -m "chore(rag): add direct rag controls and telemetry"
```

## Phase 7: Cache Acceleration For Embeddings, Reranking, And Generation

- [ ] Make cache layers explicit.
  - Use LMCache for repeated LLM generation prefixes: system prompts, stable RAG synthesis instructions, tool/evidence schemas, and static document context in whole-document mode.
  - Use Django cache/Redis via `aquillm/apps/documents/services/rag_cache.py` for retrieval work: query embeddings, rerank endpoint capability, rerank results, document access refs, document lookup refs, and image data URLs.
  - Do not treat LMCache as an embedding/reranker cache; those are separate model calls and should stay in `rag_cache`.

- [ ] Improve embedding cache hit rate.
  - Normalize retrieval queries before `get_cached_query_embedding()`:
    - trim/collapse whitespace
    - lowercase only where retrieval semantics allow it
    - preserve exact technical terms separately for exact-term search
  - Include embedding model, embedding endpoint, input type, and fitted `APP_EMBED_DIMS` in the query embedding model signature.
  - Increase `RAG_QUERY_EMBED_TTL_SECONDS` for stable local collections once invalidation is reliable.
  - Add an optional direct-RAG prewarm step: when a collection is selected, warm document-access refs; when a query is rewritten, warm the rewritten query embedding before vector search.

- [ ] Improve reranker cache hit rate and latency.
  - Keep the existing rerank-result key shape of `(query_signature, candidate_ids, top_k, model)` because candidate order matters.
  - Add query normalization before `query_signature_for_rerank()` so follow-up retries and whitespace-only variants hit cache.
  - Cache failed reranker capability probes longer, so vLLM endpoint discovery does not retry unsupported `/rerank`, `/v2/rerank`, `/score` shapes on every request.
  - Add `RAG_RERANK_BYPASS_CANDIDATE_COUNT`: skip reranker when candidate count is already `<= top_k`.
  - Add `RAG_RERANK_MIN_CANDIDATES`: use fallback order for tiny candidate sets where reranking costs more than it helps.
  - Add optional async/background rerank prefetch for likely follow-up modes, such as figure/method/math queries after an academic summary.

- [ ] Use direct RAG to increase cache locality.
  - Deterministic query rewriting should produce stable queries for common follow-ups.
  - Evidence packet generation should preserve candidate IDs in stable order so rerank-result caching is useful.
  - Retry requests must reuse the previous direct-RAG query and evidence packet when possible instead of recomputing retrieval.

- [ ] Use LMCache where it fits.
  - Keep synthesis prompt scaffolding stable and early in the prompt, with variable evidence appended after stable instructions.
  - For whole-document mode, place static document text before the latest user question where the provider/runtime can reuse prefix cache.
  - Add config documentation for `LMCACHE_ENABLED` and `LMCACHE_EXTRA_ARGS` in the RAG rollout notes.
  - Measure LMCache benefit separately from retrieval cache benefit with TTFT and total generation timing.
  - Keep profile memory and prompt-only skill bodies after direct evidence in the budget priority order.

- [ ] Add cache observability.
  - Emit per-stage hit/miss metrics:
    - `rag_cache.query_embed`
    - `rag_cache.rerank_result`
    - `rag_cache.rerank_capability`
    - `rag_cache.doc_access`
    - `rag_cache.document_lookup`
    - `rag_cache.image_data_url`
  - Add request-level summary fields:
    - `query_embedding_cache_hit`
    - `rerank_cache_hit`
    - `rerank_capability_cache_hit`
    - `lmcache_enabled`
    - `direct_rag_reused_evidence`

- [ ] Tests.
  - `aquillm/apps/documents/tests/test_chunk_search_query_cache.py` verifies normalized query variants reuse the same embedding.
  - `aquillm/apps/documents/tests/test_rerank_http_cache.py` verifies normalized query variants and repeated candidate sets skip HTTP.
  - `aquillm/apps/chat/tests/test_direct_rag_pipeline.py` verifies "try again" reuses prior query/evidence instead of reranking again.
  - `aquillm/tests/integration/test_vllm_lmcache_plumbing.py` verifies LMCache env wiring remains intact.
  - Rerank image payload tests verify caption-only rerank is used unless the query is figure/plot-specific.

Commit:

```bash
git add .env.example aquillm/apps/documents/services/rag_cache.py aquillm/apps/documents/services/chunk_search.py aquillm/apps/documents/services/chunk_rerank_local_vllm.py aquillm/apps/chat/services/rag_pipeline.py aquillm/apps/documents/tests/test_chunk_search_query_cache.py aquillm/apps/documents/tests/test_rerank_http_cache.py aquillm/apps/chat/tests/test_direct_rag_pipeline.py aquillm/tests/integration/test_vllm_lmcache_plumbing.py
git commit -m "perf(rag): cache embeddings reranks and stable generation prefixes"
```

## Phase 8: Retrieval Quality Improvements

- [ ] Add contextual chunk metadata during ingestion.
  - Add optional `TextChunk.metadata["contextual_header"]`.
  - Generate from document title, section heading, page/figure metadata, and nearby headings.
  - Search/rerank payloads should include contextual header plus chunk content.

- [ ] Improve exact-term and title matching.
  - Add title/abstract search for paper names, acronyms, figure labels, table labels, equations, and model names.
  - Preserve exact matches as candidates before reranking.

- [ ] Add retrieval modes.
  - `summary`: broad top-K, diversified, figure-aware.
  - `specific`: narrower top-K, exact-term boost.
  - `figure`: image/caption/OCR emphasis.
  - `math`: more adjacent chunks around equations and definitions.
  - `context`: central chunk first, then alternating adjacent chunks outward.

- [ ] Tests.
  - Acronym/model-name queries keep exact matches in candidate set.
  - Figure queries rank image chunks or related figures into evidence.
  - Math queries include adjacent chunks.
  - `more_context`-style expansion always retains the central chunk under small `TOOL_CHUNK_CHAR_LIMIT`.

Commit:

```bash
git add aquillm/apps/documents/services aquillm/apps/documents/tests aquillm/apps/chat/services/rag_pipeline.py
git commit -m "feat(rag): improve contextual retrieval quality"
```

## Phase 9: RAG Evaluation Harness

- [ ] Add `aquillm/apps/chat/evals/rag_cases.yaml`.
  - Include representative local cases:
    - paper summary with figures
    - follow-up explanation
    - math-heavy question
    - exact figure request
    - multi-document comparison
    - no selected collections
    - irrelevant question

- [ ] Add `aquillm/apps/chat/evals/run_rag_eval.py`.
  - Run retrieval-only metrics when expected doc/chunk IDs are known:
    - Recall@K
    - Precision@K
    - MRR
  - Run answer checks:
    - has citations
    - no invalid citations
    - answer contains image markdown when expected
    - no generic fallback
    - latency budgets

- [ ] Optional later integration.
  - Add Ragas or a lightweight LLM-judge job for faithfulness and response relevance.
  - Keep this outside the main unit-test path until deterministic enough.

Commit:

```bash
git add aquillm/apps/chat/evals aquillm/apps/chat/tests
git commit -m "test(rag): add retrieval and answer quality eval harness"
```

## Verification Commands

Run focused tests after each phase:

```bash
pytest aquillm/apps/chat/tests/test_document_search_intent.py
pytest aquillm/apps/chat/tests/test_direct_rag_pipeline.py
pytest aquillm/lib/llm/tests/test_rag_citations.py
pytest aquillm/apps/chat/tests/test_tool_result_images.py
pytest aquillm/apps/documents/tests/test_chunk_search_candidate_tuning.py
pytest aquillm/apps/documents/tests/test_rerank_http_cache.py
```

Run the broader relevant suite before final merge:

```bash
pytest aquillm/apps/chat/tests aquillm/apps/documents/tests aquillm/lib/llm/tests
pytest aquillm/tests/integration/test_cache_settings_flags.py
git diff --check
```

Manual verification:

- Ask: "Hi aquilm tell me about the paper and show me a figure or two, be thorough you are an academic system and I am a scientist."
- Confirm first answer is not generic fallback.
- Confirm first retrieval starts without a preceding LLM tool-selection retry.
- Confirm figures render with `/aquillm/document_image/<uuid>/`.
- Ask: "can you explain the math behind figure 2 in more detail?"
- Confirm follow-up rewrites and retrieves from the same paper, cites sources, and does not search for only "figure 2" or "try again."

## Rollout Notes

- Ship behind `RAG_DIRECT_ENABLED=1`, default on in local/dev after tests pass.
- Keep the model-driven tool loop intact for astronomy tools, uploaded-file processing, and ambiguous multi-tool requests.
- Keep thinking enabled for Qwen; the direct RAG path removes the fragile first tool-selection step instead of disabling reasoning.
- Keep `APP_EMBED_DIMS=1024`; the 2048-to-1024 truncation warnings are expected compatibility behavior until the vector schema is migrated.
- If direct RAG causes regressions, set `RAG_DIRECT_ENABLED=0` and the existing tool loop remains available.
