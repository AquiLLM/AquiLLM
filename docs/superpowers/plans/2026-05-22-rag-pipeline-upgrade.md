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

## Desired Behavior

- A selected-collection paper question starts retrieval immediately, without waiting for an LLM tool-call attempt.
- Follow-up questions rewrite into a standalone query using prior conversation context before retrieval.
- "Show figures/images" retrieves text evidence plus relevant figure rows and preserves markdown image URLs.
- Single small paper questions can use whole-document context when it is cheaper and better than chunk search.
- Multi-document or broad questions use hybrid search, reranking, and diversified evidence across documents.
- Final answers cite only retrieved chunks/documents and include a source block when citations are enabled.
- The UI never receives the generic clean-response fallback when tool evidence exists; it gets either a synthesized answer or a transparent evidence-based fallback.
- Latency is measurable by stage: intent, rewrite, retrieval, rerank, evidence packing, synthesis, validation.

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
  - Set a private runtime marker on the latest `UserMessage`, for example `metadata["rag_intent"]` if message metadata exists, or a sidecar on the consumer if not.

- [ ] Add tests in `aquillm/apps/chat/tests/test_document_search_intent.py`.
  - Existing "paper + figures" prompts classify as direct RAG.
  - Follow-up "try again" inherits prior RAG intent.
  - Plain physics/general chat has no tools.
  - Astronomy/FITS local tool requests still use normal tool routing.

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
  - If exactly one accessible document is selected and full text is under a configurable token threshold, call the same logic as `whole_document`.
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

- [ ] Add figure retrieval policy.
  - If `wants_figures`, include up to `RAG_MAX_FIGURES_PER_TURN` relevant figures.
  - Prefer figures whose captions or OCR text match the query.
  - If text chunks come from a parent paper that has related figures, include the most relevant related figures even if the vector search did not rank image chunks high enough.

- [ ] Tests.
  - Figure/image requests produce an evidence packet with markdown-safe `/aquillm/document_image/<uuid>/` URLs.
  - Chunk citations survive compaction.
  - Evidence packet never includes raw base64 image payloads in model text.

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

- [ ] Tests.
  - Blank synthesis over evidence returns an evidence fallback, not generic failure.
  - Invalid citations are rejected and retried.
  - Figure requests include at least one markdown image when evidence contains figures.

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

- [ ] Make slow knobs explicit in `.env.example`.
  - `RAG_DIRECT_ENABLED=1`
  - `RAG_QUERY_REWRITE_ENABLED=0`
  - `RAG_DIRECT_TOP_K=10`
  - `RAG_MAX_FIGURES_PER_TURN=3`
  - `RAG_DIRECT_WHOLE_DOC_TOKEN_LIMIT=80000`
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

## Phase 7: Retrieval Quality Improvements

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

- [ ] Tests.
  - Acronym/model-name queries keep exact matches in candidate set.
  - Figure queries rank image chunks or related figures into evidence.
  - Math queries include adjacent chunks.

Commit:

```bash
git add aquillm/apps/documents/services aquillm/apps/documents/tests aquillm/apps/chat/services/rag_pipeline.py
git commit -m "feat(rag): improve contextual retrieval quality"
```

## Phase 8: RAG Evaluation Harness

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
