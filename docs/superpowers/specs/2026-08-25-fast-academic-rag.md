# Fast Academic RAG Design

## Goal

Reduce selected-collection chat latency without reducing the evidence, citations,
reranking, knowledge-graph contribution, or reasoning available to the final
academic answer.

## Quality boundary

- Tool-selection and retrieval-routing turns may disable model thinking.
- Final evidence synthesis must retain the deployment's configured thinking mode.
- Every retrieval query must continue through the existing vector, trigram,
  exact-match, optional knowledge-graph, and reranker pipeline.
- Multi-part academic questions may issue up to three deterministic retrieval
  queries. These searches run concurrently and are merged with reciprocal-rank
  fusion, citation deduplication, and the existing per-document evidence cap.
- The final synthesis receives real cited chunks, not summaries invented by the
  retrieval layer.
- The 131,072-token model context limit and clickable chunk citations remain
  unchanged.

## Runtime behavior

1. Classify an obvious selected-collection question into direct RAG.
2. Build the original retrieval query and, only for genuinely multi-part text,
   up to two question/sentence subqueries without an LLM call.
3. Execute the existing `vector_search` tool for each query concurrently. Each
   call therefore retains local reranking and any enabled graph expansion.
4. Merge results by stable citation identity with reciprocal-rank fusion and cap
   the merged evidence at `RAG_DIRECT_TOP_K`.
5. Build the citation-preserving evidence packet and perform exactly one final
   synthesis with normal thinking enabled.
6. If direct RAG is not applicable or fails, use the existing tool loop. Initial
   mechanical tool routing passes `thinking_budget=0`; post-tool synthesis does
   not override thinking.

## Knowledge-graph observability

Graph diagnostics remain private transport data and must never be included in
model-visible tool text or the browser API. Direct-RAG logs may emit only the
already validated graph status, timings, counts, and SHA-256 signatures. This
allows deployment verification to distinguish graph `hit`, `miss`, `timeout`,
and `error` without exposing graph labels, paths, queries, document identifiers,
or user content.

## Deployment constraints

- Workflow is local commit, push `origin/development`, remote pull, then deploy.
- Never add, commit, print, or push the remote server `.env`.
- Remote environment changes are made only on the server after inspecting the
  current values and graph readiness.
- Keep `OPENAI_COMPAT_ENABLE_THINKING=1` for final answers.
- Enable direct RAG remotely only after tests pass.
- Enable graph traversal flags only when the projection/gateway health checks and
  selected-collection readiness checks pass.

## Acceptance criteria

- A multi-part collection question performs no LLM tool-selection generation.
- Each retrieval query invokes the existing reranker-backed search path.
- Distinct cited chunks from multiple searches survive merge and evidence
  packing; duplicate citations appear once.
- Final synthesis is thinking-enabled and receives the retained cited evidence.
- Fallback tool selection is non-thinking while post-tool synthesis remains
  thinking-enabled.
- Logs prove whether the graph contributed candidates without leaking content.
- Citation source expansion sends a valid CSRF token.
- The direct-RAG test suite, LLM provider tests, retrieval tests, React tests, and
  an end-to-end remote UI request pass.
