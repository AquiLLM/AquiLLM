# RAG Citation Enforcement Design

**Date:** 2026-03-31  
**Status:** Implemented  
**Related:**

- [RAG Citation Enforcement Implementation Plan](../roadmap/plans/completed/2026-03-31-rag-citation-enforcement-implementation.md)
- [Code Style and Quality Guide](../documents/standards/code-style-guide.md)

## Problem

RAG responses were grounded in retrieved chunks, but citation behavior was not enforced. The model could answer without explicit source references, making it difficult to verify whether specific claims were retrieved or hallucinated.

## Goals

1. Require verifiable, chunk-level citations in post-retrieval answers.
2. Ensure cited references map only to chunks actually returned by tools.
3. Keep behavior fail-safe: if the model does not comply, produce a cited fallback.
4. Preserve existing tool and provider contracts.

## Non-goals

- Building a full claim-to-sentence semantic verifier.
- Changing retrieval ranking behavior.
- Replacing existing post-tool summary flow outside citation enforcement.

## Current-State Anchors

- `aquillm/lib/tools/search/vector_search.py`
- `aquillm/lib/llm/providers/complete_turn.py`
- `aquillm/lib/llm/providers/fallback_heuristics.py`

## Design Summary

Citation enforcement is applied on post-tool assistant synthesis turns:

1. Build a citation allow-list from recent tool payload rows.
2. Require answer citations in format: `[doc:<doc_id> chunk:<chunk_id>]`.
3. Reject citations not present in the allow-list.
4. If missing/invalid citations are detected, retry once with strict rewrite instructions.
5. If still invalid, return an extractive cited fallback synthesized from retrieved rows.

## Retrieval Payload Contract

Search result rows now include explicit citation fields:

- Verbose payloads: `citation`
- Compact payloads: `ref`

Both fields encode `[doc:<doc_id> chunk:<chunk_id>]` and are generated directly from returned chunk metadata.

## Enforcement Flow

### 1) Allow-list construction

Collect valid citation tokens from recent assistant-facing tool messages (chunk-search style list payloads).

### 2) Prompt guidance

Append strict citation requirements and allow-list references to the post-tool synthesis system prompt context.

### 3) Validation

Validate model output:

- must include at least one citation token when allow-list exists
- all citation tokens must be members of allow-list

### 4) Recovery

If validation fails:

- send one constrained retry prompt with allow-list and invalid-citation feedback
- if still non-compliant, synthesize a short extractive answer with inline allowed citations

## Safety and Compatibility

- Feature is env-gated (`RAG_ENFORCE_CHUNK_CITATIONS`, default enabled).
- No schema or API-breaking changes in tool wiring.
- Existing retrieval output remains backward compatible with additive fields.

## Testing Strategy

Added tests cover:

- citation allow-list extraction from verbose and compact rows
- detection of missing/invalid citations
- cited fallback synthesis
- post-tool retry behavior when first answer lacks citations
- payload-field preservation checks for new `citation`/`ref` fields

## Success Criteria

- Post-retrieval answers include verifiable chunk citations.
- Unverifiable citations are rejected.
- System still returns a safe, cited response when compliance fails.
- Changes remain within architecture and quality guardrails.

