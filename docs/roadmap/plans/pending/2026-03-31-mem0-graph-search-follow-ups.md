# Mem0 Graph Search Follow-Ups

## Current State

- Durable profile fact promotion is working and `profile_facts` is now consistently non-zero in development.
- Episodic retrieval is working end-to-end for users because local fallback retrieval still returns memories when Mem0 graph search is slow.
- Mem0 graph search now uses the Memgraph compatibility shim with:
  - relation cleanup and normalization
  - seeded candidate fetches
  - NumPy scoring
  - graph-search timing instrumentation
- On deployed `mem0ai==1.0.7`, `Memory.search` and `AsyncMemory.search` do not expose an `enable_graph` kwarg even though current docs show per-request graph toggles.
- Compatibility was updated so AquiLLM now uses:
  - a graph-configured Mem0 client for graph-first searches
  - a vector-only Mem0 client without `graph_store` for fallback
- This removed the incorrect retry loop around unsupported `enable_graph` kwargs.

## Verified Behavior

- Remote runtime version: `mem0ai==1.0.7`
- Installed source for `AsyncMemory.search` uses `self.enable_graph` at the client level.
- Installed source for `Memory.search` also lacks an `enable_graph` parameter.
- Current dev logs show:
  - graph client search timing out at `3.0s`
  - vector-only client fallback succeeding quickly enough for `episodic_memories=3`

## Remaining Problem

Graph-enabled Mem0 retrieval is still too slow for the configured `MEM0_GRAPH_SEARCH_TIMEOUT_SECONDS=3` budget.

This is now a performance problem, not a correctness bug:

- graph-first attempt times out cleanly
- vector-only fallback succeeds
- user-visible memory injection still works

## Next Todo Items

1. Decide whether graph retrieval should stay enabled in user-facing chat or be limited to graph writes only.
2. If graph retrieval stays enabled, measure where graph time is spent on the deployed stack:
   - Mem0 vector search orchestration
   - graph relation expansion
   - reranking
   - embedding latency
3. Check whether repeated identical websocket requests are causing multiple graph attempts for the same user query.
4. Consider caching or suppressing repeated graph attempts for the same normalized query within a short TTL.
5. Evaluate whether `MEM0_GRAPH_SEARCH_TIMEOUT_SECONDS=3` and `MEM0_GRAPH_SEARCH_CANDIDATE_LIMIT=128` are still the right defaults after the client split.
6. Decide whether to disable graph search in production retrieval while keeping graph writes on.
7. Revisit Mem0 package upgrades only if a newer release clearly documents and ships per-request graph toggles in the installed Python signatures.

## Useful Runtime Checks

```bash
docker compose -f deploy/compose/development.yml exec web python -c "import mem0; print(mem0.__version__)"
docker compose -f deploy/compose/development.yml exec web python -c "import inspect; from mem0 import AsyncMemory; print(inspect.signature(AsyncMemory.search))"
docker compose -f deploy/compose/development.yml exec web python -c "import inspect; from mem0 import Memory; print(inspect.signature(Memory.search))"
```

## Recommended Short-Term Stance

Leave the current compatibility fix in place and treat graph retrieval as optional optimization work. The retrieval path is already safe because vector-only fallback preserves user-facing memory behavior.
