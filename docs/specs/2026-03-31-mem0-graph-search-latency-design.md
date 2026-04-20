# Mem0 Graph Search Latency Design

## Goal

Reduce Mem0 graph-search latency in development and production-facing request paths without removing graph memory support.

## Approved Approach

Use three coordinated changes:

1. Replace Python-loop cosine similarity with NumPy-based batched similarity scoring in the Memgraph compatibility shim.
2. Cap the number of graph candidates fetched and scored per user so search work is bounded.
3. Shorten the graph-enabled async search attempt budget so requests fall back to vector-only faster when graph search is slow.

## Why This Approach

The current hot path spends time in multiple places:

- fetching every candidate node with embeddings for a user
- scoring each candidate in Python
- querying related edges after nearest-node selection

NumPy helps the scoring stage, but only materially if candidate volume is also bounded. A smaller graph-only timeout improves user-facing latency even when the graph path remains slower than vector-only retrieval.

## Design Details

### Memgraph Compatibility Shim

In `aquillm/lib/memory/mem0/memgraph_compat.py`:

- add a configurable graph candidate cap with a balanced default
- fetch at most that many candidate nodes from Memgraph
- convert candidate embeddings and query embeddings to NumPy arrays
- compute cosine similarity in batch instead of per-candidate Python math
- keep a safe fallback path for malformed embeddings or shape mismatches

### Async Search Budget

In `aquillm/lib/memory/mem0/operations.py`:

- preserve off-main-loop execution using `asyncio.to_thread`
- give graph-enabled async search attempts a smaller timeout budget than the total Mem0 timeout
- continue fail-open fallback to vector-only search

This does not make graph search free, but it reduces worst-case user-visible wait time.

## Testing

Add focused regression coverage for:

- candidate fetch queries respecting the graph candidate cap
- NumPy similarity ranking preserving expected nearest-node behavior
- async graph search falling back sooner than the full timeout budget

## Risks

- A candidate cap can reduce recall if set too low.
- NumPy conversion can fail on malformed embeddings, so the shim must degrade safely.
- A shorter graph timeout may reduce graph-hit rate, but that tradeoff is acceptable for end-user latency.
