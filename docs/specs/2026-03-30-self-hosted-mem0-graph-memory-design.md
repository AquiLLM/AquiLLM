# Self-Hosted Mem0 Graph Memory Design

## Objective

Extend AquiLLM's existing Mem0 OSS integration with optional self-hosted graph memory support so episodic memory can include relationship context (entities and edges), while preserving current vector-only behavior by default.

## Current State

AquiLLM currently supports Mem0-backed episodic memory with:

- `llm` config
- `embedder` config
- `vector_store` config (Qdrant)

The current Mem0 config in `aquillm/lib/memory/mem0/client.py` does not include `graph_store`, so graph extraction and relation retrieval are not active.

## Scope

### In Scope

- Optional graph support for Mem0 OSS via environment-driven configuration.
- Initial self-hosted provider target: Memgraph (`bolt://...`).
- Graph toggles for add/search operations.
- Fail-open fallback to vector-only behavior when graph backend fails.
- Documentation updates for local and deployment usage.
- Automated tests for config wiring and fallback behavior.

### Out of Scope

- UI changes that render graph relations.
- New persistent Django models for graph data.
- Multi-provider production hardening for every graph backend in first pass.
- Historical memory backfill/migration to graph store.

## Requirements

### Functional Requirements

1. Preserve current runtime behavior when graph support is disabled.
2. Enable graph behavior only when explicitly configured.
3. Inject Mem0 `graph_store` config when graph mode is enabled and valid.
4. Allow per-request graph toggles for `add` and `search`.
5. On graph backend failures, continue serving vector-only memory (if fail-open enabled).

### Non-Functional Requirements

1. Backward compatibility for existing `.env` and compose configurations.
2. Minimal latency overhead when graph mode is disabled.
3. Clear structured logs for graph activation and fallback events.
4. Test coverage for graph-enabled and graph-disabled paths.

## Proposed Configuration Contract

Add new environment variables (all optional, safe defaults):

- `MEM0_GRAPH_ENABLED=0|1` (default `0`)
- `MEM0_GRAPH_PROVIDER=memgraph` (default empty)
- `MEM0_GRAPH_URL=bolt://memgraph:7687`
- `MEM0_GRAPH_USERNAME=memgraph`
- `MEM0_GRAPH_PASSWORD=...`
- `MEM0_GRAPH_DATABASE=...` (optional provider-specific)
- `MEM0_GRAPH_CUSTOM_PROMPT=...` (optional)
- `MEM0_GRAPH_THRESHOLD=0.75` (optional)
- `MEM0_GRAPH_FAIL_OPEN=1|0` (default `1`)
- `MEM0_GRAPH_ADD_ENABLED=1|0` (default `1`)
- `MEM0_GRAPH_SEARCH_ENABLED=1|0` (default `1`)

Notes:

- If `MEM0_GRAPH_ENABLED=0`, no `graph_store` block is passed to Mem0.
- If enabled but incomplete configuration is supplied, behavior depends on fail-open setting.

## Architecture and Data Flow

1. AquiLLM builds Mem0 OSS config in `get_mem0_oss()`.
2. When graph is enabled and valid, builder adds `graph_store`.
3. Write path (`memory.add`) can run with `enable_graph=True`.
4. Search path (`memory.search`) can run with `enable_graph=True`.
5. If graph operation fails and fail-open is enabled, retry with `enable_graph=False`.

This preserves existing vector retrieval semantics and adds graph context where available.

## Error Handling Strategy

### Graph Disabled

- No graph behavior; existing Mem0 behavior is unchanged.

### Graph Misconfigured

- `MEM0_GRAPH_FAIL_OPEN=1`: warn and continue vector-only.
- `MEM0_GRAPH_FAIL_OPEN=0`: raise configuration/init error.

### Graph Runtime Failure

- On add/search graph error, retry once with `enable_graph=False` when fail-open is enabled.
- Emit structured warning with enough context for operations.

## Testing Strategy

Add or extend tests in `aquillm/lib/memory/tests/`:

1. Config tests:
   - graph disabled omits `graph_store`
   - graph enabled includes valid `graph_store`
   - invalid graph config fail-open behavior
2. Operation tests:
   - add/search pass `enable_graph` correctly
   - graph failure triggers vector-only retry when fail-open enabled
3. Regression tests:
   - existing mem0/vector-only behavior remains unchanged

## Observability

Add logs for:

- graph enabled/disabled at Mem0 config initialization
- graph fallback events in add/search
- fail-open decision path taken

Avoid logging secrets or credentials.

## Security and Operational Notes

- Credentials should be sourced from environment/secret managers, never hardcoded.
- Memgraph endpoint should be restricted to internal network paths.
- Keep graph opt-in for staged rollout and safe rollback (`MEM0_GRAPH_ENABLED=0`).

## Rollout Plan

1. Land code with graph disabled by default.
2. Enable in local dev with Memgraph docker.
3. Enable in staging with fail-open on.
4. Observe error rate/latency before production enablement.

Rollback is configuration-only by disabling graph mode.

## Open Questions

1. Should relation payloads be surfaced into prompts in this phase, or deferred?
2. Do we need provider abstraction now or only Memgraph-first support?
3. Should strict mode (`MEM0_GRAPH_FAIL_OPEN=0`) be allowed in production or reserved for testing?

## References

- [Mem0 Graph Memory](https://docs.mem0.ai/open-source/features/graph-memory)
- [Mem0 Memgraph Docker section](https://docs.mem0.ai/open-source/features/graph-memory#memgraph-docker)
