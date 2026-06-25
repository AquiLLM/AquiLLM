# Mem0 async memory pipeline

**Status:** Implemented  
**Date:** 2026-03-30  

## Problem

Chat WebSocket handlers call `augment_conversation_with_memory` inside `database_sync_to_async`, which runs the whole augmentation step in a thread pool. Mem0 OSS and cloud calls use the synchronous `Memory` / `MemoryClient` APIs, so I/O blocks that worker thread. Profile facts and episodic retrieval run sequentially even though they are independent.

## Goals

1. Use Mem0’s async APIs ([Async Memory](https://docs.mem0.ai/open-source/features/async-memory), `AsyncMemoryClient` for cloud) so memory search does not rely on sync SDK calls in a thread.
2. Overlap **user profile facts** (Django ORM) with **episodic retrieval** (Mem0 or pgvector) where safe, reducing wall-clock time before `llm_if.spin()`.
3. Preserve existing behavior: same scoping (`user_id`, exclusions), fallbacks (OSS → cloud → local pgvector), and env-driven configuration.
4. Keep synchronous entry points (`get_episodic_memories`, `augment_conversation_with_memory`) for code paths that are still sync-only (e.g. Celery tasks), implemented by delegating to shared helpers.

## Non-goals

- Async Mem0 **writes** in this change (e.g. `create_episodic_memories_for_conversation` / `add_mem0_raw_facts`) — still synchronous; can be a follow-up if workers move to async contexts.
- Replacing `database_sync_to_async` for all Django access in augmentation — profile facts remain ORM-backed via `database_sync_to_async`; only Mem0 search moves to native async.

## Design

### Shared configuration

`aquillm/lib/memory/mem0/client.py` builds one OSS config dict (`_build_mem0_oss_config_dict()`) used by:

- `Memory.from_config` (existing sync singleton `get_mem0_oss`)
- `AsyncMemory.from_config` (new async singleton `get_mem0_oss_async`)

Cloud: `AsyncMemoryClient` singleton `get_mem0_client_async`, mirroring `get_mem0_client`.

Concurrency: `asyncio.Lock` guards one-time init per process for async singletons (same pattern as sync globals).

### Operations

New async functions in `aquillm/lib/memory/mem0/operations.py`:

- `search_mem0_via_oss_async` — `await mem0.search(...)` with the same call-shape attempts as the sync helper.
- `search_mem0_episodic_memories_async` — OSS first, then `AsyncMemoryClient.search`, then empty (caller does pgvector).
- Optional: `asyncio.wait_for(..., timeout=MEM0_TIMEOUT_SECONDS)` around Mem0 calls for bounded latency.

### Django layer

- `_get_episodic_memories_pgvector` — extracted from `get_episodic_memories` (local embedding + `EpisodicMemory` query only).
- `get_episodic_memories` / `get_episodic_memories_async` — Mem0 attempt first when `MEMORY_BACKEND=mem0`, else pgvector helper.
- `augment_conversation_with_memory_async` — `asyncio.gather`:

  - `database_sync_to_async(get_user_profile_facts)(user)`
  - `get_episodic_memories_async(...)`

  Then `format_memories_for_system` and assign `convo.system`.

- `augment_conversation_with_memory` — thin wrapper: callers that cannot await may use sync path unchanged (internally may call shared helpers without gather).

### Call sites

- `apps/chat/consumers/chat_receive.py` — `await augment_conversation_with_memory_async(...)` (remove outer `database_sync_to_async` for augmentation).
- `apps/chat/consumers/chat.py` — same for WebSocket `connect()` augmentation.

## Testing

- Unit tests: async search delegates to async OSS client (monkeypatch `search_mem0_via_oss_async` or `get_mem0_oss_async`).
- Existing Mem0 OSS tests updated if imports change; keep sync tests for backward compatibility.

## Risks / mitigations

| Risk | Mitigation |
|------|------------|
| Double connection pools (sync + async Mem0) | Acceptable short-term; long-term optional single client type per deployment. |
| `asyncio.wait_for` cancels mid-flight | Align timeout with `MEM0_TIMEOUT_SECONDS`; log warnings on timeout. |
| Django async ORM | Not required; keep `database_sync_to_async` for profile facts. |

## References

- [Mem0 Async Memory (open source)](https://docs.mem0.ai/open-source/features/async-memory)
