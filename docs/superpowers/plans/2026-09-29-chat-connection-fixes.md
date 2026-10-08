# Chat Connection Fixes Implementation Plan

> **For agentic workers:** Use parallel agents with disjoint file ownership, test-first fixes, and a final integration review.

**Goal:** Resolve all six findings in the approved chat connection audit.

**Architecture:** Give the frontend a socket lifecycle independent of render state and require hydration before sending. Protect database transcript writes against stale writers, avoid writes for unchanged startup snapshots, restore authorized pending tools, and close fatal startup connections explicitly.

**Tech Stack:** React/TypeScript/Vitest; Django/Channels/Python/pytest/PostgreSQL.

**Spec:** `docs/audits/2026-09-29-chat-connection.md`, approved for repair by the user.

## Global Constraints

- Work only in this isolated checkout; preserve unrelated working directories.
- Prefix every shell command with `rtk` (use `rtk proxy` for raw commands).
- No provider/network calls in tests; use a dedicated test database for database tests.
- Keep existing turn error recovery, transcript ordering, selected collections, and publication cancellation behavior.
- Do not commit, push, or modify git state from concurrent workers. The controller integrates and reviews the complete patch.
- Protocol agreement: fatal initialization payloads retain `exception` and may add `fatal: true`. Permanent closes are 4401 (authentication), 4404 (conversation access/not found), and 4409 (stale conversation; refresh required). Transient server/overload closes are 1011/1013. The frontend must not reconnect permanent closes; fatal errors disable input immediately. The initial `conversation` snapshot establishes hydration; socket open alone does not.

## Task 1: Frontend lifecycle, readiness, and banner (audit findings 1, 5, 6)

Owner: frontend agent. Files: `react/src/features/chat/hooks/useChatWebSocket.ts`, its tests, `react/src/features/chat/components/Chat.tsx`, its tests, and `react/src/features/chat/types/index.ts`; additional focused frontend test helpers if useful.

- [x] Add lifecycle tests with real open/close transitions and fake timers; demonstrate failures before implementation.
- [x] Keep successful reconnects alive, schedule one bounded retry per failed attempt, cancel timers and ignore events on retired sockets, and reset on conversation change.
- [x] Keep send/input disabled until authoritative hydration; classify fatal startup errors and permanent close codes; bound hydration stalls.
- [x] Present neutral initial/reconnect status separately from red exceptions. Preserve recoverable turn errors and spinner settlement.
- [x] Run focused frontend tests and TypeScript validation; report failures unrelated to this patch distinctly.

Expected tests: healthy first load; successful retry remains open; timeout creates one retry; exhausted retries; cleanup/unmount; old-socket events ignored; conversation ID change; missing hydration; saved selection precedes sending; permanent close; fatal error versus recoverable turn error.

## Task 2: Protect transcript persistence (audit finding 2)

Owner: persistence agent. Files: `aquillm/aquillm/message_adapters.py`, `aquillm/apps/chat/models/conversation.py`, migration if needed, a focused persistence service module if needed, persistence tests. Do not edit `chat.py`, `chat_delta.py`, or frontend files.

- [x] Write a regression where two loaded copies race and the stale copy must not delete/overwrite a newer append.
- [x] Implement an explicit database concurrency check under a row lock or atomic compare-and-swap, with a clear recoverable conflict outcome. A transaction alone is insufficient. Preserve legitimate save/load and explicit replacement contracts; do not silently merge incompatible snapshots.
- [x] Cover sequential saves from the same consumer, stale overlapping edits, timestamps/ratings/collections considerations, and fresh conversation creation.
- [x] Run persistence tests against a dedicated local test database. Provide any required consumer conflict-handling interface to the lifecycle agent/controller.

The lifecycle agent removes unchanged startup writes, but that is only one layer; stale writers on live sockets must also be protected.

## Task 3: Backend initialization and recovery (audit findings 3, 4; startup part of 2)

Owner: backend lifecycle agent. Files: `aquillm/apps/chat/consumers/chat.py`, `chat_delta.py`, `chat_ws_errors.py`, `chat_receive.py` as needed, `aquillm/lib/llm/types/conversation.py`, and dedicated backend lifecycle/tool recovery tests. Do not edit message adapters or model/migration files owned by persistence.

- [x] Reproduce fatal-open dead consumers, failed pending-tool recovery, and saves on unchanged initial hydration.
- [x] Authenticate and authorize before acceptance where practical; cover all initialization failures; emit fatal metadata when accepted and close with the agreed code.
- [x] Restore pending tool execution from currently authorized tool factories, or explicitly settle unavailable tools. Never silently strand the UI.
- [x] Do not run memory/spin/save work on an idle initial snapshot. In delta publication, do not save an unchanged transcript. Preserve legitimate resume and final publication behavior.
- [x] Remove redundant N+1 collection scan if no longer needed by the initialization flow.
- [x] Run focused backend tests; coordinate conflict handling with the persistence owner.

## Integration and Review

- [x] Review each task's patch against its acceptance cases.
- [x] Run combined frontend and backend regression suites, type checks, formatting/lint as appropriate, and migration checks if changed.
- [x] Dispatch an independent final review of all six fixes and repair actionable findings.
- [x] Document final verification and any limitations, then deliver the isolated branch/worktree.
