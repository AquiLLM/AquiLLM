# Production tool-calling backport

Base: `origin/main` at `6193a3c5b681b22c8d526edcc63ef105a1a39432`,
PR #222 (development merged June 25, 2026).
Source: `origin/development` at `4b604539`.
Branch: `codex/production-tool-calling-fixes`.

The selected changes fix mechanical tool selection, malformed tool-call recovery,
duplicate execution, collection evidence routing, post-tool synthesis, reconnect
persistence, and stale tool status in chat. Each backport commit records its
original SHA. The branch contains no knowledge-graph runtime, schema migrations,
ASR deployment, conversation-search feature, or citation-sources UI rollout.

## Commit map

| Development | Backport | Changes retained |
| --- | --- | --- |
| `19c6cba5` | `9b11aa19` | Configurable local thinking control; subsequently corrected below. |
| `ead505df` | `4f3fafe3` | Provider, tests, and thinking setting only; restore the final enabled default. |
| `ba19d662` | `16ff61f0` | Send vLLM thinking configuration through `extra_body`. |
| `e7e0924e` | `86802022` | Preserve selected collection retrieval. |
| `65a35370` | `cf93ce92` | Whole-document chunk citations, serialized payload budget, compression concurrency, prompt budgeting. |
| `4ffc5724` | `fafe6dab` | Evidence-first retrieval, reuse duplicate tool results, disable thinking for tool selection. |
| `6641562f` | `c07a22f5` | Bound mechanical tool calls and recover malformed or truncated tool output. |
| `92251da0` | `29272d12` | Best-effort transport, durable deltas, and direct RAG on reconnect. |
| `4b309f39` | `4b6f1f85` | Safe request-stage observability required by synthesis and answer recovery. |
| `7283cf9b` | `86b95117` | Selected collection evidence routing and reuse of prior retrieval queries. |
| `a4ee6ca1` | `c148f626` | Direct-synthesis token configuration and its test only. |
| `3de18239` | `92a0743c` | Direct-synthesis budget handling and authoritative chat ordering/tool status, with tests. |
| `8c3d520c` | `8972a61b` | Disable redundant thinking during cutoff continuation, with its test. |
| `5ec5df0f` | `2aa56572` | Recover missing visible answers and handle missing collection scope. |

## Production adaptations

- Preserve main's document authorization and inline figure helpers when resolving
  document-tool conflicts; do not import development's graph authorization layer.
- Omit `MessageSources` changes from `4ffc5724`; that component does not exist on
  main. Omit graph-only logging arguments and retain compatible core timing,
  correlation, and count metrics. Existing retrieval diagnostic behavior remains.
- Omit MTP deployment changes from `ead505df`, reranker changes from `3de18239`,
  and memory/reranker changes from `8c3d520c`.
- Adapt the reconnect smoke fixture to main, which does not have `build_memory_tools`.
- Add the minimal frontend test setup for the imported WebSocket regression:
  Vitest configuration, test script, testing-library/react, and jsdom.
- Split added direct-RAG regressions into `test_direct_rag_recovery.py` to keep
  both direct-RAG test modules within the existing 300-line limit.
- Fix a bug discovered in the imported result-reuse code: assign a fresh UUID
  to the appended cached-result message so persistence cannot overwrite the
  original tool evidence. The regression assertion failed before this fix and
  all eight tool-budget tests passed after it.

## Validation

- **318 backend tests passed**, covering the complete LLM-provider tests,
  database-independent chat tests, search-tool payload tests, and the three
  backend CI smoke modules (architecture boundaries, URL context processors,
  document image view).
- **1 frontend regression passed**: authoritative final-message ordering settles
  the live tool spinner (`npm test -- src/features/chat/hooks/useChatWebSocket.test.tsx`).
- **Frontend production build passed** (`npm run build`).
- **Django system check, import boundaries, hygiene, and `git diff --check` passed**.
- Independent review confirmed the selected fixes and conflict adaptations;
  its cached-result UUID finding was fixed and rechecked.
- **Database tests remain unverified locally.** The initial main baseline had
  245 passing tests and 39 database-setup errors because the `db` hostname was
  unavailable. Docker's Linux daemon is stopped, and the installed local
  PostgreSQL lacks pgvector. Final runs excluded the database-dependent modules
  `test_chat_consumer_append.py`, `test_collection_prompt_skills.py`,
  `test_conversation_persistence.py`, `test_feedback_capture.py`, and
  `test_message_adapters.py`. Run them with PostgreSQL/pgvector before release.
- **The file-length gate still fails on 22 legacy files already over 300 lines
  in main.** Seven existing oversized files grew with the imported fixes. No
  additional file crosses the limit; the expanded direct-RAG tests were split.

Backend validation used Python 3.13, `aquillm.settings_test`, and dummy provider
credentials. Frontend validation used Node 22.14.0. No live model calls or
production deployment were performed.
