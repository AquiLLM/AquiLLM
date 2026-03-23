# Large-file remediation — continuation handoff (2026-03-24)

**Purpose:** Resume [2026-03-23-large-file-remediation-commit-plan.md](./2026-03-23-large-file-remediation-commit-plan.md) and [2026-03-19-large-file-remediation-lib-tools-and-splits.md](./2026-03-19-large-file-remediation-lib-tools-and-splits.md). **Plan commits 1–23 from the large-file commit plan are done** except optional **commit 16** (compact tool summary / `base.py` shrink) and any further splits driven by `check_file_lengths.py`.

**Code quality / hygiene (separate track):** For runtime, security, settings, logging, and React hygiene items (mount map barrels, `IngestRow` dedup, `.gitignore`, CI), use [2026-03-19-aquillm-structure-code-quality-remediation.md](./2026-03-19-aquillm-structure-code-quality-remediation.md) and [2026-03-21-architecture-remediation-commit-plan.md](./2026-03-21-architecture-remediation-commit-plan.md).

**Previous snapshot:** [2026-03-23-large-file-remediation-handoff.md](./2026-03-23-large-file-remediation-handoff.md) (historical; superseded by this file for “what’s done”).

---

## Commits on `development` (reference — verify with `git log`)

### Large-file plan: backend + tests (earlier)

| Plan | Short hash | Subject |
|------|------------|---------|
| **15** | `09e3304` | `refactor(openai): extract multimodal token and overflow helpers` |
| **17** | `2ebd2d4` | `test(chat): split monolithic test_messages into focused modules` |

### Large-file plan: frontend + docs + allowlist (2026-03-23 session)

| Plan | Short hash | Subject |
|------|------------|---------|
| **18–21** | `8e0d1a5` | `refactor(frontend): move chat, collections, documents, and admin UI to features` |
| **22–23** | `1693dd5` | `docs: document React features layout; chore(structure): refresh file-length allowlist` |

*If you need exact full hashes: `git log --oneline -15` on `development`.*

---

## What is now implemented

### OpenAI provider — token / overflow modules

| Module | Role |
|--------|------|
| `aquillm/lib/llm/providers/openai_tokens.py` | Env helpers, `context_reserve_tokens`, flatten/estimate, `trim_messages_for_overflow`, `preflight_trim_for_context(cls, …)` |
| `aquillm/lib/llm/providers/openai_overflow.py` | `strip_images_from_messages`, `retry_args_for_context_overflow`, `retry_args_for_timeout` |
| `aquillm/lib/llm/providers/openai.py` | Thin `OpenAIInterface` delegators; tests can still patch `_estimate_prompt_tokens` / `_trim_messages_for_overflow` on the class |

### Chat tests — `test_messages.py` removed

| File | Role |
|------|------|
| `aquillm/apps/chat/tests/chat_message_test_support.py` | Shared fakes + `@llm_tool` stubs — not a test module |
| `test_message_adapters.py` | Pydantic ↔ Django adapters + `build_frontend_conversation_json` |
| `test_conversation_persistence.py` | Save/load, ratings, conversation title |
| `test_multimodal_messages.py` | OpenAI fallback parsing, context overflow/retry, reserve scaling, image token estimator |
| `test_tool_result_images.py` | Tool result redaction + markdown injection after `complete()` |
| `test_llm_complete_retry.py` | Tool-use retry + max-token cutoff continuation |

### React — `features/*` layout (commits 18–21)

| Area | Location | Notes |
|------|----------|--------|
| Chat | `react/src/features/chat/components/Chat.tsx`, `ChatShell.tsx`, `ChatInputDock.tsx`, `ChatCollectionsModal.tsx` | WebSocket logic in `features/chat/hooks/useChatWebSocket.ts` |
| Chat mount | `react/src/main.tsx` | `ChatComponent` from `./features/chat/components/ChatShell` |
| Shim | `react/src/components/ChatComponent.tsx` | Re-exports `ChatShell` |
| Collections | `react/src/features/collections/components/CollectionView.tsx` | Imports shared UI via `../../../components/*`; uses `../../documents/...` and `../../platform_admin/...` for moved widgets |
| Shim | `react/src/components/CollectionView.tsx` | Re-export |
| Documents | `react/src/features/documents/components/FileSystemViewer.tsx` | Shared deps: `../../../components/*`, `../../../types/*` |
| Platform admin | `react/src/features/platform_admin/components/UserManagementModal.tsx` | Same `../../../` pattern |
| Shims | `react/src/components/FileSystemViewer.tsx`, `UserManagementModal.tsx` | Re-exports |
| Ingestion | `react/src/features/ingestion/components/IngestRowsContainer.tsx` | Slim container; polling in `hooks/useIngestUploadBatchPolling.ts`, submit loop in `utils/runIngestRowSubmissions.ts`, status UI in `IngestRowStatusBlocks.tsx` |

### Docs + structure (commits 22–23)

- **README.md** — “Module layout” includes a bullet on `react/src/features/*` and shim pattern.
- **`scripts/check_file_lengths.py`** — Allowlist updated: removed shims and slimmed `IngestRowsContainer`; **still allowlists** the three large feature files below plus `CollectionsPage.tsx` and all prior backend hotspots.

**Current frontend allowlist entries (still &gt;300 lines):**

- `react/src/components/CollectionsPage.tsx`
- `react/src/features/collections/components/CollectionView.tsx`
- `react/src/features/documents/components/FileSystemViewer.tsx`
- `react/src/features/platform_admin/components/UserManagementModal.tsx`

---

## Gotchas (read before import churn)

1. **Circular import (`tool_wiring` ↔ `ChatConsumer`)** — `apps/chat/consumers/__init__.py` uses **`__getattr__`** for lazy `ChatConsumer`. Do not eager-import `chat.py` at package init without fixing the cycle.
2. **`lib` must not import `apps.*`** — run `python scripts/check_import_boundaries.py` after Python refactors.
3. **Wiring type hints** — `ChatConsumer` stays forward-quoted in `tool_wiring` where needed.
4. **React path depth** — Files under `features/<domain>/components/` typically need **`../../../`** to reach `src/components`, `src/types`, `src/utils`. Cross-feature imports use `../../other_feature/...` (e.g. collections → documents).

---

## Remaining work

### Large-file plan (optional / follow-up)

| Item | Work |
|------|------|
| **Commit 16 (optional)** | If `aquillm/lib/llm/providers/base.py` should leave the allowlist, extract compact tool summary (e.g. `summary.py`) per [2026-03-23-large-file-remediation-commit-plan.md](./2026-03-23-large-file-remediation-commit-plan.md). |
| **Backend allowlist** | When `chat.py`, `base.py`, or `openai.py` drops ≤300 lines, remove only that path from `_ALLOWLIST` in `scripts/check_file_lengths.py` (proof: script exit 0). |
| **React &gt;300** | Split `CollectionView`, `FileSystemViewer`, and `UserManagementModal` until each is ≤300 lines, then remove their paths from the allowlist. |

### Architecture / ingestion backend (separate plan)

- [2026-03-21-architecture-remediation-commit-plan.md](./2026-03-21-architecture-remediation-commit-plan.md) commits **11–13** (e.g. ingestion `api.py`, `chunks.py` services, Zotero split) — schedule after or in parallel worktrees.

### Structure and code quality remediation

- [2026-03-19-aquillm-structure-code-quality-remediation.md](./2026-03-19-aquillm-structure-code-quality-remediation.md) — chat append regression tests, ingest consumer auth, settings/toolbar security, logging cleanup, **`main.tsx` barrel imports**, **`IngestRow` dedup**, `.gitignore` / hygiene CI.

---

## Verification commands

**Structure (after line-count or import changes):**

```powershell
cd c:\Users\jackj\Github\AquiLLM
python scripts/check_file_lengths.py
python scripts/check_import_boundaries.py
```

**Backend — no DB (quick):**

```powershell
cd aquillm
$env:DJANGO_DEBUG='1'
$env:OPENAI_API_KEY='dummy'
$env:GEMINI_API_KEY='dummy'
python -m pytest apps/chat/tests/test_multimodal_messages.py apps/chat/tests/test_llm_complete_retry.py apps/chat/tests/test_tool_result_images.py lib/llm/tests -q --tb=short
```

**Backend — full chat + LLM (needs Postgres reachable; use Compose if `POSTGRES_HOST=db`):**

```powershell
cd aquillm
$env:DJANGO_DEBUG='1'
$env:OPENAI_API_KEY='dummy'
$env:GEMINI_API_KEY='dummy'
python -m pytest apps/chat/tests lib/llm/tests -q --tb=short
```

**Frontend:**

```powershell
cd react
npm ci
npm run build
```

---

## Suggested next session order

1. **Code quality blockers** — [2026-03-19-aquillm-structure-code-quality-remediation.md](./2026-03-19-aquillm-structure-code-quality-remediation.md) Chunk 1 (tests + consumers) and settings slice; run pytest where DB is available.
2. **Architecture plan** — [2026-03-21-architecture-remediation-commit-plan.md](./2026-03-21-architecture-remediation-commit-plan.md) commits 11–13 as needed.
3. **Large-file polish** — Optional commit 16; React splits for allowlisted feature files; trim backend allowlist only with `check_file_lengths.py` proof.

---

## Doc / command drift

Older plans may still reference `apps/chat/tests/test_messages.py`. Prefer:

```text
apps/chat/tests/test_message_adapters.py
apps/chat/tests/test_conversation_persistence.py
apps/chat/tests/test_multimodal_messages.py
apps/chat/tests/test_tool_result_images.py
apps/chat/tests/test_llm_complete_retry.py
```

---

*Handoff for large-file remediation and related code-quality continuation; large-file commit plan 1–23 is effectively complete except optional 16 and incremental allowlist/split work.*
