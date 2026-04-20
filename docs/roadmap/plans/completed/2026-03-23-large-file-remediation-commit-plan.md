# Large-File Remediation â€” Commit Plan

> **For agentic workers:** Execute task-by-task; run the listed verification command before moving on. Prefer `superpowers:subagent-driven-development` or `superpowers:executing-plans` for multi-session work.

**Goal:** Finish [2026-03-19-large-file-remediation-lib-tools-and-splits.md](./2026-03-19-large-file-remediation-lib-tools-and-splits.md) with a reviewable, bisect-friendly history: one subsystem per commit, behavior unchanged, tests green.

**Prerequisites (local verification):** From repo root, use a venv with `requirements.txt` installed. For Django/pytest, set at least `DJANGO_DEBUG=1`, `OPENAI_API_KEY`, `GEMINI_API_KEY` (dummy values are fine for CI-style runs), and a reachable PostgreSQL matching `aquillm` settingsâ€”or run tests inside Docker Compose the way your team already does.

**Default chat/LLM verification (run after most backend commits):**

```powershell
cd aquillm
python -m pytest apps/chat/tests lib/llm/tests -q --tb=short
```

**Structure scripts (after touching line counts or imports):**

```powershell
python scripts/check_file_lengths.py
python scripts/check_import_boundaries.py
```

---

## Commit 1 â€” Chat refs extraction

**Message:** `refactor(chat): extract CollectionsRef and ChatRef to apps.chat.refs`

**Files:**

- `aquillm/apps/chat/refs.py` (create)
- `aquillm/apps/chat/consumers/chat.py` (imports only; remove inline classes)

**Verify:** `cd aquillm; python -m pytest apps/chat/tests -q --tb=short`

---

## Commit 2 â€” Consumer utils (env, text, UUID, image resize)

**Message:** `refactor(chat): move consumer helpers to apps.chat.consumers.utils`

**Files:**

- `aquillm/apps/chat/consumers/utils.py` (create)
- `aquillm/apps/chat/consumers/chat.py`

**Notes:** Keep `_resize_image_for_llm_context` here unless Commit 3 is done in the same session; if you move pure Pillow logic to `lib`, do Commit 3 immediately after so `chat.py` does not churn twice.

**Verify:** `cd aquillm; python -m pytest apps/chat/tests -q --tb=short`

---

## Commit 3 (optional) â€” Pure LLM image resize helper in lib

**Message:** `refactor(llm): extract data-url resize helpers for chat context`

**Files:**

- `aquillm/lib/llm/utils/images.py` (create, if not already present)
- `aquillm/apps/chat/consumers/utils.py` (delegate to lib)
- `aquillm/apps/chat/consumers/chat.py` (imports only if needed)

**Verify:** `cd aquillm; python -m pytest apps/chat/tests -q --tb=short`

**Skip** if you prefer to keep image resize next to chat config (valid per source plan).

---

## Commit 4 â€” Dedupe WebSocket send/stream helpers

**Message:** `refactor(chat): dedupe websocket send/stream helpers in ChatConsumer`

**Files:**

- `aquillm/apps/chat/consumers/chat.py` only

**Verify:** `cd aquillm; python -m pytest apps/chat/tests -q --tb=short`

---

## Commit 5 â€” Tool wiring scaffold + vector search factory

**Message:** `refactor(chat-tools): add tool_wiring and lib vector search factory`

**Files:**

- `aquillm/apps/chat/services/__init__.py` (create or extend)
- `aquillm/apps/chat/services/tool_wiring.py` (create)
- `aquillm/lib/tools/search/__init__.py` (create or extend)
- `aquillm/lib/tools/search/vector_search.py` (create)
- `aquillm/apps/chat/consumers/chat.py` (switch to factory + wiring)

**Verify:**

- `cd aquillm; python -m pytest apps/chat/tests -q --tb=short`
- `python scripts/check_import_boundaries.py`

---

## Commit 6 â€” More-context tool in lib + wiring

**Message:** `refactor(chat-tools): extract more-context tool to lib.tools.search`

**Files:**

- `aquillm/lib/tools/search/context.py` (create)
- `aquillm/lib/tools/search/__init__.py`
- `aquillm/apps/chat/services/tool_wiring.py`
- `aquillm/apps/chat/consumers/chat.py`

**Verify:** same as Commit 5.

---

## Commit 7 â€” Document listing tool

**Message:** `refactor(chat-tools): extract document list-ids tool to lib.tools.documents`

**Files:**

- `aquillm/lib/tools/documents/__init__.py` (create or extend)
- `aquillm/lib/tools/documents/list_ids.py` (create)
- `aquillm/apps/chat/services/tool_wiring.py`
- `aquillm/apps/chat/consumers/chat.py`

**Verify:** same as Commit 5.

---

## Commit 8 â€” Whole-document tool

**Message:** `refactor(chat-tools): extract whole-document tool to lib.tools.documents`

**Files:**

- `aquillm/lib/tools/documents/whole_document.py` (create)
- `aquillm/lib/tools/documents/__init__.py`
- `aquillm/apps/chat/services/tool_wiring.py`
- `aquillm/apps/chat/consumers/chat.py`

**Verify:** same as Commit 5.

---

## Commit 9 â€” Single-document search tool

**Message:** `refactor(chat-tools): extract single-document search tool to lib.tools.documents`

**Files:**

- `aquillm/lib/tools/documents/single_document.py` (create)
- `aquillm/lib/tools/documents/__init__.py`
- `aquillm/apps/chat/services/tool_wiring.py`
- `aquillm/apps/chat/consumers/chat.py`

**Verify:** same as Commit 5.

---

## Commit 10 â€” Astronomy tools (sky / flat / point source)

**Message:** `refactor(chat-tools): move astronomy tools to lib.tools.astronomy`

**Files (adjust if you collapse into one module; keep each file under ~300 lines):**

- `aquillm/lib/tools/astronomy/__init__.py` (create)
- `aquillm/lib/tools/astronomy/sky_subtraction.py` (create)
- `aquillm/lib/tools/astronomy/flat_fielding.py` (create)
- `aquillm/lib/tools/astronomy/point_source.py` (create)
- `aquillm/apps/chat/services/tool_wiring.py` and/or `aquillm/apps/chat/consumers/chat.py` (wire callables)

**Verify:** `cd aquillm; python -m pytest apps/chat/tests -q --tb=short`

---

## Commit 11 â€” Debug weather tool

**Message:** `refactor(chat-tools): extract debug weather tool to lib.tools.debug`

**Files:**

- `aquillm/lib/tools/debug/__init__.py` (create or extend)
- `aquillm/lib/tools/debug/weather.py` (create)
- `aquillm/apps/chat/consumers/chat.py`

**Verify:** same as Commit 10.

---

## Commit 12 â€” LLM base: fallback heuristics

**Message:** `refactor(llm): extract fallback heuristics from base provider`

**Files:**

- `aquillm/lib/llm/providers/fallback_heuristics.py` (create)
- `aquillm/lib/llm/providers/base.py`

**Verify:** `cd aquillm; python -m pytest lib/llm/tests -q --tb=short`

---

## Commit 13 â€” LLM base: tool evidence helpers

**Message:** `refactor(llm): extract tool evidence helpers from base provider`

**Files:**

- `aquillm/lib/llm/providers/tool_evidence.py` (create)
- `aquillm/lib/llm/providers/base.py`

**Verify:** same as Commit 12.

---

## Commit 14 â€” LLM base: image / markdown context helpers

**Message:** `refactor(llm): extract image and markdown context helpers from base provider`

**Files:**

- `aquillm/lib/llm/providers/image_context.py` (create)
- `aquillm/lib/llm/providers/base.py`

**Verify:** same as Commit 12.

---

## Commit 15 â€” OpenAI provider token helpers

**Message:** `refactor(openai): extract multimodal token and overflow helpers`

**Files:**

- `aquillm/lib/llm/providers/openai_tokens.py` (create)
- `aquillm/lib/llm/providers/openai.py`

**Verify:** same as Commit 12.

---

## Commit 16 (optional) â€” Compact tool summary split

**Message:** `refactor(llm): extract compact tool summary helper from base provider`

**Files:**

- `aquillm/lib/llm/providers/summary.py` (create, only if `base.py` still over budget)
- `aquillm/lib/llm/providers/base.py`

**Verify:** same as Commit 12.

**Skip** if Commit 12â€“14 already bring `base.py` under your target.

---

## Commit 17 â€” Split chat message tests by concern

**Message:** `test(chat): split monolithic test_messages into focused modules`

**Files (names may follow existing test classes):**

- `aquillm/apps/chat/tests/test_message_adapters.py` (create)
- `aquillm/apps/chat/tests/test_multimodal_messages.py` (create)
- `aquillm/apps/chat/tests/test_tool_result_images.py` (create)
- `aquillm/apps/chat/tests/test_messages.py` (shrink or remove)

**Verify:** `cd aquillm; python -m pytest apps/chat/tests -q --tb=short`

---

## Commit 18 â€” Frontend: chat feature module

**Message:** `refactor(frontend): move chat UI to features/chat`

**Files (typical):**

- `react/src/features/chat/components/Chat.tsx` (create)
- `react/src/features/chat/hooks/useChatWebSocket.ts` (create)
- `react/src/components/ChatComponent.tsx` (re-export shim)
- `react/src/main.tsx` (registry / imports)

**Verify:** `cd react; npm ci; npm run build`

---

## Commit 19 â€” Frontend: collections feature module

**Message:** `refactor(frontend): move collection view to features/collections`

**Files (typical):**

- `react/src/features/collections/components/CollectionView.tsx` (create)
- `react/src/components/CollectionView.tsx` (re-export)
- `react/src/main.tsx`

**Verify:** `cd react; npm run build`

---

## Commit 20 â€” Frontend: documents + platform admin moves

**Message:** `refactor(frontend): move filesystem viewer and user management modal to features`

**Files (typical):**

- `react/src/features/documents/components/FileSystemViewer.tsx` (create)
- `react/src/features/platform_admin/components/UserManagementModal.tsx` (create)
- `react/src/components/FileSystemViewer.tsx` (re-export)
- `react/src/components/UserManagementModal.tsx` (re-export)
- `react/src/main.tsx`

**Verify:** `cd react; npm run build`

---

## Commit 21 â€” Frontend: ingestion container split (if still >300 lines)

**Message:** `refactor(frontend): split IngestRowsContainer into subcomponents`

**Files:**

- Under `react/src/features/ingestion/components/` (new presentational pieces + thinner container)

**Verify:** `cd react; npm run build`

**Skip** if `IngestRowsContainer.tsx` is already within policy after earlier work.

---

## Commit 22 â€” Docs: lib/tools and wiring pointers

**Message:** `docs: document lib.tools layout and chat tool wiring`

**Files:**

- `README.md` (short subsection: `lib/tools`, `apps/chat/services/tool_wiring.py`, dependency rule)
- Optionally `docs/documents/architecture/aquillm-current-architecture-mermaid.md` (one diagram note)

**Verify:** proofread only (no code).

---

## Commit 23 â€” Hygiene: shrink file-length allowlist

**Message:** `chore(structure): trim file-length allowlist after splits`

**Files:**

- `scripts/check_file_lengths.py` (remove paths that no longer exceed `MAX_LINES`)

**Verify:** `python scripts/check_file_lengths.py` (exit 0)

---

## Appendix A â€” Related backlog (architecture plan, not 2026-03-19)

If you are sequencing **all** structural debt in one branch, schedule these **after** Commit 23 (or in parallel worktrees) using [2026-03-21-architecture-remediation-commit-plan.md](../completed/2026-03-21-architecture-remediation-commit-plan.md) commits **11â€“13**:

- Ingestion `api.py` decomposition  
- `chunks.py` embedding/search/rerank services  
- Zotero task split and threaded ORM removal  

---

## Commit hygiene rules

1. **One primary subsystem per commit** (chat tools vs LLM providers vs React feature).
2. **Run at least one verification command** before committing; after Commits 5â€“11, run `check_import_boundaries.py`.
3. **Do not mix** large React moves with Python refactors in the same commit.
4. **Behavior-preserving extractions first**; follow-up behavior changes belong in separate commits with explicit messages.
5. If a commit would touch **more than ~12 files** without a single cohesive extraction, split it (e.g. separate commits 7â€“9 instead of one mega-commit).

---

**Plan saved as `docs/roadmap/plans/active/2026-03-23-large-file-remediation-commit-plan.md`.**





