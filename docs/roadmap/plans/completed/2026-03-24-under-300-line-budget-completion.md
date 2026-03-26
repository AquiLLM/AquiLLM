# Underâ€“300-Line File Budget â€” Completion Plan

> **For agentic workers:** Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to execute this plan. Track steps with checkbox (`- [ ]`) updates. **Do not claim a task done until** `python scripts/check_file_lengths.py` exits **0** and the listed tests/build pass.

**Goal:** Remove every path from `scripts/check_file_lengths.py` `_ALLOWLIST` by bringing each file to **â‰¤300 lines** (same counting as the script: one line per newline in the file)â€”**without** sacrificing a clear module story or forcing artificial cuts.

**Architecture:** Split by **responsibility** (helpers vs HTTP vs orchestration), keep **public APIs stable** (re-export shims in the original module where URLs or imports would otherwise churn), and follow existing patterns (`apps/*/services/`, `lib/`, `react/src/features/*`).

**Tech stack:** Django 5.x, Celery, Channels, React/TypeScript, existing `pytest` and `npm run build` gates.

**Evidence rule:** After each mergeable chunk, run `python scripts/check_file_lengths.py && python scripts/check_import_boundaries.py`. Remove a path from `_ALLOWLIST` only when that file is â‰¤300 lines and the script still exits 0.

### Structure and readability (non-negotiable)

The line budget is a **guardrail**, not the product. Prefer staying slightly over budget briefly over a split that confuses readers.

- **One reason to open a file:** Each module should answer a clear question (e.g. â€œlocal rerank HTTP client,â€ â€œcollection JSON payload builders,â€ â€œOpenAI message shapingâ€). If you cannot name the file in one short phrase, merge or rename.
- **Colocate what changes together:** Extract chunks that are edited as a unit (one feature, one pipeline stage, one integration). Avoid â€œ`part2.py`â€ dumps of unrelated leftovers.
- **Keep entrypoints thin:** Views = HTTP + auth + delegation; tasks = registration + thin wrappers; `settings` = composition; React containers = wiring; heavy logic lives in services/helpers/hooks with tests.
- **Stable imports:** Prefer re-exports from the original public module so call sites and URLs do not churn. New files should be **discoverable** (same package or documented sibling, not scattered one-offs).
- **Depth vs sprawl:** Prefer a **small subpackage** with obvious names (e.g. `chunk_rerank/parse.py`) over many ambiguous top-level filesâ€”**if** the package has a clear index (`__init__.py` or a single facade). Do not create packages with no narrative.
- **Readability check before merge:** Skim the diff as a new teammate would: can you follow the flow without jumping through more than one indirection? If not, adjust boundaries before shrinking the allowlist.

**Anti-patterns (reject or redo):** splitting mid-function; â€œmisc helpersâ€ buckets; circular imports fixed by lazy hacks without simplifying ownership; settings split into arbitrary alphabetical files; React components split only by line count with shared mutable state threaded through props.

---

## Current allowlist snapshot (line counts = `check_file_lengths.py` logic)

| Lines | Path |
|------:|------|
| 311 | `aquillm/apps/chat/consumers/chat.py` |
| 382 | `aquillm/apps/documents/services/chunk_rerank.py` |
| 318 | `aquillm/apps/collections/views/api.py` |
| 429 | `aquillm/aquillm/ingestion/figure_extraction/pdf.py` |
| 357 | `aquillm/aquillm/ingestion/parsers.py` |
| 352 | `aquillm/aquillm/settings.py` |
| 313 | `aquillm/aquillm/tasks.py` |
| 342 | `aquillm/aquillm/zotero_views.py` |
| 485 | `aquillm/lib/llm/providers/openai.py` |
| 332 | `react/src/components/CollectionsPage.tsx` |
| 655 | `react/src/features/collections/components/CollectionView.tsx` |
| 631 | `react/src/features/documents/components/FileSystemViewer.tsx` |
| 454 | `react/src/features/platform_admin/components/UserManagementModal.tsx` |

---

## File map (target layout after splits)

| Current file | New / extracted modules (illustrative) | Notes |
|--------------|------------------------------------------|--------|
| `chat.py` | `apps/chat/consumers/chat_stream.py` or move 1â€“2 private helpers next to `utils.py` | ~10â€“20 lines over; smallest win. |
| `chunk_rerank.py` | `chunk_rerank/http_config.py`, `chunk_rerank/parse.py`, `chunk_rerank/local_vllm.py` **or** flat siblings under `services/` | Avoid circular imports; `rerank_chunks` stays thin in `chunk_rerank.py`. |
| `api.py` (collections) | `apps/collections/services/collection_api_payloads.py` (tree, type labels, figure mapping) | Views stay HTTP-only; import service functions. |
| `parsers.py` | More extraction into `lib/parsers/` or `aquillm/ingestion/extractors/*.py` | Keep `detect_ingest_type` wrapper in `ingestion/parsers.py` if tests import from there. |
| `settings.py` | `aquillm/settings/` package: `base.py`, `database.py`, `integrations.py`, `security.py`, `celery.py` â€” **or** split only the largest blocks first | Must preserve `DJANGO_SETTINGS_MODULE=aquillm.settings`; use `from .x import *` pattern carefully. |
| `tasks.py` | `aquillm/task_modules/*.py` grouped by domain; `tasks.py` registers imports for Celery autodiscover | Ensure task names remain stable if referenced by string. |
| `zotero_views.py` | `aquillm/zotero_sync_ui.py` (flatten tree + fetch helpers) | Views file: decorators + thin handlers only. |
| `pdf.py` (figure extraction) | `pdf_pipeline.py`, `pdf_text.py`, `pdf_images.py` under same package | Keep one public entry used by callers. |
| `openai.py` | `openai_messages.py`, `openai_streaming.py`, or move more into `openai_tokens.py` / `openai_overflow.py` | Preserve test patch points documented in tests. |
| React TSX | Presentational subcomponents + hooks under same feature folder | Re-export from original file optional for import stability. |

---

## Phase A â€” Near-miss Python (fast)

### Task A1: `apps/chat/consumers/chat.py` (311 â†’ â‰¤300)

**Files:** `aquillm/apps/chat/consumers/chat.py`, optional new helper module under `apps/chat/consumers/`.

- [ ] Move one or two largest **pure functions** / static helpers out of `ChatConsumer` (e.g. stream/send helpers already partially deduped â€” look for remaining blocks).
- [ ] **Verify:** `cd aquillm && pytest apps/chat/tests -q --tb=short`
- [ ] **Verify:** `python scripts/check_file_lengths.py` (exit 0); remove `aquillm/apps/chat/consumers/chat.py` from `_ALLOWLIST`.

**Commit message:** `refactor(chat): trim chat consumer under file-length budget`

---

### Task A2: `aquillm/aquillm/tasks.py` (313 â†’ â‰¤300)

**Files:** `aquillm/aquillm/tasks.py`, new `aquillm/aquillm/task_handlers/` or domain modules.

- [ ] Extract the **largest single task or helper block** (e.g. ingestion, maintenance) into an imported module; keep `@shared_task` definitions discoverable.
- [ ] **Verify:** targeted pytest for any module that imports those tasks (grep for task name); `pytest aquillm/tests -q --tb=short` if lightweight.
- [ ] **Verify:** `check_file_lengths.py` exit 0; drop allowlist entry for `tasks.py`.

**Commit message:** `refactor(tasks): extract task body module under line budget`

---

### Task A3: `apps/collections/views/api.py` (318 â†’ â‰¤300)

**Files:** `aquillm/apps/collections/views/api.py`, new `aquillm/apps/collections/services/collection_api.py` (or similar).

- [ ] Move `_normalized_type_label`, `_raw_text_type_overrides`, `_child_collection_parent_document_ids`, and any other **non-view** logic into the service module.
- [ ] **Verify:** `cd aquillm && pytest apps/collections/tests -q --tb=short` (or grep + run tests that hit collection API).
- [ ] **Verify:** `check_file_lengths.py` exit 0; remove collections `api.py` from allowlist.

**Commit message:** `refactor(collections): move API payload helpers to services`

---

## Phase B â€” Documents rerank (medium)

### Task B1: `apps/documents/services/chunk_rerank.py` (382 â†’ â‰¤300)

**Files:** `chunk_rerank.py` + 1â€“2 new modules under `apps/documents/services/`.

- [ ] Split **local HTTP rerank/score** path (largest block) into `chunk_rerank_local.py` with lazy imports if needed to avoid cycles.
- [ ] Optionally split **response parsers** into `chunk_rerank_parse.py`.
- [ ] Keep `rerank_chunks` and `rerank_document_payload` as the stable entrypoints from `chunk_rerank.py`.
- [ ] **Verify:** `cd aquillm && pytest apps/documents/tests -q --tb=short` (Postgres as per project norm).
- [ ] **Verify:** `check_import_boundaries.py`; `check_file_lengths.py` exit 0; remove `chunk_rerank.py` from allowlist (or allowlist only a sub-module if still >300 â€” prefer not).

**Commit message:** `refactor(documents): split chunk rerank HTTP helpers`

---

## Phase C â€” Ingestion and runtime config

### Task C1: `aquillm/ingestion/parsers.py` (357 â†’ â‰¤300)

**Files:** `aquillm/aquillm/ingestion/parsers.py`, `lib/parsers/` or `aquillm/ingestion/` submodules.

- [ ] Extract one **cohesive** block (e.g. zip/archive path, figure extraction glue, or media transcription wrapper) into a dedicated module.
- [ ] **Verify:** `cd aquillm && pytest apps/ingestion/tests -q --tb=short` (include `test_unified_ingestion_parsers.py` and any parser smoke tests).
- [ ] **Verify:** `check_file_lengths.py` exit 0; remove `parsers.py` from allowlist.

**Commit message:** `refactor(ingestion): shrink parsers facade module`

---

### Task C2: `aquillm/ingestion/figure_extraction/pdf.py` (429 â†’ â‰¤300)

**Files:** package under `figure_extraction/` (multiple `.py` files).

- [ ] Split by pipeline stage (parse, extract images, OCR hook, etc.) following existing naming in that folder.
- [ ] **Verify:** any tests under `aquillm` referencing PDF figure extraction; `pytest` with grep-discovered paths.
- [ ] **Verify:** `check_file_lengths.py` exit 0; remove `pdf.py` from allowlist.

**Commit message:** `refactor(ingestion): split PDF figure extraction module`

---

### Task C3: `aquillm/settings.py` (352 â†’ â‰¤300)

**Files:** `aquillm/aquillm/settings.py` â†’ optional `aquillm/aquillm/settings/` package.

- [ ] Prefer **lowest-risk** first: move **integrations** (email, third-party keys) or **Celery/cache** blocks to `settings/celery.py` etc.; re-export in `settings/__init__.py` or keep `aquillm/settings.py` as thin aggregator.
- [ ] **Critical:** Preserve `DJANGO_SETTINGS_MODULE` entrypoint; run `manage.py check` and a minimal pytest import.
- [ ] **Verify:** `cd aquillm && python manage.py check`
- [ ] **Verify:** `check_file_lengths.py` exit 0; remove `settings.py` from allowlist.

**Commit message:** `refactor(settings): modularize settings under line budget`

---

### Task C4: `aquillm/zotero_views.py` (342 â†’ â‰¤300)

**Files:** `aquillm/aquillm/zotero_views.py`, `aquillm/aquillm/zotero_sync_helpers.py` (or under `apps/integrations/zotero/` if you want stricter layering).

- [ ] Move `flatten_collections`, `fetch_library_data`, and related **non-view** logic out of the views module.
- [ ] **Verify:** smoke test or integration test for Zotero URLs if present; else manual `manage.py check` + import views.
- [ ] **Verify:** `check_file_lengths.py` exit 0; remove `zotero_views.py` from allowlist.

**Commit message:** `refactor(zotero): extract sync page helpers from views`

---

## Phase D â€” LLM provider

### Task D1: `lib/llm/providers/openai.py` (485 â†’ â‰¤300)

**Files:** `openai.py`, new siblings e.g. `openai_messages.py`, `openai_request.py`.

- [ ] Move **message normalization / multimodal assembly** and/or **streaming loop** out; keep `OpenAIInterface` as facade.
- [ ] **Verify:** `cd aquillm && pytest lib/llm/tests -q --tb=short`
- [ ] **Verify:** `check_import_boundaries.py` (no new `lib` â†’ `apps` imports).
- [ ] **Verify:** `check_file_lengths.py` exit 0; remove `openai.py` from allowlist.

**Commit message:** `refactor(openai): split provider module under line budget`

---

## Phase E â€” React (largest UI files)

### Task E1: `CollectionView.tsx` (655 â†’ â‰¤300)

**Files:** under `react/src/features/collections/components/` and `hooks/` or `components/sub/`.

- [ ] Split by **UI region** (header, list, modals, document subtree) and/or data hooks.
- [ ] **Verify:** `cd react && npm ci && npm run build`
- [ ] **Verify:** `check_file_lengths.py` exit 0; remove allowlist entry.

**Commit message:** `refactor(frontend): split CollectionView under line budget`

---

### Task E2: `FileSystemViewer.tsx` (631 â†’ â‰¤300)

**Files:** `react/src/features/documents/components/` subtree.

- [ ] Same strategy as E1; respect `../../../` import depth per handoff notes.
- [ ] **Verify:** `npm run build`; `check_file_lengths.py` exit 0.

**Commit message:** `refactor(frontend): split FileSystemViewer under line budget`

---

### Task E3: `UserManagementModal.tsx` (454 â†’ â‰¤300)

**Files:** `react/src/features/platform_admin/components/`.

- [ ] Extract table, forms, and API hooks.
- [ ] **Verify:** `npm run build`; `check_file_lengths.py` exit 0.

**Commit message:** `refactor(frontend): split UserManagementModal under line budget`

---

### Task E4: `CollectionsPage.tsx` (332 â†’ â‰¤300)

**Files:** `react/src/components/CollectionsPage.tsx` or move toward `features/collections/pages/`.

- [ ] Smallest React offender; extract 30+ lines of JSX or hooks.
- [ ] **Verify:** `npm run build`; `check_file_lengths.py` exit 0.

**Commit message:** `refactor(frontend): trim CollectionsPage under line budget`

---

## Phase F â€” Close-out

### Task F1: Empty the allowlist

- [ ] Confirm `_ALLOWLIST` is **empty** or only documents **intentional** permanent exceptions (ideally none).
- [ ] **Readability pass:** spot-check new modules (names, facades, import paths) against the **Structure and readability** section above; fix any â€œline-count-firstâ€ splits before calling the work done.
- [ ] **Verify:** `python scripts/check_file_lengths.py` exit **0**
- [ ] **Verify:** `python scripts/check_import_boundaries.py` exit **0**
- [ ] **Verify:** `cd aquillm && pytest apps/chat/tests apps/documents/tests apps/ingestion/tests lib/llm/tests tests/integration -q --tb=short` (with Postgres as required)
- [ ] **Verify:** `cd react && npm run build`

**Commit message:** `chore(structure): clear file-length allowlist after budget compliance`

---

## Optional doc updates (same PR or follow-up)

- [ ] Update [2026-03-24-large-file-remediation-handoff.md](./2026-03-24-large-file-remediation-handoff.md) allowlist section to match repo state.
- [ ] Note completion in [2026-03-21-architecture-remediation-commit-plan.md](../completed/2026-03-21-architecture-remediation-commit-plan.md) if all structural commits are done.

---

## Risk notes

- **Readability vs budget:** If the only way to get under 300 lines is an incoherent split, prefer an extra allowlisted submodule with a **clear name** and a thin parent facade, or defer trimming that file until a better seam appearsâ€”then document why in the PR.
- **`settings.py` split** is high-impact: use small steps and `manage.py check` after each slice.
- **Celery task names** must stay stable if workers or beat reference them by string.
- **`lib` must not import `apps.*`** â€” enforced by `check_import_boundaries.py`.
- **Channels / `ChatConsumer`**: avoid eager imports that reintroduce the `tool_wiring` cycle (see handoff: lazy `ChatConsumer` in `consumers/__init__.py`).

---

*Plan saved as `docs/roadmap/plans/active/2026-03-24-under-300-line-budget-completion.md`.*




