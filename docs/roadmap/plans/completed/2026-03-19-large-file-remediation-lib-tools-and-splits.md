# Large-File Remediation: lib/tools, Consumer Slimming, Provider Splits

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce oversized modules (`chat` consumer, LLM providers, monolithic tests, large React components) while honoring the dependency rules and layout in the codebase refactor designâ€”primarily by extracting chat LLM tools into `lib/tools/` with **injected callables**, slimming `ChatConsumer`, splitting `LLMInterface` helpers, and mirroring structure in tests and frontend.

**Architecture:** Keep **`lib/` free of Django/app imports** (`apps` â†’ `lib` only). Chat tools that need ORM access are **wired in `apps/chat/`** (consumer or small `services/` modules) by passing closures/query helpers into factories under `lib/tools/`. Shared non-Django helpers (image resize for LLM context, UUID cleanup) live in `lib/llm/utils/` or `lib/tools/...` as pure functions. `LLMInterface` in `base.py` becomes a coordinator; heavy static/assist logic moves to sibling modules or mixins. OpenAI-specific token counting moves to a dedicated small module. Tests split by concern. React splits follow `features/<domain>/components/`.

**Tech Stack:** Django, Channels, Pydantic, React/TypeScript, pytest.

**Spec:** `docs/specs/2026-03-18-codebase-refactor-design.md` (Principles, `lib/tools` layout, dependency rules Â§512â€“518, Phase 3â€“4 and 6).

---

## Validation (run frequently)

From repo root (adjust if your venv differs):

```powershell
cd aquillm
pytest apps/chat/tests lib/llm/tests -q --tb=short
```

After frontend chunks:

```powershell
cd react
npm run build
```

Full suite when touching many areas:

```powershell
cd aquillm
pytest -q --tb=short
```

---

## Chunk 1: Shared refs and consumer utilities

### Task 1.1: Add `apps/chat/refs.py`

**Files:**
- Create: `aquillm/apps/chat/refs.py`
- Modify: `aquillm/apps/chat/consumers/chat.py` (import refs from new module; remove inline classes)

- [ ] **Step 1:** Move `CollectionsRef` and `ChatRef` from `chat.py` to `refs.py` and export via `__all__`.
- [ ] **Step 2:** Update `chat.py` to `from apps.chat.refs import CollectionsRef, ChatRef`.
- [ ] **Step 3:** Run `pytest apps/chat/tests -q`.
- [ ] **Step 4:** Commit: `git add aquillm/apps/chat/refs.py aquillm/apps/chat/consumers/chat.py && git commit -m "refactor(chat): extract CollectionsRef and ChatRef to refs.py"`

### Task 1.2: Add `apps/chat/consumers/utils.py` (env, truncation, UUID, images)

**Files:**
- Create: `aquillm/apps/chat/consumers/utils.py`
- Optionally create: `aquillm/lib/llm/utils/images.py` (if helpers are pure enough to avoid Django)

Move from `chat.py`:

- `_env_int`, `CHAT_MAX_FUNC_CALLS`, `CHAT_MAX_TOKENS`, `TOOL_CHUNK_CHAR_LIMIT`, `MAX_IMAGES_PER_TOOL_RESULT`, `LLM_IMAGE_MAX_DIMENSION`, `LLM_IMAGE_MAX_BYTES` (constants can stay in `utils` or a small `constants` submoduleâ€”pick one place).
- `_truncate_tool_text`
- `_clean_and_parse_doc_id`
- `_resize_image_for_llm_context` (if it only needs Pillow/std lib, prefer `lib/llm/utils/images.py`; if it must stay next to chat-only config, keep in `consumers/utils.py`).

- [ ] **Step 1:** Extract functions; preserve signatures and behavior (copy tests if any reference private namesâ€”prefer testing via public tool behavior).
- [ ] **Step 2:** `chat.py` imports from `utils` / `lib/llm/utils/images`.
- [ ] **Step 3:** Run `pytest apps/chat/tests -q`.
- [ ] **Step 4:** Commit: `refactor(chat): move consumer helpers to utils and shared image helpers`

---

## Chunk 2: `lib/tools` â€” search and documents (injected callables)

**Rule:** Modules under `aquillm/lib/tools/` must **not** import `apps.*` or `django.*` models. They may import `aquillm.llm` types (`llm_tool`, `LLMTool`, `ToolResultDict`) and pure utilities.

### Task 2.1: Implement `lib/tools/search/vector_search.py`

**Files:**
- Create: `aquillm/lib/tools/search/vector_search.py`
- Modify: `aquillm/lib/tools/search/__init__.py` (re-export `create_vector_search_tool` or similar)
- Modify: `aquillm/apps/chat/services/` â€” **create** `aquillm/apps/chat/services/__init__.py`, `aquillm/apps/chat/services/tool_wiring.py` (or `tool_factories.py`) that imports `Collection`, `TextChunk`, etc., and builds the callables passed into lib factories

**Pattern (sketch):**

```python
# lib/tools/search/vector_search.py â€” receives callables, not ORM types
def create_vector_search_tool(
    *,
    truncate: Callable[[str], str],
    fetch_docs_and_search: Callable[..., ToolResultDict | dict],
) -> LLMTool:
    ...
```

The **wiring** module in `apps/chat/services/` implements `fetch_docs_and_search` using existing `Collection.get_user_accessible_documents`, `TextChunk.text_chunk_search`, and `col_ref.collections`.

- [ ] **Step 1:** Add failing or characterization tests in `apps/chat/tests/` that call the wired tool (existing tests may already cover `vector_search` via WebSocketâ€”run full chat tests first).
- [ ] **Step 2:** Implement lib factory + wiring; switch `chat.py` to use the factory.
- [ ] **Step 3:** `pytest apps/chat/tests -q`
- [ ] **Step 4:** Commit

### Task 2.2: `lib/tools/search/context.py` â€” `get_more_context_func` equivalent

**Files:**
- Create: `aquillm/lib/tools/search/context.py`
- Modify: wiring + `chat.py`

Same injection pattern as 2.1.

### Task 2.3: `lib/tools/documents/list_ids.py`, `whole_document.py`, `single_document.py`

**Files:**
- Create under `aquillm/lib/tools/documents/`
- Modify: `aquillm/lib/tools/documents/__init__.py`
- Modify: `apps/chat/services/tool_wiring.py`, `chat.py`

`whole_document` needs `token_count` and `convo`â€”inject `token_count_fn(convo, text)` and document fetch/permission checks via callables from wiring.

- [ ] **Step 1:** Move one tool at a time; run tests after each.
- [ ] **Step 2:** Commit per tool or per logical group.

---

## Chunk 3: `lib/tools` â€” astronomy and debug

### Task 3.1: Astronomy tools

**Files:**
- Create: `aquillm/lib/tools/astronomy/sky_subtraction.py`
- Create: `aquillm/lib/tools/astronomy/flat_fielding.py`
- Create: `aquillm/lib/tools/astronomy/point_source.py` (or a single `astronomy/tools.py` if split would be too thinâ€”keep each file **&lt;300 lines** per design target)

These may already delegate to existing modules; preserve behavior. Inject any `ChatConsumer`-specific state via narrow protocols or callables (e.g. `get_uploaded_fits_bytes`).

- [ ] **Step 1:** Extract; wire from `chat.py`.
- [ ] **Step 2:** `pytest apps/chat/tests -q`
- [ ] **Step 3:** Commit

### Task 3.2: Debug weather tool

**Files:**
- Create: `aquillm/lib/tools/debug/weather.py`
- Modify: `lib/tools/debug/__init__.py`, `chat.py`

---

## Chunk 4: Slim `ChatConsumer` â€” dedupe send/stream

### Task 4.1: Extract `_make_send_delta_func` and `_make_stream_func`

**Files:**
- Modify: `aquillm/apps/chat/consumers/chat.py`

Replace duplicated nested `send_func` / `stream_func` in `connect()` and `receive()` with methods on `ChatConsumer`, e.g.:

- `_async_send_delta(self, convo: Conversation)` â€” encapsulate save, delta JSON, `last_sent_sequence` update, logging
- `_async_stream_payload(self, payload: dict)` â€” single-line send wrapper if still needed

- [ ] **Step 1:** Refactor without behavior change; run chat tests and manual smoke if available.
- [ ] **Step 2:** Commit: `refactor(chat): dedupe WebSocket send/stream helpers in ChatConsumer`

---

## Chunk 5: `lib/llm/providers` â€” split `base.py`

**Target:** Bring `base.py` toward the **~300 line** guideline by moving **nonâ€“interface-coordination** code out.

### Task 5.1: Extract fallback / evidence helpers

**Files:**
- Create: `aquillm/lib/llm/providers/fallback_heuristics.py` (or `extractive.py`) â€” `_looks_like_deferred_tool_intent`, `_is_useful_fallback_sentence`, `_is_high_quality_summary`, `_first_sentence`, `_extractive_fallback_enabled`, and related static logic
- Create: `aquillm/lib/llm/providers/tool_evidence.py` â€” `_select_evidence_snippet`, `_extract_recent_tool_evidence`, snippet selection helpers
- Modify: `aquillm/lib/llm/providers/base.py` â€” import helpers; keep `LLMInterface` methods thin delegators

### Task 5.2: Extract image / markdown context helpers

**Files:**
- Create: `aquillm/lib/llm/providers/image_context.py` â€” `_sanitize_data_urls_for_llm_text`, `_serialize_tool_result_for_llm`, `_contains_markdown_image`, `_looks_like_image_display_request`, `_recent_tool_image_markdown`

### Task 5.3: Keep `spin` / `_generate_compact_tool_summary` orchestration in `base.py`

If `base.py` is still large, consider a private `aquillm/lib/llm/providers/summary.py` for the compact summary generator only.

- [ ] **Step 1:** After each extraction, run `pytest lib/llm/tests -q` and any integration tests touching LLM.
- [ ] **Step 2:** Commit in 1â€“3 commits by subsystem.

---

## Chunk 6: OpenAI provider â€” token / multimodal helpers

### Task 6.1: Add `openai_tokens.py` (or `lib/llm/utils/openai_multimodal_tokens.py`)

**Files:**
- Create: `aquillm/lib/llm/providers/openai_tokens.py`
- Modify: `aquillm/lib/llm/providers/openai.py` â€” delegate image-token overflow and related counting logic

- [ ] **Step 1:** Move code with no behavior change; run tests.
- [ ] **Step 2:** Commit

---

## Chunk 7: Split `apps/chat/tests/test_messages.py`

**Files:**
- Create (examplesâ€”adjust names to match test classes):
  - `aquillm/apps/chat/tests/test_message_adapters.py`
  - `aquillm/apps/chat/tests/test_multimodal_messages.py`
  - `aquillm/apps/chat/tests/test_tool_result_images.py`
- Modify: remove grouped tests from `test_messages.py` or delete file if empty
- Ensure `aquillm/apps/chat/tests/__init__.py` exists if needed

- [ ] **Step 1:** Move tests by class/module concern; run `pytest apps/chat/tests -q`.
- [ ] **Step 2:** Commit

---

## Chunk 8: Frontend â€” feature splits (Phase 6 alignment)

**Goal:** Move oversized files under `react/src/features/<domain>/components/` and shared bits under `react/src/shared/`, per design.

### Task 8.1: Ingestion

**Files (current â†’ target examples):**
- `react/src/features/ingestion/components/IngestRowsContainer.tsx` â€” split into container + subcomponents if still &gt;300 lines after prior refactors
- Forms already partially under `features/ingestion` â€” consolidate imports

### Task 8.2: Chat / collections / documents

**Files:**
- `react/src/components/ChatComponent.tsx` â†’ `react/src/features/chat/components/Chat.tsx` + hooks (`useChatWebSocket`, etc.)
- `react/src/components/CollectionView.tsx` â†’ split header, tree, settings per design tree
- `react/src/components/FileSystemViewer.tsx` â†’ list vs viewer split
- `react/src/components/UserManagementModal.tsx` â†’ under `features/platform_admin/`

- [ ] **Step 1:** Update `main.tsx` / route imports.
- [ ] **Step 2:** `cd react && npm run build`
- [ ] **Step 3:** Commit per feature area

---

## Chunk 9: Docs and hygiene

### Task 9.1: Update architecture or README pointers

**Files:**
- Modify: `README.md` or `docs/documents/architecture/*.md` â€” note new `lib/tools` layout and `apps/chat/services` wiring

### Task 9.2: Optional: lint rule or CI check for file length

If the team wants enforcement, add a script (e.g. `scripts/check_file_lengths.py`) excluding `migrations/`, `static/*.css`, `settings.py`â€”document in plan follow-up, not blocking this plan.

---

## Execution order (recommended)

1. Chunk 1 (refs + utils) â€” low risk, immediate line reduction in `chat.py`
2. Chunk 4 (dedupe send/stream) â€” low risk
3. Chunks 2â€“3 (`lib/tools` + wiring) â€” highest architectural value; do search/documents before astronomy
4. Chunk 5â€“6 (base/openai splits)
5. Chunk 7 (tests)
6. Chunk 8 (frontend)
7. Chunk 9 (docs)

---

## Target file structure (after completion)

Below is the **new / materially changed** layout relevant to this plan (existing repo files omitted).

```
aquillm/
â”œâ”€â”€ apps/
â”‚   â””â”€â”€ chat/
â”‚       â”œâ”€â”€ refs.py                          # NEW: CollectionsRef, ChatRef
â”‚       â”œâ”€â”€ services/                      # NEW: Django-side tool wiring
â”‚       â”‚   â”œâ”€â”€ __init__.py
â”‚       â”‚   â””â”€â”€ tool_wiring.py             # OR tool_factories.py â€” builds callables for lib/tools
â”‚       â”œâ”€â”€ consumers/
â”‚       â”‚   â”œâ”€â”€ chat.py                    # SLIM: ChatConsumer + imports only
â”‚       â”‚   â””â”€â”€ utils.py                   # NEW: env helpers, truncation, UUID parse (if not in lib)
â”‚       â””â”€â”€ tests/
â”‚           â”œâ”€â”€ test_message_adapters.py   # NEW (split from test_messages)
â”‚           â”œâ”€â”€ test_multimodal_messages.py
â”‚           â”œâ”€â”€ test_tool_result_images.py
â”‚           â””â”€â”€ test_messages.py           # REMOVED or reduced
â”œâ”€â”€ lib/
â”‚   â”œâ”€â”€ llm/
â”‚   â”‚   â”œâ”€â”€ utils/
â”‚   â”‚   â”‚   â””â”€â”€ images.py                  # NEW (optional): resize data URLs for LLM context
â”‚   â”‚   â””â”€â”€ providers/
â”‚   â”‚       â”œâ”€â”€ base.py                    # SLIM: LLMInterface orchestration
â”‚   â”‚       â”œâ”€â”€ fallback_heuristics.py     # NEW
â”‚   â”‚       â”œâ”€â”€ tool_evidence.py           # NEW
â”‚   â”‚       â”œâ”€â”€ image_context.py           # NEW
â”‚   â”‚       â”œâ”€â”€ summary.py                 # NEW (optional)
â”‚   â”‚       â”œâ”€â”€ openai.py                  # SLIM
â”‚   â”‚       â””â”€â”€ openai_tokens.py           # NEW: multimodal / overflow token logic
â”‚   â””â”€â”€ tools/
â”‚       â”œâ”€â”€ search/
â”‚       â”‚   â”œâ”€â”€ __init__.py                # EXPORTS
â”‚       â”‚   â”œâ”€â”€ vector_search.py           # NEW
â”‚       â”‚   â””â”€â”€ context.py                 # NEW: more-context tool
â”‚       â”œâ”€â”€ documents/
â”‚       â”‚   â”œâ”€â”€ __init__.py
â”‚       â”‚   â”œâ”€â”€ list_ids.py                # NEW: document id listing
â”‚       â”‚   â”œâ”€â”€ whole_document.py          # NEW
â”‚       â”‚   â””â”€â”€ single_document.py         # NEW
â”‚       â”œâ”€â”€ astronomy/
â”‚       â”‚   â”œâ”€â”€ __init__.py
â”‚       â”‚   â”œâ”€â”€ sky_subtraction.py         # NEW (or grouped)
â”‚       â”‚   â”œâ”€â”€ flat_fielding.py
â”‚       â”‚   â””â”€â”€ point_source.py
â”‚       â””â”€â”€ debug/
â”‚           â”œâ”€â”€ __init__.py
â”‚           â””â”€â”€ weather.py                 # NEW

react/src/
â”œâ”€â”€ features/
â”‚   â”œâ”€â”€ chat/
â”‚   â”‚   â”œâ”€â”€ components/                    # Chat.tsx, MessageList, â€¦ (from ChatComponent split)
â”‚   â”‚   â””â”€â”€ hooks/
â”‚   â”œâ”€â”€ collections/
â”‚   â”‚   â””â”€â”€ components/                    # CollectionView parts
â”‚   â”œâ”€â”€ documents/
â”‚   â”‚   â””â”€â”€ components/                    # FileSystemViewer parts
â”‚   â”œâ”€â”€ ingestion/
â”‚   â”‚   â””â”€â”€ components/                    # IngestRowsContainer / forms
â”‚   â””â”€â”€ platform_admin/
â”‚       â””â”€â”€ components/                    # User management modals
â””â”€â”€ shared/
    â””â”€â”€ components/                        # shared modals, buttons, etc.
```

---

**Plan complete and saved to `docs/roadmap/plans/active/2026-03-19-large-file-remediation-lib-tools-and-splits.md`. Ready to execute?**




