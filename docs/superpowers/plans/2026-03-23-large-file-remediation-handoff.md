# Large-file remediation — session handoff

**Purpose:** Continue [2026-03-23-large-file-remediation-commit-plan.md](./2026-03-23-large-file-remediation-commit-plan.md) (and parent [2026-03-19-large-file-remediation-lib-tools-and-splits.md](./2026-03-19-large-file-remediation-lib-tools-and-splits.md)) with minimal rediscovery.

**Last known state:** Backend chat/tool wiring and LLM `base.py` extractions landed in the working tree; React moves, `test_messages` split, `openai_tokens` extraction, and allowlist trimming were **not** completed in that pass.

---

## Canonical plan

- **Commit-by-commit checklist:** [2026-03-23-large-file-remediation-commit-plan.md](./2026-03-23-large-file-remediation-commit-plan.md)
- **Hygiene:** one subsystem per commit; after tool/import work run `scripts/check_import_boundaries.py`; do not mix large React refactors with Python in the same commit.

---

## What is already implemented

### Chat WebSocket consumer

| Area | Location | Notes |
|------|----------|--------|
| Ref holders | `aquillm/apps/chat/refs.py` | `CollectionsRef`, `ChatRef` |
| Consumer helpers | `aquillm/apps/chat/consumers/utils.py` | Env ints, `truncate_tool_text`, `resize_image_for_llm_context` |
| Image resize (pure) | `aquillm/lib/llm/utils/images.py` | `resize_image_data_url_for_llm` |
| Slim consumer | `aquillm/apps/chat/consumers/chat.py` | `build_document_tools` / `build_astronomy_tools`; deduped `_send_conversation_delta`, `_send_stream_payload` |
| Package exports | `aquillm/apps/chat/consumers/__init__.py` | **`__getattr__` lazy `ChatConsumer`** — see *Gotchas* |

### Tool wiring (Django) + `lib/tools` (no `apps.*`)

| Layer | Path | Role |
|-------|------|------|
| Wiring package | `aquillm/apps/chat/services/tool_wiring/` | `__init__.py` (`build_*`), `documents.py`, `astronomy.py` |
| Search formatting | `aquillm/lib/tools/search/vector_search.py`, `context.py` | Chunk packing, adjacent-chunk text |
| Documents | `aquillm/lib/tools/documents/ids.py`, `list_ids.py`, `whole_document.py`, `single_document.py` | Parsing + small payloads |
| Astronomy | `aquillm/lib/tools/astronomy/*.py` | Array-level FITS math / CSV bytes |
| Debug weather | `aquillm/lib/tools/debug/weather.py` | `get_debug_weather_tool()` |

**Removed:** monolithic `aquillm/apps/chat/services/tool_wiring.py` (replaced by the package above) so `check_file_lengths.py` stays green.

### LLM provider base split

| Module | Role |
|--------|------|
| `aquillm/lib/llm/providers/fallback_heuristics.py` | Deferred-tool detection, extractive fallback, cutoff helpers, `synthesize_from_recent_tool_results` |
| `aquillm/lib/llm/providers/tool_evidence.py` | Evidence snippets for compact summary |
| `aquillm/lib/llm/providers/image_context.py` | Serialize tool results for LLM text, markdown image helpers |
| `aquillm/lib/llm/providers/summary.py` | `generate_compact_tool_summary(llm, conversation, max_tokens)` |
| `aquillm/lib/llm/providers/base.py` | Thinner `LLMInterface`: `call_tool`, `complete`, `spin`, `_continue_cutoff_response` |

### Compatibility and docs

- **`aquillm/chat/consumers.py`** — Re-exports `ChatConsumer`, refs, utils aliases (`_truncate_tool_text`, etc.), and maps legacy `get_*_func` names to `tool_wiring` factories; `get_weather_func` → `get_debug_weather_tool`.
- **`README.md`** — “Module layout” bullet for `lib/tools` + `apps/chat/services/tool_wiring/`.

### Tests

- **`aquillm/apps/chat/tests/test_messages.py`** — `_test_image_result_tool` has a docstring (required by `llm_tool`).

---

## Gotchas (read before changing imports)

1. **Circular import (`tool_wiring` ↔ `ChatConsumer`)**  
   `tool_wiring` imports `apps.chat.consumers.utils`. Loading `apps.chat.consumers` used to eagerly import `chat.py`, which imported `tool_wiring` again. **Fix:** `apps/chat/consumers/__init__.py` only imports refs and uses **`__getattr__`** to load `ChatConsumer` on demand.  
   **Do not** revert to `from .chat import ChatConsumer` at package init without breaking that cycle (e.g. move `truncate_tool_text` out of `consumers/` or keep lazy export).

2. **`lib` must not import `apps.*`**  
   Enforced by `scripts/check_import_boundaries.py`. All ORM/user wiring stays in `tool_wiring` or the consumer.

3. **Type hints on wiring**  
   `ChatConsumer` is referenced as forward-quoted `"ChatConsumer"` in `tool_wiring/astronomy.py` and `__init__.py` to avoid importing `chat.py` from wiring at module load.

---

## Remaining work (mapped to commit plan)

| Plan commits | Work |
|--------------|------|
| **15** | Extract multimodal token / overflow helpers to `aquillm/lib/llm/providers/openai_tokens.py`; thin wrappers on `OpenAIInterface` if tests patch class methods. |
| **17** | Split `aquillm/apps/chat/tests/test_messages.py` into focused modules (plan lists `test_message_adapters.py`, `test_multimodal_messages.py`, `test_tool_result_images.py`, shrink `test_messages.py`). |
| **18–21** | React: `features/chat`, `features/collections`, documents/platform admin, optional `IngestRowsContainer` split. |
| **23** | Run `scripts/check_file_lengths.py` and **remove** allowlist entries only for files that drop **≤ 300** lines. Current hotspots still likely over budget: `chat.py` (~343), `base.py` (~395), `test_messages.py` (~1000+), `openai.py` (large). |

**Optional / already satisfied in spirit**

- **16** Compact summary: implemented as `summary.py` + slimmer `base.py` (revisit only if `base.py` is still over policy after other trims).

---

## Verification commands

From repo root unless noted.

**Structure (run after refactors touching imports or line counts):**

```powershell
python scripts/check_file_lengths.py
python scripts/check_import_boundaries.py
```

**Backend tests (full chat + LLM):**

```powershell
cd aquillm
$env:DJANGO_DEBUG='1'
$env:OPENAI_API_KEY='dummy'
$env:GEMINI_API_KEY='dummy'
python -m pytest apps/chat/tests lib/llm/tests -q --tb=short
```

**Note:** If Django points `POSTGRES_HOST` at Docker-only hostnames (e.g. `db`), DB-backed tests fail on a bare-metal shell; use Compose or local Postgres matching settings.

**Quick import sanity (no server):**

```powershell
cd aquillm
$env:DJANGO_SETTINGS_MODULE='aquillm.settings'
python -c "import django; django.setup(); from apps.chat.services.tool_wiring import build_document_tools; from apps.chat.consumers.chat import ChatConsumer; print('ok')"
```

**Frontend (after React commits):**

```powershell
cd react
npm ci
npm run build
```

---

## Suggested next session order

1. **`openai_tokens.py`** (commit 15) — keeps `openai.py` shrinking toward line budget; update tests if they patch moved symbols.
2. **`test_messages` split** (commit 17) — reduces test-file hotspot and prepares allowlist trim.
3. **React feature modules** (18–21) — isolated commits per feature; `npm run build` each time.
4. **Allowlist trim** (commit 23) — only after line counts prove files under 300.

---

## Related backlog (out of this plan’s numbered commits)

Architecture follow-ups scheduled **after** commit 23 or in parallel worktrees: [2026-03-21-architecture-remediation-commit-plan.md](./2026-03-21-architecture-remediation-commit-plan.md) commits 11–13 (ingestion `api.py`, `chunks.py`, Zotero tasks).

---

*Handoff written for continuation of large-file remediation; align commits and messages with [2026-03-23-large-file-remediation-commit-plan.md](./2026-03-23-large-file-remediation-commit-plan.md).*
