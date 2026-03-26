# Large-file remediation â€” session handoff

> **Updated:** Large-file plan commits **15**, **17**, and **18â€“23** are on `development`. For current â€œwhatâ€™s doneâ€, allowlist state, and **code-quality / architecture next steps**, use **[2026-03-24-large-file-remediation-handoff.md](./2026-03-24-large-file-remediation-handoff.md)**.

**Purpose:** Continue [2026-03-23-large-file-remediation-commit-plan.md](./2026-03-23-large-file-remediation-commit-plan.md) (and parent [2026-03-19-large-file-remediation-lib-tools-and-splits.md](./2026-03-19-large-file-remediation-lib-tools-and-splits.md)) with minimal rediscovery.

**Last known state (historical):** Backend chat/tool wiring, LLM extractions, React `features/*` moves (chat, collections, documents, platform admin), ingestion split, README note, and allowlist refresh **have** landed â€” details in the 2026-03-24 handoff.

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
| Package exports | `aquillm/apps/chat/consumers/__init__.py` | **`__getattr__` lazy `ChatConsumer`** â€” see *Gotchas* |

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

### OpenAI provider â€” token / overflow (commit 15)

| Module | Role |
|--------|------|
| `aquillm/lib/llm/providers/openai_tokens.py` | Estimation, preflight trim, `trim_messages_for_overflow` |
| `aquillm/lib/llm/providers/openai_overflow.py` | Image strip + overflow/timeout retry args |
| `aquillm/lib/llm/providers/openai.py` | Delegating `OpenAIInterface` |

### Compatibility and docs

- **`aquillm/chat/consumers.py`** â€” Re-exports `ChatConsumer`, refs, utils aliases (`_truncate_tool_text`, etc.), and maps legacy `get_*_func` names to `tool_wiring` factories; `get_weather_func` â†’ `get_debug_weather_tool`.
- **`README.md`** â€” â€œModule layoutâ€ bullet for `lib/tools` + `apps/chat/services/tool_wiring/`.

### Tests

- **Split modules** (replaces monolithic `test_messages.py`): see [2026-03-24-large-file-remediation-handoff.md](./2026-03-24-large-file-remediation-handoff.md). Stubs live in `chat_message_test_support.py` with docstrings required by `llm_tool`.

---

## Gotchas (read before changing imports)

1. **Circular import (`tool_wiring` â†” `ChatConsumer`)**  
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
| **15** | Done â€” see 2026-03-24 handoff. |
| **17** | Done â€” see 2026-03-24 handoff. |
| **18â€“21** | React: `features/chat`, `features/collections`, documents/platform admin, optional `IngestRowsContainer` split. |
| **23** | Run `scripts/check_file_lengths.py` and **remove** allowlist entries only for files that drop **â‰¤ 300** lines. Hotspots may still include `chat.py`, `base.py`, `openai.py` until further splits. |

**Authoritative next steps:** [2026-03-24-large-file-remediation-handoff.md](./2026-03-24-large-file-remediation-handoff.md).

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

1. **React feature modules** (18â€“21) â€” isolated commits per feature; `npm run build` each time.
2. **Allowlist trim** (commit 23) â€” only after line counts prove files under 300.
3. See [2026-03-24-large-file-remediation-handoff.md](./2026-03-24-large-file-remediation-handoff.md) for verification commands and file layout.

---

## Related backlog (out of this planâ€™s numbered commits)

Architecture follow-ups scheduled **after** commit 23 or in parallel worktrees: [2026-03-21-architecture-remediation-commit-plan.md](../pending/2026-03-21-architecture-remediation-commit-plan.md) commits 11â€“13 (ingestion `api.py`, `chunks.py`, Zotero tasks).

---

*Handoff written for continuation of large-file remediation; align commits and messages with [2026-03-23-large-file-remediation-commit-plan.md](./2026-03-23-large-file-remediation-commit-plan.md).*

