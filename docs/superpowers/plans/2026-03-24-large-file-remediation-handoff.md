# Large-file remediation — continuation handoff (2026-03-24)

**Purpose:** Pick up [2026-03-23-large-file-remediation-commit-plan.md](./2026-03-23-large-file-remediation-commit-plan.md) and [2026-03-19-large-file-remediation-lib-tools-and-splits.md](./2026-03-19-large-file-remediation-lib-tools-and-splits.md) after commits **15** and **17** landed on `development`.

**Previous snapshot:** [2026-03-23-large-file-remediation-handoff.md](./2026-03-23-large-file-remediation-handoff.md) (superseded for “what’s done” by this file).

---

## Commits made (this session)

| Plan | Short hash | Subject |
|------|------------|---------|
| **15** | `09e3304756` | `refactor(openai): extract multimodal token and overflow helpers` |
| **17** | `2ebd2d4f9e` | `test(chat): split monolithic test_messages into focused modules` |
| **docs** | `git log` | `docs(superpowers): hand off large-file remediation after commits 15 and 17` |

Full hashes (plan commits):

- `09e33047568bb2c044dc968d98973955d6f526c9`
- `2ebd2d4f9efeeeb711109ed0a85d874ff8f958d3`

A third commit on the same branch adds this handoff; locate it with `git log --oneline -5` on `development`.

---

## What is now implemented (additions since prior handoff)

### OpenAI provider — token / overflow modules

| Module | Role |
|--------|------|
| `aquillm/lib/llm/providers/openai_tokens.py` | Env helpers, `context_reserve_tokens`, flatten/estimate, `trim_messages_for_overflow`, `preflight_trim_for_context(cls, …)` |
| `aquillm/lib/llm/providers/openai_overflow.py` | `strip_images_from_messages`, `retry_args_for_context_overflow`, `retry_args_for_timeout` |
| `aquillm/lib/llm/providers/openai.py` | Thin `OpenAIInterface` delegators; tests can still patch `_estimate_prompt_tokens` / `_trim_messages_for_overflow` on the class |

**Note:** Overflow/retry live in a second module so each file stays under the **300-line** policy (`check_file_lengths.py`).

### Chat tests — `test_messages.py` removed

| File | Role |
|------|------|
| `aquillm/apps/chat/tests/chat_message_test_support.py` | Shared fakes + `@llm_tool` stubs (`_test_document_ids`, `_test_image_result_tool`, `_FakeLLMInterface`, `_FakeTitleLLM`) — not a test module |
| `test_message_adapters.py` | Pydantic ↔ Django adapters + `build_frontend_conversation_json` |
| `test_conversation_persistence.py` | Save/load, ratings, conversation title |
| `test_multimodal_messages.py` | OpenAI fallback parsing, context overflow/retry, reserve scaling, image token estimator |
| `test_tool_result_images.py` | Tool result redaction + markdown injection after `complete()` |
| `test_llm_complete_retry.py` | Tool-use retry + max-token cutoff continuation |

**Allowlist:** `aquillm/apps/chat/tests/test_messages.py` was removed from `scripts/check_file_lengths.py`.

---

## Gotchas (unchanged — read before import churn)

1. **Circular import (`tool_wiring` ↔ `ChatConsumer`)** — `apps/chat/consumers/__init__.py` uses **`__getattr__`** for lazy `ChatConsumer`. Do not eager-import `chat.py` at package init without fixing the cycle.
2. **`lib` must not import `apps.*`** — run `python scripts/check_import_boundaries.py` after refactors.
3. **Wiring type hints** — `ChatConsumer` stays forward-quoted in `tool_wiring` where needed.

---

## Remaining work (commit plan mapping)

| Plan commits | Work |
|--------------|------|
| **18** | React: `features/chat` — `Chat.tsx`, `useChatWebSocket`, shim `ChatComponent.tsx`, `main.tsx` |
| **19** | React: `features/collections` — `CollectionView`, shim, `main.tsx` |
| **20** | React: `features/documents` + `features/platform_admin` — `FileSystemViewer`, `UserManagementModal`, shims, `main.tsx` |
| **21** | Optional: split `IngestRowsContainer.tsx` if still >300 lines |
| **22** | Docs: README / architecture pointers (if not already satisfied) |
| **23** | `python scripts/check_file_lengths.py` — remove allowlist entries **only** for paths that are ≤300 lines after splits (`chat.py`, `base.py`, `openai.py`, etc.) |

**Optional**

- **16** Revisit compact summary / `base.py` size only if still over budget after other trims.

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

**Frontend (after React commits 18–21):**

```powershell
cd react
npm ci
npm run build
```

---

## Suggested next session order

1. **React 18 → 21** — one feature area per commit; `npm run build` after each.
2. **Commit 23** — allowlist trim once `check_file_lengths.py` reports offenders only for files still over budget.
3. **Architecture backlog** (parallel or after): [2026-03-21-architecture-remediation-commit-plan.md](./2026-03-21-architecture-remediation-commit-plan.md) commits 11–13.

---

## Doc / command drift

Older plans and snippets may still reference `apps/chat/tests/test_messages.py`. Prefer:

```text
apps/chat/tests/test_message_adapters.py
apps/chat/tests/test_conversation_persistence.py
apps/chat/tests/test_multimodal_messages.py
apps/chat/tests/test_tool_result_images.py
apps/chat/tests/test_llm_complete_retry.py
```

---

*Handoff for large-file remediation continuation; align new commits with [2026-03-23-large-file-remediation-commit-plan.md](./2026-03-23-large-file-remediation-commit-plan.md).*
