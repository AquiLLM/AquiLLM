# Architecture Remediation Commit Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the remediation history easy to review, bisect, and rollback by keeping each commit narrowly scoped and test-backed.

**Source implementation plan:** `docs/roadmap/plans/pending/2026-03-21-architecture-boundary-and-structural-remediation.md`

---

## Commit 1: Ingestion WebSocket runtime move (apps path)

**Message:** `refactor(ingestion): move websocket consumers and routing to apps.ingestion`

**Files:**
- `aquillm/apps/ingestion/consumers.py` (create)
- `aquillm/apps/ingestion/routing.py` (create)
- `aquillm/ingest/consumers.py` (compat re-export)
- `aquillm/ingest/routing.py` (compat re-export)

**Verify:**
- `cd aquillm && pytest apps/ingestion/tests/test_ingest_consumers_auth.py -q`

---

## Commit 2: ASGI + URL runtime entrypoint cleanup

**Message:** `refactor(runtime): route ASGI and URL wiring through apps modules`

**Files:**
- `aquillm/aquillm/asgi.py`
- `aquillm/aquillm/urls.py`
- `aquillm/chat/views.py`
- `aquillm/tests/integration/test_context_processors_urls.py`

**Verify:**
- `cd aquillm && pytest tests/integration/test_context_processors_urls.py -q`

---

## Commit 3: Documents chunking task extraction

**Message:** `refactor(documents): extract chunking task and progress helpers into apps.documents`

**Files:**
- `aquillm/apps/documents/tasks/chunking.py` (create)
- `aquillm/apps/documents/services/chunk_progress.py` (create)
- `aquillm/apps/documents/services/image_payloads.py` (create)
- `aquillm/apps/documents/models/document.py`
- `aquillm/apps/documents/models/chunks.py`
- `aquillm/aquillm/models.py` (compat shim only)

**Verify:**
- `cd aquillm && pytest apps/documents/tests/test_multimodal_chunk_position_uniqueness.py apps/ingestion/tests/test_multimodal_ingestion_media_storage.py -q`

---

## Commit 4: Architecture import-boundary test

**Message:** `test(architecture): prevent runtime imports from aquillm.models compatibility module`

**Files:**
- `aquillm/tests/integration/test_architecture_import_boundaries.py` (create)

**Verify:**
- `cd aquillm && pytest tests/integration/test_architecture_import_boundaries.py -q`

---

## Commit 5: Chat consumer shared refs + utils extraction

**Message:** `refactor(chat): extract refs and consumer utility helpers`

**Files:**
- `aquillm/apps/chat/refs.py` (create)
- `aquillm/apps/chat/consumers/utils.py` (create)
- `aquillm/apps/chat/consumers/chat.py`

**Verify:**
- `cd aquillm && pytest apps/chat/tests -q`

---

## Commit 6: Chat tools lib extraction (search/documents)

**Message:** `refactor(chat-tools): move search/document tools into lib.tools with injected wiring`

**Files:**
- `aquillm/apps/chat/services/tool_wiring.py` (create)
- `aquillm/lib/tools/search/vector_search.py` (create)
- `aquillm/lib/tools/search/context.py` (create)
- `aquillm/lib/tools/documents/list_ids.py` (create)
- `aquillm/lib/tools/documents/whole_document.py` (create)
- `aquillm/lib/tools/documents/single_document.py` (create)
- `aquillm/apps/chat/consumers/chat.py`

**Verify:**
- `cd aquillm && pytest apps/chat/tests -q`

---

## Commit 7: Chat consumer send/stream dedupe

**Message:** `refactor(chat): dedupe websocket send/stream helpers`

**Files:**
- `aquillm/apps/chat/consumers/chat.py`

**Verify:**
- `cd aquillm && pytest apps/chat/tests -q`

---

## Commit 8: LLM provider base split

**Message:** `refactor(llm): split base provider heuristics and evidence helpers`

**Files:**
- `aquillm/lib/llm/providers/fallback_heuristics.py` (create)
- `aquillm/lib/llm/providers/tool_evidence.py` (create)
- `aquillm/lib/llm/providers/image_context.py` (create)
- `aquillm/lib/llm/providers/base.py`

**Verify:**
- `cd aquillm && pytest lib/llm/tests -q`

---

## Commit 9: OpenAI provider token helper split

**Message:** `refactor(openai): extract multimodal token and overflow helpers`

**Files:**
- `aquillm/lib/llm/providers/openai_tokens.py` (create)
- `aquillm/lib/llm/providers/openai.py`

**Verify:**
- `cd aquillm && pytest lib/llm/tests -q`

---

## Commit 10: Chat test file decomposition

**Message:** `test(chat): split monolithic test_messages into focused modules`

**Files:**
- `aquillm/apps/chat/tests/test_message_adapters.py` (create)
- `aquillm/apps/chat/tests/test_multimodal_messages.py` (create)
- `aquillm/apps/chat/tests/test_tool_result_images.py` (create)
- `aquillm/apps/chat/tests/test_messages.py` (shrink or remove)

**Verify:**
- `cd aquillm && pytest apps/chat/tests -q`

---

## Commit 11: Ingestion API decomposition

**Message:** `refactor(ingestion): split API endpoints and orchestration services`

**Files:**
- `aquillm/apps/ingestion/services/arxiv_ingest.py` (create)
- `aquillm/apps/ingestion/services/upload_batches.py` (create)
- `aquillm/apps/ingestion/services/web_ingest.py` (create)
- `aquillm/apps/ingestion/views/api/arxiv.py` (create)
- `aquillm/apps/ingestion/views/api/uploads.py` (create)
- `aquillm/apps/ingestion/views/api/web.py` (create)
- `aquillm/apps/ingestion/views/api.py` (compat entrypoint)

**Verify:**
- `cd aquillm && pytest apps/ingestion/tests/test_unified_ingestion_api.py apps/ingestion/tests/test_ingestion_monitor_includes_non_pdf.py -q`

---

## Commit 12: Document chunk model split by concern

**Message:** `refactor(documents): move chunk embedding/search/rerank logic into services`

**Files:**
- `aquillm/apps/documents/services/chunk_embeddings.py` (create)
- `aquillm/apps/documents/services/chunk_search.py` (create)
- `aquillm/apps/documents/services/chunk_rerank.py` (create)
- `aquillm/apps/documents/models/chunks.py`

**Verify:**
- `cd aquillm && pytest apps/documents/tests apps/ingestion/tests/test_unified_ingestion_parsers.py -q`

---

## Commit 13: Zotero task split and ORM write safety

**Message:** `refactor(zotero): split sync services and remove threaded ORM writes`

**Files:**
- `aquillm/apps/integrations/zotero/services/sync_collections.py` (create)
- `aquillm/apps/integrations/zotero/services/sync_items.py` (create)
- `aquillm/apps/integrations/zotero/tasks.py` (create)
- `aquillm/aquillm/zotero_tasks.py` (compat forwarding)
- `aquillm/aquillm/zotero_views.py`

**Verify:**
- `cd aquillm && pytest tests/integration/test_legacy_models_compat_module.py -q`

---

## Commit 14: Frontend feature-domain extraction (chat + collections)

**Message:** `refactor(frontend): move chat and collection views to feature domains`

**Files:**
- `react/src/features/chat/components/Chat.tsx` (create)
- `react/src/features/chat/hooks/useChatWebSocket.ts` (create)
- `react/src/features/collections/components/CollectionView.tsx` (create)
- `react/src/components/ChatComponent.tsx` (re-export)
- `react/src/components/CollectionView.tsx` (re-export)
- `react/src/main.tsx`

**Verify:**
- `cd react && npm run build`

---

## Commit 15: Frontend feature-domain extraction (documents + platform admin)

**Message:** `refactor(frontend): move filesystem and user management UI to feature domains`

**Files:**
- `react/src/features/documents/components/FileSystemViewer.tsx` (create)
- `react/src/features/platform_admin/components/UserManagementModal.tsx` (create)
- `react/src/components/FileSystemViewer.tsx` (re-export)
- `react/src/components/UserManagementModal.tsx` (re-export)
- `react/src/main.tsx`

**Verify:**
- `cd react && npm run build`

---

## Commit 16: CI structural enforcement

**Message:** `ci(structure): enforce file-size and import-boundary checks`

**Files:**
- `scripts/check_file_lengths.py` (create)
- `scripts/check_import_boundaries.py` (create)
- `.github/workflows/hygiene-check.yml` (or new `ci-structure-checks.yml`)

**Verify:**
- `python scripts/check_file_lengths.py`
- `python scripts/check_import_boundaries.py`

---

## Commit 17: Final docs and execution notes

**Message:** `docs: record architecture remediation outcomes and ownership boundaries`

**Files:**
- `README.md`
- `docs/documents/architecture/aquillm-current-architecture-mermaid.md`
- `docs/roadmap/plans/active/2026-03-21-architecture-boundary-and-structural-remediation-execution-notes.md` (create)

**Verify:**
- `cd aquillm && pytest apps/chat/tests apps/ingestion/tests apps/documents/tests lib/llm/tests tests/integration -q --tb=short`
- `cd react && npm run build`

---

## Commit Hygiene Rules

- Keep each commit focused on one subsystem boundary.
- Include at least one targeted verification command per commit.
- Do not mix backend + frontend in the same commit unless it is API contract glue.
- Prefer compatibility shims in one commit and shim removal in a later commit.
- If a commit touches >12 files, split it unless all files are one cohesive extraction.

---

**Plan complete and saved to `docs/roadmap/plans/pending/2026-03-21-architecture-remediation-commit-plan.md`.**



