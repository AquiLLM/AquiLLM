# Collection Schema Autogeneration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace fixture schemas with persistent collection schemas and generate evidence-backed drafts asynchronously using only the local Docker services.

**Architecture:** Persist published versions, one shared draft, and durable generation runs in `apps_collections`. Reuse the existing Celery knowledge-graph queue for bounded local vLLM proposal plus GLiNER2 evidence collection, expose start/status endpoints, and let React automatically start and poll the first empty collection proposal. Publishing activates a collection-scoped ontology and schedules existing graph rebuild orchestration.

**Tech Stack:** Django 5, PostgreSQL JSON fields, Celery/Redis, local OpenAI-compatible vLLM, local GLiNER2 1.3.2, React 18, TypeScript, Vitest.

**Spec:** `docs/superpowers/specs/2026-08-24-collection-schema-autogeneration-design.md`

## Global Constraints

- All inference stays inside Docker; no external API calls.
- Generation writes a draft and never publishes or overwrites concurrent edits.
- Do not log collection text, prompts, raw model output, API keys, or credentials.
- Preserve the existing schema DTO and permission contract.
- Generation caps are exactly 32 chunks, 48,000 characters, and 180 seconds by default.
- Generated schemas contain 2-24 entities and 1-32 relations before evidence filtering.
- Collection-specific ontology activation must not supersede the global fallback or another collection's ontology.
- Use test-first development and record the expected failing test before production changes.

---

### Task 1: Persistent schema domain and collection ontology publication

**Files:**
- Create: `aquillm/apps/collections/models/schema.py`
- Create: `aquillm/apps/collections/migrations/0002_collection_schema.py`
- Create: `aquillm/apps/collections/services/schema.py`
- Modify: `aquillm/apps/collections/models/__init__.py`
- Replace: `aquillm/apps/collections/views/schema_api.py`
- Modify: `aquillm/apps/collections/views/schema_api_helpers.py`
- Modify: `aquillm/apps/knowledge_graph/services/ontology.py`
- Modify: `aquillm/apps/knowledge_graph/services/builds.py`
- Modify: `aquillm/aquillm/api_views.py`
- Modify: `aquillm/aquillm/context_processors.py`
- Test: `aquillm/apps/collections/tests/test_schema_models.py`
- Test: `aquillm/apps/collections/tests/test_schema_api.py`
- Test: `aquillm/apps/knowledge_graph/tests/test_collection_ontology_selection.py`

**Interfaces:**
- Produces models `CollectionSchemaVersion`, `CollectionSchemaDraft`, and `CollectionSchemaGenerationRun` with the fields and constraints in the spec.
- Produces `workspace_envelope(collection, user)`, `validate_draft(collection, draft_id, revision)`, `publish_draft(collection, user, operation, revision)`, and `collection_ontology(collection_id)`.
- `CollectionSchemaGenerationRun.Status` values are exact lowercase strings `queued`, `running`, `succeeded`, and `failed`.
- Generation lane consumes the models and `canonicalize_definitions(definitions)` plus `write_generated_draft(run_id, definitions, statistics)`.

- [ ] Write model tests proving one draft per collection, immutable version uniqueness, one active generation constraint, and no fixture schema for a new collection.
- [ ] Run the model tests and verify failure because the schema models do not exist.
- [ ] Implement the three models, migration, exports, canonical JSON/checksum helpers, and rerun the tests green.
- [ ] Write API tests proving workspace persistence, `If-Match` conflict handling, entity/relation CRUD, validation, publish, discard, history, diff, restore, and permission redaction.
- [ ] Run the API tests and verify the stub returns fixture data or loses mutations.
- [ ] Replace the stub with transaction-safe services and thin views, then rerun API tests green.
- [ ] Write ontology-selection tests proving collection A and B choose their own active ontology while a schema-less collection chooses the global fallback and global activation remains unchanged.
- [ ] Run the ontology tests and verify current `_active_ontology()` cannot select per collection.
- [ ] Implement collection-scoped ontology activation/selection, use it in document and collection build contexts, and enqueue the existing rebuild orchestration after schema publish.
- [ ] Run all three targeted backend suites and commit the lane.

### Task 2: Bounded local asynchronous generator

**Files:**
- Create: `aquillm/apps/collections/services/schema_generation.py`
- Create: `aquillm/apps/collections/tasks/schema_generation.py`
- Create: `aquillm/apps/collections/tests/test_schema_generation.py`
- Create: `aquillm/apps/collections/tests/test_schema_generation_task.py`
- Modify: `aquillm/apps/collections/tasks/__init__.py`
- Modify: `deploy/compose/base.yml`
- Modify: `deploy/compose/development.yml`
- Modify: `deploy/compose/no_gpu_dev.yml`
- Modify: `deploy/compose/production.yml`

**Interfaces:**
- Consumes Task 1's `CollectionSchemaGenerationRun`, `canonicalize_definitions`, and `write_generated_draft`.
- Produces `collection_source_signature(collection_id) -> str`, `sample_collection_chunks(collection_id, max_chunks, max_characters)`, `generate_schema_candidate(samples, client=None) -> dict`, `collect_candidate_evidence(candidate, samples, backend=None) -> tuple[dict, dict]`, `enqueue_schema_generation(run_id)`, and Celery task `generate_collection_schema_task(run_id)`.
- Calls only the normalized `VLLM_BASE_URL` whose default hostname is `vllm`; uses `VLLM_API_KEY` and `VLLM_SERVED_MODEL_NAME`; never falls back to OpenAI or another provider.
- Terminal error codes are `disabled`, `no_collection_text`, `source_changed`, `invalid_candidate`, `draft_conflict`, and `local_inference_failed`.

- [ ] Write sampler/config tests for deterministic balanced sampling, exact caps, source signatures, and rejection of non-local HTTP(S) hosts unless explicitly equal to the configured Docker service host.
- [ ] Run them and verify failure because the generation module is absent.
- [ ] Implement config, sampling, source signatures, and the strict local vLLM adapter; rerun green.
- [ ] Write candidate tests for JSON parsing, snake-case normalization, duplicate removal, endpoint validation, ontology-loader validation, and exact entity/relation bounds.
- [ ] Run and verify failure, then implement the prompt/parser/canonical candidate conversion and rerun green.
- [ ] Write GLiNER evidence tests proving zero-evidence definitions are removed, aggregate statistics contain only counts/confidence/source references, and no raw text/model response is persisted or logged.
- [ ] Run and verify failure, implement evidence collection, and rerun green.
- [ ] Write Celery task tests for atomic claim, retryable local failure, source change, draft conflict, success, and secret/text-safe logging.
- [ ] Run and verify failure, implement enqueue/task lifecycle using the existing knowledge-graph queue conventions, and rerun green.
- [ ] Add compose passthrough for the four generation variables, run compose config tests, and commit the lane.

### Task 3: React generation action and polling

**Files:**
- Modify: `react/src/features/collections/knowledgeGraph/schemaTypes.ts`
- Modify: `react/src/features/collections/knowledgeGraph/schemaApiRoutes.ts`
- Modify: `react/src/features/collections/knowledgeGraph/collectionSchemaApi.ts`
- Create: `react/src/features/collections/knowledgeGraph/schemaGenerationPolling.ts`
- Modify: `react/src/features/collections/knowledgeGraph/useCollectionSchemaEditor.ts`
- Modify: `react/src/features/collections/knowledgeGraph/CollectionKnowledgeGraphWorkspace.tsx`
- Modify: `react/src/features/collections/knowledgeGraph/schemaTestFixtures.ts`
- Test: `react/src/features/collections/knowledgeGraph/schemaGenerationPolling.test.ts`
- Test: `react/src/features/collections/knowledgeGraph/collectionSchemaApi.test.ts`
- Test: `react/src/features/collections/knowledgeGraph/useCollectionSchemaEditor.test.tsx`
- Test: `react/src/features/collections/knowledgeGraph/CollectionKnowledgeGraphWorkspace.state.test.tsx`

**Interfaces:**
- Consumes route keys `api_collection_schema_generate` and `api_collection_schema_generation_status`.
- `startGeneration(collectionId)` returns `{run_id, status, status_url}`.
- `getGenerationStatus(collectionId, runId)` returns `{run_id, status, error_code, statistics, workspace?}`.
- Produces an abortable bounded poll controller and hook values `generation`, `onGenerateSchema`, and accessible status copy.

- [ ] Add failing route/API tests for formatting POST start and GET UUID status URLs.
- [ ] Run them and verify failure due to absent route keys/methods.
- [ ] Add DTOs, route parsing, API methods, and rerun green.
- [ ] Add failing polling tests for queued/running/succeeded/failed, exponential backoff, maximum attempts, and abort.
- [ ] Run and verify failure, implement the poll controller, and rerun green.
- [ ] Add failing hook tests proving automatic generation occurs once only for an empty editable workspace, never for VIEW or an existing draft/published version, reloads on success, exposes retry after failure, and ignores stale collection responses.
- [ ] Run and verify failure, implement hook integration, and rerun green.
- [ ] Add failing workspace tests for the Generate action, disabled busy state, `aria-live` progress/failure, and manual retry.
- [ ] Run and verify failure, implement the UI with existing button/panel styles, rerun the full knowledgeGraph React suite, and commit the lane.

### Task 4: Integrate contracts, verify, push, and deploy development

**Files:**
- Modify: `aquillm/aquillm/api_views.py`
- Modify: `aquillm/aquillm/context_processors.py`
- Modify: `aquillm/apps/collections/views/schema_api.py`
- Modify: `aquillm/tests/integration/test_context_processors_urls.py`
- Modify: `docs/superpowers/contracts/collection-schema-api.md`
- Modify: development environment only on the remote host (credentials preserved and never printed)

**Interfaces:**
- Wires Task 2 enqueue/status into Task 1 views under the exact Task 3 route keys.
- Keeps POST generation idempotent and returns HTTP 202; status GET returns HTTP 200 for authorized collection members.

- [ ] Add failing Django route/contract tests for both generation endpoints and route template parameters.
- [ ] Run and verify failure, wire the endpoints/context keys and update the provisional contract, then rerun green.
- [ ] Merge the three lane commits and resolve only interface-level conflicts.
- [ ] Run Django model/migration checks, targeted collections and knowledge-graph backend tests, the full React knowledgeGraph suite, TypeScript checking, and compose rendering.
- [ ] Run a whole-branch review and fix all Critical/Important findings.
- [ ] Merge the feature branch into local `development`, push `development`, and confirm the remote commit.
- [ ] On the development host, preserve its environment file and credentials, pull `development`, set only the new non-secret generation variables, run migrations, rebuild/restart affected web/worker/frontend containers, and verify health.
- [ ] Enable `KG_SCHEMA_GENERATION_ENABLED=1`, run one bounded empty-schema collection smoke test, and confirm the run reaches a draft without emitting text or secrets to logs.
