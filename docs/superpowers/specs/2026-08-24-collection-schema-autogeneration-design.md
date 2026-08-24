# Collection Schema Autogeneration Design

**Status:** Approved for development-server implementation on 2026-08-24.

## Goal

Replace the collection schema editor's fixture-only backend with real collection-scoped schema persistence and a local asynchronous schema proposal workflow. A collection with usable text and no published schema automatically starts one bounded generation run when an editor opens the Knowledge Graph workspace. Editors can also request generation manually. Completed proposals become shared drafts and require validation plus an explicit MANAGE publish before graph extraction uses them.

## Constraints

- All inference remains inside the existing Docker deployment. No external API or cloud service may be called.
- Use the existing Redis/Celery knowledge-graph worker and local `vllm` and GLiNER2 services.
- The generated schema belongs only to one collection. Cross-collection synthesis and schema reconciliation remain separate work.
- Generation never publishes automatically and never overwrites a concurrently edited draft.
- Collection permissions remain authoritative: VIEW can inspect published data and generation status, EDIT can create/edit/generate drafts, and MANAGE can publish/discard/restore.
- Generated entity and relation labels must be evidence-backed, bounded, deterministic after model output, and validated through the existing ontology parser before becoming a draft.
- Collection text, prompts, model responses, and credentials must not be logged.
- Existing vector retrieval remains fail-open. Collections with incompatible graph ontologies may skip cross-collection graph expansion rather than weaken authorization or provenance checks.

## Data model

Add three models to `apps_collections`.

`CollectionSchemaVersion` is an immutable published snapshot with collection, positive version number, SHA-256 checksum, canonical definition JSON, linked graph `OntologyVersion`, publisher, publication timestamp, and summary. `(collection, version)` and `(collection, checksum)` are unique.

`CollectionSchemaDraft` is the collection's single shared mutable draft. It stores a UUID, optional base published version, monotonically increasing revision, canonical definition JSON, editor, and timestamps. The one-to-one collection relation enforces one active draft.

`CollectionSchemaGenerationRun` is a UUID-addressed durable operation with collection, requester, status (`queued`, `running`, `succeeded`, `failed`), source signature, base draft revision, result statistics, bounded error code, and timestamps. At most one queued/running run exists per collection. Runs never store sampled text or raw model output.

Definitions retain the current UI contract: `entities` and `relations` arrays with `key`, `origin`, `change_state`, `capabilities`, and `values`. Persistence canonicalizes ordering by key. The effective published schema is empty for a collection that has never published; it must not fall back to the `person`/`works_for` development fixture.

## API and editor behavior

Existing schema endpoints become persistent and revision-aware. Workspace, draft creation, definition mutation, validation, diff, publish, discard, history, version diff, and restore keep their current paths and DTO shapes.

Add:

- `POST /api/collection/<col_id>/schema/generate/` returning HTTP 202 and `{run_id, status, status_url}`. It is idempotent for an existing queued/running run with the same collection source signature. It returns 409 when a user-edited draft exists or the source snapshot changes during creation.
- `GET /api/collection/<col_id>/schema/generation/<run_id>/` returning `{run_id, status, error_code, statistics}` and, on success, the refreshed workspace envelope.

The React workspace adds a **Generate from collection** action and a live queued/running status. On initial load, if the collection has no published schema, no draft, no active generation, and the schema permission snapshot allows editing, it requests generation once. Polling uses bounded exponential backoff, aborts on collection change/unmount, and reloads the workspace on success. Failure leaves the editor usable and exposes a retry action.

## Generation pipeline

1. The POST endpoint snapshots a deterministic collection source signature and creates/reuses a queued run inside a transaction, publishing the Celery task on commit.
2. The task atomically claims the run, verifies `KG_SCHEMA_GENERATION_ENABLED=1`, and rechecks the source signature.
3. The sampler selects at most 32 deterministic text chunks and 48,000 total characters, balanced across completed documents. It fails with `no_collection_text` when nothing usable exists.
4. The local vLLM adapter calls only `VLLM_BASE_URL` (default `http://vllm:8000/v1`) with `VLLM_API_KEY` and `VLLM_SERVED_MODEL_NAME`. It requests strict JSON containing 2-24 entity types and 1-32 relation types with names, descriptions, aliases, direction, and valid endpoint type names.
5. Candidate names are normalized to lowercase snake case, duplicates and invalid endpoints are rejected, ordering is canonicalized, and the candidate is converted to ontology YAML and loaded through `load_ontology_yaml`.
6. Local GLiNER2 runs over the bounded sample using the candidate schema. Per-type mention counts, relation counts, mean confidence, and up to three document/chunk references are recorded as statistics. Types with no evidence are removed unless removal would invalidate every entity or relation; an invalid/empty candidate fails rather than fabricating a schema.
7. Inside a final transaction, the task rechecks source signature and draft revision, creates the shared generated draft, marks the run succeeded, and stores only aggregate statistics and source references. Concurrent edits cause `draft_conflict`.

Retries use Celery's existing knowledge-graph queue conventions. Provider unavailability and transient local inference errors retry; validation failures are terminal and use bounded public error codes.

## Publication and graph builds

Publishing validates the exact draft revision and checksum, writes the immutable `CollectionSchemaVersion`, and activates a collection-scoped `OntologyVersion`. Collection ontology versions use globally unique semantic versions and carry `collection_id` metadata. Activating one collection ontology supersedes only that collection's prior ontology and never the deployment-wide fallback ontology.

Graph build ontology resolution becomes `collection_ontology(collection_id)`: select the collection's active ontology when present, otherwise select the single deployment-wide active ontology. Document builds resolve the collection ID from the document before choosing the ontology. Publishing schedules the existing collection/document rebuild orchestration after commit, so new artifacts bind to the published collection ontology checksum.

## Configuration and deployment

Add compose passthrough for:

- `KG_SCHEMA_GENERATION_ENABLED` (default `0`)
- `KG_SCHEMA_GENERATION_MAX_CHUNKS` (default `32`)
- `KG_SCHEMA_GENERATION_MAX_CHARACTERS` (default `48000`)
- `KG_SCHEMA_GENERATION_TIMEOUT_SECONDS` (default `180`)

The development deployment will explicitly set generation enabled. Existing graph build, projection, and retrieval feature gates are enabled only after migrations, ontology activation, worker health, and a bounded smoke test pass.

## Verification

Backend tests cover permissions, true persistence, revision conflicts, validation, history/restore, generation idempotency, source changes, safe local endpoint selection, parsing/normalization, GLiNER evidence filtering, task terminal states, collection ontology selection, and rebuild enqueue after publish.

Frontend tests cover route parsing, API calls, automatic first-load generation, manual retry, polling success/failure/abort, stale collection responses, and accessible status/button behavior. A Django/React contract test verifies the two new route keys and DTOs. Deployment verification runs migrations, Django checks, targeted backend and React suites, compose rendering, worker health, and one development collection smoke run without logging document text or credentials.
