# Collection Schema API Contract

**Backend:** persistent collection-scoped schema versions, draft, and generation runs

**Generation:** asynchronous on the self-hosted knowledge-graph worker

**Publish mode:** synchronous — no `api_collection_schema_publish_status` route

## Verification commands

```powershell
python -m pytest aquillm/tests/integration/test_context_processors_urls.py -q
cd react
npm test -- src/features/collections/components/CollectionModeNav.test.tsx src/features/collections/components/CollectionViewShell.test.tsx src/features/collections/knowledgeGraph
npx playwright test --config playwright.schema-editor.config.js
```

## Response bounds

| Limit | Value |
| --- | --- |
| Max response bytes | 512 KiB |
| Max JSON depth | 32 |
| Max array items (definitions, issues, versions) | 500 |
| History page size | 50 |

Oversized or malformed success payloads map to client error kind `invalid_response`.

## Endpoint table

| Capability | Django route name | `window.apiUrls` key | Method | Path kwargs | Body / query | Revision headers | Success DTO | Error DTOs | Permission | Required |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Workspace | `api_collection_schema_workspace` | `api_collection_schema_workspace` | GET | `col_id` | — | — | `CollectionSchemaEnvelope` | 401/403/404/503 JSON | VIEW+ | yes |
| Create draft | `api_collection_schema_draft` | `api_collection_schema_draft` | POST | `col_id` | `{}` | — | `CollectionSchemaEnvelope` | 401/403/409/503 | EDIT+ | yes |
| Entity upsert/remove | `api_collection_schema_entity` | `api_collection_schema_entity` | PUT/DELETE | `col_id`, `entity_key` | PUT: `{ values }` | `If-Match: <revision>` | `CollectionSchemaEnvelope` | 401/403/409/422/503 | EDIT+ | yes |
| Relation upsert/remove | `api_collection_schema_relation` | `api_collection_schema_relation` | PUT/DELETE | `col_id`, `relation_key` | PUT: `{ values }` | `If-Match: <revision>` | `CollectionSchemaEnvelope` | 401/403/409/422/503 | EDIT+ | yes |
| Validate | `api_collection_schema_validate` | `api_collection_schema_validate` | POST | `col_id` | `{ draft_id, revision }` | — | `ValidationResult` | 401/403/409/422/503 | EDIT+ | yes |
| Diff | `api_collection_schema_diff` | `api_collection_schema_diff` | GET | `col_id` | — | — | `SchemaDiffSummary` | 401/403/404/503 | VIEW+ | yes |
| Publish | `api_collection_schema_publish` | `api_collection_schema_publish` | POST | `col_id` | `{ draft_id, revision, candidate_checksum, validation_result_id }` | `If-Match: <revision>` | `CollectionSchemaEnvelope` (sync) | 401/403/409/422/503 | MANAGE | yes |
| Discard | `api_collection_schema_discard` | `api_collection_schema_discard` | POST | `col_id` | `{ draft_id, revision }` | `If-Match: <revision>` | `CollectionSchemaEnvelope` | 401/403/409/503 | MANAGE | yes |
| Versions | `api_collection_schema_versions` | `api_collection_schema_versions` | GET | `col_id` | `cursor`, `limit` (≤50) | — | `SchemaHistoryPage` | 401/403/404/503 | VIEW+ | yes |
| Version diff | `api_collection_schema_version_diff` | `api_collection_schema_version_diff` | GET | `col_id`, `version_id` | — | — | `SchemaDiffSummary` | 401/403/404/503 | VIEW+ | yes |
| Restore | `api_collection_schema_restore` | `api_collection_schema_restore` | POST | `col_id`, `version_id` | `{}` | — | `CollectionSchemaEnvelope` or 409 challenge | 401/403/404/409/503 | MANAGE | yes |
| Restore replace | `api_collection_schema_restore_replace` | `api_collection_schema_restore_replace` | POST | `col_id` | `{ version_id, challenge_token, existing_draft_revision }` | `If-Match: <existing_draft_revision>` | `CollectionSchemaEnvelope` | 401/403/409/503 | MANAGE | yes |
| Generate draft | `api_collection_schema_generate` | `api_collection_schema_generate` | POST | `col_id` | `{}` | — | `202 { run_id, status, status_url }` | 401/403/409/422/503 | EDIT+ | yes |
| Generation status | `api_collection_schema_generation_status` | `api_collection_schema_generation_status` | GET | `col_id`, `run_id` | — | — | `200 SchemaGenerationStatus` | 401/403/404/503 | VIEW+ | yes |

**Not used (sync publish):** `api_collection_schema_publish_status`

## Workspace contract notes

- VIEW responses omit `draft` and draft author metadata.
- EDIT/MANAGE responses include the shared active draft when present.
- Permissions and per-definition capabilities are authoritative; never infer from collection-page flags after load.

## Conflict contract

`409` responses include `SchemaConflictInfo` with `attempted_revision`, `current_revision`, `draft_id`, and per-definition field conflicts.

## Restore challenge contract

When restore is blocked by an existing draft, `409` returns `{ challenge_token, existing_draft_revision, existing_draft_id, last_editor }`. Atomic replacement uses `api_collection_schema_restore_replace` with the challenge token and exact existing draft revision in body and `If-Match`.

## Generation contract

- Starting generation is idempotent per collection while a queued/running run has the same source signature; repeated POSTs return that active run and a changed source returns 409.
- Generation is rejected when an editable draft already exists, so generated output never overwrites manual work.
- Status values are exactly `queued`, `running`, `succeeded`, and `failed`.
- Status returns `{ run_id, status, error_code, statistics }`; successful runs also include the current `workspace` envelope.
- The default caps are 32 chunks, 48,000 sampled characters, and 180 seconds.
- Sampling, GLiNER2 evidence collection, and schema proposal run only against local Docker services. Collection text, prompts, raw model output, and credentials are never returned or logged.
