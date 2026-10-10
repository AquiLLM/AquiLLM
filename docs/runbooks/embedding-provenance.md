# Stored embedding provenance

New document and conversation vectors carry a nullable `embedding_provenance`
JSON receipt. Documents migration 0007 and chat migration 0011 add the fields
without a default or backfill. Historical vectors and explicit vector fixtures
remain unknown. Copying a document's text chunks copies a valid receipt with
its vector; a copied unknown vector stays unknown. Images always use the target
document's media and a new receipt.

The ordinary facade exposes `get_embedding_result`, `get_embedding_results` and
`get_multimodal_embedding_result`. Each returns `EmbeddingResult(vector,
provenance)` after the existing dimension adaptation. Existing list-returning
APIs retain their list return types, payload fields, configured dimensions, role
handling and provider policy. Outage recovery now caps context repair attempts, stops
batch-to-item and multimodal format/text fallbacks on outages, and lets strict
KG embedding calls accept optional deadline/timeout controls. See
[embedding outage recovery](../operations/embedding-outage-recovery.md) for the
retry, fallback and deadline behavior.

Receipt schema version 1 contains:

| Field | Meaning |
| --- | --- |
| `provider`, `route`, `role` | Actual successful provider/transport and requested input role |
| `prepared_input_sha256` | SHA256 of the actual successful request input, after truncation/formatting |
| `input_transformations` | Empty or `character-truncation`; multimodal route identifies formatting |
| `declared_identity` | Configured model, optional `APP_EMBED_MODEL_REVISION` and `APP_EMBED_PRECISION`; declarations only |
| `observed_identity` | Response model echo when present; revision and precision are always null in v1 |
| `dimensions` | Raw and fitted dimensions; `none`, `pad-zero` or `truncate` adaptation |
| `raw_vector_sha256`, `vector_sha256` | Provider and fitted vector bindings, respectively |

Input digests use UTF-8 JSON with sorted keys, compact separators and unescaped
Unicode. A text input is a JSON string; an OpenAI batch hashes each prepared
element separately after resolving response indices. The native multimodal
route hashes `{input, multi_modal_data}`; the content-block route hashes its
`input` array. Neither includes model, credentials or endpoint. Vector digests
hash concatenated big-endian IEEE754 float32 values, matching pgvector storage
and surviving database round trips. Float32-zero/overflow vectors cannot carry
a valid receipt. No input text, image bytes, endpoint plaintext or credentials
are stored in the receipt.

Cohere fallback receipts name Cohere and never inherit local revision/precision
declarations. A response model echo does **not** verify checkpoint, precision,
quantization or cross-provider compatibility. Receipts describe what was sent
and received; they do not establish model authenticity. Input hashes can still
reveal predictable input by guessing, so treat receipts as internal metadata.

Document batch/per-item/image writers and conversation publication persist each
vector and receipt in the same write. `save`, `update`, `bulk_create` and
`bulk_update` validate receipt structure and vector binding. Vector replacement
without a matching receipt clears it. Non-null receipt-only partial writes are
rejected; write both fields together. Full saves of externally supplied vectors
do not infer current runtime identity. Raw SQL and explicit base-manager bypasses
are outside these guards; the audit detects mismatched or incomplete receipts.

Run this read-only bounded audit against the selected development database:

```sh
rtk proxy python aquillm/manage.py audit_embedding_provenance --limit 1000 --database default
```

It scans at most `limit + 1` rows per model (limit range 1–10000), ordered by
primary key, and emits only counts: scanned, known, unknown, invalid,
missing-vector and whether more rows exist. “Known” means a structurally valid
receipt matches the vector, not verified model identity. It makes no provider
requests, repairs, reindexing or data writes. Its bounded sample is not whole
database coverage. Existing app startup hooks still apply when invoking Django
management commands.

Do not populate historical receipts from current configuration. Re-embedding
would create new evidence and requires a separately controlled operation.
