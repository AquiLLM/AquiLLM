# Source and embedding generation operations

This is a review candidate. The integrated representation/graph rollback
rehearsals are pending. These commands document the candidate's interfaces;
none has been run against a shared or production database by this program.
Use this with the [rollout runbook](rag-remediation-rollout.md).

## Select an explicit operation

Run management commands through the deployment's verified candidate image and
configuration. The examples below use `aquillm/manage.py` from the repository
root. Replace every uppercase placeholder with an inspected value. Both rebuild
commands accept only the `default` database. Retain their JSON reports privately:
they include owner identities and input hashes, though no document bodies.

Inspect registered chunker identities first:

```sh
rtk proxy python aquillm/manage.py reindex_document_representations --show-policies
```

An artifact-backed rebuild consumes a persisted immutable extraction revision.
It does not rerun extraction. The parser key must match that revision; the
chunker key must appear in `--show-policies`. Never substitute an invented
`latest` parser or a generic structural-chunker alias.

```sh
rtk proxy python aquillm/manage.py reindex_document_representations --model-label MODEL_LABEL --document-pkid OWNER_PK --document-id OWNER_UUID --parser-key PARSER_KEY --chunker-key CHUNKER_KEY --extraction-revision-id REVISION_PK --dry-run
```

A nonledger `RawTextDocument` can use `canonical-snapshot-v1` and its registered
snapshot chunker without `--extraction-revision-id`. This captures current raw
canonical text with its documented limited guarantees; it does not establish
missing artifact provenance. Missing, ambiguous or incompatible retained
inputs are exclusions to resolve explicitly.

For a bounded collection inventory, use an exact concrete model label, positive
collection ID, an explicit limit of 1–1000 and a nonnegative primary-key cursor.
An explicit extraction revision is not accepted for collection inventory.

```sh
rtk proxy python aquillm/manage.py reindex_document_representations --model-label MODEL_LABEL --collection-id COLLECTION_PK --limit 25 --after-pkid 0 --parser-key PARSER_KEY --chunker-key CHUNKER_KEY --dry-run
```

The representation report's `integrity_verified: false` is deliberate. Planning
reads metadata; actual execution validates retained bytes and the activation
fences. For an approved rebuild, repeat the same selection without `--dry-run`.
Record each owner's `activated`, `already_active` or exclusion/failure status.
The command can report a mixture of outcomes and then exit unsuccessfully.

## Build compatible dense generations separately

Register and independently verify the intended embedding space before invoking
the vector rebuild. `SPACE_KEY` is its lowercase SHA-256 key. Current model
configuration or a response model echo cannot supply that verification; see
[embedding provenance](embedding-provenance.md).

```sh
rtk proxy python aquillm/manage.py reindex_document_embeddings --model-label MODEL_LABEL --document-pkid OWNER_PK --document-id OWNER_UUID --space-key SPACE_KEY --timeout-ms 120000 --dry-run
rtk proxy python aquillm/manage.py reindex_document_embeddings --model-label MODEL_LABEL --collection-id COLLECTION_PK --limit 25 --after-pkid 0 --space-key SPACE_KEY --timeout-ms 120000 --dry-run
```

The embedding inventory accepts limits of 1–100, default 25. Its timeout range
is 1–120000 milliseconds. Exact owner and collection selection are mutually
exclusive. The report's `input_integrity_verified: false` and
`dense_coverage_verified: false` prevent interpreting metadata as verified
vectors. Removing `--dry-run` schedules eligible owners through the real worker
path. `queued` is not proof of a ready or active generation.

Wait for terminal worker outcomes and inspect the generation's exact current
representation, input manifest, serving bindings, receipts and active pointer.
Test a real authorized query against that same verified space. A populated
vector column or successful queue delivery alone is insufficient. Authorized
lexical retrieval remains the fallback when compatible dense coverage is absent.

For both commands, `next_after_pkid` is an inventory cursor, not a success
watermark. Record exclusions and failures before advancing it; retry those
owners explicitly. Do not silently lose failures between pages.

## Rehearse rollback and retention

Retain an operation manifest containing the candidate code/image identities,
model and owner identities, source/output/revision hashes, parser and chunker
keys, embedding-space verification, old and new generation IDs, requested
bounds, observed statuses, and the relevant test artifacts. Keep private
identities and evidence outside the public repository.

1. Build source generation A and a verified dense generation for A. Confirm
   current authorized retrieval, exact source slices and a usable citation.
2. Build B through the same validated representation command. Verify A's
   historical citation under current permission and B's current retrieval.
   A's dense rows must not supply B's dense results, even for equal text.
3. Select A's exact retained extraction revision and rebuild it as fresh A2.
   Preserve A/B history. Confirm A2 has a new identity and the actual current
   predecessor; rebuild dense rows for A2 before claiming dense coverage.
4. Separately change the configured verified embedding profile while a turn
   retains its original capture. Confirm stale dense results are withdrawn;
   current authorized lexical results may remain. A fresh turn may capture the
   new profile. Restore an earlier profile only where its generation still
   matches the current representation and all verified bindings.
5. Exercise failure after lease capture and after graph/lexical publication
   preparation. A concurrent replacement must fence the stale activation.
   Check persisted state and transaction rollback, rather than exception text
   alone. Exercise current permission revocation on historical citations too.

Configuration rollback preserves integrity guards. A previous binary that
treats all retained chunks as current cannot safely consume used representation
history. Used-schema reverse guards intentionally refuse destructive downgrade.
Keep old generations, artifacts and media; generation activation grants no
cleanup permission. Assign retention ownership and separately review deletion
policy, including backup and historical-citation requirements.

## Measurements and capacity planning

The retained synthetic fidelity observation covers five inputs and eight
outputs, including repeated archive members, an empty member, an unsupported
member and an uncached spreadsheet formula. Its fixture digest is
`b59290e9a5cdc89d0fe343a17872a2f0c188a452be6dc7a8a784341b26c5d220`.
All fixture expectations passed. The quality decision is `not_evaluated`.
The [measurement artifact](../audits/rag-remediation-implementation/source-index-synthetic-measurements.json)
records the exact command, tested revision, raw-log hash and all observations.

| Observed quantity | Fixed character chunks | Structural chunks |
| --- | ---: | ---: |
| Windows | 8 | 14 |
| Source byte-token units including overlap | 5791 | 5791 |
| Embedding byte-token units including context | 5791 | 6057 |
| Added structural-context byte-token units | 0 | 266 |
| Invalid source windows | 0 | 0 |
| Sum of measured chunking time, ms | 1.350017 | 2.760885 |
| Largest single-output Python allocation peak, bytes | 14163 | 19090 |

Both variants receive the same actual extracted canonical text, with configured
size 2048 and overlap 384. The tokenizer is the synthetic UTF-8 byte counter,
not a serving model tokenizer. Extraction totaled 94.85887 ms across the five
cases, with largest case Python-allocation peak 999279 bytes. Archive extraction
is shared by its four outputs and counted once. These are single-process,
single-sample observations; they are not p95 values, RSS, GPU memory, provider
cost or production throughput.

For a deployment estimate, measure bounded batches on the exact candidate
runtime and a representative approved corpus. Record observed inputs, outputs,
chunks, model-token counts, failures, provider requests, elapsed time, queue
drain time, and actual database/object/index storage deltas. Stratify by format,
size and partial extraction; preserve unsuccessful outcomes. Estimate time from
measured throughput at the intended concurrency and reserve retry capacity.
Estimate provider cost from actual requests/model tokens and the deployment's
rate schedule. No production time or price estimate is supported by this fixture.

Storage must include retained artifacts and extraction revisions, every retained
chunk/media generation, each compatible dense generation, PostgreSQL indexes
and graph projection/history. Shared content may deduplicate at some layers;
count actual persisted bytes and references instead of multiplying input size
by generation count. The 14-versus-8 window result illustrates why chunk count
and embedding-input growth need measurement before a broad rebuild.
