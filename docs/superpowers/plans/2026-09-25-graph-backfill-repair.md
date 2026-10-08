# Graph Backfill Repair Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development or
> superpowers:executing-plans to implement the tasks with focused verification.

**Goal:** Complete large graph builds without truncation or permanent-failure loops,
and expose accurate progress through rebuild successors.

**Architecture:** Keep existing immutable build identities, leases and activation
fences. Add bounded admission policies and separate rebuild corpus limits from
read authorization limits; repair existing work through normal lifecycle hooks.

**Tech Stack:** Python, Django, PostgreSQL, Celery, Memgraph, Docker Compose, pytest.

**Spec:** `docs/superpowers/specs/2026-09-25-graph-backfill-repair.md`

## Global Constraints

- Document source limit: 10,000,000 characters; no source truncation.
- Collection default: 50,000 entities; large bucket: 100,000 entities.
- Canonical rebuild: 1,024 active collection artifacts, 250,000 entities,
  1,000,000 source links, 2,000,000 provenance
  rows, 5,000,000 decisions. Read authorization limits remain unchanged.
- Preserve existing checksum semantics and small-collection default identities.
- New Python files stay within 300 lines; existing reviewed maxima never increase.
- Deploy only development (149.165.150.254); main and production require separate
  explicit approval after validation. Never duplicate the all-scope backfill.
- Never print credentials, source passages or raw private logs.

## Task 1: Align document admission

Files: `extraction/limits.py`, `extraction/pipeline.py`,
`resolution/coreference.py`, `resolution/coreference_validation.py`,
`tests/test_coreference_capacity.py` under `aquillm/apps/knowledge_graph/`.

- [x] Reproduce aggregate and per-source resolution rejection below extraction's
  admitted maximum, and rejection beyond the shared maximum.
- [x] Introduce `DOCUMENT_SOURCE_MAX_CHARACTERS = 10_000_000`; use it in both stages.
- [x] Raise typed `ExtractionCapacityError(CHARACTER_LIMIT)` on overflow.
- [x] Run coreference and extraction capacity tests and a large synthetic probe.

```powershell
rtk proxy python -m pytest aquillm/apps/knowledge_graph/tests/test_coreference.py aquillm/apps/knowledge_graph/tests/test_coreference_capacity.py -q
```

## Task 2: Admit large collections and stop permanent-error successors

Files: `resolution/collection.py`, `resolution/candidate_pool.py`,
`graph/assembly.py`, `graph/recovery.py`, `services/builds.py`,
`services/collection_context_policy.py`, and focused collection capacity/context tests.

- [x] Reproduce failure with 89,052 distinct entities and assembly orphans.
- [x] Preserve default checksums while raising finite hard configuration ceilings.
- [x] Generate one lexical candidate pool at a time, preserving selection order.
- [x] Reproduce unnecessary resnapshot on permanent context errors using real requests.
- [x] Select the 100,000 bucket only when needed; respect supplied overrides.
- [x] Persist typed `collection_entity_limit` / `collection_document_limit` codes;
  report `capacity_blocked` in recovery and resnapshot only `StaleBuildError`.
- [x] Run lifecycle and context regressions with real PostgreSQL.

```python
resolver, assembly = select_capacity_configs(89_052)
assert resolver.max_entities == assembly.max_entities == 100_000
assert assembly.max_orphan_entities == 100_000
```

```powershell
rtk proxy python -m pytest aquillm/apps/knowledge_graph/tests/test_collection_capacity.py aquillm/apps/knowledge_graph/tests/test_collection_context_failures.py aquillm/apps/knowledge_graph/tests/test_collection_context_policy.py aquillm/apps/knowledge_graph/tests/test_build_idempotency.py -q
```

## Task 3: Scale canonical rebuilding and correct hook gating

Files: `resolution/canonical.py`, focused canonical capacity/checksum helpers,
`projection/runtime.py`, `tests/test_canonical_capacity.py` and projection hook tests.

- [x] Reproduce corpus-size, provenance and dense candidate budget failures.
- [x] Apply separate finite rebuild envelopes while retaining read caps.
- [x] Index edges per component, select only consumed ORM fields, and stream
  checksum records without changing serialized bytes or audit ordering.
- [x] Compare streamed and original checksum serialization and measure a synthetic
  corpus comparable to the observed 92,140 entities / 1.4 million name pairs.
- [x] Reproduce worker membership hook suppression with hook enabled/read disabled.
- [x] Use the explicit activation hook gate for membership publication.
- [x] Run canonical permissions, resolution, persistence and projection suites.

```powershell
rtk proxy python -m pytest aquillm/apps/knowledge_graph/tests/test_canonical_capacity.py aquillm/apps/knowledge_graph/tests/test_canonical_permissions.py aquillm/apps/knowledge_graph/tests/test_projection_runtime.py -q
```

## Task 4: Report effective progress

Files: `services/inspection.py`, `services/inspection_progress.py`, focused tests.

- [x] Create a root with a replaced child and a running successor; assert only
  the successor's counters contribute to current progress.
- [x] Add nested progress for effective leaves, live status, enumeration state,
  document counts, collection status counts and pending resnapshots.
- [x] Count active document/collection artifacts separately from retained historical
  activations; preserve top-level immutable audit fields.
- [x] Test bounded output, empty/incomplete enumeration, terminal failures and scoped
  successors using PostgreSQL.

## Task 5: Review, validate and roll out

- [x] Run focused tests, existing lifecycle/lease regressions and repository checks.
- [x] Review the integrated diff and ratchet down reduced legacy file budgets.
- [x] Commit, push and open a development PR; require relevant CI to pass (PR #237).
- [ ] Deploy development using its existing compose configuration; verify health,
  large-input admission, exact small-input identities and graph retrieval fixture.
- [ ] After separate explicit approval, merge tested development changes into main
  and deploy production preserving
  all runtime overrides and previous rollback image identities.
- [ ] Retry only incomplete scopes; verify valid extraction checkpoints are reused.
- [ ] Refresh canonical membership, reconcile projections and verify public health.
- [ ] Record exact revisions/check results and distinguish historical failures from
  remaining current work; restore the 30-minute monitor if work remains asynchronous.

```powershell
rtk proxy python scripts/check_file_lengths.py
rtk proxy python scripts/check_import_boundaries.py
rtk proxy powershell -ExecutionPolicy Bypass -File scripts/check_hygiene.ps1
rtk git diff --check
```

## Development validation follow-up

- [x] Reproduce and fix the remaining global 128-artifact write limit with a
  196-artifact database regression; preserve authorized read caps and fail on overflow.
- [x] Reproduce resume of committed extraction containing punctuation-only labels.
  Preserve extraction rows/fingerprints and explicitly audit excluded mentions and
  affected relations, while retaining complete coverage of valid mentions.
- [x] Verify clean-input compatibility, malformed-audit refusal, and resumed build
  completion; obtain an independent review of the follow-up changes.
- [ ] Merge the follow-up into development after CI, redeploy only 149.165.150.254,
  and complete canonical refresh plus the existing temporary retrieval fixture.
