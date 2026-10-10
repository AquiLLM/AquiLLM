# G1 contextual graph identity implementation and review package

Worktree: `C:/Users/jackj/.codex/worktrees/rag-graph/AquiLLM`

Branch: `codex/rag-remediation-g1`

Pinned baseline: `940cea7aeca98bf7ca6aa366f062a69f6733cdb1`

Implementation checkpoint: `fix: require identity anchors for automatic graph merges`

## Specification review

- The immutable, exact-type `IdentityEndpointV2` and `IdentityDecisionV2` implement the specified interface. Missing identifiers are empty strings; malformed/noncanonical identifiers fail closed. Type, identifier, and version conflicts precede binding handling. Matching canonical stable identifiers or an explicit binding in the same nonnull document can be automatic. Unanchored proposals are candidates.
- Document, collection, and canonical resolvers call the shared helper before contextual proposals can join components. Existing single-document validation, source coordinate provenance, whole-component conflict checks, sparse caps, exhaustive/sparse partition requirements, and singleton source mappings remain in place. Context similarity cannot mint automatic identity under V2.
- `KG_IDENTITY_POLICY` defaults to `legacy`, accepts `contextual_v2`, and rejects other values. The three required contextual resolver names participate in effective result/configuration checksums, cluster keys, document build identity, and artifact allocation. Legacy document checksum payloads omit the added optional outcome field. Legacy collection golden checksum tests remain unchanged. Explicit custom legacy canonical namespaces remain supported, but cannot claim the contextual resolver version.
- Canonical registry uniqueness already includes resolver version and identity key; no additive namespace column was necessary. Contextual reconciliation selects contextual collection artifacts only; legacy reconciliation excludes them. Actual persisted registry replay demonstrates distinct V1/V2 identities and leaves V2 audit rows retained when switching back to legacy.
- V2 document decisions persist with explicit policy, resolver version, and candidate/rejected/automatic outcomes in the run audit. Persistence replays the captured policy against locked source mentions and the captured ontology before writing or reusing rows; checksums alone cannot promote candidates. Existing row-fingerprint and marker checks continue to apply. Legacy callers do not require the new ontology argument.
- Collection persistence retains source singleton assignments and existing replay/marker validation. Contextual canonical name/acronym candidates persist as suppressed audits and are excluded from current links. Actual authorized seed lookup sees two persisted same-name components as ambiguous; an anchored exact identifier remains resolvable. Actual contextual canonical lookup retains collection, document, artifact, and resolver-version fences and excludes candidates.
- There is no optional R4 dependency, new provider, model, package, serving authority, or deployment activation.

## Approved scope extensions and schema proposal

Original owned paths include resolution modules, configuration, builds, the new policy module/test, and the six specified existing test files. Coordinator-approved additional paths are:

- `resolution/persistence.py` for explicit decision audit and policy replay;
- `extraction/pipeline.py` for optional captured `identity_policy` in document artifact identity (default legacy);
- `tests/test_direct_seeds.py` and `tests/seed_lookup_fixtures.py` for actual persisted homonym ambiguity and anchored controls;
- `models/associations.py` and allocated migration `0012_contextual_candidate_methods.py`, depending on `0011_projection_bulk_chunk_fence`.

The shared model proposal changes both Python candidate-method guards and the matching SQL constraint. Candidate `exact_name_or_alias` and `defined_acronym` are accepted only with `canonical-resolution-v2-contextual`. Legacy and unknown versions retain the old rejection; embedding similarity remains incapable of automatic identity. ORM and direct SQL tests cover both methods, accepted contextual versions, rejected wrong versions, and non-serving status. The coordinator alone integrates/applies this proposal serially after review; it has only been applied in the task-owned disposable test environment.

Migration reverse first checks for contextual name/acronym candidate rows. Empty or legacy-only schemas can restore the legacy constraint; used schemas refuse before replacing constraints and delete no data. Configuration rollback selects retained legacy artifacts and does not relabel V2 rows.

## TDD and verification evidence

All full Django checks use the isolated synthetic harness project `aquillm-rag-g1-test`; every shell command starts `rtk`. No shared `.env`, corpus, credential, provider, development database, or production service was used.

Observed meaningful red regressions included missing contextual policy arguments/helper, unanchored automatic name merges, missing explicit persisted audit, contextual build identity retaining the legacy version, allocated artifact identity retaining the legacy checksum/version, ORM and direct SQL rejecting legitimate contextual candidate methods, unused-schema reversal failing unconditionally, forged automatic decisions passing persistence with recomputed checksums, undefined contextual acronym surfaces being rejected rather than candidate, and custom legacy namespace rejection. Test setup errors encountered along the way were repaired and are not presented as feature failure evidence.

Acceptance command:

```text
rtk proxy python scripts/run_rag_remediation_tests.py --project aquillm-rag-g1-test -- aquillm/apps/knowledge_graph/tests/test_contextual_identity_policy.py aquillm/apps/knowledge_graph/tests/test_coreference.py aquillm/apps/knowledge_graph/tests/test_coreference_sparse_partition.py aquillm/apps/knowledge_graph/tests/test_collection_resolution.py aquillm/apps/knowledge_graph/tests/test_canonical_resolution.py aquillm/apps/knowledge_graph/tests/test_build_idempotency.py aquillm/apps/knowledge_graph/tests/test_canonical_permissions.py aquillm/apps/knowledge_graph/tests/test_direct_seeds.py aquillm/apps/knowledge_graph/tests/test_seed_lookup_postgres.py aquillm/apps/knowledge_graph/tests/test_seed_lookup_resolver_postgres.py -q
```

First full acceptance after core fixes: **384 passed**, six pre-existing initialization/deprecation warnings, exit 0 (147.90 seconds). Final acceptance including custom-legacy compatibility, contextual collection persistence, and empty/legacy-only reverse-schema coverage: **387 passed**, exit 0 (141.83 seconds). A subsequent targeted run of `test_task7_artifact_identity_uses_the_shared_concrete_resolver_version` passed (1 test, exit 0): the pinned legacy checksum is `fcfd07a023b6186d86f8490b72f2c184b8cb20d64b322ebe77550536065efc6c`, explicit/default legacy identities are equal, and contextual identity differs.

The isolated command `rtk proxy python scripts/run_rag_remediation_tests.py --project aquillm-rag-g1-test --manage makemigrations --check --dry-run` exited 0 with **No changes detected**. Its existing initialization/HNSW prewarm warning is not a quality or readiness benchmark.

Local Ruff check over all changed Python paths and `rtk git diff --check` passed. Harness logs remain in the ignored task-owned `.superpowers/sdd/rag-harness/aquillm-rag-g1-test` directory.

## Quality review, limits, and handoff

Self-review inspected the final diff against each G1 requirement, including source provenance, unanchored bridges, component checks, default legacy checksums, mixed-version artifact rejection, actual persistence/replay, suppressed audit serving fences, and reversible unused schema. Unrelated formatting was removed from shared build code.

Contextual document persistence adds a deterministic replay under existing locks and existing caps; this is correctness verification, not an elapsed-time or contention benchmark. Canonical reconciliation retains the existing locking/elapsed-time limitation assigned separately to G5. Synthetic identity regressions establish behavior, not answer-quality improvement or activation readiness. Explicit alias strings in higher tiers carry no local provenance authority; document acronym evidence supplies the existing concrete binding path.

The historical `docs/superpowers/plans/2026-09-29-retrieval-graph-evaluation.md` referenced by the brief is absent at the pinned worktree. Available capacity, audit, evidence-preservation, execution-decision, testing, and G1 brief documents were read; historical files were not changed.

Independent correctness/provenance review must be arranged by the coordinator (workers may not recruit subagents). Review should focus on shared constraint/migration consistency, contextual outcome replay, source mappings, V1/V2 registry separation, and the complete permission path. No push, PR, integration, deployment, shared seeding, or quality activation was performed by this worker.
