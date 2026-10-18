# Fresh membership proof implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development, with independent task review and serialized shared database tests.

**Goal:** Make current graph membership validation fit the existing retrieval lifetime without accepting stale or unverified graph authority.

**Architecture:** A bounded background certifier computes the existing full membership checksum C and a PostgreSQL source-input fingerprint H in one repeatable-read snapshot. Each serving boundary freshly recomputes H and accepts a signed H/C certificate only when its native algorithm, actual key, and current projection/state identities match. All other checks remain unchanged.

**Tech Stack:** Python, Django, PostgreSQL 17, existing Celery projection queue and restricted source/state aliases.

**Spec:** docs/superpowers/specs/2026-10-13-membership-source-proof.md

## Global constraints

- Existing 15-second retrieval and 5-second graph limits, 45 source revisions, scoring limits, deadline propagation and cancellation remain unchanged.
- Both initial and final graph boundaries verify fresh membership validity. Never cache current permissions, source state, representation validity, vocabulary validity or graph generation.
- Exact SQL input predicates must match the existing five-value membership source relation, including explicit null decisions and duplicate rejection.
- The private SQL cost gate passed with identical results and 473–486ms scoped queries. No measured fingerprint or readiness proof alone counts as accepted retrieval.
- Existing signed native codec behavior remains unchanged; custom/unsupported or unavailable proofs retain the existing full derivation within its original budget.
- Feature KG_MEMBERSHIP_CERTIFICATE_ENABLED (HybridRetrievalSettings.membership_certificate_enabled) defaults off. No mutation triggers, source epochs, graph rebuild shortcut, new periodic scheduler, or unrelated role hardening.
- Root alone runs SSH, live migrations, backfill, image operations, merge and deployment. No new GitHub SSH key.
- Existing default database superuser and applications holding the projection key remain trusted. Restricted role and signature checks do not claim otherwise.
- Prefix shell commands with rtk. Work only in the existing codex/graph-development-repair worktree; other checkouts remain untouched.
- Independent PostgreSQL tests use the existing isolated local stack, serially scheduled by root. No live data writes from tests.

## Frozen interfaces

Task 1 owns projection/membership_source_fingerprint.py:

```python
@dataclass(frozen=True, slots=True)
class MembershipSourceFingerprintV1:
    collection_id: int
    input_sha256: str
    entity_count: int
    source_rows: int
    duplicate_links: bool

SOURCE_FINGERPRINT_FORMAT = "kg-membership-source-sql-v1"

def load_membership_source_fingerprints(
    *, collection_ids: tuple[int, ...], using: str, check, require_fresh: bool = True
) -> dict[int, MembershipSourceFingerprintV1]:
    ...
```

The production function supports the original 128-collection maximum, not the disposable probe's fixed collection IDs. Its SQL includes current transaction isolation in the bounded result. require_fresh=True requires READ COMMITTED; require_fresh=False requires REPEATABLE READ. Neither setting skips isolation validation. Empty requested sets return an empty map without SQL; all nonempty IDs must be exact positive signed-64 integers, sorted and unique. The query returns every requested collection, including empty source relations. Every database operation/result passes the supplied original check. A duplicate flag is returned as data so both consumers explicitly reject it.

Task 2 owns projection/membership_certificate_contracts.py and membership_certificate_store.py:

```python
def native_membership_identity(codec) -> str | None: ...
def sign_membership_certificate(payload: dict, codec) -> str: ...
def verify_membership_certificate(
    *, payload: dict, signature: str, codec, projection, fingerprint
) -> bool: ...
def publish_membership_certificate(
    *, payload: dict, signature: str, previous_signature: str | None,
    using: str, check
) -> bool: ...
```

Payload fields are exactly: format, projection_id, generation_key, collection_id,
artifact_id, schema_version, projection_version, identifier_key_version,
membership_epoch, membership_checksum, source_format, source_sha256, entity_count,
derivation_identity. UUIDs are canonical strings. Digests/signatures are lowercase
64-hex strings. Integers reject booleans. Format is membership-certificate-v1.
The identity binds the actual secret using a separate HMAC domain, native codec
version, registry generation and derivation/serialization version. Signing uses
another separate HMAC domain and canonical UTF-8 JSON. Pin native semantics and
codec fields across acceptance; never expose the secret.

Task 2 adds CollectionGraphMembershipCertificate with projection one-to-one PK,
payload JSON, signature and created/updated timestamps. Migration 0016 depends on
0015. A qualified migration-owned state function validates payload shape and
CASes the existing ready projection/state/artifact identities without changing
them. Existing certificate signature is an optimistic concurrency token:
previous_signature=None requires absence; replacement requires exact prior
signature; identical evidence is idempotent. PUBLIC has no EXECUTE; existing state
role receives only this function; source role receives SELECT. No raw source data
belongs in certificate payloads or logs.

Task 3 owns projection/membership_certification.py:

```python
def certify_ready_membership(
    *, projection_id: UUID, settings=None,
    source_using="projection_source", state_using="projection_state"
) -> bool: ...
```

Task 4 owns retrieval/membership_certificate_authority.py:

```python
def certified_membership_checksums(
    *, projections: tuple, using: str, codec, check
) -> dict[UUID, str]: ...
```

It returns only positively proven projection checksum entries, from fresh
fingerprints and verified certificates. A miss never removes present authority:
the caller performs the existing full membership derivation.

## Task 1: exact fresh SQL input fingerprint

Files: new projection/membership_source_fingerprint.py; new focused PostgreSQL
tests test_membership_source_fingerprint.py and support module only if needed.
Do not edit memberships.py or redefine its checksum.

- [x] Reproduce the missing behavior with a real PostgreSQL fixture containing
  active automatic, inactive/manual, explicit-null and empty-collection cases.
- [x] Assert exact relation parity with _load_membership_source_rows and the
  current full F; mutate relevant entity/artifact/link/canonical fields through
  raw SQL and ORM and verify changed F cannot preserve accepted H.
- [x] Include Unicode/escaping, large IDs, duplicate links, wrong requested sets,
  null versus empty values and malformed cross-scope mappings already admitted
  by the old loader. Do not add a stricter join that hides those inputs.
- [x] Implement ordered typed JSON aggregation and SHA256 with explicit UTF-8
  and format prefix; preserve every input row. No LIMIT on source entities.
- [x] Add original-check expiry before SQL, after execute/fetch and during result
  validation; confirm exact result coverage and transaction-isolation rejection.
- [x] Run focused real PostgreSQL tests and capture RED/GREEN receipts. Review
  actual statement count and compare old full derivation on the same snapshot.
- [x] Freeze source hashes and obtain independent review before dependent tasks.

## Task 2: signed certificate and protected storage

Files: new model membership_certificates.py, models/__init__.py export,
migration0016 plus private SQL module if needed, contracts and store modules,
pure tests and focused PostgreSQL storage/role tests.

- [x] Write failing controls for tampered/wrong-occurrence payloads, malformed
  types/keys, changed actual secret under the same version, unsupported/custom
  codecs, changed native functions and mid-operation identity change.
- [x] Implement exact payload validation, distinct HMAC domains and constant-time
  verification. No key material in repr, exceptions, logs or receipts.
- [x] Write failing database controls for stale projection/generation/state,
  forged shape, missing grant, source-role write denial, PUBLIC execute denial,
  concurrent prior-signature CAS and idempotent repeated valid evidence.
- [x] Implement additive model/table and narrow state publication function.
  Follow existing lock order, fixed search_path and qualified object names.
- [x] Run migration forward/backward in isolated DB, Django migration-state
  consistency and focused pure/PostgreSQL tests; freeze and review.
- [x] No live schema changes or broad role changes in this task.

Tasks 1 and 2 may proceed in parallel with disjoint files. Root serializes DB waves.

## Task 3: bounded snapshot certification and normal automation

Files: membership_certification.py, focused snapshot helper if needed,
dedicated certificate task module, minimal projection/tasks.py post-ready dispatch,
bounded maintenance management command, focused snapshot/task/command tests.

- [x] Build a real two-connection regression that changes membership inputs
  between H and F. Show READ COMMITTED can be torn; top-level RR must bind both
  computations to one snapshot.
- [x] Reject preexisting/nested source transactions. Start RR READ ONLY before
  the existing projection_database helper's first pg_settings query.
- [x] Capture ready projection, artifact/state and previous certificate signature;
  compute H and complete existing F; require F equal both expected checksums.
- [x] Recheck original operation and native identity after source transaction
  cleanup, then publish via the restricted state alias/CAS. No long source locks.
- [x] Prove a source mutation after snapshot cannot yield accepted proof against
  fresh H, and a rebuilt occurrence prevents stale CAS. Check timeout/cleanup
  and job revocation leave no new accepted proof.
- [x] Add best-effort bounded dispatch after project_generation returned ready
  and its operation closed. Certification failure must never retry staging or
  change ready=True. No dispatch when disabled or not ready.
- [x] Add bounded management command to certify current ready occurrences,
  reporting counts and typed failures only. It must not rebuild Memgraph or
  rewrite expected membership checksums.
- [x] Verify task broker publication has a finite bound, job <=300s and each
  native call <=5s or tighter inherited limit. No new periodic scheduler.
- [x] Run focused tests and independent review.

## Task 4: fresh serving proof at both existing boundaries

Files: membership_certificate_authority.py; minimal ready_scope_repository_v2.py;
existing HybridRetrievalSettings/loaders and .env.example; focused integration tests.

- [x] Add default-off KG_MEMBERSHIP_CERTIFICATE_ENABLED with existing strict bool
  parsing and runtime settings capture; projection config includes the same flag.
- [x] At the existing whole-authority loader, prepare fresh certificates/H in a
  bounded batch, preserving per-row version/state/artifact checks and full original
  scope checks. Skip only load_automatic_membership_assignments/checksum for a
  verified native certificate entry matching the already validated expected C.
- [x] A miss/custom/unsupported/non-PostgreSQL path retains full F. Any deadline or
  source-context identity loss rejects normally; it never renews an operation.
- [x] Test changed source/permissions/representation/vocabulary/graph generation
  between initial and final boundaries; existing shared failures must still
  invalidate all graph additions. No certificate can turn present bad authority
  into absence.
- [x] Test wrong signature/key/identity, partially returned fingerprints, duplicate
  links, pending/building authority and lazy-import/native replacements.
- [x] Verify a valid hit invokes one fresh input-fingerprint query per boundary
  and no full per-entity Python derivation, using actual DB query observations.
- [x] Run legacy/custom/off-path and graph publication/cancellation regressions;
  obtain independent review.

Task 4 can develop against frozen Tasks1/2 while Task3 certification tests run,
but shared database execution remains serial.

## Task 5: integration and guarded rollout

Root owns this task.

- [ ] Review complete diff and all receipts; update CI focused tests and run
  relevant combined suite, migration checks, file-size/logging/import guards.
- [ ] Commit/push exact reviewed source to draft PR246 with issues4/6 linked.
  Wait for all six unique exact-head CI jobs to pass.
- [ ] Update private exact-source builder allowlist only for reviewed docs/config;
  verify all committed Python files in both images. Preserve current rollback.
- [ ] Review additive migration rollout: table/function only, bounded locks,
  old app remains compatible with flag off. Existing no-schema-change deploy
  helper cannot silently accept this migration; revise and test that gate.
- [ ] On254, verify live schema/roles/worker aliases and backup; apply the reviewed
  additive migration once with feature off. Grant the isolated replay role only
  the required certificate SELECT, then verify read-only constraints.
- [ ] Certify31 existing ready occurrences through bounded maintenance. Retain
  mismatches as failures; never copy stored C as proof or force ready.
- [ ] Fresh matched off/on replay under unchanged corpus/permissions/15s/5s/source/
  pair caps. Require accepted final passages with actual graph-attributed novel
  evidence. Label transferred old questions unscored until gold is reviewed.
- [ ] Verify candidate ASGI chat, merge reviewed PR to development, SSH fast-forward
  pull and deploy immutable verified images via reviewed CPU-only phases.
- [ ] Verify public WSS, fresh source/permission fences and healthy services.
  Roll back to restored graph-off configuration on failed acceptance.
- [ ] Update PR/internal issue evidence and current-development versus historical
  call-out comparison, distinguishing operational contribution from quality gains.
