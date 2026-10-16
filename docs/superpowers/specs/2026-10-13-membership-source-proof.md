# Fresh source fingerprint and membership certificate design

Status: approved for isolated implementation under the user's existing remediation authorization. Root retains exclusive live rollout ownership.

The explicit selected-collection SQL predicate was measured on 2026-10-13 in one read-only repeatable-read snapshot. Across 31 collections and 26,749 entities, all digests, counts and duplicate flags matched the original query. Original timings were 813.531/716.773/721.578ms; selected-filter timings were 472.637/486.446/477.438ms. These uncontrolled-cache measurements establish a plausible reduction in membership proof cost, not accepted graph retrieval. The implementation remains default-off until matched serving and publication tests pass. Baseline scheduling is a separate measured bottleneck.

## Decision and actual evidence

Prefer a freshly computed PostgreSQL source-input fingerprint H and a previously
verified H-to-membership-checksum C certificate. Keep both existing request-time
membership validity boundaries. Replace only Python membership reconstruction
and checksum derivation at those boundaries, not any other authority check.

The isolated cbfa full ready selector took 2601.333ms for 31 collections and
530 documents. Collection 226 contained 23509 assignments; its membership load
took 1330.492ms and checksum took 88.021ms. The serving cbfa trace instead reached
that load with only 818ms left and expired before identifier encoding. The
isolated result is not a prediction of serving latency. Two comparable full
proofs already sum to roughly 5203ms before traversal; there is no demonstrated
margin under the unchanged 5s graph allowance.

A database aggregate still reads the eligible source rows; it is not O(1).
The expected saving is fewer SQL round trips and no transfer of 23509 rows,
Python tuple/model construction, identifier HMAC work, or Python record/checksum
serialization on each request. The private SQL probe established the cost stated above. Scope predicate compilation is a separate root-owned
investigation; this design does not assume its result.

## Why the trigger proposals are abandoned

A global source epoch, including a set of global per-table epochs, makes
unrelated ordinary ingestion invalidate other collections. Per-table epochs do
not solve that problem for any common membership-affecting table. Collection
epochs avoid ordinary global churn but need parent mapping/fanout concurrency,
deferred-FK, delete/reinsert, TRUNCATE, role, and writer lock-order machinery.

The existing Python models intend entity/artifact/link mappings to be immutable,
but legitimate lifecycle code updates status and uses base QuerySet methods.
DB UPDATE immutability alone would not prevent a delete/reinsert of the same
parent PK under initially deferred FKs from changing a link's owner. Parent
KEY SHARE can protect parent identity against deletion, but does not conflict
with non-key artifact status updates; a status writer can still miss an entity
inserted concurrently into a new affected collection. Eliminating that race
without SHARE locks requires additional enforced cross-table ownership and
identity-reuse constraints, backfill, and writer changes. Raw child writes also
lock the child before trigger parent locks, risking reversal against the normal
parent-first resolver. These are larger changes than this recovery requires.

Actual live prerequisites now confirm PostgreSQL 17.6, READ COMMITTED,
initially deferred source FKs, a restricted projection_source SELECT role, and
default aquillm as superuser/table owner. The trigger-enforcement proposal would
therefore require an additional role trust decision. Fresh H eliminates mutation
triggers, epoch ownership discovery, global dirty periods, and new parent locks.
No epoch or trigger work belongs in the selected implementation.

## Exact proof

Let R be the eligible membership input relation, K the exact supported native
codec/key identity, and F(R,K) the current complete membership function,
including record validation, explicit null decisions and its final checksum.

A trusted bounded certifier proves, from one source snapshot:
- canonical source fingerprint H(R) = H0;
- current complete F(R,K) = C0;
- C0 equals that occurrence's current registry and projection expected checksum.

It records immutable evidence binding H0, C0, K and the expected authority
identity. At retrieval entry and again before publication, the reader obtains
fresh H(Rcurrent), the bound certificate and current expected authority facts.
If H(Rcurrent) equals H0 and every bound identity/checksum matches, collision
resistance plus deterministic native F implies the same membership checksum.
This is equivalent membership evidence, not permission to skip source, dense,
body, representation, vocabulary, artifact inputs, graph topology/provenance,
generation, authorization, or original-operation fences.

The proof assumes a trusted certifier, correct source SQL/schema/algorithm, and
the same cryptographic collision-resistance assumptions already used by the
checksum. It does not defend against a malicious superuser replacing trusted
database objects, source algorithms, or certificate evidence.

## H input and canonical framing

Use the exact current memberships._load_membership_source_rows predicates:
entity collection in the requested set and active; referenced artifact active,
scope_type collection, and evaluation_only false. Do not narrow entities to the
one advertised active artifact or the selected document subset. The current
function includes every entity satisfying these predicates, including malformed
cross-scope rows if their actual collection/predicate qualifies.

For each eligible entity, preserve entity PK, artifact resolver_version and
resolution_config_checksum, and every eligible automatic active link whose
canonical entity is active, carrying canonical PK and decision_checksum.
An entity with no eligible link has an explicit null canonical/null decision
row. LEFT JOIN a prefiltered eligible-link/canonical relation; filtering those
predicates in an outer WHERE can silently remove explicit null memberships.

Duplicate eligible links must remain detectable and rejected. No DISTINCT,
first-link selection, lossy counts, MIN/MAX substitution, or unordered sums/XOR.
Include a duplicate flag or maximum eligible-link count in the bounded result;
certification and proof use reject duplicates. Null and empty text must differ.
Preserve Unicode bytes without normalization, whitespace stripping or collation
equivalence. Use integer ordering and a typed, length-framed binary encoding with
a format/domain prefix, explicit null markers and UTF-8 text. A JSON array
encoding is acceptable only with equally explicit stable version and parity
tests; arbitrary delimiter concatenation is not.

Specify ORDER BY inside each aggregate. Outer query ordering or a sorted
subquery alone is not a proof of aggregate order. A fixed-length SHA256 of each
fully framed input row followed by ordered aggregation of those digests is an
option to reduce intermediate payload; its row count, null encoding and
duplicate semantics still need tests. The H encoding need not equal C's bytes;
it must distinguish all inputs that can alter accepted F or its rejection.

Return exactly one result for every requested collection, including an explicit
empty-relation digest and count zero. Missing result rows, duplicate collection
rows, unknown format, malformed digest/count or extra collection rows reject.
The request set remains bounded by existing ready-scope rules. Do not truncate
the entity/link relation or silently drop work to fit a limit. Retain the
original SQL deadline and fail closed on SQL/aggregate resource failure.

H can conservatively change for data that leaves F unchanged, causing extra
certification or fallback. The essential rule is that equal accepted H cannot
conceal changed F inputs. Inactive source rows need not enter H: an eligibility
change includes or removes their complete row on the next fresh query.

## Certificate identity, authentication and trust

Use a separate per-projection certificate model/table; do not overload registry_epoch or mutate
the existing membership_checksum to make certification pass. Bind at least:
- collection, projection occurrence ID and generation, artifact ID;
- projection schema/format/key version and registry epoch/expected checksum C;
- source fingerprint format/predicate version, H and input count;
- membership derivation/record/checksum version and MEMBERSHIP_REGISTRY_GENERATION;
- codec version, exact supported native semantic binding and actual key identity.

Equal key-version text is insufficient after a secret changes. Use a
domain-separated known-input HMAC fingerprint of the actual secret, and an HMAC
over a canonical certificate payload to bind every field. Separate fingerprint
and certificate domains from projection identifiers. Store neither raw secret
nor raw entity IDs/text in evidence logs; use constant-time authentication checks.
The signature detects tampering without the key and binds the payload, but is
not isolation from the live web role: web already holds the projection key and
default is a database superuser. State this boundary honestly, like the existing
trusted ready-publication path. No unrelated role-hardening project is included.

Only exact supported native codec/derivation semantics may use the certificate.
Use eager native identity checks so lazy imports cannot canonize replacements.
Pin/capture the codec fields and relevant function identity through the whole
acceptance operation. Unsupported/custom or changed semantics retain the existing
full derivation or reject within the original budget; never reuse an old proof
under changed code. Deployment algorithm versions must change when proof inputs
or transformation semantics change. A version string alone does not permit a
runtime monkeypatch to use stale native evidence.

A migration-owned state publication function with qualified objects and fixed
search_path restricts normal state-role writes to validated payload shape and a
short expected-identity CAS. PUBLIC execution is revoked. This reduces accidental
or unprivileged writes; the function cannot independently prove an application
HMAC computation merely by receiving H/C, and superuser remains trusted.

## Certifier snapshot and short publication protocol

A separate bounded maintenance operation can certify an already-ready generation;
do not call project_generation merely for this, because it claims a build lease.
The source connection is read-only and starts a new top-level REPEATABLE READ
transaction before any snapshot-bearing statement. Nested atomic blocks under
an existing READ COMMITTED transaction do not establish that guarantee.

The current projection_database helper opens atomic and executes pg_settings
before yielding. Issuing SET TRANSACTION only inside its yielded body is too late.
Establish isolation before that helper's first query, or add a narrowly owned
certificate snapshot context preserving its deadline guards. Reject an already
active source transaction rather than silently accepting a reused snapshot.

Inside that one bounded snapshot:
1. Capture exact registry/projection/artifact expected identity and native K.
2. Compute H and the existing complete F with all original validation.
3. Require F == current expected C and no duplicate/malformed source inputs.
4. Recheck original operation, codec/algorithm identity and deadline; close
   source cursor/transaction before accepting a payload for publication.

No broad source, collection, entity or artifact locks are held while scanning.
The existing 300s job and min(5s, configured shorter, job remainder, inherited)
native data-call limits apply; fixed metadata/control and transaction cleanup
retain the previously documented cooperative-boundary limitation. Recheck after
cleanup; a late or canceled result cannot publish.

Then the restricted state alias performs a short CAS in the established
collection -> artifact -> membership state -> projection order where those
locks are required, followed by the certificate row. It checks the exact
expected occurrence/generation/artifact, registry epoch, expected C, versions and
supported payload identity; it cannot update those authority facts. A competing
rebuild or newer occurrence makes the old job fail. Existing equivalent evidence
can be idempotent. Reject conflicting same-identity evidence or obsolete-version
overwrite unless a new complete bounded certification explicitly replaces it.

No source-epoch CAS is required. The source may change after the RR snapshot:
the resulting certificate remains evidence for H0, not a claim that H0 is still
current. A reader must freshly recompute H and match it. The CAS prevents stale
jobs from rewriting evidence under a new authority identity; fresh H prevents
old source evidence from being accepted after source changes. Never set a
standalone clean boolean that readers can trust without H. Conservative post-scan
H rechecks may avoid wasted publication but cannot replace the reader proof.

When complete F differs from expected C, leave the certificate unavailable and
use the normal membership advancement/rebuild lifecycle. Do not overwrite C or
call ready prematurely. An unchanged ready occurrence needs no Memgraph rebuild.

## Exact normal-worker automation seam

Use tasks.project_knowledge_graph_projection after its _run_redacted call to
project_generation has returned ready=True, before returning the existing
ready/failure result. By then the heartbeat, owned graph driver, state ContextVar
and original projection_operation are closed. Keep project_generation itself
unchanged and gate new automation on an additive flag defaulting OFF.

Best-effort bounded dispatch a separate certification task, with its own explicit
operation budget and the same source/state aliases. Dispatch failure is recorded
without propagating into _run_redacted or changing ready=True; certification
failure likewise never retries staging, marks the projection failed, or changes
the ready occurrence. Do not certify inside _project_generation: its exception
handler and post-ready cleanup deadline would make certification failure affect
a successful projection. Do not reuse its closed operation object.

The certification task uses only the new service; retries, if permitted, are
separate bounded certification attempts, never project_generation retries.
A bounded maintenance command can process existing ready occurrences and recover
missed dispatches. Do not silently add a new periodic scheduler or retry flood.
Tests must prove no enqueue for non-ready/disabled cases, exactly one normal
enqueue for ready, unchanged ready results after dispatch/certification failure,
and no source access through a closed projection operation. Broker publication
must use the existing bounded publication policy or explicitly bounded client;
an unbounded broker call would defeat the task seam.

## Serving read and fallback

Fetch certificate rows in a bounded batch and recompute all selected collections'
H in one fresh source SQL statement at each existing load_authorities_v2 call.
Existing selected-row validation and shared-corruption semantics stay intact.
This is two fresh fingerprint computations per successful native graph attempt.

Use authoritative source reads. Confirm actual isolation is READ COMMITTED or
use a dedicated fresh statement/connection; reusing a long-lived RR snapshot at
entry and final defeats mutation detection. A certificate fetch separate from
the H statement is safe only because every field is bound and the current
authority comparisons plus final proof remain; a joined statement can simplify
snapshot reasoning. Preserve callbacks/checks before execute, after results,
after context/DB cleanup, and before stage/cache/fusion/provenance publication.

Missing, stale, forged, unsupported or malformed certificates do not turn a
present projection into absence. Preserve present-pending/building/corrupt
authority handling. For initial compatible deployment, a private feature flag
defaults to the existing full derivation. When enabled, unavailable proof may
take that same bounded full path; if it cannot finish, the existing graph
fallback applies. Never extend the graph clock, refund sources, bypass the
shared failure barrier, or publish a late result. No serving recertification.

## Concrete implementation sequence and parallel ownership

0. Root: finish the read-only SQL probe before approving code. Measure all selected
   collections and 226 separately under the same source role and original
   deadline; return only counts/digests/timing. Record actual plan/resource
   behavior privately without source text. One fast warm result alone is not a
   serving improvement claim. Stop this design if both fresh H computations
   cannot plausibly fit alongside the other measured graph work.

1. Source-proof owner: new bounded source fingerprint reader and immutable
   fingerprint contract, with exact SQL predicates/framing and pure plus isolated
   PostgreSQL differential tests. No changes to current membership F. Reader
   accepts the captured original check/deadline; no new clocks or positive cache.

2. Certificate owner after the fingerprint contract freezes: certificate
   model/migration, protected state CAS, canonical payload/authentication,
   native derivation identity and bounded maintenance certifier. Build existing
   full F/H in one RR snapshot. Add focused role/CAS/deadline/snapshot controls.
   Root alone manages migration/runtime aliases and new state function grants.

3. Native integration owner after both contracts freeze: the smallest private
   seam in load_authorities_v2 plus flag/configuration and diagnostic reason.
   Both initial and final callers keep all other guards. Exact custom/legacy/
   unsupported paths remain compatible. No worker or public API redesign.

4. Independent reviewer can build adversarial parity and concurrency controls
   against the frozen contracts in parallel with integration. Review source/
   statement freshness, duplicate/null semantics, key/identity drift, certificate
   trust, deadline and final publication separately. Serialize shared DB waves.

5. Root: additive migration with feature OFF; verify schema and actual worker
   source/state aliases (web state alias is intentionally disabled at port1 in
   inspected environment). Run bounded maintenance for each existing ready
   occurrence. Inspect mismatches instead of retrying blindly or rebuilding all
   Memgraph generations. Shadow compare fresh H/C acceptance to full F on the
   same snapshot. Enable only after complete differential/integration evidence.

6. Root: exact-image source verification, normal CI and one guarded matched
   graph-on/off replay with unchanged scopes/settings/fences and original caps.
   Require actual accepted graph-attributed novel final passages and healthy
   chat/ASGI behavior; readiness/certificate presence alone is not graph quality.
   Rollback disables the new flag and uses existing full-proof/graph-off behavior;
   additive unused certificate data is harmless. Never claim benefit before
   accepted replay and the user's evaluation questions demonstrate it.

## Required meaningful tests

- Every relevant four-table I/U/D/TRUNCATE and eligibility transition changes H
  exactly when needed to detect changed F; unrelated collections remain stable.
  Include raw updates, duplicate active links, inactive canonicals, no-link
  explicit null, resolver/config mutation, malformed cross-scope artifact/entity
  mappings, empty collections, large IDs, Unicode/escaping and null versus empty.
- Differential property: equal valid fingerprint for the same native K implies
  equal accepted current F; any F rejection cannot be certified. Compare exact
  requested sets, counts and bytes without redefining the old checksum.
- Prove READ COMMITTED H/F certification can be torn with a real two-connection
  mutation between reads; the implemented top-level RR certifier stays coherent.
  Mutate after snapshot and before CAS: old evidence never accepts against fresh
  changed H. Mutate between entry and final: final publication rejects.
- No epoch-trigger assumptions: deferred parent creation/replacement, canonical
  status changes and delete/reinsert identities are observed by fresh H; no new
  parent lock requirements. ABA returning to exactly the same F inputs may
  accept, because no membership meaning changed and other identity guards remain.
- Wrong or rotated key under identical version, codec/algorithm replacement before
  lazy import or mid-operation, malformed certificate/signature/version, wrong
  collection/occurrence/generation, partial result set and obsolete CAS reject.
- Original deadline expires during SQL, after fetch, cleanup, native binding
  callbacks or final stage publication: no late proof/caches/provenance accepted.
  Existing graph-off, legacy/custom, source-body, permissions, corruption and
  explicit-null contracts remain passing.
- Query shape/count evidence confirms one bounded fingerprint query per complete
  selected proof and no per-entity Python HMAC/full membership derivation on the
  supported hit path. Certifier still runs the complete old function.

## Evidence and references

Local source: projection/memberships.py; retrieval/ready_scope_repository_v2.py;
projection/identifiers.py; projection/state_repository.py. Live inspection:
membership-certificate-live-prerequisites.log. Cost evidence:
freshscope/diagnostic-cbfa-readiness.readiness.json and diagnostic-cbfa-graph.phases.jsonl.

PostgreSQL 17 supplies bytea SHA256 and explicit UTF-8 conversion, so the proof
does not inherently require a new extension:
[Binary string functions](https://www.postgresql.org/docs/17/functions-binarystring.html).
Aggregate input order and empty/null behavior must be controlled explicitly:
[Aggregate functions](https://www.postgresql.org/docs/17/functions-aggregate.html).
READ COMMITTED and REPEATABLE READ provide different statement snapshot behavior:
[Transaction isolation](https://www.postgresql.org/docs/17/transaction-iso.html).
