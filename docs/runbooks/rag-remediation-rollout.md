# RAG remediation rollout candidate

This runbook describes the integrated implementation candidate. Focused rehearsal
evidence is retained at its recorded source revisions; final job records are
available in the release manifest. Review the PR before deciding to merge into
`development`; deployment requires a separate decision after review and verification.
This document does not authorize deployment, register a production embedding
profile, rebuild shared data, or activate an experimental treatment.

Use the [remediation program](../superpowers/plans/2026-10-10-rag-remediation-program.md)
and [execution decisions](rag-remediation-execution-decisions.md) together with
the final release manifest and F01–F19 closure ledger. A code fix, a data rebuild,
a passing quality gate and a deployed behavior are separate states. Historical
task test results do not establish that the final integrated candidate passes.

## Release order

| Release | Behavior | Required evidence before activation |
| --- | --- | --- |
| C1: deterministic integrity | Source artifacts and output recovery; final citation/publication checks; memory intent; bounded reranker failure | Integrated invariants, compatible schema and consumers, actual stream and persistence checks |
| C2: embedding compatibility | Exact query/document serving identity and compatible dense generations | Registered verified identity, current representation coverage, fenced workers, lexical fallback and rollback |
| C3: retained representations | New source/chunk generations, retained citations and generation-bound graph rebuilds | Populated migration, concurrency, old citation access, durable rebuild and rollback rehearsal |
| Q1: retrieval treatments | Lexical fusion, vocabulary/rewrite proposals, source/window and support treatments | Separate treatment-specific quality and operational decisions |
| Q2: graph treatments | Contextual identity, bounded topology, graph ranking and vocabulary vetoes | Absolute identity/graph gates and the shared answer/evidence gates |

Ship compatible readers and workers before allowing a generation-aware producer
to publish. Drain incompatible chunk, embedding and graph producers during the
transition. A previous binary that enumerates all retained chunks as current is
not a safe rollback target after representation history exists. Re-enable workers
only after their actual task discovery, durable delivery and schema compatibility
have been verified. The final release manifest must name the tested image and
code identities; a branch name alone is insufficient.

The private V2 frontier request checksum changes with bounded index batching.
Deploy the app and topology gateway as a tested compatible pair: drain in-flight
graph requests during replacement, then verify the versions and request checksum
before resuming them. Mixed old/new peers fail closed for graph reads; do not
bypass checksum validation to restore availability. This protocol change requires
no stored projection or index schema migration.

A protocol rollback likewise restores a tested compatible app/gateway pair;
rolling back only one peer is not sufficient. Preserve retained graph data and
all existing authority, integrity and deadline checks. Version compatibility does
not activate a treatment or satisfy its separate quality and rollout gates.

## Schema and retained data

Use Django's migration dependency plan. The program adds these chains:

- Documents: embedding generations 0009, concurrent HNSW 0010, lexical projection
  0011, concurrent GIN 0012, retained representations 0013 and history guards 0014.
  Documents 0015 records the bounded deletion lifecycle ORM state. Documents
  0013 also depends on ingestion 0004 and contenttypes 0002.
- Ingestion: immutable source artifacts 0003, then the output ledger 0004.
- Knowledge graph: contextual candidate methods 0012, then nullable captured
  representation bindings 0013 and restricted projection representation validation
  0014. Knowledge graph 0014 depends on documents 0014.
  Knowledge graph 0015 then replaces only the two projection chunk-store/coordinate
  validation joins with bounded input-driven primary-key lookups. It rewrites no data
  and retains the existing function ownership, grants, lease and coordinate checks.
- Collections: immutable vocabulary versions and explicit attachments 0004, followed
  by publication transaction provenance 0005. The database assigns each new version
  its publication transaction ID and permits entries only in that transaction.
  Reversal refuses removal of this provenance while published versions exist.
- Memory: profile-fact provenance 0003.

Inspect the complete plan on the candidate binary before applying it. Concurrent
index migrations must retain their non-atomic execution and catalog validation;
a migration record or an index name alone does not prove a usable index. Exercise
fresh and populated databases in the isolated rehearsal environment before a
deployment decision.

Normal rollback preserves used additive history. Reverse guards deliberately
refuse removal of representation, output-ledger, vocabulary or graph-binding
provenance once used. Reversing an unused disposable schema is a different test
from rolling back an application with citable history. Do not bypass those guards
or clear history to make a downgrade succeed.

Knowledge graph 0015 is independently reversible to the previous store function.
That reversal preserves stored rows and authority checks but restores the earlier
stale-statistics performance risk; it is not a performance-safe rollback promise.
Review the actual forward/reverse restricted-role and populated-load results in
the final job records before selecting a deployment rollback.

Retained source artifacts, extraction revisions, chunks and media continue to
consume storage. This program does not authorize retention cleanup. Assign an
operator to retention and a separately reviewed cleanup policy; preserve the
exact current permission checks on historical source access. Owner deletion and
bounded lifecycle cascades remain explicit operations with their existing limits.

## Bounded rebuilds

Both rebuild commands support the `default` database only. Select an exact
allowlisted concrete model, positive primary key and UUID, or use the command's
explicit bounded collection inventory. Never infer ownership from a UUID alone.

`reindex_document_representations --show-policies` lists the currently registered
chunker identities. Artifact-backed representation rebuilds require an exact
persisted parser key and a registered chunker key. An explicit extraction revision
is available only with an exact owner. Collection inventory requires an explicit
limit from 1 to 1000 and a nonnegative primary-key cursor. Artifact-backed work uses
an existing persisted extraction revision; it does not run a parser or invent
provenance for legacy rows. A nonledger `RawTextDocument` instead supports
`canonical-snapshot-v1` with its registered snapshot chunker and no extraction
revision. That snapshot retains its limited source/structure guarantees.

`reindex_document_embeddings` requires a registered lowercase 64-character
verified space key. Its inventory limit is 1 to 100, default 25; the cursor is
nonnegative. Its timeout must be 1 to 120000 milliseconds. A model name, response
echo or copied historical vector is not a verified space registration. See
[stored embedding provenance](embedding-provenance.md) for the distinction
between a receipt and verified serving identity.

Start with each command's metadata-only `--dry-run`. A representation plan does
not verify artifact bytes or build success. An embedding plan does not verify
input integrity or dense coverage. In an execution report, `queued` means durable
work was scheduled; it does not mean an embedding generation became active.
Inspect each owner's terminal state, exact manifests and compatible active
pointer before claiming coverage.

The reported `next_after_pkid` is an inventory cursor, not a success watermark.
When a batch reports excluded or failed owners, retain their exact identities
and retry them explicitly. Advancing the cursor must not silently skip failed
work. Capture a credential-free operation manifest with requested identities,
policies, limits, observed statuses and retained artifact hashes.

## Rollback boundaries

Representation rollback uses the same validated build and activation path to
create a **fresh** generation from an explicitly selected retained revision. It
does not reactivate or rewrite a retired row. In an A → B → A2 rehearsal, A2 has
its own identity and current predecessor B; old A/B chunks remain available for
their historical citations under current authorization. The source occurrence,
origin output, artifact, parser and currently registered chunker must match.

The predecessor fence begins when the actual build captures its lease. A dry-run
observation is advisory and is not a transaction fence. A different active
generation established before that capture may legitimately be the predecessor;
a replacement after capture must make the stale activation fail. Re-read actual
persisted state to identify the resulting generation.

Only policies registered by the candidate can be rebuilt. This procedure does
not promise arbitrary historic tokenizer/chunker availability. A valid partial
extraction retains its partial status. A figure remains bound to its exact
immutable output/parent/revision/media; another parser revision cannot be
substituted for that figure. Editing raw canonical text creates new source input,
not an artifact-backed rollback.

Dense rollback is separate. A compatible earlier profile may serve only the
same current representation and exact receipt/input bindings. A fresh A2 cannot
reuse A's or B's old dense rows merely because text is equal. Keep authorized
lexical retrieval available until a verified generation for A2 is ready.

Disable a rejected optional treatment through its existing configuration gate.
Preserve citation/publication checks, memory intent safeguards, current source
authorization and compatibility checks. Restore only a tested compatible binary
or configuration. Actual rollback results and their limitations must be entered
in the final release manifest before any release is described as ready.

## Quality and runtime decisions

Treatments remain off when their decision is inconclusive. Synthetic correctness
fixtures prove mechanisms and validators; they do not establish scientific answer
quality, live GPU latency, production coverage or independent human review.
Record a separate `activate`, `reject` or `inconclusive` decision for each exact
configuration, including observed rather than merely configured identities.

Independent trust is read from the administrator-controlled
`/etc/aquillm/evidence-review-trust.json`. Candidate code, report fields and command
arguments cannot enroll their own reviewer trust. No production trust manifest
is created by this program. Calibration and current-run activation require
separate pinned evidence. Missing reviews or runtime identities stay missing.

Use the existing frozen 80-case/four-arm and 30-case operational suites, the
additional graph identity and paired-query gates, and treatment-specific held-out
evidence. Count failures, timeouts and missing seeds. Do not turn one improved
tagged example into acceptance when broad or stratum outcomes regress. Preserve
all actual frontier observations; missing, multiple or unknown observations must
not be replaced by a convenient selected event.

Protocol warm-up is opt-in through `RAG_RERANK_PROTOCOL_WARM_ENABLED`. It uses
bounded per-worker startup/renewal attempts and does not attest real model
readiness or quality. Verify actual current binding readiness. The six-pair
admission limit is per process, not a fleet-wide guarantee. Measure cold/warm
latency, cancellation work and deployment capacity on the intended runtime before
making operational claims.

## User-visible behavior and final checks

Final-answer callbacks wait for complete-answer classification and validation.
With `LLM_STREAM_FINAL_ANSWER_ONLY=0`, ordinary answers therefore no longer emit
incremental final-answer tokens. Tool/progress events retain their own behavior.
The main model generates a cited answer from the delivered evidence. Usable text
survives the final access, evidence, presentation and citation checks. Unknown
semantic support remains unknown; a publication receipt and high ranking score
do not establish factual correctness or calibrated confidence. Whole-source
attributed excerpts are a fallback for unusable or citation-invalid drafts.
Failed authority checks still produce a limited answer without source content.
Evaluate generated answer correctness against the study references in addition
to checking successful publication.

Default retrieval uses the original private authority/budget carrier without
enabling source/window feature routing. Explicit human `/search` requests may
create one exact authorized document child under that same budget; automatic
tools and planners cannot inherit that expansion. Old citation endpoints may
serve retained historical bytes under current permission, while new answers
must retain their originally captured current generation.

Before the PR merge decision, bind the complete source tree to final integrated
test runs, rollback results and the closure ledger. Report pinned baseline failures
separately from new failures, warnings and tests not run; include the immutable code
revision and independent review findings. Before a separate deployment decision,
complete the runtime identities, migration/rebuild manifests, resource and cost
estimates, and retention ownership. These operational requirements remain pending;
source review or focused test passes alone do not establish a verified rollout.
