# Support-policy evidence contract (A3)

Status: disabled, pending independent evidence. The administrator manifest at
`/etc/aquillm/evidence-review-trust.json` cannot be selected by evaluator arguments,
candidate JSON, model self-assessment, or a matching artifact digest. No semantic
policy or source/window treatment is activated by this change. A2 continues to
reject arbitrary calibration IDs. Synthetic loader positives test plumbing only.

## Private review and enrollment

Use the existing evaluator and its `review_subject`, `validated_review` and
`attach_evidence` contracts. Historical observations remain inspectable using the
existing subject schema; new fields participate when present. Old observations
lacking support or exact publication evidence cannot pass the support gate.

`load_support_policy(attachment, runtime_identity=..., scope_kind=...)` takes only
`artifact` and `artifact_sha256`. The path resolves to a private JSON policy
artifact. The six-field frozen capability identifies an empirically accepted
decision policy; it is never proof of a future answer or permission to publish.
The loader rechecks enrollment and expiry on every call; callers must not cache
capabilities across revocation or runtime changes. No production caller is added.

An independently controlled administrator must review actual human provenance
out of band, then install a separately reviewed trusted-manifest change. The file
must be regular, administrator-owned and read-only to the application and corpus
uploaders. No production file is installed by this work. The parser rejects
oversized files (256 KiB), duplicate keys, unknown schema fields, malformed hashes,
boolean versions, nonfinite numbers and ambiguous artifact roles. Private pinned
artifacts are bounded to 32 MiB. Missing/malformed trust is empty and is read anew
on every use. The schema is `support-review-trust-v1`, with positive integer
`version` and separate `calibration` and `activation` SHA-to-pin maps. A pin maps
the complete artifact SHA-256 to `reviewer`, `provenance_sha256`,
`criteria_sha256`, and `subjects_sha256` (the digest of both frozen split digests).
The private provenance artifact must exist and match its pinned hash. The full
policy artifact pins observations and human labels, not just answer digests.
Enrollment is never inferred from the artifact's own reviewer name. Removing a
pin revokes it; rotation requires new artifacts, new pins and independent review.
Hash checks authenticate the enrolled bytes, not the truth of a human judgment.

Calibration JSON includes kind `independent_support_policy`, policy/decision version,
runtime/model/tokenizer/input-preparation identities, scope kind, corpus SHA-256,
reviewer and provenance artifact path/hash, timezone-aware expiry,
`predeclared_at`, `heldout_opened_at`, criteria, observations and split hashes.
Each observation binds the original primary question, actual support question,
source scope, support decision, runtime/policy/corpus identity, SDK payloads and
delivered spans, source bindings, and exact final stream/delta/persisted answer.
Persistence is read from PostgreSQL by conversation and canonical assistant
message UUID; refresh snapshots are not concatenated as token streams.
The trace retains the packet from the last actual SDK dispatch through context
teardown and snapshots its final support after publication invalidations. A newer
unsent packet cannot replace it. Support evidence/scope digests must match that
delivery, with exact payload and mapped-span digests; absent or changed bindings
block eligibility. A retry replaces the binding only at its actual dispatch.

Review labels retain quote accuracy, entailment, inference, numerical/condition
correctness, contradiction freedom and coverage, plus false completion,
unsupported answer, abstention and abstention correctness. Citation coordinates
and authorization remain separate mandatory mechanical checks. Operational
coverage and semantic support remain separate. Rank scores are never confidence.

Criteria require numeric `confidence`, positive integer `minimum_samples`,
explicit `domains`, both routine/deeper `depths`, explicit `modes`, method `wilson`, and numerical
`max_false_complete`, `max_unsupported_answer`, `min_abstention_quality`, and
`min_answer_coverage`. There are **no default acceptance thresholds**. Independent
reviewers must choose and freeze these before opening held-out outcomes.
The evaluator computes Wilson uncertainty for every split/domain/depth/arm cohort;
missing samples or abstention judgments remain inconclusive. An observed false
completion blocks acceptance even when a declared tolerance is permissive.
Current paired support/faithfulness/qualification comparisons include domain
checks; nDCG cannot substitute for these. Existing latency and reserve gates apply.

`independent_evidence.support_policy` attaches the reusable calibration file.
`independent_evidence.support_activation` separately attaches the current-run
artifact with kind `independent_support_activation`, its current Git revision,
`policy_sha256`, corpus/criteria/provenance, observations, expiry and
`subjects_sha256`. Activation pins additionally require `revision` (40 hex) and
`policy_sha256`. `support_subjects_sha256` binds sorted current report subjects.
Every report subject **and its exact human review** must occur in the separately
enrolled activation artifact, with the exact current report/code revision and
accepted policy digest. Historical calibration observations need not have the
current code revision; applicability still requires unchanged runtime, scope,
corpus and reviewed criteria. Any changed input preparation needs new calibration.
Administrator manifest versions are independent of evaluated Git revisions, so
enrollment does not mutate the evaluated source or create a circular commit pin.
No ordinary answer inherits individual certification from a calibrated policy.
Reannotation, changed evidence, final sink bytes, scope, runtime, corpus or policy
invalidates that attachment. Existing `human_operator` flags still require their
own measurement artifacts. No parallel activation authority is introduced.

## Credential-free checks and later experiment recipe

Run scoped tests only with the program's isolated wrapper/project. The targeted
CLI test executes:

```text
python aquillm/apps/chat/evals/run_evidence_quality_eval.py --backend fixture --mode baseline --split development --report artifacts/rag-remediation/a3-fixture.json --require-activation
```

It writes 40 plumbing observations and exits 2. Fixture reports are never eligible.
The public supplement has 32 synthetic cases, frozen development/heldout hashes,
and explicitly no independent human labels. It is a regression resource, not a
validation study. Neither the original frozen corpus nor its labels were changed.

A separately authorized experiment should:

1. Pin the code, exact corpus/source generations and split hashes; preserve the
   primary questions. Freeze the decision policy and independent numeric criteria
   before held-out access. Keep questions, text and reviewer packets excluded.
2. Pin answer/reranker/embedding, tokenizer/input preparation and hardware
   identities. The runtime manifest must provide `answer_digest`,
   `tokenizer_digest` and `input_preparation_digest`; missing fields yield unknown.
3. Collect the existing baseline, selection, preservation and combined arms under
   matched cold/warm, routine/deeper conditions. Preserve existing pair, action,
   context and deadline ceilings. Use fixed numeric latency targets and measured
   completion reserves; do not fill absent graph/source-generation observations
   with success flags. Those lanes must supply reviewer-bound actual observations.
4. Have independent humans label exact final sink text against actual SDK spans
   on development and untouched held-out samples. Record all listed dimensions,
   abstentions, failures and domains, including supplemental adversarial cases.
5. Enroll only artifacts passing independent review and all numerical/cohort
   checks. Rescore saved observations with the existing runner's `--observations`,
   `--reviews`, `--rollout-evidence`, four-arm `--compare`, `--targets` and
   `--operational-reports` inputs. Keep private artifacts outside public Git.
6. R5 consumes the existing activation gate now extended with A3 requirements.
   G6 supplies graph evidence; P2 supplies experiment results; P3 reconciles the
   integrated evidence. Missing real human reviews, runtime identities, held-out
   observations, lane evidence or latency/reserve measurements keep treatments off.

This recipe does not authorize provider calls, shared data access or deployment.
