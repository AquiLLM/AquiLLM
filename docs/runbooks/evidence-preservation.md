# Evidence preservation: local verification and gated rollout

All preservation controls remain legacy/off. The implementation report is
[here](../superpowers/reports/2026-09-22-evidence-preservation-implementation.md).
The adaptive selector and PPR retain their own activation gates; these results do
not establish PPR quality or resolve its measured local maximum-cap overhead miss.
No live quality, deployment, human answer review or GPU measurement has run locally.

## Controls and activation order

| Control | Default | Visible effect when enabled |
| --- | --- | --- |
| `RAG_RERANK_TEXT_MODE` | `legacy` | `windowed` scores exact bounded source windows; incomplete coverage uses explicit rank fallback. |
| `RAG_EVIDENCE_TEXT_MODE` | `legacy` | `source` protects selected exact spans through provider shaping and recovery; unavailable context yields a visible limited answer. |
| `RAG_DOCUMENT_CAPACITY_MODE` | `legacy` | `budgeted` allows complementary passages beyond the legacy three per document. `RAG_DOCUMENT_HARD_CAP=0` uses the final passage cap. |
| `RAG_FOLLOWUP_EVIDENCE_ENABLED` | `0` | Resolve plural/ordinal references from validated displayed history, then load currently authorized revisions. |
| `RAG_ITERATIVE_RETRIEVAL_ENABLED` | `0` | Coverage assessment may acquire additional evidence within the same cumulative ledger. |
| `RAG_EVIDENCE_SELECTION_MODE` | `legacy` | Independent `adaptive` relevance/diversity selector; `shadow` serves legacy output. |
| `RAG_RERANK_SHADOW_SCORING_ENABLED` | `0` | Additional shadow inference only with a shared budget and explicit workload allowance. |
| `RAG_PAIR_CAPABILITY_WARM_ENABLED` | `0` | Explicit bounded worker warm inference, only with windowed or opted-in shadow scoring. |

Enable only after reviewed live evidence passes: source/windowing, capacity,
follow-ups, then iterative acquisition. Change one stage at a time and rerun the
matching comparisons. Preserve pair context 1024, current models, GPU allocation,
source revisions and provider context limits. Adaptive selection is a separate arm;
PPR is neither enabled nor imported by preservation modules.

One turn permits at most 3 acquisition actions, 45 unique sources, 250,000
materialized and 1,000,000 tokenized codepoints, 90 acquisition plus 45 reserved final
pairs, 6 in-flight pairs, 2 planner calls (2 seconds and 512 output tokens each),
and a 15-second retrieval deadline with 3-second final scoring allowance. Existing
operator action/tool limits can be stricter. The 1-second completion reserve must
be checked against measured authorization/packet p95; the deadline is a safety
ceiling, not a routine responsiveness target. Sealed synthesis has its own bounded
provider dispatch/recovery lease and retains the original ledger counters.

## Per-serving-worker pair capability

`aquillm.asgi.application` wraps the existing HTTP/WebSocket application with an
ASGI lifespan owner. Uvicorn workers initialize their own registry. Management
commands, another process, shared Redis metadata, WSGI, or a launcher disabling
lifespan do not initialize this capability. Unsupported launchers remain unknown;
use the deployed Uvicorn lifespan path before enabling windows. Startup is not
blocked on warm inference; requests arriving before verification use unknown
coverage. With all controls off there is no warm thread, tokenizer load or probe.

Mount an operator-verified attestation at `RAG_PAIR_CAPABILITY_ATTESTATION`. It is
UTF-8 JSON, at most 64 KiB, with these fields:

```json
{
  "verified_by": "named operator",
  "verification_record": "content-addressed deployment verification artifact",
  "verified_at": 1790000000,
  "expires_at": 1790000300,
  "endpoint": "http://actual-reranker:8000/score",
  "served_model": "exact APP_RERANK_MODEL value",
  "pair_context": 1024,
  "tokenizer_name": "exact APP_RERANK_TOKENIZER value",
  "template": "exact deployed chat template bytes",
  "identity": {
    "model_revision": "40 lowercase hex characters",
    "tokenizer_revision": "40 lowercase hex characters",
    "code_revision": "40 lowercase hex characters",
    "template_sha256": "SHA256 of exact UTF-8 template",
    "runtime_digest": "sha256:64 lowercase hex characters"
  }
}
```

The illustration is not usable proof. Verify actual immutable image/code/model,
tokenizer and template identities outside web workers; match the pinned
`APP_RERANK_*_REVISION` settings. Prepare tokenizer assets in the existing cache.
The initializer loads only pinned local assets, never downloads them, and checks
actual `/tokenize` counts, normal `/score` pairs and an overflow rejection. A model
name or `/models` response is insufficient. Keep attestations valid for at most
300 seconds and renew them through the existing deployment process.

Each worker has one daemon slot, no queue, a whole-attempt caller limit of 15
seconds and at most three charged score pairs. Tokenizer/HTTP/filesystem hangs can
occupy this one slot but cannot create more workers or publish after timeout.
Renewal targets 60 seconds before expiry with at least 30 seconds between attempts;
failed attempts also back off 30 seconds, and
valid attestations are checked in bounded background reads every second. Removal,
expiry, identity change, shutdown or failed renewal revokes held counters too.
The canonical verified scorer is reachable without Redis endpoint-cache priming.

Before replacing/restarting a reranker, withdraw its attestation and disable warm
and window flags, let workers invalidate, then restart serving workers if immediate
revocation is needed. Verify the replacement afresh, renew the attestation and
observe each worker's warm result. Never reuse an old attestation across a changed
endpoint runtime. Record serving-worker count and multiply warm costs accordingly.

The optional cache uses 4 bounded worker slots, no unbounded queue, a 400 ms
caller ceiling, and 150 ms Redis connection/socket options with no transport
retry. Process-local usable pointers are capped at 1024 plus 32 capability slots;
staging values are at most 1 MB and expire within 300 seconds. Late cancelled
writes may leave undiscoverable expiring objects. Cross-process reuse is reduced;
socket settings alone do not prove cold/warm cost. Reauthorization and exact
revision/query/tokenizer/template/window identities remain required for reuse.

## Quality runner and independent human review

`aquillm/apps/chat/evals/evidence_quality_cases.json` is the frozen independently
reviewed 40-development/40-heldout synthetic V5 corpus. Its full SHA256 is
`6eb8e3617ef8e07a083b33bc502d9615017a1a210997a2c333d3cd822ddc3b07`; canonical heldout
SHA256 is `01b3c9fed145c3ef46bf96552e1b4c0cc662260891af6033c114a2a2bac67395`.
See the adjacent provenance document. Do not tune using heldout answers or change
labels to match retrieval output. Mandatory recall uses `required_claims.support_ids`;
optional and scenario-expected context stays separate. nDCG stays null without
independent graded relevance labels. Model self-review never fills human judgments.

Fixture plumbing (no provider calls, always ineligible):

```text
rtk proxy python aquillm/apps/chat/evals/run_evidence_quality_eval.py --backend fixture --split development --mode combined --report artifacts/evidence/development-fixture.json
```

For live runs, configure an explicitly isolated development database/provider
environment and `RAG_EVAL_ISOLATED=1`. The runner never reads `.env`. Use synthetic
data only. `--seed` explicitly creates an inactive evaluation principal, dedicated
collections and authored chunk boundaries using real embeddings. It refuses an
existing manifest. The same manifest (including actual document/chunk/vector
bindings) must be reused across the four arms for each split. Record verified
`runtime.answer_digest`, `reranker_digest`, `embedding_digest`, `hardware_digest`
in that manifest before measured runs; missing identities block comparisons.

```text
rtk proxy python aquillm/apps/chat/evals/run_evidence_quality_eval.py --backend live --split development --mode baseline --live-manifest artifacts/evidence/dev-manifest.json --seed --concurrency 4 --cache-state cold --report artifacts/evidence/dev-baseline.json
```

After seeding, omit `--seed`; repeat with baseline, selection, preservation and
combined, matching source/model/hardware/configuration and concurrency. The runner
sets only the documented arm controls. Baseline and selection retain actual legacy
packing and provider shaping; only exact unambiguous final SDK text maps to support.
Ambiguous provenance is unknown. Use independent processes per arm. Alternate arm
order across repeated runs. Cold runs get an isolated cache namespace; warm runs
perform a real preconditioning pass in the same process/namespace, recording its
cost, then measure. Neither touches shared-user cache keys. Record physical model,
database and filesystem warm state separately. Worker warm inference is reported
separately from user-turn pairs. Capture multiple repetitions and both cache states
as external measurement artifacts; no latency claim follows merely from a label.

Answer reviews are JSON keyed by case ID with `kind: human`, named `reviewer`,
`answer_sha256` (SHA256 of exact UTF-8 answer), `answer_faithful`,
`citation_entailment`, and `claims[claim_id]`: boolean `faithful` plus every frozen
claim label (units/conditions/negation/date/etc.). Unknown/missing is not pass.
Attach reviews with `--observations saved-report.json --reviews reviews.json` to
re-score immutable observations without calling providers again.

`--rollout-evidence` attaches named `human_operator` evidence bound to exact
`revision`, `corpus_sha256` and `snapshot_digest` (canonical digest of case-ID-sorted
snapshot objects). Each proof record has `status: passed`, `reviewed_by`, `artifact`
and verified `artifact_sha256`. Required keys are `runtime_verified`,
`corpus_human_reviewed`, `deterministic_regressions_passed`, `cold_warm_verified`,
`completion_reserve_measured`; the reserve record also supplies measured
`authorization_packet_p95_ms` and `configured_reserve_ms`; the latter must equal
the actual reserve observed in every turn's comparison controls. Preserve all underlying
raw records. A named operator owns the claims; these hashes are binding checks,
not a replacement for review. Raw SDK traces contain synthetic content and stay
outside tracked/private production data.

## Operational evidence and activation-v2

`evidence_operational_cases.json` is ops-v2, full SHA256
`ae60c47e40d7fa3574d6e8fa6c32e40eba51c30121e54741316d50314f8d3136`, preserving ops-v1
parent `57c199c7b254f4a10277380ceb24130ee8574e618c31a443a9a0e7d29f222da0`. Its 168
all-authorized sources and five workloads were authored before live feedback.
The eight calibration logs provide prospective window pressure; character counts do not
establish tokenizer/window demand or guarantee exhaustion.

```text
rtk proxy python aquillm/apps/chat/evals/run_evidence_operational_eval.py --backend live --mode preservation --profile pilot --repetition 1 --live-manifest artifacts/evidence/ops-pilot.json --seed --report artifacts/evidence/ops-preservation-pilot-r1.json
```

Execute all 30 predeclared IDs: both modes, repetitions 1–3, `pilot` and
`one_action` profiles, all matching workloads. Reuse the same profile manifest,
omit `--seed` thereafter, separate processes per profile/mode. Alternate mode order
by repetition as recorded in the corpus. Keep every run; do not stop after success.
No oracle hits, forced planner results, hidden sources, artificial delays, budget
pre-debits or limit expansion are permitted. First-pass support success is valid
quality but unexercised refinement. Real planner/action/ledger/window/transport
events and exact final SDK support determine whether refinement actually occurred.

Operational human reviews are keyed by exact run ID and answer SHA. Refinement
uses the same claim/answer review schema. Limit reviews additionally require
`bounded_partial` booleans `preserves_usable_support`, `explicit_limit`,
`explicit_missing_aspects`, `no_fabrication`, `qualifications_correct`,
`citations_entailed`, `losses_explained`, `finalization_reserve_preserved`, with
`usable_fact_count`, `retained_fact_count` and an `empty_reason` if no fact is usable.
Retain at least one supported factual portion whenever usable evidence exists.
The pair-pressure case's 24 ideal spans are diagnostic, not a mandatory complete
answer after exhaustion; its empty fixed denominator is never an automatic pass.

The activation-v2 gate retains raw `frozen_v5_oracle` outcomes and all 80 quality
labels/denominators. Universal candidate authorization/revision/citation/bound/
publication safety must be observed; unknown blocks. Baseline/selection new-ledger
and cancellation failures remain visible diagnostics, while ordinary baseline
authorization/citation violations still block. Trigger-conditional guarantees can
be unexercised only with complete observations, never missing instrumentation.

Both candidate modes additionally need actual live refinement and actual shared
acquisition-pair-reserve OR retrieval-deadline exhaustion. Action/source limits,
final-phase-only fallback and planner/HTTP timeouts cannot substitute. Acquisition
pair denial leaves reserved final scoring/synthesis valid; global retrieval sealing
and terminal cancellation are distinct. New optional work after its actual closure
is forbidden. Operational reports cannot replace quality arms or enter their means,
latency percentiles or bootstrap intervals.

Use `--compare` with exactly four reviewed quality reports and
`--operational-reports` with all operational report files. `--targets` is JSON
`{routine: {first_grounded: milliseconds, completion: milliseconds}, deeper: {...}}`
with numeric user-facing targets set on the development deployment. Supply
`--require-activation` for exit 2 on any blocked gate. Strict frozen split joins,
matched snapshots/concurrency/cache state, positive targeted support bootstrap lower
bound, no aggregate/cohort quality decline, zero applicable authority/citation
violations, routine p95 ratio upper bound <=1, deeper <=1.2 with support improvement,
absolute targets, human review, verified worker/runtime and all operational evidence
are required. Inconclusive uncertainty is ineligible.

## Rollback and evidence retention

Restore preservation modes to `legacy`, switches to `0`, and withdraw the worker
attestation. Restart serving workers for immediate process-local capability/cache
invalidation; do not flush shared production caches. Preserve report/corpus/trace,
runtime and manifest hashes and original failure outcomes. Evaluate rollback on
the unchanged baseline. Development acceptance precedes a separate graph-free
main adaptation; refresh target-main observations then, never broadly cherry-pick
development graph ancestry. No deployment or backport is performed by these runners.
