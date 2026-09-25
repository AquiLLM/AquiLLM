# Evidence preservation implementation status

Tasks 1–6 were independently reviewed and accepted locally. Task 6 implements
the quality/operational runners, serving-worker capability lifecycle, CI contracts,
runbook and exact-commit backport inventory. Independent review found seven bounded
issues (I1–I7, including the deferred malformed historical identity); fixes are locally
implemented and accepted after two correction rounds. Whole-branch review then
identified scoring-only timeout fallback and evaluator Git portability failures;
their bounded corrections await re-review and the development PR's CI rerun.
All controls remain legacy/off. The controller published a draft development PR;
deployment, live provider evaluation, activation and main backport remain pending.

The [runbook](../../runbooks/evidence-preservation.md) specifies controls, runtime
attestation, commands, human review, activation-v2, measurements and rollback.
The [backport manifest](2026-09-22-evidence-preservation-backport.md) lists exact
accepted implementation/fix commits and required non-graph selector adaptations.
It is a local adaptation inventory; target-main observations remain provisional.

## Implemented and locally exercised

- One cumulative turn ledger and ASGI-owned cancellation lifetime; bounded
  acquisition, current source authorization/revision checks, exact source windows,
  final selection, protected synthesis, and bounded recovery/cache publication.
- Plural/ordinal continuity and direct, manual `/search` and normal model-tool
  handoffs use the shared selection/synthesis path. Prior displayed citations are
  reauthorized; history is not permission authority.
- Default-off per-Uvicorn-worker pair capability warm/renewal/invalidation with
  finite native-work slots, pinned local tokenizer assets, immutable deployment
  identity attestation and actual complete-pair/overflow protocol verification.
  Missing proof leaves capability unknown. Pair context remains 1024.
- Actual isolated DB/ASGI/SDK quality backend alongside explicit fixture mode.
  Baseline/selection preserve original packing; exact SDK-delivered source slices
  are mapped to frozen revision/fingerprint/offset identities in all four arms.
  Cancellation retains what the SDK already received, separately from forbidden
  publication after disconnect. Raw V5 oracle mismatches remain visible.
- Frozen independently reviewed V5 80-case corpus and separate predeclared ops-v2
  30-turn operational workload. No labels derive from implementation answers.
  Mandatory claim support excludes optional context; ungraded nDCG and missing
  human judgments remain unknown. Fixture reports cannot qualify activation.
- Strict four-arm joins, aggregate/cohort support and human qualification metrics,
  paired bootstrap uncertainty, p50/p95 stage timing, p95 ratio uncertainty,
  absolute targets, concurrent/cache matching, window-count bias, fallback and
  inference/cache cost reporting. Operational trace/human evidence is attached
  separately, never added to quality or latency denominators.
- A real four-arm KG-off regression exposed an adaptive-only authorization
  dependency: valid served adaptive selection now uses the same existing selected
  scope Django builder as preservation. Current VIEW/EDIT/MANAGE, supplied-context,
  scope and revision checks remain enforced; legacy/shadow behavior is unchanged.
- Verified-window ASGI regression demonstrates positive acquisition pair charges
  and zero extra final scoring for compatible reuse. This is local deterministic
  runtime-path proof with external boundaries faked, not live inference evidence.
- Legitimate one-word completed answers now survive storage and frontend rendering.
  Interim, tool and reasoning-only text remains suppressed; the stored base system
  prompt stays separate from runtime memory augmentation.

## Verification and remaining gates

The broad local gate enumerated 187 modules, including 164 current CI modules:
1,887 tests and 11 subtests passed, with one existing Task21 container smoke test
reserved for its reviewed cloud gate. The separate required PostgreSQL seed gate
passed all four tests. Its 110 warnings comprise four Pydantic deprecations, the
Google SDK deprecation, Django startup database access, and 104 missing local
`prod_static` directory warnings. No warnings were suppressed.

After that run, standalone CLI verification exposed an unwanted transitive Django
import. The observation sink now lives in application-independent `lib`, and the
pure comparison helper has no application imports. The final 58-module covering
run passed 429 tests and 11 subtests; its one new subprocess test initially omitted
the required `TurnLimits` constructor argument and was corrected: all six import/CLI
and review tests then passed. A final reserve-binding negative test then prevented
operator records from claiming a larger configured reserve than the actual runtime;
its covering review/ASGI/gate/operational suite passed 21 tests. Exact
commands, outputs and final focused results are retained in the local handoff.
Both runners' fixture commands remain explicitly ineligible. System check reports
no issues; all 51 changed/new Python files pass Ruff and formatting, and file-length,
import, logging, retrieval-logging and tracked-path hygiene pass without ratchet
expansion.

The actual GitHub Python 3.12 runtime is not established by this Windows Python 3.13
run. Existing frontend `npm ci` and `npm run build` passed on unchanged inputs under
Node 22; CI uses Node 20. Locked dependency audit findings were not changed.

Still required before activation: whole-branch correction acceptance and actual deployed
model/tokenizer/template/runtime identity and each worker's capability; real
embedding/reranker/answer runs on both frozen splits; named human answer review;
snapshot-matched four-arm support/quality improvements; measured completion reserve;
representative concurrent cold/warm latency and cost; numeric deployment targets;
and actual refinement plus shared pair-or-retrieval-deadline exhaustion in each
candidate mode. Source/action-only stops or untriggered scenarios cannot discharge
that operational requirement. Unknown actual safety or missing observations block.

The earlier adaptive plans are implemented, not superseded by preservation. See
their [implementation report](2026-09-22-adaptive-retrieval-implementation.md).
PPR remains independent and default-off: representative live quality/concurrent
cost is unmeasured, and its local maximum-cap p95 overhead missed the proposed
target. Preservation tests make no claim to resolve that gate.

## Task 6 review fix round 1

Human records bind exact original observations, runtime/code/configuration and actual
run identity; rescoring and attachment consumption revalidate them. Bounded partial
answers require observed acquired/delivered facts, cited answer portions and reviewed
loss or observed empty-source explanations. Pilot limit proof cannot be replaced by
one_action results. Observer failures preserve application outcomes and invalidate
evaluation evidence. Effective arm/config validation precedes live work. Short final
answers now survive actual completion and streaming through persistence, while
nonterminal/tool/interim suppression remains. Malformed historical document identities
retain unavailable notices without discarding valid historical or current sources.

Focused local regressions and hygiene are recorded in the controller handoff; they
are not live evaluation or activation evidence. Default-off and independent adaptive/
PPR, deployment, human review, serving-worker identity and measured latency gates remain.

## Task 6 review fix round 2

R1–R3 corrections add actual final-selector source/span/resource exclusion evidence,
complete common completion-policy snapshot matching, and one supported typed-SDK
normalization boundary for initial review, partial proof and saved/rescored reports.
Local verification and exact fix commits are recorded in the scoped handoff and
backport manifest. Scoped Task 6 review accepted these corrections; all live
activation gates remain open.

## Whole-branch review corrections

Scoring-only expiry now preserves the prepared authorized pool for rank fallback,
with current revision/access revalidation and finalization bounded by the original
retrieval ledger. Initial preparation and scoring keep their hard phase allowance.
Portable native Git supplies evaluator revision/dirty provenance; both standalone
fixture runners and saved-observation rescoring work without RTK. Affected local
covering tests passed 254 tests with the six known warnings. Independent correction
review and actual development CI remain required before merge/deployment.
