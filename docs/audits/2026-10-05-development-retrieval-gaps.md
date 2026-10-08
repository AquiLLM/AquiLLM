# Development retrieval gaps — investigation and rollout audit

Status: deployed and verified on development. Known recall and broad-scope latency gaps remain documented below.

## Scope and reference
Development only: `149.165.150.254`, `aquillm-dev2.cis260251.projects.jetstream-cloud.org`.
Starting application revision: `9136d3b70c09a34748e32b0d20527bc89f463e73`.
Internal findings reviewed at [AquiLLM-internal revision fe7a3b4](https://github.com/AquiLLM/AquiLLM-internal/tree/fe7a3b4f2b884fccc5183896504386e075aa00d4), particularly the production findings note, recall study, retrieval roadmap, and web-rewrite experiment.

The relevant development study is the parent collection 226 (31 documents), plus optional figure collections 227–256 (530 total documents across the combined scope). Production collection/chunk IDs cannot identify this corpus. The runs use an existing authorized principal; no permission bypass or conversation write.

## Confirmed defect and correction
A direct graph alias query remained active for more than 38 seconds after its 4.5-second caller deadline. Synchronous executor work continued after cancellation, occupying a worker slot; repeated requests then reported backend unavailable. Releasing the lease before the SQL actually finishes would permit unbounded work.

Applying PostgreSQL `join_collapse_limit=1` only while fully consuming the alias query reduced two observed alias lookups to 85–118 ms with identical results. A broader planner change that disabled nested loops slowed name lookups and was rejected.

The correction propagates the existing absolute deadline into direct seed reads, caps each statement by remaining time (preserving a stricter caller timeout), checks between tiers/spans, and restores transaction-local settings. It changes no permissions, source/provenance predicates, candidate caps, branch budgets, or worker-pool size.

Whole-branch expiration now reports `direct_branch_timeout` or `extended_branch_timeout`; an actual extractor timeout remains distinct. Fixed-label stage logs cover ontology, extraction, entity resolution, topology and materialization. The dedicated INFO logger is wired into configured handlers. Logs contain no prompts, passages, entity/document IDs or exception messages. Timings cap at 5000 ms; absent or late completion events must not be interpreted as precise duration or assigned to a subsequent request.

A scoped Q2 profile using the diagnostics code showed parent entity resolution 466 ms and topology 293 ms; the broader figure scope took 814 ms direct resolution, 1462 ms direct topology, and roughly 1746 ms cumulative extended resolution followed by 2770 ms extended topology. That extended branch exceeded its existing deadline. This is a remaining broad-scope performance issue, not an extractor failure.

## Recall and input-contract experiments
Retrieval-only probes run the production turn with generation blocked and synthesis replaced by a final-packet recorder. Cache is disabled locally. Candidate-pool inclusion, rerank inclusion, packet inclusion, and surviving answer text are distinct measurements.

A manually source-checked subset contains seven potentially supporting passages across Q1, Q3, Q4. These are provisional diagnostics, not complete expert-confirmed gold. Q2's terminology question is not directly explained in the selected corpus and is not treated as a missing-passage regression. Q5–Q10 remain diagnostics without an independently confirmed development gold mapping.

Graph-off comparisons preserve the ten question texts, changing depth or reranker character cap one variable at a time. The initial scratch harness misencoded curly apostrophes in Q5/Q8; those two rows were rerun from decoded original JSON and substituted before computing the following aggregate timings. Q1/Q3/Q4 measurements were unaffected.

| Experiment | Parent target passages in pool | Figures target passages in pool | Target passages in final packet |
|---|---:|---:|---:|
| Current effective depth 17, 1600 characters | 0 | 0 | 0 |
| Depth 30 | 0 | 0 | 0 |
| Depth 60 | 3 | 2 | 0 |
| 2048 characters | 0 | 0 | 0 |
| Depth 60 + 2048 characters | 3 | 2 | 0 |

The corrected graph-off latency measurements below are one observation per question/scope, not a production load test. Column maxima are maxima across ten queries, not stable tail estimates.

| Experiment | Parent mean / maximum (ms) | Figures mean / maximum (ms) |
|---|---:|---:|
| Current | 2499 / 2969 | 2736 / 2995 |
| Depth30 | 2820 / 3224 | 3082 / 3474 |
| Depth60 | 3298 / 3636 | 3584 / 4035 |
| 2048characters | 2559 / 2994 | 2755 / 3106 |
| Depth60 +2048characters | 3401 / 3820 | 3589 / 4121 |

At depth 60, one Q3 and two Q4 passages enter the parent pool; both Q4 passages enter the combined scope. They are still absent after reranking and from the final packet. Q1 targets remain absent. Consequently these settings are not promoted. A character-cap experiment is not a windowed-evidence experiment.

The local text embedding path validates input_type but sends raw strings; it does not apply a distinct query instruction. Dimension fitting can truncate/pad without normalization. The deployed model is Qwen3-VL-Embedding-2B, whose [official model card](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) describes a multimodal chat input format. Applying a text-model instruction convention to this existing index is not justified by these tests. Query/document formatting and normalization require a compatibility-controlled experiment before any re-indexing or production change.


The maintained CLI then ran the exact decoded ten-question manifest against a verified archive of d9df6ad9, with seven source/chunk/answer-span fingerprints mapped against authorized documents before retrieval. All twenty turns completed with complete observation and generation blocked. Parent scope averaged 4721 ms (10/10 graph hits); figures averaged 7419 ms (5/10 graph hits). All seven provisional supporting targets were absent from the final packet in both scopes, so answer-span coverage remained false. This is retrieval evidence, not an answer-quality pass. Runs use existing warm services, not a cold-start or concurrent load benchmark.


The maintained runner compared graph arms at ee64fb44 (60 additional turns; all complete, no generation). Disabled branches may report their existing no-seeds reason; those are not enabled-branch failures.

| Arm | Scope | Mean / maximum ms | Graph hits | Covered provisional spans |
|---|---|---:|---:|---:|
| graph-off | parent | 2512 / 2799 | 0/10 | 0 |
| graph-off | figures | 2805 / 3170 | 0/10 | 0 |
| direct-only | parent | 2857 / 3318 | 7/10 | 0 |
| direct-only | figures | 4747 / 5987 | 5/10 | 0 |
| extended-only | parent | 4616 / 4943 | 10/10 | 0 |
| extended-only | figures | 7464 / 7935 | 4/10 | 0 |

No arm recovered a provisional answer span. Direct-only is faster but the labels are insufficient to establish that disabling extended retrieval preserves other answers, so both-branch deployment settings are retained. Broader-scope failures and graph contribution counts remain development test targets.

## Evidence-mode gates
Existing authorization/fallback, replay, quality-gate, evaluation, and operational tests are exercised. Source preservation, iterative retrieval and windowed reranking remain disabled.

The established activation-v2 gate requires live per-worker tokenizer/model/template/runtime attestation, warm-up and overflow verification, frozen development/held-out cases, operational cases, and independent human review. Capability warm-up is disabled and attestation unavailable in this development build. Fixture tests do not satisfy that gate; no live 80-case activation evaluation or independent human gate was manufactured.

## Retention
Read-only baseline: database approximately 5.1 GB; 350,649 mentions, 45,321 superseded (12.9%). This differs substantially from the production note's 91% superseded estimate. Large tables include collection entity/document links (~1.8 GB), canonical links (~1.15 GB), and collection entities (~1.06 GB).

The existing pruner keeps active/referenced artifacts, retains the two newest superseded generations, requires a coherent terminal age older than 30 days, and revalidates under scope locks before deletion. Its default batch is at most 100 artifact/run candidates. Initial dry run found six eligible artifacts and six runs. Historical failed projection records are not removed through ad hoc SQL.

A guarded live cleanup deleted the six eligible artifacts and six runs, with zero eligible candidates remaining. Inside one transaction, exact before/after snapshots confirmed that all 2,089 active artifacts, 202 membership rows and 202 ready unpruned projections were unchanged. The existing pruner revalidated each candidate under its scope lock. No historical failed projection rows were manually removed.

Daily scheduling remains opt-in and is enabled only on the development graph maintenance scheduler after validation.

## Validation and rollout record
Completed scoped checks (counts overlap; they are not a total suite count): graph diagnostics/scheduler contracts101; configured logging/redaction77; direct seed/PPR integration31; isolated PostgreSQL cancellation/planner9; cross-task authorization/graph/replay57; production authorization reachability12; evidence quality/evaluation/operational suites34; isolated retention/projection29. Existing Pydantic/Google SDK deprecation and Django startup-query warnings remain. Temporary test PostgreSQL uses pgvector17 with synthetic rows, restricted projection roles and no persistent volume.

All four implementation tasks passed independent spec/code review. The final combined regression set passed 381 tests on an isolated PostgreSQL instance and two Git-dependent fixture tests in the local checkout (383 distinct tests across 42 files). The first archive run exposed harness omissions and inherited production environment flags; adding the missing deployment files and clearing inherited environment fixed those test failures without application changes.

The final whole-branch review found no critical or important issues. Its minor stale pruning-task docstring was corrected. Deployment and post-deployment details follow.

## Development deployment
Application revision: `7a2e145c6a0f06794bf8b92085b9e78972d6cf22`, merged and pushed to development. All 18 unrelated local draft files were verified unchanged by SHA-256 before and after the fast-forward. The subsequent audit-only commit records rollout results; the server checkout and application images remain at the verified application revision above.

All ten application services were rebuilt and recreated from this revision: web, main/memory/graph/schema/projection workers, both maintenance schedulers, query extractor, and topology gateway. Each running image's OCI revision label was verified. Web, extractor and gateway health checks passed; all five Celery workers returned ping responses. Public root and `/ready/` returned HTTP 200.

The proxy retained the previous web container address after recreation; validating and reloading nginx restored public access. Future recreations and rollback must include this reload. An initial deployment-helper read encountered a non-UTF-8 comment in the existing environment file; no file write occurred, and a byte-preserving update changed only the two pruning keys. A replay startup check hit Git's repository ownership guard; the invocation then scoped `safe.directory=/app` to that process after verifying the host revision, without changing global Git configuration.

Development pruning is enabled at 86,400 seconds only on the graph beat service, using the extraction queue at priority 9. Existing 300-second graph recovery/reconciliation and 60-second application maintenance remain present. Application beat explicitly has both graph/pruning gates off. Retention stays 30 days/keep 2. The temporary PostgreSQL test container was removed; it had no persistent volume.

The image build retained an existing frontend chunk-size warning. This change adds no migration or model/index replacement.

## Post-deployment retrieval verification
The maintained CLI ran all ten exact decoded questions twice per scope: 40 total retrieval-only turns, all completed, generation blocked, with `git_verified` revision `7a2e145c6a0f06794bf8b92085b9e78972d6cf22`. Cache was disabled only in the replay processes; deployed adaptive selection and both graph branches were preserved.

| Scope | Turns | Mean / maximum (ms) | Graph hits | Supporting spans covered |
|---|---:|---:|---:|---:|
| Parent | 20 | 4691.5 / 5504.7 | 20/20 | 0 |
| Parent + figures | 20 | 7473.5 / 8231.7 | 10/20 | 0 |

There were **zero backend_unavailable** reasons across all forty turns. Parent diagnostics included one extractor_timeout and two extractor_provenance failures on the first three turns after restart; none recurred in the second pass, and the extended sibling supplied graph results. This observation does not establish the exact startup failure cause.

Figures produced eight direct_branch_timeout and sixteen extended_branch_timeout reasons (a turn may have both), plus six direct_no_seeds. Fallback kept the retrieval turns handled; these branch failures remain real. All seven provisional target passages stayed out of the final packets. Completed turns are not a study-answer quality pass, and these sequential runs are not a load test or a controlled cold-start benchmark.

Configured INFO output contained 360 graph-stage events with only fixed branch/stage labels, bounded timing and standard logging fields. The CLI correctly leaves request-correlated graph substage timing unavailable; ordinary late completion logs are not attributed to a later request.

## Rollback
Protected rollback directory on 254:
`/home/exouser/.local/state/aquillm-rollbacks/retrieval-gaps-20261005`.

It contains a mode-600 environment backup, exact pre-deploy image IDs and preserved image tags for all ten services, a Compose image override, and a guarded rollback script. Image-tag identities and script syntax were verified; rollback was not executed.

To restore application code/config/images, run as exouser on 254:

```sh
python3 /home/exouser/.local/state/aquillm-rollbacks/retrieval-gaps-20261005/rollback.py
```

The script requires a clean checkout at the recorded application revision (or original base), checks preserved image identities, checks out base `9136d3b70c09a34748e32b0d20527bc89f463e73` detached, restores the environment file, recreates only the ten application services with preserved images and no build, waits for health, and validates/reloads nginx. It does not restore the deliberately pruned eligible terminal records or alter database/model volumes.

## Remaining test and research work
- **Missing support:** establish independently reviewed development gold, then isolate Q1 candidate retrieval and Q3/Q4 reranker demotion. Larger pools alone did not deliver the supporting text.
- **Broad-scope graph timeouts:** profile and reduce repeated projection resolution/topology work on the 530-document scope, retaining the current authorization and deadline bounds. Do not interpret successful vector fallback as successful graph retrieval.
- **Extractor startup:** distinguish service warm-up/overload from provenance failures and test readiness after restart. Initial post-restart diagnostic failures must remain visible.
- **Embedding/reranker contracts:** compare validated input formats, token overflow and normalization against a compatible index before changing model-facing requests.
- **Evidence-mode activation:** obtain the required live attestations, frozen development/held-out and operational runs, and independent review before enabling source/windowed/iterative modes.
