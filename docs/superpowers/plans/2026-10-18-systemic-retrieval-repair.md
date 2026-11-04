# Systemic Retrieval Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Improve general evidence discovery and grounded synthesis without study-specific behavior.

**Architecture:** Extend bounded planner context and unify grounding in the existing retrieval/publication flow. Version graph extraction behavior before evaluating any schema or endpoint changes. Reuse existing source authorization, immutable artifacts and shared turn budgets.

**Tech Stack:** Python, Django, pytest, PostgreSQL, GLiNER2, OpenAI-compatible model interface, Docker.

**Spec:** `docs/superpowers/specs/2026-10-18-systemic-retrieval-repair-design.md`

## Global Constraints

- Work in `codex/graph-development-repair`; do not create another branch.
- No study question IDs, expected answers, corpus document IDs or special-case aliases in production code.
- Keep main-model synthesis and exact source authorization; do not default to excerpt-only answers.
- A successful graph build, valid citation or model self-critique is not proof of semantic correctness.
- Keep one shared turn budget: at most 3 actions, 2 planner calls, 45 unique sources and current byte/token/scoring limits; preserve stricter operator deadlines.
- Do not widen retrieval caps or require graph evidence merely to improve benchmark scores.
- No request-time graph construction; extraction changes require versioned background artifacts.
- Keep private source text, study traces and corpus-specific diagnostic scripts out of tracked files.
- Root alone runs SSH, shared database tests, model calls and GPU experiments; serialize these resources.
- Keep the serving development stack unchanged while isolated candidate verification is incomplete.

### Task 1: Restore meaningful handoff regression coverage

**Files:**
- Modify: `aquillm/apps/chat/tests/test_rag_fix_handoff.py`
- Modify: `aquillm/apps/chat/tests/test_rag_fix_tool_deadlines.py`

**Interfaces:** Consume the existing `acquire_evidence` stop reasons and `PLANNER_SYSTEM` contract. Produce tests that still prove loss of support after final selection and strict deadlines with fenced late results. No production changes belong to this task.

- [x] Reproduce the four known failures using the root-owned archived test runner. Baseline evidence exists in private `publication-db-20261018T101014040538Z`.
- [x] Inspect why the fake planner fails and update it to the actual provider call contract. Identify planner calls using the imported contract, not obsolete prose. Preserve the asserted single planner call, single synthesis and missing source/tail support:

```python
assert acquired.assessment.support
assert not packet.coverage_assessment.support
assert runtime.budget.planner_calls == len(planners) == 1
assert len(llm.calls) == 2
```

- [x] Update timeout diagnostics to the actual phase-specific contract while preserving the timing and late-worker assertions:

```python
assert task.result().stop_reason == "acquisition_timeout"
assert late == []
assert budget.actions_used == 2
```

- [x] Ask root to run both test files with the archived runner. Review the diff, commit only these test changes, and report the exact evidence path.

### Task 2: Bounded planner memory and useful source scope

**Files:**
- Modify: `aquillm/apps/chat/services/rag_acquisition.py`
- Modify: `aquillm/apps/chat/services/rag_coverage.py`
- Modify: `aquillm/apps/chat/services/rag_planner_contract.py`
- Create: `aquillm/apps/chat/services/rag_planner_context.py`
- Create: `aquillm/apps/chat/tests/test_rag_planner_context.py`
- Modify relevant existing acquisition/coverage tests.

**Interfaces:** Extend `assess_coverage(..., attempts=(), latest_source_ids=())` with an optional bounded history of action kind/query/document/chunk and cumulative available-source/new-source counts. Existing injected assess callbacks accepting keyword arguments remain compatible. `planner_views(evidence_views, max_chars=16000, latest_source_ids=())` returns complete immutable source views, ordered in document round-robin. Prioritize groups with newly acquired sources and new sources within each group, retaining stable order otherwise. This is planner context only; final ranking remains unchanged.

- [x] Add failing synthetic tests: a first document with many chunks cannot hide a second document; a later planner sees newly acquired source views; an oversized source is not clipped into mismatched offsets; source text totals stay within 16000 characters; repeated source identities do not consume context twice.
- [x] Add failing orchestration tests showing the second planner receives the first attempted query/outcome, and initial external handoff is represented without inventing an unknown query. History is bounded by admitted actions, includes no raw source text, and records action failures without provider bodies.
- [x] Add a provider-independent test rejecting a document action for an authorized but unseen document. Validate against precisely the planner-visible source set; adjacent references must also bind to that set.
- [x] Implement small context helpers and thread history through the existing acquisition loop. Use source-grounded terms for a new global search; use document search for a known relevant source, adjacency only for local context. Do not force document search for every question.
- [x] Keep canonical duplicate detection conservative: case/Unicode/whitespace normalization may reject identical requests, but negations, numbers, scientific punctuation and different document scope must remain distinct. Do not add a semantic deduplicator.
- [x] Run focused context, coverage, acquisition, handoff and budget tests through root, inspect observed prompts/counts, commit and review.

```python
views = planner_views((a1, a2, b1), max_chars=len(a1.text) + len(b1.text))
assert views == (a1, b1)
assert sum(len(view.text) for view in views) <= 16000
```

### Task 3: Version graph extraction behavior consistently

**Files:**
- Create: `aquillm/lib/knowledge_graph/extraction_identity.py`
- Modify: `aquillm/apps/knowledge_graph/extraction/pipeline.py`
- Modify: `aquillm/apps/knowledge_graph/services/builds.py`
- Modify: `aquillm/apps/knowledge_graph/evals/run_kg_eval.py`
- Create focused identity tests alongside existing extraction/build tests.

**Interfaces:** A pure `extraction_identity(settings)` function returns a canonical behavior fingerprint fitting the existing 128-character extractor field. It binds provider, immutable checkpoint, adapter/schema-contract version and effective inference settings. Production construction, worker comparison and evaluation use this same function. No model import is permitted in this helper.

- [x] Write failing tests: stable identity for equivalent settings; identity changes with adapter/schema-contract/inference behavior; operational cache location is not semantic identity; production and evaluation agree.
- [x] Inspect actual adapter inference options and pinned package version, then define a canonical sorted JSON manifest and prefixed SHA256 identity. Avoid hashing environment secrets or whole environment variables.
- [x] Replace duplicated identity construction and verify existing artifacts are not rewritten, while newly requested artifacts receive distinct build keys. Preserve existing source/ontology/resolver/filter bindings.
- [x] Run focused pure and database build identity regressions; inspect all configured-extractor call sites, commit and review.

```python
identity = extraction_identity(settings)
assert len(identity) <= 128
assert identity == extraction_identity(settings)
```

### Task 4: One grounding contract for every synthesis path

**Files:**
- Modify: `aquillm/lib/llm/providers/complete_turn.py`
- Modify: `aquillm/lib/llm/providers/synthesis_retry_prompt.py`
- Modify: `aquillm/lib/llm/providers/rag_citations.py`
- Modify: `aquillm/apps/chat/services/rag_support_delivery.py`
- Modify: `aquillm/lib/llm/tests/test_direct_synthesis_grounding.py` and focused retry/repair tests alongside it.

**Interfaces:** Use a shared instruction function/constant for evidence-backed synthesis across direct and post-tool initial/recovery/continuation paths. Citation repair keeps the original evidence and treats the previous answer as an untrusted draft. Coverage metadata explicitly identifies unassessed aspects without declaring the evidence absent.

- [x] Add failing path-matrix tests that inspect actual provider requests: each path preserves exact source text, source citation coordinates, qualifier requirements and allowance for a concise partial answer.
- [x] Remove fixed 400-900-word and unrelated methods/math requirements from post-tool recovery. Preserve relevant user requirements and main-model generation.
- [x] Ensure citation repair can remove unsupported assertions and restore source qualifiers; never tell it to preserve every previous claim. Do not enable the rejected model critic or claim deterministic entailment.
- [x] Preserve publication authorization, current evidence receipts, source-loss fallback and useful unknown-support synthesis. Run focused synthesis/retry/repair/publication regressions, commit and review.

```python
assert supplied_source_text in captured_request
assert source_citation in captured_request
assert "400-900" not in recovery_instruction
```

### Task 5: Measured graph coverage and candidate acceptance

**Files:** Private evidence under `.superpowers/sdd/2026-10-18-systemic-retrieval-repair/`; tracked documentation only contains aggregate findings and reproducible general procedure.

**Interfaces:** Consume the reviewed candidate from Tasks 2-4, current active ontology and pinned corpus/runtime manifests. Produce extraction rejection-stage evidence, full-study/held-out comparison, and an explicit decision on graph schema/adapter changes.

- [x] Inspect the saved read-only active ontology/build stats. Freeze general synthetic relation examples and real held-out passages before candidate extraction. Include supported relations, unsupported/ambiguous endpoints, negation and alternate dataset/metric types.
- [x] Run bounded offline extraction with the existing adapter and record raw-output, normalization and endpoint-validation counts. Do not infer passage-level rejection reasons from document aggregates.
- [x] Evaluate the versioned description-preserving graph compiler on the frozen sample with unchanged provenance and endpoint validation. Reject both tested variants after zero accepted relations within their resource bounds; remove the unused compiler from product code. Retain shared extraction identity and private diagnostic evidence. No activation or graph rebuild follows from this experiment.
- [x] Stage immutable r10 at `079ab3a895d3e7f8b9f540275f0c8dbffaf2574c` and complete all 26 matched cells: ten study questions in two arms plus six pre-frozen held-outs. Preserve model prose and independently review acquisition, final evidence, content, citations, latency and failures. Live evaluation is complete; quality acceptance has not passed.
- [x] Repair the later resolver-identity test fixture in `44957ae0cb2aed157fd187d8143d6a76325daaa5`; the focused local regression run passed 171 tests. This test/CI repair does not change the frozen r10 evaluation head.
- [x] Independently review P1's 40 frozen requests and reject the added navigation-output contract: no generic routing gain, repeated structural/span failures, and higher token/latency cost.
- [x] Independently review all 48 P2 attempts, retaining three 32k timeouts as failures. Reject the 32k context increase and navigation contract: more visible pointers do not produce a clean generic gain or saved-question recovery.
- [x] Independently review P3's 48 completed requests. The support-free shape yields no useful-action gain at 16k (both arms 4/8 generic); it remains experimental. Diagnostic 32k contract improvements do not authorize a production cap increase.
- [x] Complete and independently review P4's frozen 48 long-case and 32 small-control requests at 16k. Reject balanced packing and the combined treatment: the combination improves one generic comparison case but balanced packing misroutes the external-method diagnostic in both blocks under both strategies. Do not promote pointer-only strategy: no generic gain and the release-comparison gain is inconsistent across blocks. Small controls pass 16/16 per strategy with no gain or regression. Retain frozen expectations and failures; no product contract or budget change follows.
- [ ] Run repository checks and review the whole change. Update PR247 and the authorized internal tracking issue with code, artifact, evaluation and deployment status. Keep unproven quality improvements explicit; do not merge/deploy on unit tests alone.

### Additional confirmed defects (within systemic repair scope)

- **Graph compiler experiment (Task 3/5):** rejected and removed; the description-loss diagnosis does not establish a successful extraction repair. Shared behavior versioning remains, with existing endpoint validation and serving artifacts unchanged.
- **Large source snapshot serialization:** completed a reversible migration replacing quadratic work while preserving canonical bytes and immutable hash guards. Equivalence, Unicode/escaping and realistic large-source integration checks passed without shrinking the fixture.
- **Local anaphora:** completed general same-question noun/pronoun handling with explicit-source and ambiguity barriers. Regression checks passed and the frozen held-out questions remained unchanged; routing repair alone does not establish answer correctness.
- **Returned source order:** committed `d15430ab`, preserving retrieval-provenance order among the same verified, deduplicated sources instead of cache insertion order. Revision filtering, planner latest-source priority and caps remain unchanged. The 32-test source-order/context/acquisition suite and independent review passed; this does not establish Q3/Q4 success.
- **Lexical settings wiring:** committed `888203a6`, exposing the three existing full-text, RRF and protected-exact controls through `env_bool` with independent false defaults. Protected exact remains effective only with RRF. The 24-test settings/control suite and independent review passed; the settings file allowance ratcheted from 485 to 480 lines. This repairs operability without activation or a recall claim.
- **Optional rewrite budget:** committed `409f9ef4`, the verified repair reserves capacity for the original search and gives an admitted vocabulary expansion precedence over a rewrite. Normal handoff explicitly identifies an already completed initial acquisition. After RED (5 failures / 6 passes), the seven-file GREEN suite passed 151 tests with three existing warnings in 391.99 seconds; independent review, file-length and diff checks passed. Query rewriting remains disabled in the frozen r11 evaluation profile.

Root continues to serialize shared tests, model calls and commits. Remaining
experiments and acceptance review are separate from the completed implementation.

Separate candidate-readiness inspection reports 530 documents, 2,701 text and
499 image chunks, zero current FTS projections and zero structured sections.
All three lexical flags remain false. The mixed-modality FTS gate is intentional;
changing it or the native-to-scored fallback requires a separate decision. The
frozen neighbor policy is rejected: it reaches zero missing targets and removes
observed acquisition paths for five final-selected sources. The 16 text-only
FTS probes completed without errors: 12 returned zero matches and none of the
11 frozen targets matched anywhere. Reject this literal English plain-query
treatment as a remedy for the recorded misses, not FTS generally. Inputs were
unchanged unresolved-context questions, not production-routed queries.

The separate all-lexeme OR SQL contrast completed all 44 arm observations and
passed mechanical checks. It gains 4/11 frozen targets in the first eight versus
plain FTS, but all four were already acquired and delivered in r10; Q3/Q4 remain
0/4 at that cutoff. All three generic controls lose their useful source from
the returned eight. Reject unconditional OR replacement, integration and
projection backfill. This computed lexical counterfactual does not exercise
production admission, final evidence or answers; it establishes no r11 result.

## Verification checkpoint after r10 live evaluation

Tasks 1–4 have focused regression evidence and independent review. Planner tests
passed in two 88-test groups. Grounding passed 146 regressions and a 20-test
review follow-up. The serializer migration passed 40 guard/integration tests and
13 final focused tests. Local question routing passed 497 tests. Final extraction
identity/default-adapter checks passed 72 tests; all existing artifact bindings
and strict endpoint validation remain in place.

The description-preserving graph compiler did not pass its offline quality and
resource checks. Both tested variants produced no accepted relations within
their bounds. The unused compiler was removed from product code; experiments
and source adjudications are retained privately. Shared extraction versioning
remains implemented. Existing graph extraction quality is an open limitation,
and no graph rebuild or compiler activation is authorized by these test results.

All 26 r10 cells completed operationally, but acceptance has **not passed**.
Study questions Q1/Q2 remain partial, Q3/Q4 fail required answer coverage, and
Q5–Q10 preserve their core answers with remaining citation or scope issues in
some cells. These outcomes do not establish a general planner or graph gain.

The six held-outs improve from 8 fully correct / 1 partial / 12 missing required
claims to 18 / 3 / 0; six of seven frozen evidence targets are acquired and
delivered, versus four at baseline. Core coverage, incidental assertions and
citation precision remain separate scores; one negative-scope judgment is
explicitly borderline. No acquired target is lost during final selection.

The 171-test fixture repair, 32-test source-order repair, 24-test settings repair
and 151-test rewrite-budget repair are separate from immutable r10. P1/P2 reviews reject
the navigation contract and 32k increase; P3 supplies no 16k routing benefit.
P4 is complete: reject balanced
packing, the combined treatment and promotion of the pointer-only strategy.
Generic useful-action counts are 4/8 for control, pointer-only and balanced-only,
versus 6/8 for the combination; saved diagnostics separately score 2/4, 3/4, 0/4
and 1/4. Small controls score 16/16 per strategy with no gain or regression.
Prompt and packing experimentation ends for this investigation; any next
source-candidate diagnosis is a separate workstream. These results do not change
the r10 findings or production caps.
The serving development stack remains unchanged; neither operational completion
nor regression tests authorize quality acceptance, merge, deployment or a graph
rebuild.

## r11 verification and next confirmed defect

The isolated r11 candidate at `29ff6e01ebc210fd1a253c91c20da104a334e741`
completed all 26 frozen conversations. All answers persisted, the original chat
remained unchanged, and no runtime, observer or acquisition failure was recorded.
Whole-turn latency was 31.59–103.00 seconds (median 46.56); this includes model
synthesis and repair, not just retrieval. All 78 per-cell trace/result/summary
artifacts agree with the batch and the seven independent quality reviews.

Quality acceptance still fails. Q1/Q2 remain partial; Q3/Q4's known reference
passages remain absent before final selection. Q4 removes the earlier unsupported
no-change conclusion but still does not answer the catalog comparison. Q5–Q10
retain all 12 core answers, with incidental qualification/citation defects graded
separately. Held-outs score 17 fully correct / 4 partial / 0 missing required
claims, versus r10's 18 / 3 / 0 and the original baseline's 8 / 1 / 12. Six of
seven evidence targets remain acquired and delivered. Strict full-answer semantic
success improves from two to three cases; losing one explicit simulation
qualification lowers the required-claim score. Contextual sensitivities are
reported separately, not used to replace the strict result. Single stochastic
outputs and a final-scoring readiness difference prevent ordering-only causal
claims. Neither source-order preservation nor valid citations establish recall.

Publication fixture repair `1a8bf59b` retains the real model-answer SQL barrier,
the configured optional-action timeout and late-result fence, and exact source
delivery through scoring, sealing and SDK payloads. RED reproduced nine failures;
the combined follow-up passed 37 with one remaining two-sentence citation fixture
failure. Correcting that fixture passed all 11 suffix tests. Independent reviews,
file-length and diff checks passed. This test-only commit is outside r11; full
GitHub CI remains separate from these focused results.

The read-only source inventory confirms 30 PDFs, 499 figure-image documents and
one raw-text document, with 3,200 chunks and the same frozen representation digest.
All 560 stored references are present by stat/HEAD; this does not prove content
integrity or original-file provenance. No document has an exact new output ledger
or extraction revision; all active representations are legacy and all structured
sections remain absent. Existing reindexing cannot directly rebuild the PDFs or
figures. Unchanged native PDF parsing produces page paragraphs and would not
recover semantic headings or publication titles. No reparse, backfill, migration,
activation or serving change followed from this inventory.

The separately frozen existing query-rewrite viability check timed out on its
first question after 5.011 seconds without a provider response. The remaining
15 questions were unattempted, not failures. The request inherited thinking and
had no strict output schema; code inspection also found that truncated responses
could be admitted. The shared model was verified idle afterward. A narrow repair
now has RED evidence (25 failed / 4 passed): reuse the planner's supported
structured/non-thinking request contract, reject reported truncated/nonterminal
or malformed responses with safe failure categories, preserve the exact original
query and existing filters/caps,
and version the bound policy. The implementation uses policy v3 and passed
212 focused regressions across 12 files in 413.47 seconds, with three existing
warnings. Independent code review passed. The isolated r12 treatment at
`1c2b839b4a536f0934751e846b4864d462e0fe99` verified all 1,183 runtime file
hashes and completed all 16 original questions on the configured local vLLM
stack under the unchanged five-second limit: 0.18–0.39 seconds each (median 0.26),
with valid terminal structured responses and one recorded SDK call per question.
Existing filters accepted four
proposals, rejected eleven exact duplicates and rejected one source-name match.
The latter is a conservative heuristic triggered by a punctuation change; it
was not relaxed. Independent request, budget, result and artifact review passed.

This repairs proposal generation, not missing-evidence acquisition. Q3's output
is unchanged, and Q4's only removes question framing. No search, source admission,
reranking or answer generation was performed in this proposal-only check.
The baseline stopped after its first timeout, so there is no paired baseline
latency for the other 15 questions. Rewriting stays disabled. Serving 254, its
model settings and its stored source representations remain unchanged; full CI
and end-to-end retrieval benefit are separate requirements.

## Bounded lexical ranking result

The separately frozen BM25 diagnostic completed 38 valid rank observations:
sixteen unchanged literal questions and three generic cases, each with BM25 and
an identical IDF=1 ablation. Normalization and exact returned-source identity
checks passed. BM25 reaches seven of eleven frozen reference targets within
eight candidates, versus four for the earlier OR contrast and zero for plain
FTS. One is new to the recorded acquisition histories: a held-out
report-date passage, at rank six. All three generic useful sources remain, and
no earlier plain/OR target is lost. This meets the predeclared candidate-stage
criterion. Q3/Q4 still reach zero of their four targets within eight.

IDF=1 also reaches seven targets, including the new date passage at rank five;
the corpus-target gain cannot be attributed specifically to IDF. BM25 retains
the version-specific generic control that IDF=1 loses (three of three controls
versus two). Do not choose a different treatment or increase the cutoff after
seeing these results. Two earlier harness failures are retained: an incorrect
Django app label stopped startup, and a JSONB decoding error prevented recording
the executed SQL's rankings. The final repair changed neither inputs, SQL,
scoring nor grading criteria.

The corpus batch took 12.48 seconds for all sixteen questions, both ranking
arms, corpus statistics and identity checks. This is neither per-query latency
nor a production serving implementation; request-time corpus tokenization is
not proposed. The diagnostic used the authorized active text scope and returned
no source bodies. Serving and source representations remained unchanged.
The next evidence needed is source admission, displacement, final ranking and
actual supported answers under the existing 30/45 source ceilings and shared
work limits. Candidate-stage success does not authorize BM25 activation,
projection backfill, merge or deployment.

## Paused checkpoint after bounded admission comparison

The frozen comparison completed all 32 arms (sixteen literal questions,
baseline versus the fixed BM25 candidate treatment) on isolated r12. Every arm
completed with model scoring; source, authorization, representation, capability
and serving guards passed. Both arms delivered six of eleven frozen useful
targets with matching source bytes, offsets and actual score-window coverage.
There were zero new previously unacquired useful targets and zero losses among
those mapped baseline targets. The admission criterion failed; do not promote
the earlier candidate-stage result into a retrieval or answer-quality gain.

Q3/Q4's four targets remain absent even from candidate metadata. The held-out
date target is present in metadata in both arms but is not materialized or
accepted in either. Its loss therefore precedes final reranking. The precise
allocation cause still needs source-level diagnosis. A separate supplemental
review found Q5–Q10's core evidence retained in all six pairs; that review does
not change the frozen primary target set or prove general absence of regression.
This experiment generated no answers and made no serving change. The ranking
inputs were computed offline, so the run does not measure live BM25 cost.

Full CI at 0555a2fe passed frontend and isolated projection checks, but a later
backend stage exposed a second obsolete successful-rewrite fixture. Root
reproduced two failures in the retry/recovery copies, then added terminal
completion metadata to those mocks and the identity-rewrite integration mock.
No product validation or assertions changed. The five-file follow-up passed
65 tests with three existing warnings in 43.88 seconds; independent diff review
and whitespace checks passed. Full CI for this later fixture repair is pending.

The next identifier-format experiment exists only as private pure code and
controls. Eight checker methods pass, including complete versus partial name
matches, protected exclusions and bounds. It has no SQL integration, corpus rank
result, product implementation or activation. Before resuming it needs review
of overlap with existing English scoring, a frozen ranking package and controls
that can expose displacement. It is not a demonstrated remedy for Q4.

Work is paused at the user's request. All live comparison and local regression
jobs have completed. Keep the current branch and draft PR247; retain the private
artifacts and handoff. Serving 254, source representations and graph builds are
unchanged. Quality acceptance, full CI and the remaining acquisition,
preprocessing and graph-quality work remain open.

## November 3: integration with development

PR247 incorporates development at `2439ca86`: PDF ligature and line-break
cleanup (#248), native PDF NUL removal (#249), background embedding recovery
(#250), and release metadata (#225). The October runs remain evidence for their
original deployment and corpus only. Cleaning affects new extraction; recovering
stalled embedding builds does not add missing contextual text to existing
passages. No reparse, reindex, change to graph artifacts, or deployment follows
from merging these changes into the repair branch.

The cleanup changed canonical PDF text and offsets without changing the native
parser key. Ingestion includes that key in the output identity for a source
occurrence. PR247 versions new native PDF extractions as v2, preserving the
stored extraction and offsets for v1. A real-PDF regression binds both revisions
to one occurrence and verifies idempotent persistence of each. Direct legacy
document saves are not rewritten. The PDF parser suites passed locally (32
tests); a separate PostgreSQL 16 run covering revision identity, embedding
recovery and rewrite validation passed 55 tests.

The integrated embedding worker could enqueue a continuation after its delivery
deadline when the handoff transaction committed late. A regression reproduced
the late dispatch. Checking the original deadline immediately before enqueue
keeps staged rows recoverable by scheduled redelivery. On a fresh PostgreSQL
16 instance, all 37 recovery, deadline and extraction-revision tests passed
after the fix, including the handoff that expires precisely at commit.

A replay of the original admission traces reproduces both 30-source charge
sequences for the held-out report date. Ten slots went to the initial search
lanes and six to new graph results, leaving fourteen slots. The date passage
was 32nd in the remaining baseline order and 30th with BM25. Its sixth-place
BM25 position became 15th after nine trigram entries were prepended. Across the
16 treatment questions only five of 59 candidates appearing exclusively through
that lexical suffix were charged. These counts measure exposure, not usefulness.
The allocator followed its contract. For Q3/Q4, the four targets were absent
from first-stage candidate lists; a change to final reranking cannot recover
them. The current study does not justify a feature rollout.

A second offline replay alternated trigram and BM25 candidates at the same
cutoff and held observed graph outputs fixed. Of the 15 interpretable pairs,
no additional mapped target could be charged and three previously selected
baseline source occurrences would definitely be displaced. Q3 was inconclusive
because its original graph phase boundary was ambiguous. This does not measure
live graph effects or answer quality.

In a separate PostgreSQL test with seven synthetic ten-source pools, the useful
source made the first eight in five pools with baseline ranking, six with full
identifier-format matching and seven with adjacent component matching added.
Wrong-version distractors also gained score. Both saved Q3 target bodies contain
the partial identifier match, but the whole identifier does not match; this does
not establish a top-eight rank on the corpus. Q4 contributes no identifier keys.
Its saved passages ranked 45th and 24th globally and 11th and 7th within the
relevant paper in the October diagnosis. Test source context separately; do
not force a paper or add benchmark-specific aliases. No identifier experiment
or document-context change has been enabled in the product.

## November 4: real-collection identifier comparison

The original isolated r12 runtime reproduced all 16 baseline lexical rankings
against the authorized development collection after maintenance. The same
530 documents, 2,701 text chunks, 499 other chunks, representation digest,
title inputs and full text input digest passed the October checks. Baseline,
full-format matching, and full-plus-partial matching each placed the same
seven of eleven mapped passages in the top eight, at the same ranks. Neither
Q3 method passage entered the shortlist. Q4 had no eligible identifier.

All seven newly shortlisted Q3 passages were read and checked against their
content hashes. The highest-scoring new source uses the SDSS field name but
describes catalog selection and error scaling, not the HSC fitting procedure.
The other passages discuss comparisons, calibration, or figure labels. Four
of these reads were obtained on an isolated copy of the integrated PR runtime
with 1,184 verified runtime files; three were identical to previously captured
source bytes. The gain on synthetic controls did not transfer to the study
question. Reject this ranking treatment; no activation or answer change.

Fetched `origin/development` still points to `2439ca86`. All four merges
since #246 are contained in this PR. A read-only check found that the ten
application files changed by #248, #249 and #250 in the **running web
container** also match development. This is a scoped file comparison, not a
claim that the entire image, deployed workers, or active corpus are identical
to the branch.

The last full CI run for `cd824799` failed in the remaining graph tests on an
incomplete vocabulary test fixture. The database regression reproduced the
missing extraction-device error; the fixture now starts with complete
extraction settings while retaining its synthetic provider/model and all
repinning and authorization assertions. The focused three-file PostgreSQL 16
run passed 32 tests. Full CI for the fixture correction is separate.

On the integrated PR candidate, a separately bound 16-question comparison used
the native global and document-scoped retrieval paths on the same query
embedding. A document with at least three of the first five global dense hits
could supply the tail of the 17-slot dense lane; the first nine global hits
and the other search lanes were retained. The candidate union included 9/11
mapped passages versus the current baseline's 7/11. The two new hits were
both Q4 passages. Q3 still missed both. This was not universal improvement:
46 candidate paths disappeared, one belonging to a previously selected Q7
source, and a broad-query control lost its only useful minority-document hit.
Do not replace the global tail unconditionally. The earlier attempt to enforce
October dense order stopped at its first question and is not included.

A subsequent downstream test initially stopped **before acquisition** because
the post-maintenance reranker process had no current verification proof. The
container, image and creation identity were unchanged, but its start time had
changed. An explicit operator verification of all 14 pinned artifacts, runtime
arguments and packages, template and tokenizer cache restored the deployment
record. The ordinary timer issued a later successful renewal; a new, separate
Q4 baseline completed with actual model scoring. The earlier failed arm is
retained as a readiness failure, not credited as a successful search.

All four post-repair arms then completed with local model scoring. Each pair
started with the same returned source identities. For Q4 the diagnostic added
one same-query document action instead of replacing the initial retrieval.
It spent 9.49 seconds and charged three more sources (33 total). Both reference
passages reached the scoped dense lane; only one returned as evidence and was
scored (raw rank 19/33). Neither survived final selection. The ten selected
identities and their order were unchanged. On Q3 the rule chose the wrong
paper, spent 8.89 seconds, and changed two selected identities without recovering
either reference. Both document tools completed despite their internal graph
branch reaching its deadline. No model planner selected this route and no
answers were generated. Do not enable majority-document routing. The remaining
Q4 gaps are returning relevant candidates and selecting explanatory passages;
the Q3 gap still begins at acquisition.

## November 4 follow-up on primary sources and graph deadlines

We checked the internal faithfulness references and the scored development
passages separately. For Q4, development has both primary catalog lists. The
older paper is a preprint of the journal paper used in the original study,
with different chunk boundaries and an opaque filename. The two lists support
the SDSS version change, the six additional catalogs, and the retained QSO
catalog citation. The DR14 label needs another source. For Q3, the collection
contains the two component fluxes and a pointer to the pipeline paper. The
full fit procedure still lacks a verified passage in this collection.

We tried splitting literal comparison questions into two searches under the
same 17-slot budget. Across 16 questions, only two qualified. The candidate
union retained 7 of 11 mapped targets in both arms. Neither Q4 subquery found
the primary catalog lists, and one control lost useful evidence. We did not
enable this routing policy.

We also found a separate defect in document-scoped follow-ups. The graph child
could time out after reserving source slots, leaving already-ranked baseline
candidates unread while the parent search still had time. The revised native
schedule sets aside up to 2.5 seconds from the graph allowance to revalidate
and read those candidates for an eligible single-document follow-up. It keeps
the original turn, source and byte limits and the final closure reserve.
Permission loss and hard expiry still invalidate the search. A PostgreSQL
regression covers expiry on each side of the point where the sources enter the
cache. The scoped, closure and replay run passed 127 tests.
Another 53 admission, publication and provenance tests passed on PostgreSQL.

Five isolated local-vLLM answer replays on 254 separated source sufficiency
from automatic retrieval. The replays used the real publication authorization
check, acquired its row locks and rolled the diagnostic transaction back. An
earlier read-only harness prevented those locks and produced five limitation
messages. Those messages cannot establish a serving defect.

Given the ten sources from the ordinary Q4 selection, the model inferred no
intrinsic survey changes and missed both the SDSS update and all six additions.
With both primary catalog passages supplied, it cited the DR12-to-DR16 change
and named all six. With their title pages also supplied, it explicitly
identified the retained QSO catalog. Q3's ordinary selection supported a
reference to the method paper. Supplying the two component passages yielded a
cited, appropriately limited explanation. These were individual outputs with
controlled source packets. Only the selection replays used ordinarily
retrieved sources.

The six-arm comparison on the isolated development candidate at `ebcf7cb4`
gave the scoped recovery a concrete benefit and two rollout blockers. Each
question used the same literal query, initial ordered sources and authorized
collection across its old and new arms. The one-document follow-up was an
experimental routing choice. It wasn't chosen by the model planner.

| Case | Old acquisition and final scoring | With scoped recovery |
| --- | --- | --- |
| Q4 | 33 sources scored. The full later catalog never reached the scorer. | 45 sources scored. The full later catalog ranked 28th, outside the final nine. The earlier catalog was outside the chosen paper. |
| Q3 | All 29 acquired sources scored in 35 windows. | 41 acquired, 39 scored in 45 windows. Two previously scored sources lost their scores, including one previously selected. The follow-up still chose the wrong paper. |
| Q7 | All 32 sources scored. Final evidence contained the reported maximum of 4.119. | The extra 13 sources exhausted the cumulative tokenization allowance before final evidence handoff. A separate instrumented run reproduced the failure with 999,973 of 1,000,000 codepoints used and 4.2 seconds left. |

Keep scoped recovery disabled by default. The measured behavior is available
through `RAG_SCOPED_BASELINE_RECOVERY_ENABLED=1` with native closure. Protect
the ordinary candidates' scoring windows and final text-preparation budget
before adding recovered sources. Raising the graph timeout would not address
either downstream constraint.

With the gate in place, 93 scoped timeout, dispatch and replay tests passed
on PostgreSQL. The native replay report records the effective version and
the time reserved for scoped recovery, including zero when it is disabled.

One old Q4 arm logged a generic graph error. In a separate reproduction the
source-preparation deadline check stopped graph evidence hydration. The
trace did not distinguish which nested time limit expired first. That guard
raises a different exception from the graph deadline classifier.
The recovery experiment intentionally requires an explicitly classified graph
deadline. Fixing the missing classification requires distinguishing expiry of
the graph child from loss of the original authorized search.

We also tested a two-paper candidate search on the same development corpus.
It identified both primary papers from self-descriptions already present in
the first five ordinary Q4 results despite their opaque titles. It reused
the original embedding and split the 17 dense slots into nine global and
four per paper. The resulting candidates contained the full later list and
a partial earlier list. The full earlier list ranked first in its paper's
trigram search but missed its top 17 dense results. Q3 did not qualify and
kept its candidate set. This policy was designed after inspecting Q4.

A subsequent Q4 comparison charged both arms for the same five source reads
and two scoped metadata searches, within the existing three-action budget.
Both used ordinary graph admission and the local final scorer. The paired
search placed the later and earlier passages at dense ranks 15 and 16,
respectively. Both appeared in graph seed metadata, but neither received one
of the 30 source reads. The reranker never saw either passage. Two other
acquired sources changed. Both arms selected the same 11 sources, with
identical order and excerpt bytes, and delivered none of the mapped comparative
facts. We stopped before answer generation. This experiment isolates an
admission problem as well as the separate ranking miss seen when a
single-document follow-up did return the full later catalog. Do not enable
the paired routing policy on the strength of candidate-level recall.
