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
unchanged unresolved-context questions, not production-routed queries. One
separate all-lexeme OR contrast is being prepared; no adoption, activation or
projection backfill follows from this result.

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

The 171-test fixture repair, 32-test source-order repair and 24-test settings
repair are recorded separately from immutable r10. Completed P1/P2 reviews reject
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
