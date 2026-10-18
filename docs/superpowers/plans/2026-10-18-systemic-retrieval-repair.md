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

- [ ] Reproduce the four known failures using the root-owned archived test runner. Baseline evidence exists in private `publication-db-20261018T101014040538Z`.
- [ ] Inspect why the fake planner fails and update it to the actual provider call contract. Identify planner calls using the imported contract, not obsolete prose. Preserve the asserted single planner call, single synthesis and missing source/tail support:

```python
assert acquired.assessment.support
assert not packet.coverage_assessment.support
assert runtime.budget.planner_calls == len(planners) == 1
assert len(llm.calls) == 2
```

- [ ] Update timeout diagnostics to the actual phase-specific contract while preserving the timing and late-worker assertions:

```python
assert task.result().stop_reason == "acquisition_timeout"
assert late == []
assert budget.actions_used == 2
```

- [ ] Ask root to run both test files with the archived runner. Review the diff, commit only these test changes, and report the exact evidence path.

### Task 2: Bounded planner memory and useful source scope

**Files:**
- Modify: `aquillm/apps/chat/services/rag_acquisition.py`
- Modify: `aquillm/apps/chat/services/rag_coverage.py`
- Modify: `aquillm/apps/chat/services/rag_planner_contract.py`
- Create: `aquillm/apps/chat/services/rag_planner_context.py`
- Create: `aquillm/apps/chat/tests/test_rag_planner_context.py`
- Modify relevant existing acquisition/coverage tests.

**Interfaces:** Extend `assess_coverage(..., attempts=(), latest_source_ids=())` with an optional bounded history of action kind/query/document/chunk and result/new-source counts. Existing injected assess callbacks accepting keyword arguments remain compatible. `planner_views(evidence_views, max_chars=16000, latest_source_ids=())` returns complete immutable source views, ordered in document round-robin. Prioritize groups with newly acquired sources and new sources within each group, retaining stable order otherwise. This is planner context only; final ranking remains unchanged.

- [ ] Add failing synthetic tests: a first document with many chunks cannot hide a second document; a later planner sees newly acquired source views; an oversized source is not clipped into mismatched offsets; source text totals stay within 16000 characters; repeated source identities do not consume context twice.
- [ ] Add failing orchestration tests showing the second planner receives the first attempted query/outcome, and initial external handoff is represented without inventing an unknown query. History is bounded by admitted actions, includes no raw source text, and records action failures without provider bodies.
- [ ] Add a provider-independent test rejecting a document action for an authorized but unseen document. Validate against precisely the planner-visible source set; adjacent references must also bind to that set.
- [ ] Implement small context helpers and thread history through the existing acquisition loop. Use source-grounded terms for a new global search; use document search for a known relevant source, adjacency only for local context. Do not force document search for every question.
- [ ] Keep canonical duplicate detection conservative: case/Unicode/whitespace normalization may reject identical requests, but negations, numbers, scientific punctuation and different document scope must remain distinct. Do not add a semantic deduplicator.
- [ ] Run focused context, coverage, acquisition, handoff and budget tests through root, inspect observed prompts/counts, commit and review.

```python
views = planner_views((a1, a2, b1), max_chars=len(a1.text) + len(b1.text))
assert views == (a1, b1)
assert sum(len(view.text) for view in views) <= 16000
```

### Task 3: Version graph extraction behavior consistently

**Files:**
- Create: `aquillm/lib/knowledge_graph/extraction_identity.py`
- Modify: `aquillm/apps/knowledge_graph/services/pipeline.py`
- Modify: `aquillm/apps/knowledge_graph/services/builds.py`
- Modify: `aquillm/apps/knowledge_graph/evals/run_kg_eval.py`
- Create focused identity tests alongside existing extraction/build tests.

**Interfaces:** A pure `extraction_identity(settings)` function returns a canonical behavior fingerprint fitting the existing 128-character extractor field. It binds provider, immutable checkpoint, adapter/schema-contract version and effective inference settings. Production construction, worker comparison and evaluation use this same function. No model import is permitted in this helper.

- [ ] Write failing tests: stable identity for equivalent settings; identity changes with adapter/schema-contract/inference behavior; operational cache location is not semantic identity; production and evaluation agree.
- [ ] Inspect actual adapter inference options and pinned package version, then define a canonical sorted JSON manifest and prefixed SHA256 identity. Avoid hashing environment secrets or whole environment variables.
- [ ] Replace duplicated identity construction and verify existing artifacts are not rewritten, while newly requested artifacts receive distinct build keys. Preserve existing source/ontology/resolver/filter bindings.
- [ ] Run focused pure and database build identity regressions; inspect all configured-extractor call sites, commit and review.

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

- [ ] Add failing path-matrix tests that inspect actual provider requests: each path preserves exact source text, source citation coordinates, qualifier requirements and allowance for a concise partial answer.
- [ ] Remove fixed 400-900-word and unrelated methods/math requirements from post-tool recovery. Preserve relevant user requirements and main-model generation.
- [ ] Ensure citation repair can remove unsupported assertions and restore source qualifiers; never tell it to preserve every previous claim. Do not enable the rejected model critic or claim deterministic entailment.
- [ ] Preserve publication authorization, current evidence receipts, source-loss fallback and useful unknown-support synthesis. Run focused synthesis/retry/repair/publication regressions, commit and review.

```python
assert supplied_source_text in captured_request
assert source_citation in captured_request
assert "400-900" not in recovery_instruction
```

### Task 5: Measured graph coverage and candidate acceptance

**Files:** Private evidence under `.superpowers/sdd/2026-10-18-systemic-retrieval-repair/`; tracked documentation only contains aggregate findings and reproducible general procedure.

**Interfaces:** Consume the reviewed candidate from Tasks 2-4, current active ontology and pinned corpus/runtime manifests. Produce extraction rejection-stage evidence, full-study/held-out comparison, and an explicit decision on graph schema/adapter changes.

- [ ] Inspect the saved read-only active ontology/build stats. Freeze general synthetic relation examples and real held-out passages before candidate extraction. Include supported relations, unsupported/ambiguous endpoints, negation and alternate dataset/metric types.
- [ ] Run bounded offline extraction with the existing adapter and record raw-output, normalization and endpoint-validation counts. Do not infer passage-level rejection reasons from document aggregates.
- [ ] If the evidence identifies a correctable adapter/schema defect, write a focused design addendum and failing regression before a versioned implementation. Require source provenance and typed endpoints; do not disable validation or invent graph facts. Evaluate the repair on the frozen set before any background rebuild.
- [ ] Stage an immutable isolated candidate on 254. Run the full ten-question study and pre-frozen held-out questions serially with matched settings; record actions, recall, final evidence, factual correctness, citations, latency and failures separately. Preserve real model prose for independent review.
- [ ] Run repository checks and review the whole change. Update PR247 and the authorized internal tracking issue with code, artifact, evaluation and deployment status. Keep unproven quality improvements explicit; do not merge/deploy on unit tests alone.

### Additional confirmed defects (within systemic repair scope)

- **Graph compiler experiment (Task 3/5):** pinned GLiNER2 drops relation descriptions. Add a deterministic description-preserving structure/field compiler, version its behavior, retain endpoint validation and matched inference options, and repeat the frozen offline sample. Do not activate or rebuild unless semantic precision/recall evidence supports it.
- **Large source snapshot serialization:** add a new reversible migration replacing the quadratic canonical string serializer with an equivalent bounded-work implementation. Preserve canonical bytes and immutable hash guards. Test realistic large mixed text, full escaping/Unicode boundaries, old/new equivalence, and normal adjacent-edit integration under a bounded statement timeout.
- **Local anaphora:** repair same-question noun/pronoun handling using general grammar, retaining explicit-source and ambiguity barriers. Test generic local nouns, real historical-source references, ambiguous cases and the pre-retrieval routing boundary. Keep frozen held-out questions unchanged.

Disjoint SQL, graph and synthesis implementation may proceed in parallel as the
user requested. Root serializes shared tests, model calls and commits; each change
receives independent review before final candidate evaluation.
