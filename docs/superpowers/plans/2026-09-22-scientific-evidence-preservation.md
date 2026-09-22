# Scientific Evidence Preservation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent avoidable loss of scientific evidence during reranking, packing, follow-ups and retrieval refinement, with measurable quality and bounded work.

**Architecture:** Retain authorized source text and provenance until a shared final selector chooses the evidence. Add verified scoring windows, explicit capacity policy, source continuity and a turn-level acquisition controller around the separately owned relevance/diversity selector. Validate on development before preparing a non-graph main backport.

**Tech Stack:** Python >=3.12, Django, existing LLM/reranker providers and cache, pytest/pytest-django, current retrieval evaluation runner.

**Spec:** [Scientific evidence preservation design](../specs/2026-09-22-scientific-evidence-preservation-design.md).

**Status:** Plan only. No application behavior or deployment is changed by committing these documents. The user requested implementation on development and a later backport; this task records the plan first.

## Global Constraints

- Python >=3.12; use existing Django, provider and cache infrastructure. No new service, dependency, database migration or model replacement.
- Keep the reranker pair context at 1024 tokens and existing GPU allocation. Never solve overflow by silently increasing either.
- Preserve authorization checks, source revalidation, citation identity and stored conversation history.
- Preserve bounded concurrency, cancellation, retry limits and a monotonic turn deadline; a fallback must not restart the turn's budget.
- Keep public tool row and citation schemas compatible; place new provenance in a private typed sidecar.
- Use one shared final relevance/diversity selector and one final answer synthesis. Existing bounded citation repair/cutoff recovery remains part of synthesis and must be counted separately.
- Introduce no imports from knowledge graph modules in the new evidence-preservation modules.
- New Python modules must remain within the repository's 300-line limit; extract focused helpers instead of raising reviewed limits.

## Ownership, order and validation environment

Gap 3 belongs to the separate adaptive evidence selection effort. Read its committed design/plan when available. This plan requires its fused candidate pool, comparable-score adapter and single final selector; implement/test the source modules independently, but do not integrate against an invented parallel selector. Resolve three contract amendments before integration: budgeted effective per-document capacity, window-pair accounting/coverage, and a legacy ordering compatibility adapter through the shared selector entry point. The design reproduces these amendments without depending on another uncommitted file.

Task order: 1 -> 2 -> 3 -> 4 -> 5 -> 6. Window preparation and follow-up resolution can be developed independently after Task 1's contracts settle; their integrations still follow Task 3. Keep each task disabled until its tests pass. Preserve unrelated work and use an isolated development worktree at execution time.

Existing paths below are relative to the repository root. New types live in their named modules; the other workstream's selector types remain owned there. Tests run from the repository root in Python >=3.12, with dependencies from `requirements.txt` and the CI-compatible disposable `pgvector/pgvector:pg16` instance. Check `.github/workflows/test-backend-frontend.yml` at execution time.

Use process-local `DJANGO_DEBUG=1`, `DJANGO_TESTING=1`, `DJANGO_SETTINGS_MODULE=aquillm.settings_test`, `LLM_CHOICE=OPENAI`, `MEM0_ENABLED=0`, dummy provider API keys, and explicit isolated local PostgreSQL settings. Do not source operator `.env` or use production. Django fixes its test database name to `test`: changing `POSTGRES_NAME` alone does not isolate concurrent runs. Use a separate PostgreSQL instance or reviewed per-run test settings with database/extension creation permission. Do not weaken authorization or skip database tests to obtain green results.

All test commands below assume that environment and use `--ds=aquillm.settings_test`. Start with `rtk proxy python aquillm/manage.py check --settings=aquillm.settings_test`. The existing `run_rag_eval.py` and WebSocket smoke tests use mocked retrieval/provider behavior; retain them as routing regressions, never as proof of scientific answer quality.

## Task 1: source, coverage and shared work contracts

**Files:** Create `aquillm/lib/retrieval/__init__.py`, `evidence.py`, `turn_budget.py` in that package and `aquillm/apps/chat/services/rag_preservation_config.py`; create `aquillm/apps/chat/tests/test_rag_turn_budget.py`, `test_rag_source_types.py`. Modify `.env.example` and `aquillm/apps/chat/services/rag_config.py` only to expose explicit opt-in configuration. Shared source/ledger types belong in `lib.retrieval` so provider/tool modules can consume them without importing `apps.*`; inject authorization/hydration callbacks from the application layer.

**Interfaces:** Immutable `SourceEvidence(chunk_id: int, document_id: str, chunk_number: int, source_fingerprint: str, text: str)`; `SourceSpan(chunk_id: int, source_fingerprint: str, start: int, end: int, text: str)`; `PreparedEvidence(source: SourceEvidence, spans: tuple[SourceSpan, ...], evidence_fingerprint: str, estimated_tokens: int, source_coverage: str)`. Separately record `prepared_input_scoring_coverage` on score results. Both coverage fields validate the literal values `complete`, `partial`, `unknown`; scoring all of a chosen span does not mean the full source was covered.

`TurnBudget(limits, *, clock)` exposes `reserve_action(signature) -> bool`, `admit_source(identity) -> bool`, `reserve_pairs(count, *, phase) -> bool`, `reserve_text(count, *, kind) -> bool`, `remaining_ms() -> int`, `close(reason) -> None`, `can_publish() -> bool`. `TurnLimits` carries the numerical pilot limits from the spec; phases are `acquisition` or `final`; text kinds are `materialized` or `tokenized`. Source identities include revision; duplicate admission returns true without charging twice. Reservations are atomic across concurrent callers. No caller receives a fresh ledger during fallback. `can_publish` governs late retrieval/cache writes; closing retrieval does not forbid answering from an already validated frozen packet.

- [ ] Add tests before implementation for exact Unicode span slicing, fingerprint changes, duplicate admissions, 45 cumulative unique sources, 3 actions, reserved final pairs, retry charges, 250,000 materialized/1,000,000 tokenized code-point limits, closure and a fake monotonic deadline. Oversized sources must be rejected before full hydration, without a silent prefix substitute. Include this concrete budget assertion:

  ```python
  budget = TurnBudget(TurnLimits(), clock=lambda: 0.0)
  assert budget.reserve_pairs(90, phase="acquisition")
  assert not budget.reserve_pairs(1, phase="acquisition")
  assert budget.reserve_pairs(45, phase="final")
  assert not budget.reserve_pairs(1, phase="final")
  budget.close("cancelled")
  assert not budget.can_publish()
  ```

- [ ] Run `rtk proxy python -m pytest --ds=aquillm.settings_test aquillm/apps/chat/tests/test_rag_turn_budget.py aquillm/apps/chat/tests/test_rag_source_types.py -q`; confirm failure reflects missing contracts, not missing test infrastructure.
- [ ] Implement frozen data records, exact span validation and a lock-protected ledger; parse independent flags with legacy/off defaults and reject invalid limits. Use a monotonic clock injected for tests. Hard caps are clamped to available parent allowances, never raised by nested callers.

  ```python
  # Atomic reservation logic; retries use this same path.
  if closed or clock() >= deadline or used[phase] + count > allowance[phase]:
      return False
  used[phase] += count
  return True
  ```

- [ ] Repeat the focused tests, including concurrent reservation and late-publication tests. Commit only task-owned files as `feat: define source evidence and bounded retrieval contracts`.

## Task 2: score complete source windows with truthful coverage

**Files:** Create `aquillm/apps/documents/services/chunk_rerank_windows.py`, `chunk_rerank_window_scores.py`; modify existing `chunk_rerank_budget.py`, `chunk_rerank_local_vllm.py`, `chunk_rerank_payload.py`, `chunk_rerank_config.py`, and the shared score/cache adapter supplied by the selection workstream. Create `aquillm/apps/documents/tests/test_chunk_rerank_windows.py`, `test_chunk_rerank_window_scores.py`; extend `test_rerank_http_cache.py`.

**Interfaces:** `prepare_source_windows(query, source, *, pair_counter, pair_limit=1024) -> WindowPlan`. `WindowPlan` contains exact query identity, ordered `SourceSpan` windows, tokenizer/template identity, required window IDs and preparation coverage. `pair_counter` returns the complete model-pair token count or `None` for unknown. `score_window_plan(plan, *, scorer, budget) -> WindowScoreSet` records each successful exact pair, retries and coverage. `aggregate_window_scores(scores) -> ChunkWindowScore` produces `window-max-v1` and complete/partial/unknown status. Adapt this to the selection workstream's score types rather than defining competing `PassageScore` semantics.

- [ ] Add deterministic fixtures: decisive result after character 1200; negation on a window boundary; query longer than pair capacity; Unicode units/equations; unequal numbers of windows; 400 retry to a shorter input; unknown server truncation; timeout after one window. Test actual tail coverage, not merely number of windows.

  ```python
  text = "Background. " * 120 + "Treatment did not improve survival at 5 mg/kg."
  source = SourceEvidence(1, "paper-a", 0, "revision-a", text)
  plan = prepare_source_windows("Did survival improve?", source,
                                pair_counter=fake_pair_counter, pair_limit=1024)
  assert plan.windows[-1].end == len(text)
  assert all(w.text == text[w.start:w.end] for w in plan.windows)
  assert any("did not improve survival" in w.text for w in plan.windows)
  ```

- [ ] Run `rtk proxy python -m pytest --ds=aquillm.settings_test aquillm/apps/documents/tests/test_chunk_rerank_windows.py aquillm/apps/documents/tests/test_chunk_rerank_window_scores.py aquillm/apps/documents/tests/test_rerank_http_cache.py -q` and confirm the new regressions fail on legacy preparation.
- [ ] Implement full-pair-first preparation, boundary-aware windows with the specified overlap/tail coverage, verified tokenization capability, and rank fallback when the complete question cannot fit. Keep source offsets exact; use conservative estimation without a false complete-coverage claim when verification is unavailable.

  ```python
  if verified_pair_count is not None and verified_pair_count <= 1024:
      return full_source_plan(query, source)
  # Each window is an exact slice; the last ends at len(source.text).
  return covering_window_plan(query, source, overlap_tokens=64)
  ```

- [ ] Wire global pair reservations, <=6 in-flight pairs and finite transport deadlines into every request/retry. Charge repeated tokenization input, reuse offset mappings and check time between source/window operations. Cache only the exact successful query/window inputs. Unknown or incomplete final prepared-input scoring coverage maps to whole-pool rank fallback. Closed ledgers reject late cache writes/results.
- [ ] Run new and existing reranker budget/cache tests. Verify shadow mode requires a separate explicit scoring opt-in and cannot spend unbounded extra inference. Commit as `feat: preserve source coverage in reranker preparation`.

## Task 3: full evidence delivery and explicit document capacity

**Files:** Create `aquillm/apps/chat/services/rag_source_preparation.py`, `rag_context_budget.py`; modify `rag_evidence.py`, `rag_evidence_handoff.py`, `rag_config.py`, `services/tool_wiring/documents.py`, `consumers/utils.py` under chat and `aquillm/lib/tools/search/vector_search.py`. Integrate the selection workstream's candidate hydration/score preparation/packet assembly seams. Create `aquillm/apps/chat/tests/test_rag_source_preparation.py`, `test_rag_context_budget.py`; extend evidence, handoff and payload compaction tests. Update `.env.example`.

**Interfaces:** `prepare_evidence(source, *, question, windows, token_ceiling) -> PreparedEvidence` retains full text if it fits the ceiling, otherwise exact qualified spans. `available_evidence_tokens(*, model_context, prompt_tokens, output_reserve, safety_margin, configured_budget) -> int`. `resolve_document_cap(*, mode, legacy_cap, explicit_hard_cap, final_passage_limit) -> int` supplies the shared selector; no quota filter after selection.

- [ ] Add fixtures proving tail text survives retrieval -> selection -> synthesis input; five complementary same-paper passages fit; explicit hard cap remains enforced; an oversized skipped candidate does not consume slots; a later candidate still fits; small-context requests include image/tool/history/citation overhead. Preserve public citation IDs and full/compact tool shapes.

  ```python
  assert resolve_document_cap(mode="budgeted", legacy_cap=3,
                             explicit_hard_cap=0, final_passage_limit=10) == 10
  assert resolve_document_cap(mode="budgeted", legacy_cap=3,
                             explicit_hard_cap=4, final_passage_limit=10) == 4
  assert available_evidence_tokens(model_context=8192, prompt_tokens=3000,
      output_reserve=4096, safety_margin=256, configured_budget=3500) == 840
  ```

- [ ] Run `rtk proxy python -m pytest --ds=aquillm.settings_test aquillm/apps/chat/tests/test_rag_source_preparation.py aquillm/apps/chat/tests/test_rag_context_budget.py aquillm/apps/chat/tests/test_rag_evidence.py aquillm/apps/chat/tests/test_rag_evidence_handoff.py aquillm/apps/chat/tests/test_tool_payload_compaction.py -q`; confirm targeted failures.
- [ ] Preflight source lengths and hydrate full sources independently of clipped public previews; prepare one representation per source before comparable final scoring. Optional acquisition scores can guide oversized spans; final scores are computed only after preparation. Use full chunks when feasible; oversized representations retain scored/requested spans and qualifications. Pass identical prepared text to score provenance, redundancy, token cost and synthesis; remove any subsequent prefix clipping of selected evidence.

  ```python
  cap = min(final_passage_limit, explicit_hard_cap or final_passage_limit)
  ceiling = max(0, min(configured_budget,
      model_context - prompt_tokens - output_reserve - safety_margin))
  # Legacy mode still resolves the old cap explicitly; no silent reinterpretation.
  ```

- [ ] Delegate relevance/order to the shared selector once, using its legacy compatibility adapter or adaptive policy according to the supported mode matrix in the design. Both consume the resolved cap; no old top-k/quota pruning precedes them. Reject unsupported feature combinations at configuration validation. Revalidate at handoff; if scope/content changed, exclude or rehydrate within remaining budget, never refill from stale cached text. Keep stored history unchanged. Zero available context yields an explicit limited outcome instead of sending an overflowing prompt.
- [ ] Run focused tests plus `test_document_tool_figures.py` and the selection workstream's tests. Demonstrate five necessary short passages survive while redundant passages are handled by the existing selector. Commit as `feat: preserve scientific evidence within explicit context budgets`.

## Task 4: resolve follow-up sources and rehydrate relevant evidence

**Files:** Create `aquillm/apps/chat/services/rag_source_continuity.py`; modify `rag_query.py`, `rag_pipeline.py`, `rag_evidence_handoff.py`. Create `aquillm/apps/chat/tests/test_rag_source_continuity.py`; extend `test_rag_query.py`, `test_rag_evidence_handoff.py`.

**Interfaces:** `resolve_source_anchors(question, history) -> SourceAnchors`, carrying ordered document/chunk identities, resolution basis and unresolved references. `rehydrate_prior_evidence(anchors, *, user, selected_scope, budget) -> tuple[SourceEvidence, ...]` uses current authorization/storage and charges the common ledger. History supplies references, never raw trusted evidence or authorization.

- [ ] Add two papers with reversed alphabetical/display order and different measurements. Test “compare their measurements,” “both papers,” “the second paper,” an explicit citation, duplicate titles, a topic switch, and ambiguous references. Add revoked/deleted/moved/edited source fixtures and an immutable original-conversation assertion.

  ```python
  anchors = resolve_source_anchors("Compare their measurements", two_paper_history)
  assert anchors.document_ids == ("paper-z", "paper-a")  # conversational order
  ordinal = resolve_source_anchors("Explain the second paper", two_paper_history)
  assert ordinal.document_ids == ("paper-a",)
  ```

- [ ] Run `rtk proxy python -m pytest --ds=aquillm.settings_test aquillm/apps/chat/tests/test_rag_source_continuity.py aquillm/apps/chat/tests/test_rag_query.py aquillm/apps/chat/tests/test_rag_evidence_handoff.py -q` and observe the old first-title behavior fail.
- [ ] Implement explicit identity/citation resolution followed by contextual plural/ordinal resolution; ambiguous materially different anchors remain unresolved. Rehydrate relevant references, apply current scope, and admit them to the common pool. Do not restore all old tool text or old assistant assertions into the evidence packet.

  ```python
  for identity in anchors.chunk_identities:
      source = authorized_current_source(identity, user, selected_scope)
      if source is not None and budget.admit_source(source_identity(source)):
          candidates.append(source)
  ```

- [ ] Repeat tests with source authorization revoked between hydration and synthesis. Confirm changed content is rescored and previous citations cannot bypass the new allowlist. Commit as `feat: preserve authorized source continuity in follow-up questions`.

## Task 5: targeted acquisition rounds and meaningful completion

**Files:** Create `aquillm/apps/chat/services/rag_acquisition.py`, `rag_coverage.py`; modify `rag_pipeline.py`, `rag_synthesis.py`, `rag_metrics.py`, and the document tool adapters. Extract a focused helper from `aquillm/lib/llm/providers/complete_turn.py` if the provider seam needs extension; do not grow its reviewed limit. Create `aquillm/apps/chat/tests/test_rag_acquisition.py`, `test_rag_coverage.py`; extend direct pipeline and synthesis tests.

**Interfaces:** `assess_coverage(question, evidence_views, anchors, *, llm, budget) -> CoverageAssessment`; record `requested_aspects`, validated support span references, `unresolved_aspects`, optional `next_action`, and assessment certainty. `AcquisitionAction(kind, query, document_id, chunk_id)` permits only vector/single-document/adjacent actions valid for the selected scope. `acquire_evidence(..., budget) -> AcquiredEvidence` returns sources, assessment, budget stop reason and rounds; it does not synthesize. Recheck coverage by source revision and containment of support offsets in actual delivered spans, not merely by selected chunk IDs.

- [ ] Create scripted fake-search/fake-planner fixtures: initial method-only result then second query finds the requested measurement; sufficient first-round result; zero initial results then recovery; follow-up with no new spans; duplicate query; invented/out-of-scope ID; malformed planner JSON; planner timeout; cancellation; global budget exhaustion; direct-path fallback. Include a retained chunk whose supporting tail span was omitted, and a third-action recovery with exactly two planner calls. Spy on final selection/synthesis counts.

  ```python
  result = await run_scripted_turn("What was the measured yield?", scenario="second_search")
  assert result.acquisition_actions == 2
  assert "yield-evidence" in result.final_packet_ids
  assert result.selection_calls == result.final_synthesis_calls == 1
  assert result.rerank_pairs <= 135
  assert result.final_scoring_pairs <= 45
  ```

- [ ] Run `rtk proxy python -m pytest --ds=aquillm.settings_test aquillm/apps/chat/tests/test_rag_acquisition.py aquillm/apps/chat/tests/test_rag_coverage.py aquillm/apps/chat/tests/test_direct_rag_pipeline.py aquillm/apps/chat/tests/test_rag_synthesis.py -q`; verify missing refinement and incorrect stop behavior fail before implementation.
- [ ] Run one initial action, then validate at most two planner-proposed actions. Pass budgeted evidence views with exact source references and omission state; no unbounded full-pool planner prompt. Use the same provider without a new selection judge. Enforce the spec's action, planner, pair, source, concurrency and monotonic time limits throughout.

  ```python
  for decision_index in range(2):  # Calls can propose actions two and three.
      if budget.remaining_ms() <= 0:
          break
      assessment = await assess_current_pool()
      if not assessment.next_action or not reserve_valid_action(assessment.next_action):
          break
      additions = await execute_scoped_action(assessment.next_action)
      if not adds_new_authorized_source_or_span(additions):
          break
  # Comparable final scoring and selection happen once, outside acquisition.
  ```

- [ ] After action three, make no additional planner call. Validate already known support structurally and label new evidence unassessed; final synthesis may use it normally without the controller claiming semantic completeness. Add assertions for two planner calls, three actions and honest final coverage state.

- [ ] Convert deadline/provider failure into an explicit partial/unknown retrieval outcome; preserve the available authorized evidence. Cancellation propagates and fences late work. A normal-loop fallback inherits the ledger and existing stricter tool limits; it cannot restart exhausted retrieval.
- [ ] Synthesize supported portions at the depth requested, retaining units, conditions and limitations. Do not claim complete support based on a planner decision if final selection omitted its cited spans. Keep output reserves, numeric/citation validation and bounded repair/continuation; remove only unconditional brevity conflicting with the user's request.
- [ ] Run focused tests, existing direct WebSocket smoke tests and `aquillm/lib/llm/tests/test_spin_tool_budget.py`. Commit as `feat: refine retrieval within a shared scientific evidence budget`.

## Task 6: quality gates, rollout evidence and future backport manifest

**Files:** Create `aquillm/apps/chat/evals/run_scientific_evidence_eval.py`, focused helpers `scientific_evidence_eval.py` and `scientific_evidence_cases.json` in that directory, `aquillm/apps/chat/tests/test_scientific_evidence_eval.py`, and `docs/runbooks/scientific-evidence-preservation.md`. Leave the existing canned routing runner intact. Extend existing redaction tests and `.env.example` comments. Reuse the selection corpus by stable case ID when committed; do not silently change its held-out labels.

**Evaluation interface:** The new runner accepts `--cases PATH`, `--report PATH`, `--split development|heldout`, `--mode baseline|selection|preservation|combined`, and `--backend fixture|live`. Fixture mode tests metric/plumbing behavior; only live mode with actual isolated retrieval, reranking and answer providers can qualify an activation gate. Live mode uses a dedicated evaluation collection and licensed/public/synthetic content, not a production user's data. Fixtures record case ID, split, question/turns, authorized source revisions, gold support offsets, required numerical qualifications, permitted citations and expected missing aspects. Results record packet support recall, human-reviewed answer faithfulness, numerical/condition accuracy, nDCG where available, stage timings, inference pairs, coverage/fallback state and stop reason. Report run revisions and label fixture-mode results ineligible for quality claims.

- [ ] Create 40 development and 40 held-out cases, stratified across the four gaps, boundary/contradiction cases and authorization failures. Use licensed/public or synthetic papers; no production private content. Freeze held-out labels before tuning. Define six mandatory regression classes: tail/boundary; >3 complementary same-paper passages; plural/ordinal follow-up; second-search recovery; revoked/stale evidence; cancellation/global limits.
- [ ] Write evaluator tests with deliberately omitted support, swapped units, dropped negation and unsupported citations. Verify metrics catch each failure and paired comparisons join by case ID rather than file order.

  ```python
  # Gold support is source-offset based, not "answer is longer".
  assert support_recall(required={"tail-result", "dose-condition"},
                        delivered={"tail-result"}) == 0.5
  assert support_recall(required={"tail-result", "dose-condition"},
                        delivered={"tail-result", "dose-condition"}) == 1.0
  ```

- [ ] Run `rtk proxy python -m pytest --ds=aquillm.settings_test aquillm/apps/chat/tests/test_scientific_evidence_eval.py aquillm/apps/chat/tests/test_rag_eval_runner.py -q`, first red then green. Implement the separate runner with the specified flags and source-span metrics; save versioned JSON reports outside tracked private data. Example after configuring the isolated evaluation environment: `rtk proxy python aquillm/apps/chat/evals/run_scientific_evidence_eval.py --cases aquillm/apps/chat/evals/scientific_evidence_cases.json --report artifacts/evidence-preservation/development-combined.json --split development --mode combined --backend live`.
- [ ] Evaluate baseline, selection-only, preservation-only and combined modes on identical model/source revisions. Record positive paired support-recall change on targeted cases, no observed aggregate quality decline, bootstrap uncertainty and zero authorization/citation violations. Include reranker window-count bias and the rate at which incomplete windows force rank fallback. Inconclusive evidence does not justify activation.
- [ ] Run concurrent-load and cancellation tests. Record p50/p95 per stage and end-to-end; enforce the 15-second retrieval deadline plus measured cancellation overhead and <=20% end-to-end p95 regression versus the matching baseline cohort. Verify first-round sufficient answers stop early and no late work publishes after closure.
- [ ] Run the existing CI checks and targeted Python suites; review code independently before enabling anything. Run `rtk git diff --check`, `rtk proxy python scripts/check_file_lengths.py`, `rtk proxy python scripts/check_import_boundaries.py`, `rtk proxy python scripts/check_logging_conventions.py`, `rtk proxy python scripts/check_retrieval_logging.py`. Resolve actual findings without expanding unrelated scope.
- [ ] Explicitly run tests not discovered by `pytest.ini`: `rtk proxy python -m pytest --ds=aquillm.settings_test aquillm/lib/tools/search/tests/test_vector_search_pack.py aquillm/apps/collections/tests/test_retrieval_authorization.py aquillm/apps/collections/tests/test_django_retrieval_authorization.py -q --tb=short`. Also run current citation/context/answer-completion regressions: `test_citation_api.py`, `test_document_move.py`, `test_conversation_persistence.py`, `test_numeric_rag_citations.py`, `test_direct_synthesis_grounding.py`, `test_context_packer.py`, `test_prompt_budget.py`, and `test_llm_complete_retry.py` in their existing test directories. On development include `aquillm/tests/integration/test_retrieval_authorization_propagation.py`, `test_production_retrieval_authorization_reachability.py`, and existing document graph-overlay/reranker-authority tests to detect integration regressions without porting graph code to main.
- [ ] Write the runbook with default-off flags, exact tested revisions/configurations, metrics, limits, disable/rollback procedure and independent activation order: source/windowing -> capacity -> follow-ups -> iterative acquisition. Shadow scoring needs its own workload allowance; record user-visible effects of each flag.
- [ ] Create a backport manifest listing exact development commits, required selector dependencies, target-main authorization adapters, tests and configuration differences. Keep graph/PageRank changes excluded. Commit evaluation/runbook work as `test: gate scientific evidence preservation on quality and latency`.

## Later integration and deployment checklist

- [ ] After development acceptance, create a separate main-based backport branch/PR. Review every dependency and adapt authorization without introducing graph imports or weakening production safeguards.
- [ ] Run identical scientific fixtures on development and the backport, plus target-main runtime/deployment checks. Explain any measured differences in the PR; do not use a successful build as evidence of retrieval quality.
- [ ] After the backport is merged and deployment is authorized, take a verified backup, deploy the exact merged revision, verify runtime config/image identity, health/worker state and a representative real retrieval including tail/follow-up evidence.
- [ ] Record rollback commands and deployed revision. Enable only modes that passed the development and target-main quality/latency gates.

**Completion of the current request:** Commit this plan and its companion design to `development` only. Application implementation, backport, feature activation and deployment are not performed in this planning task.
