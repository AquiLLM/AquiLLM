# Retrieval evidence acceptance ledger

Status: review candidate; final composed verification remains pending.
**All twelve treatment decisions are inconclusive; no activation is authorized.** Passing invariants, fixture reports, source review,
quality acceptance and deployment are separate outcomes. This ledger changes no
configuration. See the [treatment register](../../runbooks/rag-remediation-treatments.md),
[evidence gates](../../runbooks/evidence-preservation.md) and
[rollout boundaries](../../runbooks/rag-remediation-rollout.md).

## Evidence delivered

| Evidence | Completed observation | Acceptance limit |
| --- | --- | --- |
| [Whole-source delivery](../../../aquillm/apps/chat/tests/test_rag_suffix_evidence_delivery.py) and [window reuse](../../../aquillm/apps/chat/tests/test_rag_suffix_window_reuse.py) | Synthetic whole turns retain a 2048-codepoint source and suffix after 1600 through scoring, sealed SDK request and real publication; current authority, conditions and compatible score reuse are checked. The integrated delivery/capture/catalog/fixture slice passed 72 tests. | Protocol/publication invariants; no semantic calibration, deployment or independent four-treatment acceptance. |
| [Fixture CLI schedule](../../../aquillm/apps/chat/tests/test_evidence_fixture_cli.py) | 22 actual subprocesses: eight quality reports covering 40 development and 40 held-out cases in four arms, twelve operational reports covering all 30 scheduled IDs, and two comparisons. Twenty exits were zero; both activation-required comparisons exited two. | **Fixture backend only: zero live-model studies and zero independent human reviews.** Blocked comparison decisions remain blocked. The finite 80/four-arm and 30-operation structure is not evidence that a live study occurred. |
| [Graph capture](../../../aquillm/apps/chat/tests/test_evidence_graph_capture.py) and [treatment catalog](../../../aquillm/apps/chat/evals/retrieval_treatment_manifest.py) | Original event order/payloads and artifact digests are retained; configured and observed fields remain distinct. | Neither a catalog entry nor complete frontier counters creates missing runtime, source-generation, execution, attachment or reviewer identity. |
| [Source/index synthetic artifact](source-index-synthetic-measurements.json) | Five cases/eight outputs, original extraction/structure statuses, byte-token counts and single-sample Python measurements are retained. | Fixture fidelity only; no provider/GPU operation, semantic quality, production cost, capacity or p95 result. |

R4 and I5 are integrated into the verification candidate. Native structural,
whole-source and fixture CLI controls have executed; query and projection
integration repairs are under verification. Historical results do not certify the
final composed candidate. Original failure reports remain in the evidence bundle.

## Independent treatment decisions

Every row additionally requires the common human/runtime evidence below. “Off”
means no optional treatment activation from this program; integrity, citation and
current-authority safeguards remain in force. Existing baseline behavior remains.

| Treatment | Current evidence / missing treatment-specific proof | Decision and default impact |
| --- | --- | --- |
| Window/source | Delivery/reuse invariants delivered; matched legacy/window/source/both quality and operational results missing. | **Inconclusive/off**; text modes remain legacy. |
| Retrieval fusion | Implementation/invariants delivered; matched exact/FTS/RRF/protected-slot quality outcomes and final channel-competition verification remain pending. | **Inconclusive/off**; retain reviewed baseline fusion configuration. |
| Query proposals | Original/vocabulary/rewrite controls have executed; query integration repairs are under verification. Matched semantic acceptance remains missing. | **Inconclusive/off**; optional vocabulary/rewrite disabled. |
| Candidate depth | No accepted independent bounded-depth comparison or denial/latency study. | **Inconclusive/off**; baseline depth and shared limits unchanged. |
| Structure | Native structural delivery and exact row-membership ordering controls passed. Matched semantic acceptance and parser-only transition remain unmeasured. | **Inconclusive/off**; no serving section/downweighting policy activated. |
| Embedding preparation | Verified producer/binding mechanisms delivered; merged coverage and matched semantic comparison pending. | **Inconclusive/off** for treatment; only genuinely compatible current generations may serve. |
| Graph | Capture and full-route controls have runtime observations; projection integration repairs and final composed verification remain pending. Strict G6 acceptance is not established. | **Inconclusive/off**; retain reviewed baseline graph settings. |
| Document capacity | Independent capacity study and supported exclusion accounting missing. | **Inconclusive/off**; capacity mode legacy. |
| Follow-up | Independent current-source rehydration and quality decision missing. | **Inconclusive/off**; follow-up disabled. |
| Iterative acquisition | Actual refinement/shared exhaustion and independent operational decision missing. | **Inconclusive/off**; iterative acquisition disabled. |
| Adaptive selection | Independent matched selection/quality gate not accepted. | **Inconclusive/off**; selection mode legacy. |
| Reranker/model | No accepted separately preregistered replacement study or measured deployment resource identity. | **Inconclusive/off**; no model replacement or stale-capability reuse. |

The completed exclusive [physical-vector diagnostic](physical-vector-observations.json)
retains 800 paired comparisons and 640 explained SQL statements across
1/8/64/512/4096 eligible chunks. Raw and validated recall@1/5/10 were 1.0, with
zero underfill or observed ineligible output. No plan used HNSW despite a valid
index, so **no ANN speedup is established**. Original limits and source/index
pre/post identity were preserved. Timings describe synthetic geometry on shared,
unreserved Docker host capacity, with no explicit per-container CPU or memory
reservation/limits; background activity and mixed cache state remain limitations.

## Evidence still required for an acceptance decision

- Actual matched live frozen 80-case/four-arm quality and 30-run operational
  reports, including every failure, cohort outcome and applicable exhaustion/
  refinement observation. Fixture reports cannot discharge these requirements.
- Independent human judgments on the exact answers, citations, support spans,
  conditions and omissions, with trusted identities bound to original reports.
  Calibration and current-run activation need their separate pinned artifacts.
- Observed serving model/tokenizer/template/engine identities, code/image and
  source/profile/generation bindings, hardware allocation, concurrency/cache
  conditions and target-runtime latency. No configured digest fills an absent
  observation; process-local handles are not deployment attestations.
- Final composed source-to-test bindings for native/query/graph and physical-vector
  controls, current-authority/publication and applicable rollback evidence, followed
  by independent final review. Unsupported/partial sources retain their status.

Trust remains administrator-controlled; missing D17 or strict G6 identity evidence
keeps the decision inconclusive/off. No production trust enrollment, cost saving,
semantic improvement, fleet capacity, p95 compliance or deployed F06 closure is
claimed. No production rebuild or deployment has been performed by this evidence
work. Record future decisions against exact immutable artifacts using the
[isolated verification procedure](../../runbooks/rag-remediation-testing.md) and
[source/index rollback contract](../../runbooks/rag-source-index-generations.md).
