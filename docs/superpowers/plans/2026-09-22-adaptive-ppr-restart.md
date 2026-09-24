# Adaptive PageRank Restart Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adapt the seed-restart versus graph-propagation ratio to the retrieval query and available branch signals without increasing retrieval or inference budgets.

**Architecture:** Preserve the authorized topology snapshot and legacy PPR path. A pure bounded policy consumes private seed summaries and admitted topology signals; a versioned execution adapter validates the original snapshot, runs the existing kernel once with the selected restart, and binds the effective computation into provenance.

**Tech Stack:** Existing Python 3.12+, Django, frozen dataclasses, hashlib/json, deterministic PPR kernel, pytest. No new dependencies or model calls.

**Spec:** [Adaptive PageRank Restart Design](../specs/2026-09-22-adaptive-ppr-restart-design.md).

**Parent:** [Adaptive retrieval and evidence-selection plan](2026-09-22-adaptive-evidence-selection.md). This is workstream B (pipeline step 6); workstream A covers step 8.

**Status:** Implemented and reviewed in development (PR #230); live representative quality/concurrent-latency gates remain pending. See [current implementation report](../reports/2026-09-22-adaptive-retrieval-implementation.md). Features remain default off. Preservation composes with the adaptive selector; the independent PPR activation gate remains open. This status update does not authorize activation or deployment.

## Global Constraints

- Preserve authorized scope, graph readiness, source identities, branch isolation, and final reranker authority.
- Keep existing graph node/edge/hop/seed/candidate caps and branch deadlines authoritative.
- Keep eight PPR iterations in the first production experiment; do not claim convergence.
- Run at most one PPR computation per enabled branch in serving mode; no online parameter sweep.
- Add no model calls, embeddings, graph fetches, services, database migrations, or graph rebuilds for the policy.
- Preserve existing seed masses, edge weights, and graph-to-passage scoring for the first experiment.
- Preserve fixed-mode output and legacy snapshot/config validation exactly.
- Record effective restart and execution identity separately from the original topology snapshot identity.
- Keep raw questions, entity text, document IDs, and score vectors out of normal logs.
- Enable and roll back PageRank adaptation independently of evidence selection.
- This draft changes documentation only; activation and benchmarks remain future work.

## File map and execution environment

Paths are relative to the repository root. Existing large modules should delegate to the new focused modules.

| File | Responsibility |
| --- | --- |
| aquillm/apps/knowledge_graph/retrieval/ppr_policy.py (new) | Closed signal/decision types and pure query/restart policy |
| aquillm/apps/knowledge_graph/retrieval/ppr_seed_support.py (new) | Numeric direct/extended support summaries; prepared seed type |
| aquillm/apps/knowledge_graph/retrieval/projected_ppr_execution.py (new) | Validated effective configuration, one kernel run, execution provenance |
| aquillm/apps/knowledge_graph/retrieval/projected_ppr.py | Share original validation and transition preparation; preserve legacy entry point |
| aquillm/apps/knowledge_graph/retrieval/production_direct.py; production_extended.py | Retain support summaries from already-performed preparation |
| aquillm/apps/knowledge_graph/retrieval/production_runtime.py; production_runtime_support.py | Mode dispatch, branch deadlines, explicit effective algorithm signature |
| aquillm/apps/knowledge_graph/retrieval/scheduler.py; scheduler_support.py | Pass query explicitly to extended preparation |
| aquillm/lib/knowledge_graph/retrieval_config.py | Closed fixed/shadow/adaptive configuration |
| aquillm/apps/knowledge_graph/evals/run_ppr_restart_eval.py (new) | Fixed-policy and adaptive offline comparison |
| aquillm/apps/knowledge_graph/evals/ppr_restart_cases.json (new) | Synthetic deterministic graph fixtures |
| docs/documents/operations/adaptive-ppr-restart.md (new) | Measurements, independent rollout and rollback |

Run tests from the repository root using the configured development environment and PostgreSQL/pgvector where required. Do not start model/gateway services merely to execute pure policy tests. Production parity checks use the repository's existing explicit evaluation capabilities; PostgreSQL is not a new production fallback.

## Task B1: Define and test the bounded policy

**Files:** Create ppr_policy.py and aquillm/apps/knowledge_graph/tests/test_ppr_policy.py.

**Interfaces:**

~~~python
@dataclass(frozen=True, slots=True)
class PPRPolicySignalsV1:
    branch_kind: HybridBranchKind
    intent: Literal["focused", "balanced", "relational"]
    seed_count: int
    support_status: Literal["supported", "insufficient", "unknown"]
    cap_pressure: bool
    outward_seed_mass: float
    support_digest: str

@dataclass(frozen=True, slots=True)
class PPRPolicyDecisionV1:
    policy_version: str
    restart: float
    iterations: int
    reason: str

def classify_ppr_intent(query: str, *, context_ambiguous: bool = False) -> str:
    import re
    if type(query) is not str or type(context_ambiguous) is not bool:
        raise TypeError("query and context_ambiguous must have exact types")
    if context_ambiguous or ":" in query or "\n" in query:
        return "balanced"
    text = " ".join(query.casefold().split())
    relational = (
        r"\b(?:relationship between|connection between|connected to|"
        r"path between|linked through)\b"
    )
    if re.search(relational, text):
        return "relational"
    if re.match(r"^(?:who|when|where|how many|define)\b", text):
        return "focused"
    return "balanced"

def choose_ppr_restart(
    signals: PPRPolicySignalsV1, *, mode: str
) -> PPRPolicyDecisionV1:
    # Construction of signals validates all closed fields and numeric bounds.
    if mode not in ("fixed", "shadow", "adaptive"):
        raise ValueError("unsupported restart mode")
    reason, restart = "balanced_intent", 0.20
    if mode == "fixed":
        reason = "fixed_mode"
    elif signals.support_status == "unknown":
        reason = "unknown_signals"
    elif signals.support_status == "insufficient":
        reason = "seed_support_insufficient"
    elif signals.cap_pressure:
        reason = "cap_pressure"
    elif signals.intent == "focused":
        reason, restart = "focused_supported", 0.35
    elif signals.intent == "relational":
        if signals.seed_count >= 2 and signals.outward_seed_mass >= 0.50:
            reason, restart = "relational_supported", 0.15
        else:
            reason = "insufficient_connections"
    return PPRPolicyDecisionV1("ppr_restart_policy_v1", restart, 8, reason)
~~~

The classifier deliberately abstains for colon-bearing and multiline queries: current rag_query.py can prepend a retrieved title as "title: question" without preserving a separate context marker. This may also abstain on legitimate identifiers containing colons; document and evaluate that tradeoff. Shadow proposes the adaptive decision but never changes the serving computation. Signals must have exact types, a positive bounded seed count, finite outward mass in [0,1], and a valid 64-hex support digest. Decisions permit only the versioned ratios, eight iterations, and spec-listed reasons.

- [ ] Add a failing numerical policy regression:

~~~python
def test_relational_exploration_requires_support_and_connections():
    from dataclasses import replace
    from apps.knowledge_graph.retrieval.ppr_policy import (
        PPRPolicySignalsV1, choose_ppr_restart,
    )
    from apps.knowledge_graph.retrieval.topology.contracts import HybridBranchKind
    signals = PPRPolicySignalsV1(
        HybridBranchKind.DIRECT, "relational", 2, "supported", False, 1.0, "a" * 64,
    )
    assert choose_ppr_restart(signals, mode="adaptive").restart == 0.15
    assert choose_ppr_restart(replace(signals, support_status="unknown"),
                              mode="adaptive").restart == 0.20
    assert choose_ppr_restart(replace(signals, cap_pressure=True),
                              mode="adaptive").restart == 0.20
    assert choose_ppr_restart(replace(signals, intent="focused"),
                              mode="adaptive").restart == 0.35
~~~

- [ ] Test comparison-only questions remain balanced, relational cues override "who", context-enriched/ambiguous questions stay balanced, and word substrings do not trigger profiles. Include non-English/unsupported phrasing as balanced.
- [ ] Test false/NaN/Infinity masses, invalid modes/statuses, zero seeds, out-of-range counts, stable field-order-independent input digests, and fixed-mode fallback.
- [ ] Run the new tests and observe failure for the missing policy, then implement the pure types and logic.
- [ ] Run `rtk python -m pytest aquillm/apps/knowledge_graph/tests/test_ppr_policy.py -q` and require a passing result.
- [ ] Review and commit the pure policy during implementation.

## Task B2: Add a separate, validated PPR execution contract

**Files:** Create projected_ppr_execution.py and aquillm/apps/knowledge_graph/tests/test_projected_ppr_execution.py; modify projected_ppr.py only to share validation/transition preparation. Preserve ppr.py's fixed caps and literal legacy traces.

**Interfaces:**

~~~python
@dataclass(frozen=True, slots=True)
class ProjectedPPRExecutionV1:
    ranking: ProjectedPPRResultV1
    decision: PPRPolicyDecisionV1
    algorithm_signature: str
    execution_signature: str
    policy_input_digest: str
~~~

Produce execute_adaptive_projected_ppr(*, snapshot: ProjectedAuthorizedGraphSnapshotV1, seeds: tuple[ProjectedSeedV1, ...], base_config: PPRAlgorithmConfig, signals: PPRPolicySignalsV1, expected_branch: HybridBranchKind, deadline_check: Callable[[], None]) -> ProjectedPPRExecutionV1. It checks original snapshot/config/seeds first, requires signals.branch_kind to equal the runtime's expected_branch, and recomputes outward seed mass and cap pressure from admitted topology instead of trusting a caller to invent those values. Combine seed cap pressure with measured topology cap pressure. Produce prepare_projected_ppr_inputs(*, snapshot, seeds, config) -> tuple[WeightedEdge[str], ...] in projected_ppr.py by sharing the current validation and replay code without changing its order or admission semantics.

- [ ] Write failures for altered base signatures, unauthorized seed identities, reordered equivalent inputs, mismatched signal branch/seed counts, and unsafe effective changes to any field except restart.
- [ ] Add a hand-solvable two-node graph, A->B with B dangling and s(A)=1. For one kernel iteration expect p(A)=r and p(B)=1-r. Separately compare the adapter's full eight steps to the same recurrence; do not change serving iterations for the hand calculation.
- [ ] Prove adaptive r=0.20 produces the exact legacy score vector and rank trace, but has its own explicit algorithm identity. Fixed mode continues to use the legacy entry point and exact legacy identity.
- [ ] Build effective_config = dataclasses.replace(base_config, ppr_restart=decision.restart). Validate unchanged fields and eight iterations. Run run_ppr_kernel once, passing deadline_check before preparation and throughout the kernel. Never call the legacy scorer first merely to validate it.
- [ ] Canonicalize policy inputs using sorted keys and float.hex values; hash an algorithm payload containing domain ppr_projected_execution_v1, canonical effective config, and policy version. Hash an execution payload containing that digest, original snapshot checksum, seed checksum, and policy-input digest. Use no raw question or entity text.
- [ ] Verify different effective restarts change algorithm/execution hashes, different absolute seed-support signals change execution hashes even when normalized seeds match, and the original topology checksum remains unchanged.
- [ ] Test deadline interruption does not emit a partial success or mutate the snapshot. No result cache is added.
- [ ] Run:

~~~powershell
rtk python -m pytest aquillm/apps/knowledge_graph/tests/test_projected_ppr_execution.py aquillm/apps/knowledge_graph/tests/test_projected_ppr.py aquillm/apps/knowledge_graph/tests/test_projected_ppr_caps.py aquillm/apps/knowledge_graph/tests/test_retrieval_ppr.py -q
~~~

- [ ] Review and commit the execution adapter during implementation.

## Task B3: Prepare signals and integrate both independent branches

**Files:** Create ppr_seed_support.py, tests/test_ppr_seed_support.py; modify production_direct.py, production_extended.py, production_runtime.py, production_runtime_support.py, scheduler.py, scheduler_support.py, aquillm/lib/knowledge_graph/retrieval_config.py, and .env.example. Extend the runtime/scheduler/config tests listed below.

**Interfaces:** Produce frozen PPRSeedSupportV1(status: str, cap_pressure: bool, summary: tuple[tuple[str, int | float | bool], ...], digest: str) and PreparedPPRSeedsV1(seeds: tuple[ProjectedSeedV1, ...], support: PPRSeedSupportV1, intent: str). Summary has at most 16 sorted unique numeric keys from a closed per-branch allowlist. Produce summarize_direct_support(outcome: DirectSeedOutcomeV1, *, max_seeds: int) -> PPRSeedSupportV1 and summarize_extended_support(*, ranked_seeds: tuple[GraphExpansionSeed, ...], mapped_chunk_ids: frozenset[int], vector_chunk_ids: tuple[int, ...], trigram_chunk_ids: tuple[int, ...], exact_chunk_ids: tuple[int, ...], retained_identity_count: int, max_seeds: int) -> PPRSeedSupportV1.

Direct summary keys: deduplicated_spans, resolved_spans, ambiguous_spans, retained_matches, retained_seeds, exact_tier_matches, minimum_extraction_score, cap_pressure. Extended keys: requested_chunks, mapped_top_chunks, required_top_chunks, rank_one_mapped, rank_one_channel_agreement, retained_seeds, cap_pressure. Unknown values yield unknown status; do not invent confidence zero. Hash the canonical versioned summary; identifiers are used transiently for joins and are absent from the summary.

Produce prepare_direct_with_policy(runtime, *, query, scope, deadline) -> PreparedPPRSeedsV1 | DirectBranchFailureReason and prepare_extended_with_policy(runtime, *, query, baseline, shared, authorization, settings, deadline) -> PreparedPPRSeedsV1 | ExtendedBranchFailureReason. Share current preparation internals so each path performs the same single extractor/mapping sequence. Preserve legacy seed-only wrappers for fixed mode.

- [ ] Write direct tests where high and low extraction scores normalize to the same single seed but yield different support decisions; ambiguity, cap pressure, and embedding-only support must retain 0.20.
- [ ] Write extended tests where rank-1 channel agreement plus complete top-seed mapping passes the spec's support guard, while missing mappings, missing channels, and cap pressure abstain. Assert rank/mass values are never treated as calibrated confidence.
- [ ] Preserve exactly the existing seed identities, ordering, masses, extractor calls, lookup calls, and authorization checks after adding summaries. No waiting for the other branch.
- [ ] Pass query explicitly through HybridGraphBranchScheduler._extended and HybridBranchRuntime.prepare_extended; update production and test doubles together. Retain run_extended's prepared-object input. Empty/unknown query context produces balanced intent; do not set shared query state in run_direct.
- [ ] Add KG_PPR_RESTART_MODE to the exact config allowlist and HybridRetrievalSettings, default fixed. Accept fixed/shadow/adaptive; malformed values raise the existing HybridRetrievalConfigError. Do not change KG_GRAPH_ALGORITHM's topology contract or assume legacy KG_OVERLAY_PPR_RESTART controls production.
- [ ] Dispatch fixed to unchanged ppr_projected_v1. Shadow gathers/proposes policy but runs only legacy PPR. Adaptive validates/prepares and calls the new execution adapter once.
- [ ] Extend success_envelope with keyword execution_algorithm_signature: str | None = None. Only legacy/fixed callers use None; adaptive callers pass the computed algorithm digest. Retain original snapshot/seed/ready/candidate checksums, and test them independently.
- [ ] Emit a separate closed aggregate event for mode, proposed/effective restart, policy version, reason, branch kind, and timing. Do not append arbitrary fields to BranchSafeDiagnosticsV1 or expose the execution record in public tool payloads. Private replay may serialize the versioned record with existing opaque identities.
- [ ] Test direct timeout or invalid-policy input preserves successful extended results, and vice versa; invalid source authorization must still fail. Verify cooperative deadline exceptions map to existing branch-local timeout behavior.
- [ ] Run:

~~~powershell
rtk python -m pytest aquillm/apps/knowledge_graph/tests/test_ppr_seed_support.py aquillm/apps/knowledge_graph/tests/test_production_hybrid_runtime.py aquillm/apps/knowledge_graph/tests/test_retrieval_branch_scheduler.py aquillm/apps/knowledge_graph/tests/test_retrieval_scheduler_lifecycle.py aquillm/apps/knowledge_graph/tests/test_branch_contracts.py aquillm/lib/knowledge_graph/tests/test_retrieval_config.py aquillm/apps/documents/tests/test_hybrid_graph_authorization.py aquillm/apps/documents/tests/test_hybrid_graph_reranker_authority.py -q
~~~

- [ ] Review and commit the opt-in integration during implementation. Leave .env.example at KG_PPR_RESTART_MODE=fixed.

## Task B4: Evaluate against fixed ratios and the evidence selector

**Files:** Create run_ppr_restart_eval.py, ppr_restart_cases.json, tests/test_ppr_restart_eval.py, and docs/documents/operations/adaptive-ppr-restart.md at the mapped locations. Add relevant tests to .github/workflows/test-backend-frontend.yml; coordinate with workstream A's CI/docs edits.

**Interface:** main(argv=None) -> int accepts --cases, --split, --policy (fixed_015, fixed_020, fixed_035, fixed_050, adaptive_v1), and --output. The deterministic runner loads graph snapshots and numeric support summaries, produces actual graph candidates, and scores their identities against labels. Private corpus snapshots stay outside committed fixtures.

- [ ] Create a synthetic two-node fixture with version 1, nodes [A,B], edges [[A,B,1.0]], seeds [[A,1.0]], and expected one-step vectors {fixed_020: [0.20,0.80], fixed_035: [0.35,0.65]}. The loader maps synthetic names to deterministic opaque keys and creates valid authorized test snapshots. Reject duplicate nodes, negative/nonfinite weights, outside-scope evidence, and unsupported policy names.
- [ ] Add branch-local fixtures for strong/weak identical normalized seeds, relational chains, factual hub distraction, dead ends, cap pressure, unknown extended support, and a comparison question that needs no graph walk change.
- [ ] Compare all fixed controls and adaptive on the same snapshots, seeds, limits, eight iterations, and unchanged graph-to-passage scoring. Include the best fixed value chosen only on development data.
- [ ] For an offline reference only, use the normalized recurrence with max 500 iterations and L1 step tolerance 1e-10. Report residual/stopping reason and candidate-rank changes against eight steps. Do not raise the legacy PPRAlgorithmConfig iteration cap to run this reference.
- [ ] Add graph labels to the parent plan's frozen 40 development/40 held-out questions. Measure candidate/support-group recall, irrelevant expansion, final evidence quality, abstention, latency, and memory. Include the seed and policy quality strata explicitly; high gate-abstention can make the adaptive arm equivalent to fixed.
- [ ] Compare four end-to-end arms: current pipeline, A only, B only, A+B. Keep answer model, graph caps, per-document/passage/token budgets, and question set constant. Preserve workstream A's separate same-candidate selector evaluation.
- [ ] Extend private evaluation output with schema version, effective config/policy, topology/seed/execution digests, and graph candidate labels. Do not silently retrofit variable query-dependent signatures into legacy fixed-algorithm benchmark reports; keep their strict validation intact.
- [ ] Run:

~~~powershell
rtk python -m pytest aquillm/apps/knowledge_graph/tests/test_ppr_restart_eval.py aquillm/apps/knowledge_graph/tests/test_projected_snapshot_parity.py aquillm/apps/knowledge_graph/tests/test_projected_topology_adapter.py -q
~~~

From aquillm/:

~~~powershell
rtk python -m apps.knowledge_graph.evals.run_ppr_restart_eval --cases apps/knowledge_graph/evals/ppr_restart_cases.json --split regression --policy adaptive_v1 --output ppr-restart-regression.json
~~~

- [ ] Measure added policy/adapter p95 (proposed target <=5 ms at existing caps) and allocations. Verify one PPR call per enabled branch, no extra graph/model calls, unchanged deadlines and cap enforcement. Report absolute measurements; do not assume zero overhead.
- [ ] Apply the spec's held-out gates, including comparison to the best fixed baseline. If the heuristic does not help, retain the better fixed option; do not enable adaptation solely because it is more dynamic.
- [ ] Document rollout: offline evaluation -> shadow decisions with fixed serving -> adaptive canary after gates -> joint A+B canary after factorial comparison. KG_PPR_RESTART_MODE=fixed rolls back B independently of RAG_EVIDENCE_SELECTION_MODE.
- [ ] Run the B1-B4 focused suites, `rtk python scripts/check_file_lengths.py`, and `rtk git diff --check`; review and commit implementation artifacts. No deployment or service restart is part of completing this draft.

## Self-review and dependency order

- [x] Covers pipeline step 6 without conflating it with final passage weighting.
- [x] Separates seed distribution, restart probability, and passage relevance.
- [x] Preserves original topology validation and binds actual execution parameters.
- [x] Uses available direct/extended signals without inventing calibrated confidence.
- [x] Keeps one run, eight iterations, existing graph limits, and independent branch failures.
- [x] Requires fixed-baseline and four-arm evaluation with independent rollback.

B1 precedes B2/B3; execution and signal preparation can be developed independently with distinct file ownership. B3 integrates them. B4's deterministic replay can start after B2; its final A+B evaluation requires workstream A. The parent plan remains the shared roadmap.
