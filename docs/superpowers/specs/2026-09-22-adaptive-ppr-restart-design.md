# Adaptive PageRank Restart Design

**Status:** Implemented and reviewed in development (PR #230); live representative quality/concurrent-latency gates remain pending. See [current implementation report](../reports/2026-09-22-adaptive-retrieval-implementation.md). Features remain default off. Preservation composes with the adaptive selector; the independent PPR activation gate remains open. This status update does not authorize activation or deployment.
**Date:** 2026-09-22
**Scope:** Workstream B of the [adaptive retrieval plan](../plans/2026-09-22-adaptive-evidence-selection.md), covering pipeline step 6 in both production graph branches.

## Intended behavior and the three distinct weights

Choose one restart probability per retrieval query and branch, then hold it fixed during that PageRank run. Focused questions can concentrate on seed neighborhoods; explicit relational questions can give graph connections more influence when the available seed/topology signals support doing so. Uncertain inputs retain the current policy.

The current recurrence is:

~~~text
p_next = r * s + (1 - r) * (P_transpose * p_current + dangling_mass * s)
~~~

Here s is the normalized seed distribution, P is the normalized graph transition matrix, and r currently defaults to 0.20. The 20% term returns to s; it is not 20% of the previous importance vector. The remaining 80% propagates the current scores, with dead-end mass returned to s.

Keep three concepts separate:

1. **Seed masses s:** where restarts land. These are already nonuniform: direct matches use extraction/tier/similarity signals; extended seeds use retrieval-rank weights distributed over linked entities.
2. **Restart probability r:** how strongly diffusion stays anchored to those seeds. This workstream adapts it.
3. **Passage relevance and redundancy weights:** which retrieved passages enter the answer. Workstream A owns these.

Do not multiply graph probability into the final reranker score or reweight seed masses as part of the first experiment. Those would confound evaluation of the restart change.

## Alternatives and research grounding

- **Tune one fixed value:** cheapest strong baseline; it may outperform a heuristic policy and must be evaluated.
- **Bounded query/branch policy: recommended experiment.** Inspectable rules reuse existing signals, choose one value, and add no neural calls.
- **Learned or node-dependent restart:** potentially more expressive but needs labels and a different algorithm/validation contract. Defer.

HippoRAG tuned its PPR parameter to 0.5 using 100 MuSiQue training examples. This supports validation-based selection, not copying 0.5 into a different graph. [HippoRAG, section 3.4](https://arxiv.org/html/2405.14831v1#S3.SS4)

SuRe studies learned node-dependent restart for graph ranking/link prediction. It supports the broader idea of adaptive restart but does not validate the query rules, ratios, or RAG benefits proposed here. [Jin, Jung and Kang, 2019](https://doi.org/10.1371/journal.pone.0213857)

## Global constraints

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

## Available signals and their limits

DirectSeedOutcomeV1 exposes retained matches, extraction confidence, resolution tier, similarity, and ambiguity/coverage counts. production_direct.py currently discards that information when returning opaque seeds. Retain a bounded numeric summary internally, without leaking spans or identifiers. A normalized seed mass of 1.0 is not a confidence score.

Direct resolution caps matches before computing diagnostics, so unresolved counts can include cap-dropped matches. Treat reaching the seed cap as possible truncation and abstain from adaptation; do not call that a measured linkage error rate.

The extended branch has GraphExpansionSeed.rank, the frozen vector/trigram/exact candidate IDs, and the seed-chunk-to-entity mappings it already loads. These support mapping coverage and retrieval-channel agreement checks. They do not provide calibrated confidence or post-reranker scores: this branch runs before final reranking. Do not add reranking or wait for the direct branch to manufacture confidence.

Both branches have an authorized bounded snapshot. Derive structural signals only from that snapshot and the exact transition groups admitted for scoring. High degree, seed entropy, or a high raw PPR score alone does not establish relevance.

## Initial deterministic policy

All numbers below are experimental, versioned starting points to tune on development data.

| Intent when support guards pass | Restart r | Graph propagation 1-r |
| --- | ---: | ---: |
| focused factual lookup | 0.35 | 0.65 |
| balanced/default | 0.20 | 0.80 |
| explicit relational/multi-hop lookup | 0.15 | 0.85 |

The range is limited to 0.15..0.35 in adaptive v1. No per-iteration change is allowed. Unknown intent uses 0.20. A literature review or comparison is not automatically a multi-hop query; do not reuse the evidence selector's breadth label as an exploration instruction.

Use the actual resolved retrieval query, consistently supplied to both branches. Match token boundaries and bounded phrases. Initial relational cues are "relationship between", "connection between", "connected to", "path between", and "linked through". They take precedence over factual cues (who/when/where/how many/define). Other requests remain balanced. Prefix/context ambiguity keeps balanced intent; do not strip arbitrary query text or infer intent from a document title. These rules need explicit paraphrase and nested-question evaluation.

In v1, colon-bearing or multiline queries conservatively remain balanced because the current query builder may prepend "title: question" without a separate context marker. This also abstains on some legitimate colon-bearing identifiers; measure that coverage cost. A future explicit query-context contract can improve coverage without guessing at title boundaries.

Before applying an intent-based change:

- **Direct support guard:** nonempty retained matches; no ambiguous spans; all deduplicated spans resolved; retained match count and seed count below their configured seed cap; every retained match is identifier/name/alias tier with extraction score at least 0.80. This is a conservative heuristic gate, not an 80% correctness claim. Embedding-only or mixed uncertain matches keep 0.20 initially.
- **Extended support guard:** all of the first min(3, number_of_requested_seeds) ranked seed chunks map to at least one authorized entity; the rank-1 mapped chunk appears in the vector list and at least one lexical list (trigram or exact); no seed-cap pressure. Missing metadata or any failed condition keeps 0.20. Agreement is a support proxy, not proof of relevance.
- **Structural guard:** reaching the node or edge cap keeps 0.20 as a conservative response to possible truncation. For the 0.15 setting, require at least two retained seed identities and at least half the normalized seed mass on identities with positive outgoing transitions to other identities. Otherwise keep 0.20. Self-loops do not count as useful exploration.

Use fixed reason codes: fixed_mode, focused_supported, relational_supported, balanced_intent, unknown_signals, seed_support_insufficient, cap_pressure, and insufficient_connections. Invalid authorization/topology/seeds retain existing failure behavior; a policy fallback cannot rescue invalid source data.

Lower restart changes diffusion inside the loaded neighborhood. It does not load a third hop or enlarge the authorized graph. More restart also cannot repair incorrectly linked seeds; do not increase it simply because seed quality is poor.

## Execution identity and integration

Current production_runtime_support.ppr_config and projection/topology_encoding.algorithm independently build the fixed configuration. projected_ppr._validate_config checks the snapshot's hash against that full configuration. Changing only the runtime r would therefore fail validation.

Introduce a separate versioned adaptive execution adapter:

1. Validate the original snapshot against its original base configuration with the unchanged validator.
2. Validate seeds and replay the same bounded transition groups used by legacy PPR.
3. Derive the policy decision from private prepared-seed summaries, query intent, and those groups.
4. Build an effective configuration by replacing only ppr_restart; keep eight iterations and every other field equal.
5. Run the existing normalized kernel once under the existing absolute deadline.
6. Produce a domain-separated effective algorithm hash over canonical effective configuration and policy version. Produce a separate execution hash binding that algorithm hash, original snapshot checksum, seed checksum, and numeric policy-input digest.

The topology snapshot is immutable. Do not relabel its signature, bypass validation, or rewrite projection/gateway metadata. Its authorization and checksum remain authoritative. In adaptive mode, success_envelope receives the effective algorithm hash explicitly for BranchProvenanceV1.ppr_algorithm_signature; topology_snapshot_checksum still refers to the original snapshot. Fixed mode preserves its existing signature and bytes. The versioned internal execution record holds the full decision and execution hash for replay; public tool rows remain unchanged.

Do not add a PPR result cache in v1. If a future cache is introduced, key it by the execution hash and revalidate scope. Existing topology caches may remain reusable because topology membership has not changed. Do not reuse fixed-policy ranking results for a different effective restart.

Give the extended runtime an explicit query parameter through the scheduler/protocol; never obtain it by waiting for run_direct to mutate shared state. Prepared branch objects carry seeds and their private summaries. Neither branch depends on successful completion of the other.

## Iterations, cost, and observability

The current result is eight iterations of PPR, not necessarily the stationary solution. Lower restart may need more iterations to approach that solution. Offline replay must compare the eight-step result with a converged reference (tolerance 1e-10, maximum 500 iterations); report non-convergence and top-candidate changes. Do not secretly increase serving iterations to make the adaptive arm win.

Computational order remains O(I * (V + E)) time and O(V + E) working memory, with I=8 and existing V/E caps. Signal collection is a bounded scan of already available data. The design adds no model inference. Measure actual policy/adapter CPU time and allocations; no unmeasured percentage-overhead claim.

Add KG_PPR_RESTART_MODE=fixed|shadow|adaptive, default fixed. Shadow computes and records the proposed decision but serves the existing fixed run; it performs no second PPR or graph fetch. Counterfactual ranking comparisons use offline snapshots. Ordinary logs allow only branch kind, mode, policy version, effective/proposed restart, fixed reason codes, bounded counts, and durations.

## Evaluation and activation gates

Extend the existing 80-question development/held-out collection with graph-specific labels: expected seed grounding, relevant connected passages, all-supporting-passage groups, and graph-distraction examples. Include ambiguous names, weak matches, disconnected seeds, cycles, dead ends, capped graphs, and factual-looking multi-hop questions. Keep held-out questions and labels frozen.

Compare r=0.15, 0.20, 0.35, 0.50 as fixed offline baselines plus adaptive v1 on identical snapshots, seeds, caps, iterations, and graph-to-passage mapping. The best development-selected fixed value is a required baseline; changing 0.20 alone might explain the gain. Offline 0.50 is an evaluation control, not an adaptive v1 output.

Measure graph candidate recall, all-supporting-passage recall, irrelevant expansion rate, final-packet recall/nDCG, policy abstention rate, per-branch latency, memory, and eight-step versus converged rank stability. Then compare current pipeline, evidence selection only, adaptive PPR only, and both together. Keep retrieval recall evaluation distinct from same-candidate selector replay.

Activation requires invariant/regression tests passing, no held-out factual/contradiction regression in supporting-evidence recall, and an improvement on targeted relational retrieval over both current 0.20 and the best fixed baseline. Report paired uncertainty intervals and per-family results; a small test set does not establish general superiority. Proposed engineering target: added policy/adapter p95 <=5 ms at the existing cap, with no added queries/model calls/PPR runs or deadline expansion. If gains are unclear, ship the better fixed policy or keep 0.20. Roll back either workstream independently.
