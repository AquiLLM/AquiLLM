# Projected PPR restart: evaluation and rollback

The projected graph retrieval policy uses the existing graph and candidate limits. The example environment keeps the current restart value:

```dotenv
KG_OVERLAY_PPR_RESTART=0.20
KG_PPR_RESTART_MODE=fixed
```

`fixed` runs the existing 0.20 restart. `shadow` computes a private proposed restart but serves the fixed run, with no second PPR execution or graph fetch. `adaptive` may select a bounded 0.15, 0.20, or 0.35 restart from current query, seed, and topology summaries. It retains eight PPR iterations, the same authorized immutable topology snapshot, graph-to-passage mapping, branch deadlines, and candidate caps. The policy adds no model call. The restart-mode flag is independent of `RAG_EVIDENCE_SELECTION_MODE`.

There is no representative 80-question frozen labeled corpus available for this policy today. The deterministic graph fixtures establish invariants and edge behavior, not superiority. Development may use fixed or shadow mode after code verification; adaptive serving needs the following real evaluation:

| Gate | Required evidence | Current state |
| --- | --- | --- |
| Safety and parity | Authorization, provenance, fail-open, cap, deterministic-order, and regression tests pass; fixed mode preserves the current signature and output. | Code tests; development acceptance pending. |
| Fixed baselines | Compare restarts 0.15, 0.20, 0.35, and 0.50 offline on identical snapshots, seeds, caps, eight iterations, and mapping. Select the best fixed value using development labels. The 0.50 arm is an evaluation control only. | **Unmeasured** on representative questions. |
| Held-out quality | No factual/contradiction supporting-evidence recall regression, and targeted relational retrieval improves over both current 0.20 and the best development-selected fixed baseline. Report paired uncertainty intervals and per-family results. | **Unmeasured**: labeled corpus absent. |
| Retrieval diagnostics | Report graph candidate recall, all-supporting-passage recall, irrelevant expansion, final-packet recall/nDCG, abstention, branch latency, and memory. Compare eight-step ranks with a converged reference. | Synthetic replay only. |
| Cost | Added policy/adapter p95 at the existing graph cap is at most 5 ms, with no extra queries, model calls, PPR runs, or deadline increase. | A local synthetic 200-node/1,000-edge, 20-warm-sample check measured paired added p95 **32.36 ms**, above target. Development-host gate pending. |

Extend the same frozen 40-development/40-held-out question collection used for evidence selection with seed-grounding, connected supporting passages, independent evidence groups, and graph-distraction labels. Include ambiguous names, disconnected seeds, cycles, dead ends, capped graphs, and factual-looking multi-hop questions. Compare four end-to-end arms: current pipeline, evidence selection only, restart adaptation only, and both together. Report retrieval recall separately from same-candidate evidence-selection replay. Eight serving iterations need not equal the stationary solution; check whether the converged reference changes top candidates without silently increasing serving iterations.

If adaptive restart does not improve over the best fixed value, use that fixed value or retain 0.20. Do not claim a quality gate from synthetic fixtures or a small sample. Development activation and any later canary are separate operational decisions.

The local timing check measured fixed PPR p95 4.31 ms and adaptive policy/adapter p95 36.55 ms on that fixture. These are provisional local measurements, not serving latency; duplicate snapshot digest work is under investigation. Keep fixed or shadow mode while the added-cost gate is unresolved.

For rollback, set `KG_PPR_RESTART_MODE=fixed` and retain `KG_OVERLAY_PPR_RESTART=0.20` for the existing behavior. This does not delete graph state, topology caches, projection data, or source documents. Evidence selection can be rolled back separately with `RAG_EVIDENCE_SELECTION_MODE=legacy`; see [adaptive evidence selection](adaptive-evidence-selection.md). Existing topology caches remain valid because restart adaptation does not change topology membership; there is no new PPR result cache in v1.
