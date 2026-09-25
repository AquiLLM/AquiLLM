# Adaptive evidence selection: evaluation and rollout

This policy applies only to automatic direct RAG. Manual search and the ordinary LLM tool loop keep their existing evidence behavior. The feature is disabled in the example environment:

```dotenv
RAG_EVIDENCE_SELECTION_MODE=legacy
RAG_EVIDENCE_SELECTION_SCORE_TIMEOUT_MS=3000
RAG_EVIDENCE_SELECTION_SHADOW_SCORING=0
```

`legacy` serves the existing ranking and document rotation. `shadow` also serves that same legacy output and records a private comparison; initial retrieval does not capture new numerical scores or add scoring calls, so the default comparison uses rank-derived relevance. Set shadow scoring to `1` only for a bounded development experiment that permits extra private final-union scoring. `adaptive` serves the new selector and is a later canary decision. The mode accepts only `legacy`, `shadow`, or `adaptive`; the scoring deadline accepts 100–3000 ms. Invalid values select legacy as a unit and emit the fixed configuration-error code. Profile weights and relevance-gap allowances live together in the versioned policy, outside environment settings.

## Evidence needed before activation

Start with the deterministic replay in `aquillm/apps/chat/evals/evidence_selection_cases.yaml`. It has nine **synthetic regression cases**. They check mechanics and expected citation identities, not real retrieval quality. Run the four policies (`legacy`, `relevance_only`, `fixed_mmr`, `adaptive`) on identical candidate snapshots and budgets. Keep the zero-extra-final-scoring arm to measure the value of neural work separately. Keep the existing routing evaluator `run_rag_eval` as a separate check.

From `aquillm/`, replay a policy and run the existing routing evaluator:

```sh
python -m apps.chat.evals.run_evidence_selection_eval --cases apps/chat/evals/evidence_selection_cases.yaml --policy adaptive --split regression --output evidence-selection-regression.json
python -m apps.chat.evals.run_rag_eval --quiet
```

There is currently no representative 80-question labeled corpus in this repository. Before tuning or enabling adaptive serving, collect 40 development and 40 frozen held-out questions, balanced across focused facts, comparisons/synthesis, multipart questions, and contradiction/condition-sensitive questions. Label relevant chunks, grades, aspects, and required independent evidence. A gold chunk missing from the upstream candidate pool counts as a retrieval miss. Keep private question text, document IDs, and source passages outside tracked artifacts and ordinary logs. Tune coefficients and cross-document attenuation on development cases only; choose the winning policy before opening held-out results.

Activation requires all of these gates:

| Gate | Required evidence | Current state |
| --- | --- | --- |
| Correctness | Authorization, citation, current source identity, budget, deterministic order, and regression tests pass. | Code tests only; development acceptance pending. |
| Held-out quality | Final-packet nDCG and supporting-passage recall are at least the legacy point estimates, with paired bootstrap intervals and per-family results. | **Unmeasured**: labeled corpus absent. |
| Required evidence | Contradictory/supporting evidence recall does not fall; known displacement cases improve. | **Unmeasured** on real questions. |
| Breadth quality | Aspect coverage improves or stays equal while repeated-evidence rate falls. Document count alone is not a quality gate. | **Unmeasured** on real questions. |
| Selector cost | Pure selection p95 is at most 20 ms at 45 candidates on the development host. | Local synthetic p95 was 2.68 ms; development-host gate pending. |
| Scoring tradeoff | Added latency and pair counts are reported separately for warm, cold, and concurrent turns; quality gain justifies cost. | **Unmeasured** in serving conditions. |

If adaptive profiles do not beat a fixed profile, use the simpler fixed policy. If extra scoring does not justify its latency, keep rank fallback or legacy. Synthetic nDCG or a local microbenchmark cannot satisfy these gates.

## Development sequence

1. Verify the offline replay, current direct-RAG and score-contract tests, authorization, redaction, and strict-reranker checks. Development deployment may use `legacy` or `shadow` after code verification; deploying is a separate operational action.
2. Observe `shadow` with shadow scoring off. Compare served and proposed aggregate counts, profile, and score status in ordinary logs while serving legacy results. Compare citation-level choices only through authorized private replay fixtures; ordinary shadow logs contain no citation IDs.
3. In a bounded development experiment, explicitly set `RAG_EVIDENCE_SELECTION_SHADOW_SCORING=1`. Measure final-scoring reuse, new pairs, caller latency, provider work, and warm/cold/concurrent behavior. Return the switch to `0` when the experiment ends.
4. After the real labeled and latency gates pass, approve a limited `adaptive` canary. Compare it with legacy and the best fixed profile using the same authorized candidate snapshots, then broaden only if the canary preserves the gates.

The candidate union is capped at 45 rows (at most three searches of 15). Final comparable scoring targets the **primary resolved question**. Clause-query scores remain retrieval provenance; pointwise scores may be reused only with matching query, scorer, source, and successful input fingerprints. Listwise scores require one run over the whole merged pool. New work is bounded by 45 query/passage pairs per turn, a 3,000 ms aggregate deadline (or the lower configured value), and at most six in-flight pointwise calls. If one compatible scorer cannot cover the current union, use rank-derived relevance for the whole union. Do not mix model and fallback scales.

Model scores are min/max normalized within the current union, so they express relative ordering rather than calibrated confidence. Flat scores become stable ties. Text cost uses an approximate four-characters-per-token estimate. Existing query-count, final-passage, per-document, and evidence-text limits remain hard. Lexical three-shingle novelty is a soft redundancy signal; it cannot establish entailment or contradiction. Upstream retrieval and per-query reranking still determine candidate recall. Cancelling an HTTP request may leave remote inference running, so record provider work separately from caller latency.

Ordinary logs may contain only bounded aggregate counts, mode/profile/status, policy version, durations, budget use, score-reuse/new-pair counts, and fixed fallback reason codes. Keep raw questions, passage text, document IDs, embeddings, credentials, and per-passage scores out of ordinary logs. Score metadata stays in the private top-level sidecar or typed internal result, never in public passage rows.

## Rollback

Set `RAG_EVIDENCE_SELECTION_MODE=legacy` and `RAG_EVIDENCE_SELECTION_SHADOW_SCORING=0`, then verify the legacy direct-RAG and manual/tool-loop checks. This independently rolls back evidence selection without deleting graph state, source documents, or caches. The legacy ID-only rerank cache and versioned scored-result cache use separate namespaces and may coexist. The graph restart policy has its own rollback switch; see [adaptive PPR restart](adaptive-ppr-restart.md).
