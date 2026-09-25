# Adaptive retrieval implementation

Implemented the two independently controlled policies described in the [evidence selection design](../specs/2026-09-22-adaptive-evidence-selection-design.md) and [PPR restart design](../specs/2026-09-22-adaptive-ppr-restart-design.md).

- Evidence selection compares the bounded candidate union against the resolved primary question. Compatible numerical scores drive a relevance-first selector with a soft redundancy penalty. Incomplete or incompatible scores use one rank-derived scale for the whole pool.
- PPR can choose a bounded restart value from query, seed-support, and topology signals. Fixed mode preserves the existing 0.20 restart; shadow proposes a value while executing the fixed walk once.
- Both paths preserve authorization, graph/evidence caps, source identity, and public citation boundaries. They have separate flags and rollback paths.

## Verification and activation

The full branch received an independent code review. No blocking correctness, authorization, privacy, or default-behavior regression remained. Two scoring findings were fixed and independently re-reviewed: incompatible successful retry inputs now trigger whole-pool fallback, and cached batch endpoints receive the correct singleton batch shape.

The combined local verification run covered 58 retrieval and CI contract modules. Final results are recorded in the pull request. Standalone evidence-selection and PPR replay CLIs both succeeded on their synthetic fixtures. New or enlarged Python files respect the 300-line limit; pre-existing repository lint and file-length issues were not hidden by expanding exemptions.

The code is opt-in. Example defaults remain `RAG_EVIDENCE_SELECTION_MODE=legacy`, `RAG_EVIDENCE_SELECTION_SHADOW_SCORING=0`, and `KG_PPR_RESTART_MODE=fixed`. Development deployment and activation require the operational checks in the [evidence selection runbook](../../documents/operations/adaptive-evidence-selection.md) and [PPR runbook](../../documents/operations/adaptive-ppr-restart.md).

No representative 80-question labeled corpus was available. Synthetic regression success does not establish improved retrieval quality. Local maximum-cap PPR added p95 was 32.36 ms against a proposed 5 ms target; adaptive PPR activation remains gated. Development-host latency, concurrent scorer cost, and real held-out quality remain unmeasured.

## Implementation decisions

| Decision | Reason and tradeoff |
| --- | --- |
| Build the pure selector before the scoring coordinator. | Its candidate types define the integration contract; only task order changes. |
| Use an isolated worktree and parent-serialized commits. | Independent modules could progress in parallel while preserving the original checkout; integration required explicit ownership and reviews. |
| Include development deployment in scope after the user's implementation request. | Deployment is authorized for development; production is outside scope. |
| Keep adaptive activation gated by real evaluations. | Synthetic fixtures verify mechanics but cannot replace the missing labeled corpus; new behavior may remain disabled after code deployment. |
| Bind score reuse to PK, document, chunk, source, and effective input. | Cache identifiers alone cannot validate current evidence; incompatible scores cause conservative fallback. |
| Use the canonical current source and exact emitted excerpt during hydration. | This prevents stale or mixed source versions from entering selection; changed rows are dropped. |
| Start the aggregate scoring deadline before preparation. | Hydration consumes some of the available budget, favoring bounded caller time over additional inference. |
| Capture new initial scores only in adaptive mode. | Legacy and shadow preserve the existing initial reranker/order; default shadow therefore compares rank-derived relevance. |
| Put new coverage in dedicated test modules. | Existing oversized test files do not grow; established tests still run as regression checks. |
| Share admitted PPR policy signals between shadow and adaptive. | This avoids divergent proposal logic while preserving the original topology and separate execution identity. |
| Retain checksum validation despite measured adapter overhead. | Skipping provenance checks would weaken the contract; the latency activation gate remains unmet. |
| Correct the stale wire-cap test fixture only. | The same failures reproduced on the untouched base; production request validation remains strict. |

Private question text, source passages, credentials, per-passage scores, and local test outputs are excluded from this report and ordinary retrieval logs.
