# Task 1 implementation report

Implementation commit: `a4fe1ca128c367dfa40010c03294036b62880fb4`.
Branch: `codex/development-retrieval-gaps`; implementation base: `a704656`.
Status: implemented and locally verified; submitted to parent for review. Deployment
and development root-cause SQL corrections belong to the parent's later tasks.

## Behavior

- Whole-branch scheduler expiry reports `direct_branch_timeout` or
  `extended_branch_timeout`, independent of inferred seed or topology state.
- Ontology deadline checks use direct branch timeout. Actual extractor boundary
  errors continue to report `extractor_timeout`.
- Direct and extended PageRank deadline failures use branch timeout; extended
  seed-source deadline checks use extended branch timeout. Topology loader's
  explicitly attributed topology failures retain their existing labels.
- Branch expiry has zero aggregate counts: the scheduler cannot safely infer
  which stage completed. Existing BranchEnvelopeV1 and BranchSafeDiagnosticsV1
  field schemas are unchanged. Enum whitelist tests and transport diagnostics
  cover the two added fixed labels.
- `obs.rag.graph_stage` records only branch, stage and elapsed_ms (0..5000).
  Fixed stages: ontology, extraction, entity_resolution, topology, materialization.
  Direct entity resolution includes repository setup and resolution; extended
  entity resolution records each selected projection read. Final materialization
  uses the shared branch label. Completion/failure paths omit exception data.
- No changes to ranking, authorization, candidate caps, deadlines, or worker pool.

## Red-green evidence

1. Before implementation, changed scheduler expectations and added ontology-before/
   after deadline tests. Focused run produced **4 expected assertion failures**:
   direct scheduler returned extractor_timeout; extended scheduler returned
   extended_topology_timeout; both ontology cases returned extractor_timeout.
2. Added safe ontology timing regression. Before stage instrumentation it produced
   **1 expected failure**, no graph stage event. Actual extractor-timeout
   regression passed in the same run.
3. Added stronger real-production-runtime + real-scheduler regression: exact
   successful extractor response, then event-controlled blocking resolution.
   Temporarily restored ONLY scheduler_support.py from a704656 using git show,
   ran the regression, and restored current bytes in finally. It produced
   **1 expected failure**, extractor_timeout instead of direct_branch_timeout.
   Current implementation passes, releases the controlled worker, and preserves
   the extended no-seeds sibling outcome.
4. Initial integration additions exposed two fixture issues (late worker used
   incomplete settings; empty selected_snapshot defaults to baseline rows).
   Corrected test fixtures without production changes. Final run is green.

## Reproducible verification

All shell commands start with rtk. Global Python 3.13 was used; repository .venv
does not contain pytest. Tests use dummy client keys and localhost port 1 so no
real DB or external model operation is requested. Initial startup needed
OPENAI_API_KEY, ANTHROPIC_API_KEY and GEMINI_API_KEY in addition to the supplied
dummy environment; startup failures were configuration errors, not red evidence.

Final command (PowerShell):

```powershell
rtk proxy python -c 'import os, subprocess; os.environ.update(SECRET_KEY="local-test-only", DEBUG="1", COHERE_KEY="test", POSTGRES_HOST="127.0.0.1", POSTGRES_PORT="1", POSTGRES_PASSWORD="test", GOOGLE_OAUTH2_CLIENT_ID="test", GOOGLE_OAUTH2_CLIENT_SECRET="test", OPENAI_API_KEY="test", ANTHROPIC_API_KEY="test", GEMINI_API_KEY="test"); paths=["aquillm/apps/knowledge_graph/tests/test_retrieval_branch_scheduler.py", "aquillm/apps/knowledge_graph/tests/test_production_hybrid_runtime.py", "aquillm/apps/knowledge_graph/tests/test_branch_contracts.py", "aquillm/apps/knowledge_graph/tests/test_stage_diagnostics.py", "aquillm/apps/knowledge_graph/tests/test_retrieval_overlap.py", "aquillm/apps/knowledge_graph/tests/test_retrieval_overlap_failures.py", "aquillm/apps/knowledge_graph/tests/test_retrieval_scheduler_lifecycle.py", "aquillm/apps/documents/tests/test_hybrid_graph_diagnostics.py"]; raise SystemExit(subprocess.call(["python", "-m", "pytest", "-q"]+paths))'
```

Result: **101 passed, 5 warnings in 1.53s**, exit 0. Warnings are pre-existing
Pydantic validator deprecations and google.generativeai package deprecation.

Red runs used the same environment and test runner with:

- scheduler + runtime files, `-k "independent_budget or overall_deadline_preserves or ontology_deadline"`: 4 failed as expected.
- runtime file, `-k "ontology_timing or real_extractor_timeout"`: 1 expected failure, 1 pass.
- restored baseline scheduler helper, runtime file,
  `-k "scheduler_expiry_after_extraction"`: 1 expected failure, then restored fix.

Ruff command:

```powershell
rtk proxy python -m ruff check aquillm/apps/knowledge_graph/retrieval/branch_contracts.py aquillm/apps/knowledge_graph/retrieval/scheduler_support.py aquillm/apps/knowledge_graph/retrieval/production_direct.py aquillm/apps/knowledge_graph/retrieval/production_extended.py aquillm/apps/knowledge_graph/retrieval/production_runtime.py aquillm/apps/knowledge_graph/retrieval/stage_diagnostics.py aquillm/apps/knowledge_graph/tests/test_branch_contracts.py aquillm/apps/knowledge_graph/tests/test_retrieval_branch_scheduler.py aquillm/apps/knowledge_graph/tests/test_retrieval_overlap_failures.py aquillm/apps/knowledge_graph/tests/test_production_hybrid_runtime.py aquillm/apps/knowledge_graph/tests/test_stage_diagnostics.py aquillm/apps/documents/tests/test_hybrid_graph_diagnostics.py
rtk git diff --check
```

Both exit 0, Ruff all checks passed. Ruff format applied only to touched files.

## Touched files

- retrieval/branch_contracts.py: fixed timeout labels.
- retrieval/scheduler_support.py: scheduler timeout attribution without guessed counts.
- retrieval/production_direct.py: ontology attribution and direct stage timings.
- retrieval/production_extended.py: seed/PPR attribution and extended timings.
- retrieval/production_runtime.py: direct PPR attribution, topology/materialization timings.
- retrieval/stage_diagnostics.py: bounded, fixed-label timing helper.
- tests/test_branch_contracts.py: whitelist compatibility.
- tests/test_retrieval_branch_scheduler.py: whole-branch failure expectations.
- tests/test_retrieval_overlap_failures.py: overlapping deadline attribution.
- tests/test_production_hybrid_runtime.py: ontology, extractor, blocked resolution regressions.
- tests/test_stage_diagnostics.py: all fixed labels, timing clamps, rejected dynamic labels.
- apps/documents/tests/test_hybrid_graph_diagnostics.py: new label transport/redaction.
- docs/operations/graph-retrieval-latency.md: timing semantics and constraints.
- This report is saved in a separate documentation commit so it can name the implementation SHA.

## Concerns / review notes

The existing four-slot worker pool releases capacity only when its future really
finishes. Cancellation cannot interrupt an already-running synchronous DB call.
Safe stage timings consequently may arrive after scheduler failure; a missing
completion can indicate a stuck stage. No request IDs are logged, so normal logs
cannot associate concurrent requests unambiguously. Capped elapsed_ms deliberately
cannot show a precise duration above 5000ms. Instrumentation diagnoses this issue;
it does not fix the parent's discovered unbounded direct alias SQL.

This task does not claim deployment success or remote SQL behavior verification.
The parent independently investigates that root cause and owns review/deployment.
No server or database mutations were performed by this implementer.
