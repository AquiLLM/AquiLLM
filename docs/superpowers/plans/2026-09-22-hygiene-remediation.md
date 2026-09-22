# Hygiene remediation implementation plan

**Goal:** Make PR #230's existing structure gate pass, then merge the verified PR into `development`, as requested by the user.

**Architecture:** Split oversized modules at coherent responsibility boundaries while retaining public imports and behavior. Split tests by scenario, retaining fixtures and assertions. Preserve the existing 300-line default and all reviewed maxima; the controller only lowers or removes obsolete maxima after a module shrinks.

**Failure evidence:** `python scripts/check_file_lengths.py` currently exits 1 with 69 violations. `check_import_boundaries.py` passes. The existing failing structural check is the regression test; existing behavioral suites protect the refactor.

## Constraints

- No disabling/skipping checks, new exemptions, raised maxima, deleted assertions, dynamic code loading, or arbitrary fragment modules.
- Every new source/test module stays at or below 300 lines. Previously reviewed hotspots must shrink to their existing maximum or below; their recorded maximum then ratchets down to the actual count.
- Preserve public APIs, serialized fields, permissions, transaction boundaries, monkeypatch seams, Celery task names, and test collection.
- Work in the existing isolated PR worktree. Independent domains have exclusive file ownership; only the controller stages, commits, updates the shared line-count map, pushes, or merges.
- All shell commands start with `rtk`. Tests use the task-owned PostgreSQL service and separate databases for concurrent workers.

## Execution

1. Collections domain: split violating collection services, schema task/view modules, schema/visualization tests, and collection React hooks/components/tests. Run collection tests and relevant frontend tests/typecheck/build.
2. Graph core: split violating extraction, graph, resolution, build/ontology/task modules and their corresponding oversized tests. Run focused domain regressions, preserving transaction and patch behavior.
3. Graph infrastructure: split violating projection/retrieval modules, shared graph configuration/extractor/query-extractor modules, and their oversized tests. Run focused projection, seed, extractor, and query-extractor regressions.
4. Chat/document/provider domain: split violating chat support/tests, rerank/cache modules/tests, memory/settings modules, LLM provider/citation modules/tests, and integration test modules. Run direct-RAG, reranker/cache, citation, provider, memory/settings and compose checks.
5. Integrate: lower/remove only obsolete reviewed maxima; run file-length/import-boundary/hygiene checks, affected suites, and independent review. Fix any regression and verify collected coverage was retained.
6. Push the reviewed commits to PR #230, wait for hosted checks, and merge into `development` without bypassing required checks. No production deployment is included in this request.
