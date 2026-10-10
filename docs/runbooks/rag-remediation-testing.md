# Disposable RAG remediation testing

Use Python3.12 in the pinned test image, PostgreSQL16/pgvector, Redis7.4 and
Memgraph3.8.1. The wrapper needs only host Python3.12+ and a working Docker
Compose daemon. It needs no production credentials, shared `.env`, corpus or GPU.

From the root of each lane's worktree:

```powershell
rtk proxy python scripts/run_rag_remediation_tests.py --project aquillm-rag-review -- aquillm/tests/integration/test_rag_remediation_isolation.py -q
rtk proxy python scripts/run_rag_remediation_tests.py --project aquillm-rag-r1-test -- aquillm/apps/documents/tests/test_rerank_scores.py -q
rtk proxy python scripts/run_rag_remediation_tests.py --project aquillm-rag-i1-test -- aquillm/apps/ingestion/tests -q
```

Choose a distinct `aquillm-rag-*` project per lane and per concurrent run. Do not
run simultaneous wrappers against the same project. The runner rebuilds current
worktree code into images; dependency layers use the repository's frozen
`uv.lock` and pinned `astral/uv:0.8.17-python3.12-bookworm` runtime from the existing
test Dockerfile. CI's ASR test dependencies are also installed. Image builds can
download dependencies/tokenizer data; running containers have an internal-only
network. There are no host ports, source binds, host corpus mounts or shared env
files. The explicit empty Compose env file disables implicit `.env` loading.

The runner validates project/ownership labels, internal network, named volumes,
privileges, published ports and fixed synthetic DB/cache/model endpoints before
application execution. It refuses external or inherited connection settings and
prints setting names without their values. Never bypass a refusal with a shared
Compose file or load shared environment values to make a test pass.

All installed Django apps remain loaded. Test DB is `test_rag_test` on the owned
PostgreSQL16 service. Projection roles are provisioned exactly as CI (NOLOGIN);
the test settings mirror both projection aliases to the default database. This
does not test production separation of role permissions. Transaction, migration,
FTS/index and ORM tests use PostgreSQL. The isolation suite also proves a real
Memgraph connection; graph readiness/cap/bridge lane suites should use this
service as their integration adapter when they require a live backend.

Run migration checks without altering unrelated baseline migrations:

```powershell
rtk proxy python scripts/run_rag_remediation_tests.py --project aquillm-rag-review --manage makemigrations --check --dry-run
rtk proxy python scripts/run_rag_remediation_tests.py --project aquillm-rag-review --manage showmigrations --plan
```

Do not use alternate Django settings for new lane tests. The isolated settings
guard runs before base settings import. Existing deterministic suites can patch
providers/settings within tests; transport fakes remain available for integration
tests. The synthetic HTTP model implements deterministic embeddings, lexical
rerank scores, model/health checks, and nonstreaming chat completions. Unknown
APIs (including streaming) explicitly fail; it is not a model quality benchmark.
No model weights or corpus data are downloaded at runtime.

## Browser tests

```powershell
rtk proxy python scripts/run_rag_remediation_tests.py --project aquillm-rag-review --playwright react/tests/usersettings.spec.ts
```

The frontend image uses Playwright1.52 (matching package-lock), builds the React
bundle, and shares only a labeled build/artifact volume with the app. The runner
creates a synthetic authenticated user/session in the owned database, writes
fresh `/artifacts/browser/run-<uuid>/state.json`, starts the owned app and uses
`react/playwright.rag-remediation.config.ts` with `http://app:8000`. Specs must
resolve beneath `react/` and use relative `page.goto` URLs. Absolute host/shared
base URLs, caller-supplied storage state, shared `auth.json`, and path traversal
are refused before a browser starts. Run repeated browser tests to get fresh
session state; no existing CI browser fixture is assumed.

Browser screenshots/traces/results and server logs live in the project artifact
volume under that run directory. Backend/manage test output and exit status are
retained under `.superpowers/sdd/rag-harness/<project>/run-<uuid>/` in the caller's
worktree; these are ignored local evidence. Runtime output does not include
resolved container environments. Copy only synthetic, reviewed evidence into PRs.

## Cleanup and failure evidence

```powershell
rtk proxy python scripts/run_rag_remediation_tests.py --project aquillm-rag-review --cleanup
```

Cleanup inspects ownership again, removes only validated project containers,
network and non-artifact named volumes, and retains artifact volume and host test
logs. Images/build caches remain available. A failed test leaves the stack running
for inspection; a failed isolation check performs no cleanup of foreign resources.
Never run global prune, shared Compose down, or recursive cleanup commands.

Saved `docs/audits/2026-10-10-rag-adversarial/` probes are synthetic counterexamples
that intentionally reproduce existing failures. A probe's zero exit status means
the reviewed failure still reproduces. Lane acceptance requires new regression
tests through the actual installed application/ORM path and relevant migrations;
probe success alone cannot establish remediation.

The coordinator owns schema/migration numbering, shared contracts, evaluation
labels/statistical policies, activation gates and deployment. This stack is
disposable verification infrastructure and performs no production operations.
