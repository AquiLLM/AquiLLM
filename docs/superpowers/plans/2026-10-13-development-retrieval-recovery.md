# Development Retrieval Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Checkboxes track execution.

**Goal:** Restore working retrieval on254 and deliver reproducible old/new development comparisons.
**Architecture:** Repair existing lifetime boundaries, replay current interfaces, and independently verified dense sidecars. Keep deployment and data mutations under root control.
**Tech Stack:** Django, PostgreSQL/pgvector, Celery/Redis, Python asyncio, vLLM, Docker Compose.
**Spec:** docs/superpowers/specs/2026-10-13-development-retrieval-recovery.md

## Global Constraints

- Initial recovery base b76f2ef7afa257436e1310d55e38e585f878f4e4; that release merged through PR245. Remaining graph work uses the existing isolated worktree, codex/graph-development-repair and PR246 against development720d6c12fdf1b8999edf98a3beb1def9881f9164. The merged rag-remediation and development-retrieval-recovery branches were removed on October15; retain their commits and evidence. Do not create another integration branch for this work.
- Every shell command starts rtk. No credentials, source bodies or full environment dumps in outputs.
- Do not widen the15000ms retrieval cap or reset a spent ledger. Preserve authorization and publication fences.
- No fabricated embedding provenance; no legacy-vector relabeling or blanket regeneration/deletion.
- No GitHub SSH keys; GitHub transport remains HTTPS.
- Agents edit disjoint assigned files only and do not commit, push, SSH, deploy or change shared services. Root owns integration and server operations.
- Regression tests fail before code changes; existing assertions remain unless an intentional corrected behavior is documented.
- Preserve independent acceptance gates; development experiments are explicitly labeled experiments.

### Task 1: Correct expired retrieval and user-tool handoff

**Files:** aquillm/lib/llm/turn_context.py; aquillm/apps/chat/services/rag_pipeline.py; focused existing/new chat and llm test modules.
**Interfaces:** consumes TurnBudget, bind_execution and existing limited-result helpers; produces unchanged public signatures with corrected terminal behavior.
- [x] Reproduce with real call_tool_async and fake-clock expired TurnBudget under bound execution: dispatch_count0 must produce assistant context_limited rather than uncertain side effects.
- [x] Test direct retrieval TimeoutError at exhausted budget finishes handled with limited answer, avoiding fallback tool selection.
- [x] Test user-facing uncertain vector_search passes through real provider completion without evidence synthesis; actually started uncertain side-effect tools remain nonreplayable.
- [x] Run tests RED, implement smallest coherent guards, run focused GREEN plus adjacent lifetime/execution/publication modules.
- [x] Record files, commands, results and limitations in task1 report.

### Task 2: Repair replay and make comparisons explicit

**Files:** aquillm/apps/chat/evals/retrieval_replay.py; scripts/replay_retrieval.py; tests/test_replay_retrieval.py and relevant root CLI tests; .github/workflows/test-backend-frontend.yml; docs/operations/retrieval-replay.md.
**Interfaces:** consumes run_direct_rag_turn and current retrieval_query_runtime; produces generation-blocked replay with explicit effective settings and comparable reports.
- [x] Reproduce current AttributeError for removed rag_query._llm_rewrite_query through the existing real run_turn test.
- [x] Guard current generation/query-proposal entry points, not a fabricated obsolete attribute. Baseline explicitly disables optional proposal flags and retains generic GenerationGuard.
- [x] Preserve legacy baseline reproducibility and add only explicit modes/options needed to exercise current preservation and graph treatments without falsely naming them legacy.
- [x] Record selected evidence/graph/text/capacity/proposal modes, observed budgets and configured settings; never expose endpoint credentials or source bodies.
- [x] Include root replay tests in selected release CI. Run RED then GREEN for actual helper/CLI behavior and adjacent tests.
- [x] Document baseline semantics, supported treatment matrix and disabled generation/cache behavior; report remaining limits.

### Task 3: Remove measured retrieval bottlenecks safely

**Files:** aquillm/apps/documents/services/lexical_collection.py, chunk_search_authorization.py and narrowly related source/scope/collection authorization helpers as evidence warrants; focused authorization and lexical regressions.
**Interfaces:** consumes existing selected-scope snapshots and querysets; preserves result authorization, ordering and deadlines.
- [x] Investigate existing query shape, per-candidate authorization and repeated selected-document work; root supplies live profile (lexical9603ms, selected metadata1749ms, authorized rows1632ms for31collections).
- [x] Identify concrete avoidable work and reproduce via query-count/semantic regression before edits. Do not optimize by removing validation boundaries, increasing budgets, lowering recall or broadening scope.
- [x] Implement bounded batch reads/index-compatible queries only where semantics remain unchanged.
- [x] Verify revocation, inactive users, ancestry and document lifecycle changes remain rejected; preserve stable candidate order and exact term behavior.
- [x] Report evidence and measured local improvement; root performs final live profiling.

### Task 4: Verify and build real dense coverage

**Owner:** root.
**Files:** docs/runbooks/development-embedding-policy.md; docs/runbooks/rag-source-index-generations.md; existing embedding contracts/provider/manifest/reindex interfaces; add narrowly scoped verified registration/deployment support and tests only where required.
- [x] Inspect live embedder image, model snapshot, tokenizer, precision/quantization, preprocessing, dimensions and actual transport. Record hashes privately.
- [x] Check that serving responses and provenance contracts can bind honestly to the verified artifact; repair any missing integration with RED/GREEN tests.
- [x] Inventory selected development documents/chunks and storage; prepare reversible bounded rebuild and rollback manifest.
- [x] Register verified immutable space, schedule bounded sidecars using real worker path, wait for terminal outcomes and inspect exact active coverage.
- [ ] Configure verified space consistently in web/workers, check actual dense candidate contribution and usable citations.
- [x] Preserve all exclusions and failed owners; do not infer coverage from queue success.

### Task 5: Review, merge and deploy the correction

**Owner:** root after Tasks1-4 preparation.
- [ ] Review each independent implementation, integrate and run focused combined regression plus complete required CI.
- [x] Open a PR with current change summary, validation and links to internal#4/#6 plus relevant internal graph notes; attach it.
- [ ] Stage exact candidate images, verify source/config identity, preserve rollback and take appropriate fresh backup before data/cutover changes.
- [ ] Merge into development under current user authorization, update254 via HTTPS fetch and fast-forward, deploy only intended services.
- [ ] Verify real chat path, truthful timeout behavior, actual dense retrieval, migrations/worker routing/schedulers/HTTPS/assets and stability. Never substitute health-only checks for retrieval acceptance.

### Task 6: Run matched old/new development comparisons

**Owner:** root.
- [ ] Freeze authorized corpus selection and the10known study questions plus chat431 question, input hashes and budgets.
- [ ] Obtain safely isolated old-code baseline from pre-PR244 development; do not run an old mutable binary against new representation history.
- [ ] Run fixed new-code legacy baseline and explicit independent graph/source-window/vocabulary comparisons supported by the repaired harness.
- [ ] Record gold passage/evidence support, candidate contributions, errors, latency and effective flags; retain failed runs.
- [ ] Present measured improvements/regressions and comparison artifacts; keep independent holdout and production acceptance visibly separate.

### Task 7: Bound native graph seed preparation inside its closed stage

**Files:** graph_candidate_operation_v2.py, graph_candidate_stage_v2.py, graph_admission_seeds.py and narrow native seed/runtime/repository interfaces plus focused tests. Do not edit gateway/projection frontier transport files.
**Base:** 93697e4f44d6466d3ca54ea99764285cad5f52b9, existing codex/graph-development-repair worktree.
- [x] Reproduce repeated selected-scope proof overhead using real native paths and a 31-collection/530-document fixture; preserve failing query/row evidence.
- [x] Introduce private operation-owned seed validation for exact body-free admitted references. Bind original stage/scope/admission/runtime/header/settings/query/operation, immutable dense keys and original seed references. Reject mutation, forgery, object substitution and post-exit reuse. Keep ordinary public entrypoints unchanged.
- [x] Remove redundant full-scope scans only within that unpublished operation. Keep live fingerprint, active representation, dense owner and full-text eligibility predicates. No intermediate positives, permission cache, reusable permit or new body-loading authorization.
- [x] Batch exact projection/chunk ownership seed reads where necessary so 31 selected collections do not trigger repeated whole-scope authorization per projection. Preserve before/after current projection state and native manifest/mention/artifact predicates. All eager body reads retain fresh authorization; full original scope is fresh at entry and publication; all existing final fences remain.
- [x] Add adversarial tests for revocation/change between entry/publication, unadmitted seeds, settings/dependency substitution, mutated containers, expired operation and uncertain SQL cancellation. Root serializes real PostgreSQL tests; record RED/GREEN and native SQL counts.
- [x] Independent review before commit and candidate use. No cap/deadline increase or source-scope reduction.

### Task 8: Batch mandatory graph authority reads across generations

**Files:** frontier gateway/client/contracts/service, projection frontier adapter/index/scope readers and focused fixtures/tests. Do not edit Task7 files.
- [x] Reproduce the structural query bound for 31 participating generations under the shared64-query/5000-row/1MiB limits. Prove the old physical reads cannot initialize and finalize that scope before traversal.
- [x] Add bounded generation/key tuple reads for initial/final manifests, directory/index lookup and raw scope provenance. A batch must be one bounded datastore statement, with exact per-generation validation. Preserve complete selected-document scope and fresh raw/manifest evidence.
- [x] Bound total requests and sentinels: per-statement320-row ceiling; at most160 manifest/index pairs with two-row duplicate sentinel per key, raw at most32 pairs with ten-row sentinel. Reduce packing with the remaining row allowance; the transport enforces the existing wire/expanded limits and oversized completed frames fail closed without automatic wire splitting. These are packing bounds; never increase cumulative query/row/wire/deadline limits.
- [x] Preserve accounting of each physical query, all returned rows and wire bytes. Reject missing/duplicate/extra/cross-generation records and changed markers/checksums. Fresh final attestation resources must be reserved before exploration, including rows. Uncertain reads suppress positives.
- [x] Use a versioned lossless zlib/base64 JSON envelope on the new batch route when needed to reduce measured wire bytes. Charge every received envelope byte; retain the cumulative1MiB wire cap,320-row ceiling and1MiB expanded JSON limit per response. Strict bounded decoding must reject bombs, malformed/trailing streams, duplicate JSON fields and wrong counts, with original deadline checks before/after. The old route remains unchanged. Report compressed/expanded sizes separately.
- [x] Update contract signatures/client compatibility and test real gateway boundary plus31-generation fixture and multiple groups sharing one budget. Immutable content-addressed index cache remains allowed; manifest/raw authority may not be cached across freshness fences.
- [x] Independent review and current local integration controls before live candidate acceptance.

### Task 9: Accept and roll out the repaired graph candidate

**Owner:** root after Tasks7–8 review.
- [ ] Integrate source, gateway, CI corrections; run focused combined tests, static guards and complete exact-head CI.
- [ ] Build and verify immutable candidate web/KG/gateway sources. Keep restored development graphOFF until accepted.
- [ ] Run matched graphON/OFF with same source snapshot, query, native identities, budgets and cache policy; require accepted packet and attributed graph contribution. Then real ASGI chat with persisted supported answer/citations, original431 unchanged.
- [ ] Update PR246 and internal#4/#6 with current evidence and limitations. Merge/deploy254 under standing user authorization only after gates; verify public chat path and rollback readiness.
- [ ] Run the frozen broader study and keep historical759 findings, regression measurements and independent quality acceptance distinct.

### Task 11: Bound fresh native seed policy proofs without materializing the scope

**Status:** Authorized after the private complete-scope batching feasibility gate; no graph efficacy or latency acceptance is implied.
**Base:** e4c6a841d10bbb0ffe93209aad34156602a1400e, existing codex/graph-development-repair worktree. Root alone commits/pushes/SSH/deploys. No child agents. Prefix every shell command with rtk. Do not print credentials, source bodies, SQL parameters or raw retrieval inputs. Preserve private evidence files.
**Requirements supplement:** Read `.superpowers/sdd/2026-10-13-development-retrieval-recovery/resume-native-exact-scope-proof-design.md` in full. Its exact proof, eligibility guards, accounting and negative cases are requirements, subject to the explicit rulings here. Do not read other historical plans/reports unless required to understand an interface.
**Files:** One focused private native proof helper under apps/collections/services, minimal shared immutable-context validation extraction in retrieval_authorization.py if necessary, graph_seed_reads_v2.py for exactly-one-statement metering reuse, and narrow closed_graph_seed_validation.py/closed_seed_projection_batch.py/extended_seed_repository.py integration. Add focused tests and their existing CI selector. No frontier/protocol/copy/ready/body/public policy behavior changes. Do not remove duplicated direct transaction controls in this task.

- [ ] Reproduce current real Django policy cost at both batch boundaries on PostgreSQL with31 selected collections/530 selected documents:4 data statements/1122 returned rows. Compare against required2 statements/2 proof rows. RED must identify missing optimized behavior, not only a nonexistent symbol.
- [ ] Implement only the private exact native closed-seed route. Before and after the identity batch, independently evaluate current selected permissions P (bounded C+1) and all eight native document relations UNION ALL R (globally bounded N+1 before aggregate). Return one fixed five-integer count record and accept only(C,C,N,N,C): permission rows/distinct collections, document rows/distinct UUIDs/distinct collections. No pre-count DISTINCT or hidden deduplication, no reusable token, no fabricated authorized pairs, no serialized scope cache. Parameterize values/quote trusted metadata, keep indexed selected membership and original database deadline. Inspect an actual bounded PostgreSQL plan without logging parameters.
- [ ] Ruling: the native closed owner may fail closed when any part of its full captured selected scope drifts, even if that part supplied no seed. Public/custom partial-scope behavior is unchanged. The proof may return its one count record on false; the exact acceptance predicate grants nothing. Both fresh boundaries and all source/body/representation/projection/generation/capability/publication fences remain. Native eligibility must reject overrides/unsupported backend/fields before dispatch; any mid-read replacement or failed/uncertain dispatch aborts without fallback or retry.
- [ ] Reuse the original signed context and capability identity validation. Original principal, policy, database, method identities and immutable native field/model identities must be checked as specified in the design. Unsupported cases choose the old independently guarded route before dispatch. Do not add a caller-controlled bypass/skip flag or expose a success cache. Avoid building a generic query framework.
- [ ] Both proof statements use the original FrontierBudgetV2: reserve before dispatch, charge actual cursor rowcount, one data statement only, and poison uncertain work. Bound aggregate inputs independently of one-row result. Preserve original64queries/5000rows/1MiB and all absolute deadlines. Adjust only the native batch's unspent final-policy reservation to its true one-row maximum; keep ordinary reservation and seed overflow sentinel unchanged. No spent-row refund or counter assignment.
- [ ] Real PostgreSQL tests cover healthy scope, revoked/inactive/replaced principal, permission/document drift between the two fresh proofs, lost selected collection, selected UUID duplication across models/collections, missing+duplicate with total N, N+1 overflow, irrelevant unselected collisions, malformed counts, custom/overridden methods/native-field ineligibility, wrong aliases/context/budget, cancellation/unknown rowcount, late result and dependency mutation. Preserve integrated body/hash/representation/generation race rejection. Use smallest relevant existing fixtures; do not create hundreds of mirrored unit cases.
- [ ] Run real before/after policy counts and same original-ledger identity assertions, targeted existing authorization/closed seed/meter regressions and required static checks. Root serializes PostgreSQL ownership; do not run while another owner holds it. Copy/hash-verify changed source/tests into the named owned no-bind test stack. No destructive fixture resets or bypasses. Tests must be RED then GREEN. Report exact commands/results/warnings and limitations; do not repeat passing broad suites without a change or unresolved concern.
- [ ] Self-review and leave the product diff for independent root-dispatched review. Report remaining physical-prefix terms honestly: this task alone is known insufficient; the separate scope batching task and real native closure acceptance remain. Do not claim final graph repair, merge, deployment, or comparison completion.

### Task 12: Integrate the proven native metadata reductions

**Working root:** C:/Users/jackj/.codex/worktrees/rag-remediation/AquiLLM. **Base:** f15f2d7fa2f98e559c72ec14e3eb7a9628a02e19. **Branch:** codex/graph-development-repair, existing PR246. No new branch/worktree/PR. **Q:** .superpowers/sdd/2026-10-13-development-retrieval-recovery.

**Binding design:** Q/resume-native-metadata-product-design.md, all sections after its historical entry-gate discussion. Read Q/resume-metadata-product-entry-ruling.md first; it supersedes that draft's pending status and obsolete counts. Q/resume-native-metadata-spike-report.md and its private scripts are differential evidence, not production code to copy wholesale. All entry gates for this metadata implementation are now met; final native/protocol/rollout gates remain unmet.

**Ownership:** one implementer owns the design's named new retrieval modules, narrow existing native-entry/direct/ontology/batch integrations, cohesive regression helpers/tests and relevant CI selectors. Task11 final modules are stable and approved. Root alone owns Git integration and all SSH/server changes. You own the sole local PostgreSQL/source slot until explicit release. Do not spawn subagents, commit, push, merge, deploy, create branches, or touch other checkouts/services. Report proposed ownership expansion before editing it.

- [ ] Read the product design and actual f15 module locations. Add failing tests at genuine native entry seams before product edits. Confirm ordinary native generated GraphArtifact manager is eligible, while unsupported/custom implementations select generic behavior before work. Selected-path dependency/owner/type/method/alias drift fails closed without redispatch. No false-positive native eligibility via arbitrary observed replacements.
- [ ] Capture the exact original selected object at closed owner construction as expected inputs only, before first ontology; bind exact original owner/runtime/group/ready/codec and FrontierBudget once. Both fresh ontology calls, direct-tier calls and batch chunk proof use that checked owner, same original ledger and original lifetime. No caller-controlled bypass, cached proof, new ledger or ContextVar permit.
- [ ] Extract pure original bounded candidate QuerySet construction for both loaders. Preserve every predicate, type, ordering, six-column projection, L+1 sentinel, conversion, ambiguity and mass behavior. Implement the private native repository seam with one fresh atomic bounded membership proof plus unchanged candidate SQL per actually attempted tier. Mandatory typed header; C+1 current relation bound before aggregation; total/distinct collection/distinct artifact/exact six-field captured tuple counts all equal C. Strong captured collection/artifact pairing and ordinary one-statement snapshot are intentional. Valid-empty differs from failed proof. L+2 reservation, maximum130; charge every header/candidate/overflow record with unchanged meter.
- [ ] Implement separate fresh bounded ontology artifact proofs at both original call sites, using the actual YAML completion/parser through a shared pure refactor. No fabricated artifact rows. Preserve original selected input checks, C+1 relation bound, exact identity-pair count, statuses, non-evaluation/collection predicates, MIXED_ONTOLOGY count/error attribution, active-YAML two-row sentinel and parsed identity. Implement the single fresh chunk-owner equality proof against validated references/request map at the original batch boundary, c+1 input bound and total/distinct/exact-pair counts. Preserve both Task11 policy barriers, both projection barriers, all identity payload rows and body/source/hash/representation/generation/final guards.
- [ ] Preserve exact native type/manager/QuerySet/compiler/model-field semantics and explicit trusted deterministic pg_catalog.C text equality; original Python equality distinguishes case, Unicode composition and trailing spaces. Any needed catalog/environment checks must be inside the bounded metered statement; no hidden setup SQL. Preserve outer branch and inner meter contexts, ALIAS join limit, original clocks/deadlines,64queries/5000rows/1MiB, reservation-before-dispatch and poison-on-uncertainty. Shared _metered_seed_rows and Task11 one-row reader remain unchanged. No context deduplication, retry, caching, locks, isolation changes, cap increases or spent-budget refund.
- [ ] Port focused meaningful real-PG semantic/accounting/race checks from the design using genuine product integration. Cover native eligibility positive and drift/custom negatives; original candidate oracle across four tiers; zero/128/129 rows; malformed proof and count/type/pair failures; original predicate/embedding/automatic-link behavior; deterministic fresh-call drift; timeout/uncertain accounting/settings restoration; both real ontology boundaries and YAML parsing; native chunk route and retained guard order. Avoid duplicating private mirrored cases or weakening existing tests to accept dropped guards. New source/test files <=300 lines; no existing ratchet growth or new exemptions.
- [ ] Demonstrate actual product SQL reduction on the existing genuine closed native_frame and original nonempty ledger, retaining its actual fixture shape. The exact C31,p7,c16,m264,t5,candidate rows[0,0,0,0,4] transcript is remote e4 evidence, not an existing integrated local fixture. Reuse the proven31-artifact/16-chunk fixtures for differential product SQL, paired query plans/control counts and reservation maxima; do not build a second corpus solely to mimic the remote counts. Record exact integrated counts and all unchanged calls/payloads, with no extra data statements. The projected remote total remains15queries294rows and sequence[1,1,1,7,1,264,1,7,1,1,1,1,1,1,5] until root measures the actual isolated native254 replay. Escalate any contradictory reduction or extra debit rather than forcing a result. The canonical scope fixture with synthetically precharged prefix already ends63queries4937rows,16allocated seeds,2candidates18identities, zero holds/uncertainty; its later maximum40 fails honestly with32available. This is not a product/native/wire gate; do not rerun unchanged private proofs or introduce protocol code in this task.
- [ ] Run focused RED/GREEN then the relevant existing regression selection listed in the design once after integration, plus all required file-length/import/diff checks. Add product checks to mandatory CI selectors as designed. Use only owned aquillm-rag-recovery-20261013-app-1 with /app/.venv/bin/python3.12, KG_REQUIRE_POSTGRES_TESTS=1, PYTHONPATH=/app:/app/aquillm, --ds=aquillm.settings_projection_test --reuse-db. Copy/hash-verify source into the no-host-bind container. No destructive fixture resets or bypasses. All shell commands start rtk. Keep secrets/raw source bodies/questions/SQL params out of tool output.
- [ ] Self-review and report exact changed files, commands, RED/GREEN results/warnings, source hashes, real measured debit/control/plan results, limitations and any scope decisions in Q/task-12-report.md. Explicitly release PostgreSQL/source slot at completion. Return concise DONE/DONE_WITH_CONCERNS/BLOCKED/NEEDS_CONTEXT, no commits. Root dispatches independent review. No claim of graph efficacy, deployment readiness or full audit completion.

### Task 13: Complete scope reads within the original frontier budget

**Entry gate:** Task12 implementation and independent review must be accepted before dispatch. Root supplies its exact reviewed commit and measured product SQL report. This task is planned, not yet authorized to edit product code. Use the existing codex/graph-development-repair worktree and PR246; no new branch, worktree or PR.

**Binding design:** Read Q/resume-complete-raw-scope-product-design.md in full. Its exact constants, two strict request schemas, fixed query, scope state, original-ledger accounting, failure ordering, allowed file boundaries and nine acceptance steps are requirements. Q is .superpowers/sdd/2026-10-13-development-retrieval-recovery. The design's synthetic15/294 prefix is projected remote arithmetic, not an observed integrated native result. Consume Task12's truthful local fixture counts without reshaping its fixtures.

**Ownership:** one fresh implementer owns only the design's scope planner, fixed route/codec and narrow existing frontier integrations, cohesive tests and CI selectors. Reuse existing transport, deadline, worker occupancy, checkpoint and ledger helpers; do not build a parallel generic framework. New modules require a real responsibility/line-ratchet boundary. Each new file remains at most300lines; no ratchet growth or exemption. Task11/12 metadata, shared meter/budget semantics, copy traversal, graph builders and schema are outside this task. Root alone commits, pushes and operates SSH254. No child agents or unrelated service changes.

- [ ] Establish a meaningful RED on the real existing scope state/unsupported route. Implement the fixed scope-complete state from the first directory stage and preserve owner/worker/replay parity. The unchanged canonical31/530 fixture must reduce18raw reads to5 and preserve exact complete scope dictionaries; retain the measured failed prefixes as comparisons.
- [ ] Implement only the design's versioned scope endpoint and window/single modes, exact schemas/checksum/magic, bounded fixed raw query and unchanged leaf codec. Single is solely the predispatch oversized-raw-checkpoint path. No postdispatch retry, legacy downgrade, copy behavior change or new ledger. All completed statements/records/bytes are charged; unknown work poisons the original ledger.
- [ ] Follow all nine acceptance steps in the binding design, using the actual client/ASGI/worker/Memgraph/owner route for physical query/plan, wire and accounting evidence. Prove canonical all16 allocated identities, useful candidates and fresh final within64queries/5000rows/1MiB and original deadlines. Preserve320-row saturation as capacity failure with exact receipt and no partial publication. Keep the full failure/cleanup/replay and legacy compatibility checks focused; do not mirror private test cases unnecessarily.
- [ ] Use only the established owned test stack and sanctioned owned Memgraph harness. Root grants the sole local source/PostgreSQL/Memgraph slot after Task12 release. No destructive fixture reset outside that harness, guard bypass, index rebuild or production access. Copy/hash-verify exact source into the no-bind container; all shell commands start rtk. Keep secrets, raw questions/source bodies and SQL parameters out of tool output.
- [ ] Reconcile actual Task12 SQL into the same original frontier ledger, preserving truthful fixture boundaries. A separated canonical/native fixture is allowed only with explicit limits and root's mandatory closing ordinary native254 replay. Report any actual count, wire, packing or availability contradiction immediately; do not relax caps, narrow authorized scope/seeds, invent earlier stopping or force the emulator result.
- [ ] After focused RED/GREEN, run the relevant existing regression selection once and the required structure/import/diff checks. Self-review and write Q/task-13-report.md with exact commands/results/warnings, source hashes, actual SQL/RPC/row/wire counts, largest request/checkpoint, retained guards and limitations. Release the local source/database slots and return concise status without commits. Root performs independent review, then native replay, ASGI/comparison and deployment gates; fixture success alone does not establish graph efficacy.
