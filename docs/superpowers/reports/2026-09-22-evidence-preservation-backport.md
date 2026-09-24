# Evidence preservation: exact non-graph adaptation manifest

This is a future adaptation inventory, not a backport, deployment or activation.
Development Tasks 1–6 and whole-branch corrections were accepted and merged in
`612b98422aea99f9698f0b1328b1a225478f3b87`, on the adaptive-selection baseline
`52af11daab20975a5fbed6db742e9a596bd4f4da`. The later evaluator-history correction
`8664556a164679807dc27bc77565e398be573fd0` and this manifest update await review.
Live activation gates remain open. No graph branch is to be copied into main.

Target observations are **provisional and unfetched**: local `origin/main` was
`1d0468b0470225eb0b6236fc573c28668d2e3aff`; local `main` was the older
`dbc922e1dff31ab65cf48bfbdfe2c9e8364e91b0`. Refresh the actual target and repeat the
dependency audit only when creating the later main-based branch/PR after development
acceptance. These observations do not identify a current remote or deployed revision.

## Required preservation implementation and review fixes

Every row is required behavior/dependency evidence. Adapt the final implementation
together; this is not permission for wholesale cherry-picks of mixed development files.

| Exact commit | Required behavior |
| --- | --- |
| `cf709c14aa2fd139a28488a8dc3bbc5d586f7851` | Task 1: immutable exact source/span/coverage contracts, shared turn ledger, default-off independent flags. |
| `642ffe0c96c5e03cb7914c4a259121b24eaeb0ce` | Task 2: complete-pair/window preparation, verified pair protocol, exact score provenance/cache and bounded HTTP work. |
| `008737e487d829fff1807c45c9b04da39a5f2f28` | Task 2 fix: nested source/window preparation deadlines. |
| `3c3a6f10b4eb06fc255e5d07ff8f6461a50f3f4a` | Task 2 fix: recheck deadlines after shared ledger waits. |
| `b9c1d241ff8972c15bcc8da1c9713b945c94fb87` | Task 3: bounded current-source hydration, exact prepared evidence, capacity, common selection, provider protection and figures. |
| `7d67ae2d76519de2a9ec074539cdb57fd265fb59` | Task 3 fix: protected OpenAI evidence bypasses the legacy estimator. |
| `2a41d77fef468bcb9d00cb41b346be78f588b80d` | Task 4: authorized source continuity and follow-up rehydration. |
| `77fdff51556a870e79677c5d24615df4026481de` | Task 4 fix: reconcile mixed explicit/plural/ordinal references. |
| `cb49b1e47f3a0189df4f1da2121a8e472fb1e486` | Task 4 fix: singular explicit-source binding without unrelated history. |
| `d12ff8e9ecdc6d2f4bc2f3da7241118b31fc6004` | Task 5: cumulative acquisition/coverage and direct/manual/normal handoff, synthesis/recovery lease. |
| `f4da561847cf44ce7d5cdb8eb2457c925dbf227a` | Task 5 fix: actual ASGI turn ownership/disconnect cancellation, exact deterministic support, stricter tool waits, bounded cache staging/pointers and non-source behavior. |
| `3b53c40f8d2e658e5ebf4988fa7896e8aea4adb1` | Task 6: actual four-arm evaluator, frozen quality/operational corpora, activation-v2, per-worker capability lifecycle, SDK/ledger observations, adaptive-only non-graph authorization dependency fix, short-final persistence fix, CI/runbook/status. |
| `935da23524c8f88aacb10552f1772b7e3c57936f` | Task 6 local self-review fix: require completion-reserve evidence to match the actual runtime reserve in every observed turn, preventing an inflated operator-record allowance. |
| `a287509ea22413050e5c25ab193268dd5a8459c0` | Task 6 review fix I1–I7: exact human answer/evidence/run/code binding and observed bounded-partial facts; isolated observation failures; actual terminal short completion/streaming; resolved arm/runtime controls; pilot-only subtype proof; per-identity malformed historical UUID rejection retaining current/valid evidence. |
| `f80f2a1dc6d1d04b68873e8003839a9bbc357c99` | Task 6 review fix R1–R3: observed source/span-specific selection exclusion causes, actual common completion/recovery/publication policy snapshot, and stable typed SDK observation normalization across review, partial proof and serialization. |
| `78e87679f3abe6bf6f270e43d9a284f36e65ba0e` | Whole-branch I1–I2: separate hard initial preparation/scoring deadline from current-authority finalization under the original retrieval ledger; preserve whole-pool rank fallback and protected synthesis after scoring-only expiry; portable raw Git evaluator provenance without RTK. |
| `8664556a164679807dc27bc77565e398be573fd0` | Evaluator history fix: serialize the authored historical chunk ordinal alongside the real DB document/chunk identity; retain ordered plural anchors after persistence, current-text rehydration, and current authorization revocation checks. Frozen labels and application resolver behavior are unchanged. |

This document's enclosing documentation-only commit need not contain its own hash.
Append exact later implementation/review-fix SHAs before final backport acceptance.

## Prior selector dependencies to adapt

The selector requires full candidate fusion, comparable score transport, bounded
hydration/final scoring, authorization, packet assembly and provider presentation.
Porting only `rag_selection.py` would lose these invariants. These exact commits
identify the non-graph dependency closure and its fixes; mixed graph hunks are excluded.

| Exact commit | Non-graph dependency |
| --- | --- |
| `7a83a84eab63ac01d3709366d249a9722afa69e4` | Relevance/diversity selector, profiles/types/limits/feasibility and legacy adapter. |
| `8044c2fe100fd6279e23114bf502c82ef39cf2a5` | Comparable reranker scores and effective-input provenance. |
| `0182c30d8e06f16c34f3a03ed48b52eaf7ec708a` | Offline selector replay and bounded text work. |
| `f385a08add05321560c029e9774a80de8fe1291e` | Reject score reuse with unverified model identity. |
| `771220b9f734f5321dfe4aeefa123c37773e2493` | Private scores through the complete fused candidate pool. |
| `06f5fedddb125a7b0b47859b8465af890ba11a25` | One fusion vote per query and invalid-sidecar rejection. |
| `bec3d83c0a5b4b1fa0cdb4f9a0d6c9f82f0bca0d` | Typed score envelopes and search helper split. |
| `5b1201ace5118067563846cf1922763849b6bde5` | Independent selector flags/runbook; exclude PPR controls/hunks. |
| `c4fcac9549009a203717d7170ddb55309da10e9c` | Final-union scoring and hydration under a shared deadline. |
| `cf4ef15620b5504845707af1d3e2fe6b0cc18657` | Authorized adaptive chat integration and final handoff. |
| `8448a3b3ccd3d330273a5fb4a2044821e132a283` | Relevant final-scoring/selector CI contracts; exclude graph-policy jobs. |
| `da575fadd74dcc3044f87d75a47b91fb44e77e43` | Incomparable retry rejection and actual batch-shape handling. |
| `eba2c2eb7f3935f9fff7cd813c09e551c42e926a` | Partial-retrieval adaptive integration regression. |
| `f6cd0b00300d44fa2b74e54e4dc8f5a632e88da7` | Relevant structural/style changes, with no ratchet expansion. |
| `60394bda25f711c0aa034da0ddd844b925a5d0a0` | Bounded shadow diagnostics, preserving explicit scoring opt-in. |
| `64a0eedde5ae9aa6c50305e94f09a6822a636916` | Needed provider/tool/cache helper extraction and regression seams only; exclude unrelated/graph refactors. |

## Adaptation boundaries

| Source bundle | Main adaptation and invariant |
| --- | --- |
| `lib/retrieval/*`, `lib/evidence_observation.py`, chat `rag_preservation_config`, `rag_turn`, acquisition/coverage/source helpers | Keep application-independent immutable contracts, one outer ledger across routes/fallback, finite cumulative actions/sources/text/pairs/concurrency/deadline, cancellation/publication fences and explicit partial outcomes. Observation imports must not cause Django setup in the pure contract package. |
| Chat `rag_selection*`, `rag_legacy_selection`, `rag_retrieval`, evidence packet/handoff, document score transport/window/cache helpers | Retain the complete fused pool and one final relevance/diversity selection. No old top-k/per-document pruning before feasibility. Prepare once, score/redundancy/cost/send the same exact representation; reuse only compatible exact query/source/revision/model/template/window identities. |
| `retrieval_authorization`, collection authorization builders, source loading/revalidation | Implement a graph-free selected-scope adapter with current VIEW/EDIT/MANAGE or target `user_can_view` policy, exact supplied-context validation, bounded requested-document set and current revision checks. Task 6 also requires this builder for valid served adaptive selection with KG off. Invalid configuration and legacy/shadow behavior stay unchanged. |
| Mixed `chunk_search`, candidate/scoring/authorization modules and development private materialization | Adapt vector/trigram, source preflight, score-sidecar and current-authority portions to main's actual search API. Do not copy `hybrid_graph_authorization`, seed, projection or traversal imports. Development's graph metadata-only hydration adapter stays development-only; main still needs the same bounded non-graph source preflight. |
| Consumer connect/receive/publish, `rag_turn_tasks`, `rag_preservation_turn`, `rag_acquisition`, manual/source tool wiring | Preserve actual owned serialized ASGI turn/disconnect cancellation, immutable stored history, direct `/search`/normal tool integration, bounded tool waits and one selector/synthesis. Main's explicit authorized UUID search may remain broader than selected collections; never lend that manual exception to planner/follow-up acquisition. |
| `synthesis_budget`, `synthesis_dispatch`, `evidence_protection`, complete-turn/provider guards and final SDK arguments | Port the separate sealed synthesis/recovery allowance with the original retrieval counters and protected evidence. Main may lack development's `complete_turn_*`, `openai_request` and `rag_citation_extracts` splits; adapt the seams together to its actual layout. Verify Claude/Gemini/OpenAI shaped requests and overflow/recovery, including actual dispatch counts. |
| `bounded_rag_cache`, `rag_cache*`, window-score cache and pair capability registry | Main's cache/ref helper layout differs. Preserve finite 4-worker/no-queue/400ms callers, transport bounds, <=1MB expiring staging, bounded process-local usable pointers and publication fencing. A late worker may leave only an undiscoverable expiring staging object; do not replace this with an unfenced canonical cache write. |
| ASGI wrapper, `pair_worker_lifecycle`, canonical scorer, capability registry | Wire real per-serving-worker lifespan startup/renewal/invalidation. No AppConfig, management process or shared cache counts as worker initialization. Keep one finite warm slot, <=3 pairs/15s, pinned local assets, exact independent runtime attestation, held-reference invalidation and unknown fallback. Unsupported launchers remain unknown. |
| `visibility`, message adapters, persistence guards | Preserve short completed answers such as `42`, `Yes` and one-word owner/status through storage/display, without exposing interim/tool/reasoning artifacts. Base stored system prompts stay separate from runtime memory augmentation. |
| Quality/operational eval helpers, private SDK/source/ledger/cache observations, corpora and CLI | Use actual isolated DB/ASGI/retrieval/reranker/SDK paths and unchanged baseline shaping. Retain exact support identities and human unknowns, four-arm joins, strict snapshot/config provenance and activation-v2. Fixture output is always ineligible; operational results stay outside quality/latency denominators. |

Explicit exclusions: `apps/knowledge_graph/**`, `lib/knowledge_graph/**`, graph
schemas/projection maps/seed lookup/traversal/PageRank, `hybrid_graph_*`, graph-only
settings/services/migrations/tests, and graph rollout changes. Development graph
overlay/authority regressions remain required for development acceptance, but do
not become dependencies of main. No new preservation module imports KG code and
no new `lib` module imports application code.

## Configuration and runtime differences

Keep every preservation mode `legacy`, every opt-in switch `0`, selector `legacy`
and PPR off until the corresponding independent live gates pass. Preserve pair
context 1024, existing models/GPU/context limits, source revisions, current
3500-token evidence default and three-per-document legacy behavior. Resolve actual
target values rather than assuming historical deployment settings. Preserve stricter
operator action/tool controls and the cumulative pilot caps in the
[runbook](../../runbooks/evidence-preservation.md).

Main needs the new default-off `RAG_PAIR_CAPABILITY_WARM_ENABLED` and empty
`RAG_PAIR_CAPABILITY_ATTESTATION` settings documented, with independently verified
short-lived attestation and local pinned tokenizer assets in every serving worker.
The web and reranker cache mounts may differ; model names and `/models` metadata
are insufficient capability proof. No new service, dependency, migration, model,
GPU allocation or context-size increase belongs to this adaptation.

Preserve `.gitattributes` exact-byte rules for both frozen JSON assets: full V5
SHA256 `6eb8e3617ef8e07a083b33bc502d9615017a1a210997a2c333d3cd822ddc3b07`, heldout
canonical SHA256 `01b3c9fed145c3ef46bf96552e1b4c0cc662260891af6033c114a2a2bac67395`,
ops-v2 SHA256 `ae60c47e40d7fa3574d6e8fa6c32e40eba51c30121e54741316d50314f8d3136`.
Do not normalize reviewed bytes, derive labels from model output or tune on heldout.

## Verification required on the later main branch

Retain the non-graph selector replay/scoring/fusion/identity/authority tests from
the exact prior commits above. Adapt the preservation test modules listed below,
plus main's existing citation API, document move, conversation persistence,
message-adapter/visibility/interim, prompt/context budget, numerical citations,
direct synthesis and both app/lib completion-retry suites. Actual source revisions,
manual selected-scope distinctions, revoked/moved/deleted evidence, all four arms,
no-extra-final-score reuse, actual ASGI disconnect, frozen support, provider shaping,
worker startup/expiry and blocked cache transports must remain directly exercised.

Use main's real CI workflow after adapting its seams: Python 3.12 backend and
Node 20 frontend `npm ci` plus `npm run build`, system check, tracked-path hygiene,
file lengths, import boundaries, logging/retrieval redaction and exact changed-file
Ruff/format. Current local Windows Python 3.13/Node 22 evidence is not that CI runtime.
Keep source/operational reports content-addressed and compare identical frozen
fixtures between accepted development and the adapted main branch.

Activation still requires real deployment identity per worker, both frozen splits,
named human answer review, snapshot-matched four-arm quality/uncertainty, numeric
deployment targets, measured reserve and concurrent cold/warm latency/cost, zero
applicable authorization/citation/safety violations, and observed refinement plus
shared acquisition-pair OR retrieval-deadline exhaustion in the pilot profile of
both candidate modes; one_action cannot substitute.
Source/action-only stops cannot substitute. Baseline new-ledger/cancellation failures
stay visible diagnostics; ordinary baseline authority violations still block.
Unknown candidate safety or missing evidence blocks. PPR has an independent open gate.

No target merge/deployment follows from this manifest. After later main acceptance,
use the authorized backup/deploy/identity/health/rollback process and enable only
the separately proven modes. Exact target diff and measured differences belong in
that future PR, not an assumed parity claim here.

### Preservation test path inventory at the source implementation


```text
aquillm/apps/chat/tests/test_conversation_persistence.py
aquillm/apps/chat/tests/test_evidence_operational.py
aquillm/apps/chat/tests/test_evidence_bounded_partial.py
aquillm/apps/chat/tests/test_evidence_effective_config.py
aquillm/apps/chat/tests/test_evidence_operational_profiles.py
aquillm/apps/chat/tests/test_evidence_review_subject.py
aquillm/apps/chat/tests/test_evidence_loss_causes.py
aquillm/apps/chat/tests/test_evidence_completion_controls.py
aquillm/apps/chat/tests/test_evidence_sdk_normalization.py
aquillm/apps/chat/tests/test_evidence_git_portability.py
aquillm/apps/chat/tests/test_rag_scoring_timeout_fallback.py
aquillm/apps/chat/tests/test_short_final_completion.py
aquillm/apps/chat/tests/test_evidence_quality_delivery.py
aquillm/apps/chat/tests/test_evidence_quality_eval.py
aquillm/apps/chat/tests/test_evidence_quality_gates.py
aquillm/apps/chat/tests/test_evidence_quality_live.py
aquillm/apps/chat/tests/test_evidence_quality_review.py
aquillm/apps/chat/tests/test_evidence_quality_seed.py
aquillm/apps/chat/tests/test_rag_acquisition.py
aquillm/apps/chat/tests/test_rag_budget_boundaries.py
aquillm/apps/chat/tests/test_rag_commit_fence.py
aquillm/apps/chat/tests/test_rag_context_budget.py
aquillm/apps/chat/tests/test_rag_coverage.py
aquillm/apps/chat/tests/test_rag_delayed_provider.py
aquillm/apps/chat/tests/test_rag_evidence_handoff.py
aquillm/apps/chat/tests/test_rag_fix_asgi.py
aquillm/apps/chat/tests/test_rag_fix_cache.py
aquillm/apps/chat/tests/test_rag_fix_continuity.py
aquillm/apps/chat/tests/test_rag_fix_coverage.py
aquillm/apps/chat/tests/test_rag_fix_handoff.py
aquillm/apps/chat/tests/test_rag_fix_tool_deadlines.py
aquillm/apps/chat/tests/test_rag_fix_window_cache.py
aquillm/apps/chat/tests/test_rag_normal_handoff.py
aquillm/apps/chat/tests/test_rag_preservation_routes.py
aquillm/apps/chat/tests/test_rag_provider_evidence_guard.py
aquillm/apps/chat/tests/test_rag_query.py
aquillm/apps/chat/tests/test_rag_refinement_routes.py
aquillm/apps/chat/tests/test_rag_source_continuity.py
aquillm/apps/chat/tests/test_rag_source_continuity_integration.py
aquillm/apps/chat/tests/test_rag_source_document_tools.py
aquillm/apps/chat/tests/test_rag_source_figures.py
aquillm/apps/chat/tests/test_rag_source_loading.py
aquillm/apps/chat/tests/test_rag_source_pipeline.py
aquillm/apps/chat/tests/test_rag_source_preparation.py
aquillm/apps/chat/tests/test_rag_source_selection.py
aquillm/apps/chat/tests/test_rag_source_singular.py
aquillm/apps/chat/tests/test_rag_source_synthesis.py
aquillm/apps/chat/tests/test_rag_source_types.py
aquillm/apps/chat/tests/test_rag_synthesis_lifecycle.py
aquillm/apps/chat/tests/test_rag_turn_budget.py
aquillm/apps/chat/tests/test_rag_turn_lifecycle.py
aquillm/apps/chat/tests/test_rag_websocket_lifecycle.py
aquillm/apps/chat/tests/test_tool_payload_compaction.py
aquillm/apps/collections/tests/test_django_retrieval_authorization.py
aquillm/apps/documents/tests/test_chunk_rerank_pair_capability.py
aquillm/apps/documents/tests/test_chunk_rerank_window_fences.py
aquillm/apps/documents/tests/test_chunk_rerank_window_scores.py
aquillm/apps/documents/tests/test_chunk_rerank_windows.py
aquillm/apps/documents/tests/test_pair_worker_lifecycle.py
aquillm/apps/documents/tests/test_rerank_http_cache.py
aquillm/apps/documents/tests/test_retrieval_log_redaction.py
aquillm/lib/llm/tests/test_budget_observation.py
aquillm/lib/llm/tests/test_context_packer.py
aquillm/lib/llm/tests/test_direct_synthesis_grounding.py
aquillm/lib/llm/tests/test_evidence_observation.py
aquillm/tests/integration/test_production_retrieval_authorization_reachability.py
```

### Review-fix adaptation closure

Adapt the canonical review subject and bounded-partial helpers with both runner
rescore/attachment consumers; old answer-SHA-only reviews are intentionally unknown.
Preserve source-bindings and final SDK/acquisition trace mapping. Port observer
failure state and robust completion signals together so instrumentation cannot
change dispatch, reservation, cancellation or return semantics. Port terminal stop
eligibility through complete_turn, visibility and openai_streaming together with
the already-required persistence adapters. Keep the shared pure OpenAI runtime
getters and their dispatch/request consumers coupled to effective comparison
configuration; these extractions preserve existing defaults and retry behavior.
Malformed historical IDs must be skipped only during UUID parsing, keeping anchors
for unavailable notices and all current scope/principal/revision revalidation.
The new five CI modules above and existing observer, continuity, completion/retry,
streaming, visibility, authority and CLI tests are required. No graph imports,
ratchet increases or live activation are part of these fixes.

Round 2 couples the opt-in final-selection exclusion observer to the typed loss-cause
validator: source identity, current fingerprint, affected span, failed resource
admission and acquisition/exclusion/final-SDK ordering must agree. A human reason
alone is not an observed cause; unsupported cause types remain unproven. Port the
shared completion-policy getters, their production caller and snapshot together,
including continuation, output, citation repair, recovery and final-only publication.
Use the same JSON-safe normalization for SDK models, dataclasses, UUIDs and sequences
in trace capture, review subjects, partial comparisons and both report writers.
Unsupported representations invalidate observation completeness rather than changing
the application result. The three additional CI modules above cover these boundaries.

Whole-branch correction requires adapting `source_deadline` and
`rag_selection_scoring` together. Keep initial source work and scoring under their
existing hard phase deadline, but permit bounded current-source revalidation and
rank fallback under the original remaining retrieval ledger. Global closure still
fences publication, and revised/revoked rows are removed before packet handoff.
The real scorer/ledger/selection/protected-synthesis regression above belongs to
that dependency closure. Evaluation provenance uses raw standard Git at all three
subprocess sites, with unchanged failure behavior and saved-observation identity;
RTK remains only a developer workflow tool, never a runtime dependency.

The evaluator-history correction is required for faithful live follow-up fixtures:
carry the authored historical `chunk_number` into the public history row's `chunk`
field before serialization. Keep strict coordinate validation and current-source
authorization/revision revalidation intact. The existing CI seed module now verifies
the four frozen development plural histories through DB save/load, exact displayed
coordinates, plural resolution, current text and permission revocation. Earlier
diagnostics built by the incomplete history adapter remain immutable diagnostics;
collect fresh matched runs after the correction, without changing frozen labels or
claiming that those earlier runs used the fixed harness.
