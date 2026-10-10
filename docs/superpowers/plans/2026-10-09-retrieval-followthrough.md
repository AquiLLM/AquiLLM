# Retrieval Follow-through Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development with independent task and whole-branch reviews. The user's parallel-agent instruction persists: independent tasks use separate native worktrees; shared embedding tasks are sequential.

**Goal:** Deploy truthful vector provenance, bounded outage recovery and usable human-review tooling, and repair development GPU runtime consistency.
**Architecture:** Provider-bound receipts flow atomically into nullable chunk metadata. An independently reviewed recovery follow-on uses that same embedding path. Offline review preparation reuses existing immutable observation binding and never changes scoring or eligibility.
**Tech Stack:** Python, Django/PostgreSQL/pgvector, Celery/Redis, Docker Compose, vLLM.
**Spec:** docs/superpowers/specs/2026-10-09-retrieval-followthrough-design.md

## Global Constraints

- Development 149.165.150.254 only; production 149.165.169.204 excluded.
- Preserve extractor 3000 ms, direct/extended 4500 ms, overall 5000 ms, authorization, graph provenance, resource caps and zero-retry graph semantics.
- Preserve embedding wire payloads, configured storage dimensions and model/precision choices. Old embeddings remain truthfully unknown.
- Preserve frozen cases, observation subjects and independent-human activation gates. Missing proof remains unknown. No answer-key tuning or automatic quality-mode enablement.
- No private questions, source/user identifiers, raw corpus, credentials or resolved environments in Git.
- Every shell command begins with rtk. Agents edit assigned worktrees only, do not spawn agents, and do not access servers/providers, merge, push, deploy or send external messages.
- Root owns shared runtime tests, development maintenance, serial integration and rollout. Preserve the 18 unrelated primary drafts.

## Acceptance

New vectors and their receipts persist together; fallback providers and transformations are accurately recorded; old/copied-unknown vectors remain unknown. Outage tests demonstrate finite transport/worker attempts and later recovery without permanent-failure floods. Review export/import rejects stale or incomplete bindings and leaves existing evaluation eligibility unchanged. Independent task and whole-branch reviews, isolated PostgreSQL integration/migration checks, and exact-image development cold/graph-off/study replays gate rollout. Remaining human judgments and unreproduced production incidents are explicitly reported.

### Task 1: Durable embedding provenance

**Files:** Add aquillm/lib/embeddings/provenance.py and corresponding tests. Modify local.py, cohere.py, multimodal.py, aquillm/aquillm/utils.py, document/conversation chunk models and writers. Add nullable-field migrations documents0007 and chat0011 after their current leaves. Related tests and a concise provenance runbook are owned. Do not change quality review modules or outage retry policy in this task.

**Interfaces:** Add a typed EmbeddingResult containing vector and JSON-serializable provenance. Expose get_embedding_result(query, input_type="search_query"), get_embedding_results(queries, input_type="search_query"), and get_multimodal_embedding_result(prompt, image_data_url, input_type="search_document") through the ordinary facade. Existing get_embedding/get_embeddings/get_multimodal_embedding keep their exact return types and request semantics. New result APIs must follow the same policy, contract and fallback behavior, not duplicate providers or make an extra request.

- [ ] Write exact receipt schema and writer map before edits. Include schema version, actual provider and route, role, actual prepared-input digest, vector digest, raw/fitted dimensions/adaptation, separate declared and observed identity with unknown revision/precision retained. Store neither credentials, endpoint plaintext nor input text/images.
- [ ] Add RED provider-bound tests: reordered batch response indices bind receipts to the correct prepared input; truncation is recorded for the actual sent input; Cohere fallback is attributed to Cohere; missing observed identity stays unknown; dimensions/adaptation and vector binding are exact. A model-name echo must not become verified checkpoint/precision provenance.
- [ ] Implement response-bound results and compatibility wrappers; retain strict KG indexed APIs. Replace raw multimodal response/exception logging encountered in the touched path with fixed redacted diagnostics.
- [ ] Add nullable embedding_provenance JSON fields with no backfill. Wire document task bulk_create, duplicate copy, image route, TextChunk.save and conversation indexing to persist matching vector/receipt together. Null/externally supplied legacy vectors remain unknown; explicit fixture writes may remain unknown but must not inherit current runtime claims. Validate/clear stale receipts when a vector is replaced through supported model paths.
- [ ] Add RED/ GREEN Django behavior tests for known/unknown duplicate copying, new target image provenance, conversation persistence, unchanged old rows across migrations and receipt/vector integrity. Existing APIs and adjacent embedding/chunk/indexing behavior must pass.
- [ ] Provide a bounded read-only provenance coverage/integrity audit (counts/digests only), document unknown-history limits, commit and report exact tests/concerns. Root executes live checks.

### Task 2: Bounded outage recovery (after Task 1)

**Files:** Embedding configuration/client/multimodal transport; documents chunk_embeddings, task chunking, ChunkPublication model/service/recovery; conversation indexing/task; their tests and a recovery runbook. Start from reviewed Task1 so result APIs are not bypassed.

**Interfaces:** Preserve Task1 result APIs, typed EmbeddingContractError and transient EmbeddingUpstreamUnavailableError, publication source/generation fences and existing lease-based recovery. Add a narrowly typed permanent failure state/reset only if necessary to stop repeated redispatch of the same malformed source.

- [ ] Reproduce SDK implicit retry/timeouts, infinite chunk retry and outage batch-to-per-item amplification. Record exact current publication acknowledgment and reset behavior.
- [ ] Write a scoped subplan with explicit finite request/attempt limits before code. Use no SDK transport retries, bounded request timeout, bounded context-shrink attempts, and finite typed-transient worker retry; do not retry arbitrary programming errors. Preserve unchanged successful payloads.
- [ ] RED tests show one batch outage does not launch one call per chunk/window, worker invocation terminates, retained exact-source intent recovers when the provider returns, and stale task failure cannot block a newer source/generation.
- [ ] Implement minimal bounded transport/worker behavior. Permanent contract failures fail without provider/per-item fallback and cannot create an endless publication loop. Expose an explicit safe retry/reset mechanism; content changes can legitimately create a new intent.
- [ ] Verify conversation keyword-only availability where currently supported, terminal contract classification and finite task chain; preserve authorized search and transactional publication.
- [ ] Run covering/adjacent Linux tests, commit and report. Root performs a synthetic outage/recovery acceptance case on disposable infrastructure, not by stopping the shared model endpoint.

### Task 3: Offline human-review packets (parallel with Task 1)

**Files:** Add aquillm/apps/chat/evals/evidence_review_packet.py, run_evidence_review_packet.py, focused tests; extend docs/runbooks/evidence-preservation.md. Reuse evidence_review_subject, existing review validation/schema and rescore CLI unchanged.

**Interfaces:**
    prepare_packet(reports, cases, *, random_seed) -> (packet, binding_manifest)
    import_reviews(packet, binding_manifest, original_reports, responses) -> reviews_by_report
CLI export accepts --observations REPORT... --output-dir PRIVATE_DIR; import accepts packet/bindings/responses/original observations and output directory. Pure offline code must not initialize Django or providers.

- [ ] Write RED tests that export starts reviewer identity and every judgment null, preserves exact Unicode/question/history/answer/citations/spans, and omits explicit arm/mode/score/pass metadata from the first reading sheet.
- [ ] Implement shuffled opaque IDs, a readable sheet plus JSON template, exact full-observation audit supplement, and private binding manifest containing packet/report digests and canonical subjects. Describe arm-metadata blinding accurately; exact answers/traces can still reveal treatments.
- [ ] Inventory every original observation, including invalid/unreviewable subjects and safety/operational cases that need the existing separate workflow. Never silently count omitted records as complete or reviewed.
- [ ] RED/ GREEN import tests reject changed invocation/SDK payload/span/trace/snapshot, duplicate/unknown IDs, mismatched hashes/subjects, missing reviewer/audit attestation and partial/null judgments. Preserve explicit false values; do not manufacture human identity or judgments.
- [ ] Emit exactly the existing per-report review schema without cross-arm case-ID collisions; do not mutate observations, corpus, labels, scores or gates. Fixture observations remain ineligible after fixture responses.
- [ ] Offline CLI roundtrip tests use synthetic observations; unchanged templates must not produce completed reviews. Document the human step and rescore commands, commit and report. Root exports a private development acceptance bundle without inventing human reviews.

### Task 4: Development GPU maintenance (root)

- [ ] Verify loaded versus installed driver/module/library versions, bootable installed modules, sudo access, Docker startup/restart policies, durable mounts and running service inventory.
- [ ] Record reboot/service baselines in private evidence and use a controlled .254 reboot to load the installed matching driver. Do not alter model/container configuration or precision.
- [ ] Wait for SSH recovery, verify matching nvidia-smi/kernel library and device access, all existing containers, model health, synthetic embedding/rerank checks and public readiness. Recover only the previously running intended services if restart policy requires it.
- [ ] Capture allowlisted runtime/precision facts honestly; new hardware availability does not supply old embedding provenance or human approval.

### Task 5: Integrate and deploy (root)

- [ ] Independent reviews of each code task; fix findings through implementers, then cherry-pick sequentially.
- [ ] Isolated exact-source/image backend tests and migration lifecycle checks; preserve legacy unknowns, indexes and data. Run provider outage/recovery and review-packet acceptance fixtures.
- [ ] Whole-branch review, exact web/KG image build, canonical/cold candidate controls.
- [ ] Merge/push development and deploy only .254 with rollback images/environment and health gates. Run deployed cold6, controls14, exact study-question20 and metadata coverage checks.
- [ ] Publish an honest audit, preserve private evidence and primary drafts, clean owned test resources and archive managed worktrees.

## Dependency and scope decisions

Task1 and Task3 are independent and run in separate worktrees. Task2 waits for Task1 because they share provider/writer interfaces. Task4 is coordinator-only; runtime tests are sequential around maintenance. No stronger model, reindex, query treatment or quality-mode activation is bundled. Those require controlled comparison and actual independent judgments, for which Task3 supplies a practical workflow.

