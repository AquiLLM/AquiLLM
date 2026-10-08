# Development retrieval remediation design

Approved scope: implement and test the remaining internal-repository gaps and deploy verified fixes to development (149.165.150.254). This follows the five priorities discussed with the user: truthful graph diagnostics, passage recall through final evidence, reranking/truncation, graph usefulness/reliability, and storage retention.

## Design
Use the existing production retrieval and authorization paths. Record a reproducible baseline on the unchanged ten study questions, separately for the parent collection and parent plus figures. Match development documents/chunks by source content; production numeric IDs are not portable. The internal gold labels are provisional diagnostics, not an independent quality gate. Separate unanswerable Q2 from missing evidence.

Correct concrete operational defects first. Whole-branch deadlines must not falsely identify an extractor failure; stage metrics must be bounded and must not expose prompts or source text. Compare one retrieval variable at a time in a separate process, with generation blocked for retrieval-only runs. Follow evidence to the actual generator packet, including whether the supporting text survives truncation.

Use existing authorization regression tests and retrieval/evidence evaluation suites. Keep experimental preservation, iterative, and windowed modes disabled for ordinary users until their established activation requirements are satisfied. Record unmet human-review/attestation requirements explicitly; never manufacture gate evidence.

Enable conservative, opt-in, bounded graph retention scheduling only after inspecting a dry run on development. Preserve active artifacts and existing retention settings. Do not remove historical failed projections through ad hoc SQL or reclaim model volumes.

## Scope boundaries
The current release addresses confirmed defects and adds reproducible development testing. New vocabulary administration, parser replacement/re-ingestion, a different GPU model, and production rollout are separate follow-ups unless evidence demonstrates a small necessary correction. Do not hard-code study answers or query-specific synonyms. An LLM rewrite or model-input change must be evaluated for factual drift and cache/embedding compatibility before activation.

## Acceptance
- Exact development commit and relevant running images verified.
- Tests demonstrate correct timeout attribution and unchanged retrieval permissions.
- Ten-question measurements include stage membership, graph outcomes, evidence packet, and latency; failures are explicit.
- Root-cause fixes carry reproductions and regression tests.
- Storage dry run and retention policy recorded; scheduling is explicit and bounded.
- Deployment has rollback revision/images and post-deployment health/retrieval smoke results.
