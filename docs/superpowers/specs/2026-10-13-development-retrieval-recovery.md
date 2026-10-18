# Development retrieval recovery design

User authorization: fix the chat431 error, deploy the remediation properly on development254, and make meaningful old-versus-new testing possible (2026-10-13). The preceding investigation established the defects; this is execution of those repairs.

Repair the existing retrieval lifetime rather than increasing its 15000ms ceiling. An expired direct request must finish with an honest limited answer, never spin into a zero-budget retrieval tool. A tool that never started must not be described as an uncertain side effect. User-facing tool termination must bypass evidence synthesis; actually started uncertain operations retain durable replay protection.

Restore the retrieval replay against the real current query-proposal interfaces, cover it in release CI, and make baseline versus explicit development treatments observable. Keep generation and chat persistence prohibited for retrieval-only runs.

Reduce the measured lexical/authorization cost through evidence-led, semantics-preserving work. Preserve current principal, selected scope, source-generation, permission revocation and deadline validation boundaries.

Independently verify the deployed embedding model/tokenizer/precision and serving artifact, register its truthful immutable space, rebuild compatible document sidecar generations, and configure all relevant application processes consistently. Never relabel historical vectors from current configuration. Preserve data and rollback artifacts.

Integrate reviewed fixes in a PR against development and deploy254 under the existing user authorization. Verify an actual chat request, dense candidates and citations, worker outcomes, runtime modes and matched retrieval measurements. New treatments may be exercised on this development test server; experimental results do not imply production activation or held-out acceptance.

Use pre-PR244 development dae54507c6e55f3cf1fb0a0974d4f22ebe2cdb8f as the primary old-code baseline where a safe matched replay is feasible; retain759ef8d historical study outputs separately. Freeze query/corpus scope, label feature modes and budgets, preserve failed cases, and distinguish known-question regression from independent holdout acceptance.

## Native frontier source closure correction

The b09e development trace reaches frontier initialization after spending 3879.567ms of its 4710ms graph allowance, then expires without publishing graph evidence. The request currently loads provenance for all 530 selected documents before traversing any edge. Correct this architecture through a distinct native acquisition version, keeping existing legacy/default behavior.

The full requested authorization, selected ready group, allowed document and collection scope, and group checksum remain unchanged. Freshly attest every selected generation initially and finally and authenticate every selected collection's provenance. Authenticate and validate document, chunk, membership, relation, evidence and mention proofs when consumed, before they affect admitted traversal, ranking or retained evidence. Keep exact captured representation binding and fresh full requested permission, source/body and cleanup checks at publication.

For this new version, request success establishes complete proof for the consumed graph and evidence, not raw-record completeness of every unused authorized graph document. Corruption confined to unused graph source records no longer rejects a query solely because those records belong to the selected scope; the identical defect must reject when consumed. Background generation certification retains responsibility for complete stored graph integrity. This explicitly supersedes Task8's eager full-document raw provenance requirement only for the new native version. Existing V1/canonical-frontier-v2 behavior remains unchanged.

Bind this distinction into the exact algorithm tags, acquisition policy, request/session and whole-runtime signatures. Preserve all deadlines, original 15000ms turn ceiling, query/row/byte limits, all-generation copy completeness, final reservations and uncertain-work accounting. No reusable authority cache, generic validation bypass, new gateway protocol or numerical PPR change. Require focused proof/compatibility tests and the existing full-scope physical fixture before one isolated native feasibility run. Independent review, broader regressions, matched comparisons, real ASGI and deployment acceptance remain separate gates.
