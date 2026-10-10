# Retrieval follow-through design

The user authorized implementing the remaining development gaps and deploying them. This continues the accepted retrieval gap program with four concrete deliverables: truthful provenance for newly produced vectors, bounded embedding outage recovery, offline independent-human review packets, and correction of the confirmed development GPU driver mismatch.

## Design choices

Persist nullable provenance alongside document and conversation chunk vectors. Capture it at the provider response boundary, before response metadata is discarded; retain existing vector-only APIs as compatibility wrappers. Distinguish declared configuration from observed response identity. No metadata is inferred for historical vectors, no reindex is performed, and no retrieval filtering changes on this release. Copied vectors carry their original provenance, including unknown.

Bound embedding transport and worker retry attempts, stop batch outages from multiplying into one request per chunk, and retain the existing durable publication/recovery mechanism. Permanent malformed contracts need an exact-source terminal state, with an explicit reset path; transient failures remain recoverable. Preserve existing keyword availability and source-generation fencing.

Export arm-metadata-blinded quality review sheets and an exact audit supplement from immutable saved observations. Bind responses to original subjects and report digests, start every judgment unknown, and import only explicitly completed named-human reviews into the existing schema. Safety/operational records are inventoried separately; fixture data cannot gain live eligibility. This does not supply human judgments or activate new evidence modes.

The root coordinator verifies installed driver and reboot readiness, records the running development services, performs a controlled development reboot, and verifies GPU/model/application recovery. No model, precision or index change is part of that maintenance.

## Binding constraints

- Development 149.165.150.254 only; production 149.165.169.204 excluded.
- Preserve extractor 3000 ms, direct/extended 4500 ms, overall 5000 ms, authorization, graph provenance, resource caps and zero-retry graph semantics.
- Preserve embedding wire payloads, configured storage dimensions and model/precision choices. Old embeddings remain truthfully unknown.
- Preserve frozen cases, observation subjects and independent-human activation gates. Missing proof remains unknown. No answer-key tuning or automatic quality-mode enablement.
- No private questions, source/user identifiers, raw corpus, credentials or resolved environments in Git.
- Every shell command begins with rtk. Agents edit assigned worktrees only, do not spawn agents, and do not access servers/providers, merge, push, deploy or send external messages.
- Root owns shared runtime tests, development maintenance, serial integration and rollout. Preserve the 18 unrelated primary drafts.

## Acceptance

New vectors and their receipts persist together; fallback providers and transformations are accurately recorded; old/copied-unknown vectors remain unknown. Outage tests demonstrate finite transport/worker attempts and later recovery without permanent-failure floods. Review export/import rejects stale or incomplete bindings and leaves existing evaluation eligibility unchanged. Independent task and whole-branch reviews, isolated PostgreSQL integration/migration checks, and exact-image development cold/graph-off/study replays gate rollout. Remaining human judgments and unreproduced production incidents are explicitly reported.

