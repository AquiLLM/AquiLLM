# GCP Secret Manager Runbook

Status: Task21 secret contract; cloud-provider integration remains deployment-owned
Created: 2026-03-25

This document defines the secret names and handling rules needed by the
provider-neutral Task21 Memgraph hybrid rollout. It does not add a GCP-specific
application integration: the deployment owner maps these names from the
approved secret manager into the reviewed cloud environment without printing
values.

## Secret inventory

Store these values only in the approved secret manager. Inject them into the
matching service at runtime; never commit them to `.env` files, Compose
metadata, command arguments, logs, diagnostics, or Task21 evidence.

| Name | Consumer | Rotation/handling rule |
| --- | --- | --- |
| `KG_MEMGRAPH_QUERY_PASSWORD` | KG retrieval reader | Read-only traversal credential; rotate independently from the writer. |
| `KG_MEMGRAPH_PROJECTION_PASSWORD` | KG projection worker | Projection writer credential; never expose to web readers. |
| `KG_PROJECTION_POSTGRES_SOURCE_DSN` | `projection_source` | Must use exact role `aquillm_projection_source` and the authoritative PostgreSQL database. |
| `KG_PROJECTION_POSTGRES_STATE_DSN` | `projection_state` | Must use exact role `aquillm_projection_state` on that same authority; function-only lease/CAS/outbox access. |
| `KG_PROJECTION_IDENTIFIER_HMAC_KEY` | Opaque projection codec | Rotate only with `KG_PROJECTION_IDENTIFIER_KEY_VERSION` and a complete projection rebuild. |
| `KG_QUERY_EXTRACTOR_BEARER_TOKEN` | Query extractor | Keep out of evaluator output and rotate on provider or incident response. |
| `TASK21_EVIDENCE_SIGNING_KEY` | Cloud evidence publisher | Sign `task21-hybrid-cloud-evidence-v1`; retain only in the protected operator secret store. |

These are identity/configuration values rather than secrets and must still be
pinned in the reviewed environment: `KG_QUERY_EXTRACTOR_BUILD_HASH`, the
extractor model/revision, the Memgraph image digest, and
`KG_PROJECTION_IDENTIFIER_KEY_VERSION`. The fixed PostgreSQL role names are
not operator-selectable.

## Handling and rotation

Use the secret manager's versioning and access audit. Grant read access only to
the named service identity and the rollout operator. Test retrieval with a
redacted health check; do not use `echo`, shell tracing, or an inspection
command that includes environment values. Evidence may record secret names,
versions, and hashes of non-secret configuration, never secret values.

For a projection HMAC rotation:

1. Set `KG_OVERLAY_ENABLED=0` and `KG_BUILD_ENABLED=0`; restart readers and
   publishers.
2. Create a new secret version and new
   `KG_PROJECTION_IDENTIFIER_KEY_VERSION` as one change. Keep the previous
   version available for the bounded transition, but do not use it to create
   new IDs.
3. Rebuild, project, reconcile, and inspect every selected collection. A
   rotated key without a complete rebuild is invalid.
4. Run the provider-neutral cloud evaluator and obtain fresh approval before
   re-enabling builds/retrieval. Retire the old secret version only after no
   active or resumable projection references it.

For DSN, Memgraph, extractor, or signing-key rotation, keep the two feature
flags off, restart only the affected service, run its bounded health/provenance
check, and repeat the required cloud shadow/parity evidence before staged
enablement. Do not change `KG_QUERY_EXTRACTOR_BUILD_HASH` to bypass a mismatch.

## Evidence permissions and incident response

The Task21 bundle directory and every member, including `bundle.json`, must be
operator-only with mode `0600` (or the platform-equivalent ACL). The signing
key must be available only to the evidence publisher. Capture state and
redacted logs before teardown, fsync and atomically publish without overwrite,
then print only the final path/size/SHA-256 line.

If a secret may have leaked, disable graph retrieval and builds, revoke the
affected secret version, issue a new version, restart the affected service,
rebuild any projections whose opaque-key or provenance identity changed, and
repeat cloud parity approval. Do not delete durable graph state or patch
database rows by hand while investigating.
