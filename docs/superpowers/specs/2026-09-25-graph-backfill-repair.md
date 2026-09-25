# Graph backfill repair design

The first production backfill exposed four independent limits and an orchestration
error. Extraction admitted documents up to 10 million characters, but resolution
rejected aggregate source contexts above 2 million. One collection had 89,052
document entities against a 50,000-entity context cap. The global canonical rebuild
had a 10,000-entity cap despite 92,140 active collection entities. Finally, an
extraction worker permitted to publish activation hooks skipped canonical membership
hooks because the latter required the database reader's configuration gate.

Collection context exceptions were all treated as changed source snapshots. A
permanent capacity failure therefore created successors that re-extracted the same
80 documents. Inspection based on original child requests hid this new work.

## Required behavior

- Extraction and resolution share the finite 10,000,000-character admission bound.
  All source text, coordinates, mentions and provenance remain intact.
- Default collection configs remain at 50,000 entities. Automatically choose a
  100,000-entity bucket above that threshold, with matching orphan capacity. Respect
  explicit config overrides and retain independent document/link/evidence limits.
- Only actual stale source failures trigger resnapshotting. Capacity, corruption
  and unexpected preflight errors terminate with privacy-safe fixed error codes.
- Canonical rebuilds admit 250,000 entities, 1,000,000 source links and 2,000,000
  provenance rows, with a finite 5,000,000-decision budget. Authorization/read limits
  do not change. Preserve candidate/audit ordering and existing checksum semantics.
- Avoid retaining every collection candidate pool simultaneously. Index canonical
  edges by component and stream checksum serialization to reduce memory overhead.
- Membership hooks use the same explicit hook gate as activation hooks. Workers
  need no graph database read credentials to publish either hook.
- Add successor-aware aggregate progress without rewriting immutable request audit
  fields. Distinguish active artifacts from retained historical activations.

## Rollout constraints

Deploy and validate only on development (149.165.150.254). Main and production
(149.165.169.204) require a separate explicit approval after development validation.
Preserve deployment overrides, secrets,
graph identities for unchanged small inputs, and all completed graph artifacts.
Repair only the three failed documents and incomplete collections, reusing valid
extraction checkpoints where possible. Do not submit another operator-wide rebuild.
Reconcile canonical memberships and projections after current builds finish.
Historical failed request records remain an accurate audit, even after repairs.
Public health, worker health, projection consistency and retrieval must pass before
declaring the repair complete. Temporary SSH access stays until explicitly removed
at the user's request.
