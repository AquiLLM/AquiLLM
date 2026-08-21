"""Pure scoring and envelope helpers for the production hybrid runtime."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from apps.knowledge_graph.retrieval.branch_contracts import (
    BranchEnvelopeV1,
    BranchProvenanceV1,
    BranchSafeDiagnosticsV1,
    BranchStatusV1,
    DirectBranchResultV1,
    ExtendedBranchResultV1,
    GraphBranchCandidateV1,
    branch_candidate_order_checksum,
)
from apps.knowledge_graph.retrieval.ppr import PPRAlgorithmConfig
from apps.knowledge_graph.retrieval.projected_types import (
    ProjectedAuthorizedGraphSnapshotV1,
    projected_snapshot_checksum,
)
from apps.knowledge_graph.retrieval.ready_scope import SelectedReadyScopeV1
from apps.knowledge_graph.retrieval.topology.contracts import (
    HybridBranchKind,
    TopologyCapsV1,
    projected_seed_checksum,
)


@dataclass(frozen=True, slots=True)
class ProductionSharedScopeV1:
    scope: SelectedReadyScopeV1


def topology_caps(settings, kind: HybridBranchKind) -> TopologyCapsV1:
    prefix = "graph_direct" if kind is HybridBranchKind.DIRECT else "graph_extended"
    return TopologyCapsV1(
        kind,
        getattr(settings, f"{prefix}_max_seeds"),
        getattr(settings, f"{prefix}_max_depth"),
        getattr(settings, f"{prefix}_max_nodes"),
        getattr(settings, f"{prefix}_max_edges"),
        getattr(settings, f"{prefix}_max_candidates"),
    )


def ppr_config(snapshot: ProjectedAuthorizedGraphSnapshotV1, maximum: int):
    caps = snapshot.caps
    resolver = next(
        row.resolver_version
        for row in snapshot.artifact_provenance
        if row.scope_type.value == "collection"
    )
    return PPRAlgorithmConfig(
        canonical_resolver_version=resolver,
        max_seeds=caps.max_seeds,
        max_scope_documents=caps.max_scope_documents,
        max_scope_collections=caps.max_scope_collections,
        max_hops=caps.max_hops,
        max_nodes=caps.max_nodes,
        max_edges=caps.max_edges,
        max_evidence_rows=caps.max_evidence_rows,
        max_evidence_per_edge=caps.max_evidence_per_edge,
        max_mentions_per_entity=caps.max_mentions_per_entity,
        max_candidates=maximum,
        max_per_document=min(3, maximum),
    )


def graph_candidates(
    *, snapshot: ProjectedAuthorizedGraphSnapshotV1, identity_scores, maximum: int
) -> tuple[GraphBranchCandidateV1, ...]:
    """Overlay PPR identity mass on bounded associated chunk evidence."""

    if type(snapshot) is not ProjectedAuthorizedGraphSnapshotV1:
        raise TypeError("snapshot must be exact")
    if type(maximum) is not int or not 1 <= maximum <= 20:
        raise ValueError("maximum must be in 1..20")
    scores = dict(identity_scores)
    chunks: dict[str, tuple[float, str]] = {}

    def admit(identity_score: float, evidence) -> None:
        score = identity_score * evidence.confidence
        if score <= 0.0:
            return
        candidate = (score, evidence.document_key)
        current = chunks.get(evidence.chunk_key)
        if current is None or candidate > current:
            chunks[evidence.chunk_key] = candidate

    for mention in snapshot.mentions:
        admit(scores.get(mention.identity_key, 0.0), mention.evidence)
    for group in snapshot.relation_groups:
        score = max(
            scores.get(group.source_identity_key, 0.0),
            scores.get(group.target_identity_key, 0.0),
        )
        for evidence in group.evidence:
            admit(score, evidence)
    ordered = sorted(chunks.items(), key=lambda row: (-row[1][0], row[0]))
    retained, per_document = [], defaultdict(int)
    for chunk_key, (score, document_key) in ordered:
        if per_document[document_key] >= 3:
            continue
        retained.append((chunk_key, score))
        per_document[document_key] += 1
        if len(retained) == maximum:
            break
    return tuple(
        GraphBranchCandidateV1(key, rank, float(score))
        for rank, (key, score) in enumerate(retained, start=1)
    )


def ppr_failure_envelope(
    kind, reason, *, seed_count, snapshot, elapsed_ms
) -> BranchEnvelopeV1:
    """Return a valid fixed failure without exposing graph identifiers."""

    return BranchEnvelopeV1(
        kind,
        BranchStatusV1.FAILED,
        None,
        reason,
        BranchSafeDiagnosticsV1(
            seed_count,
            len(snapshot.identity_keys),
            len(snapshot.relation_groups),
            0,
            elapsed_ms,
        ),
    )


def success_envelope(kind, *, ready, seeds, snapshot, candidates, settings, elapsed_ms):
    maximum_ms = (
        settings.graph_direct_timeout_ms
        if kind is HybridBranchKind.DIRECT
        else settings.graph_extended_timeout_ms
    )
    provenance = BranchProvenanceV1(
        kind,
        ready.bundle_checksum,
        projected_seed_checksum(seeds),
        len(seeds),
        projected_snapshot_checksum(snapshot),
        snapshot.algorithm.algorithm_signature,
        len(candidates),
        branch_candidate_order_checksum(candidates),
        maximum_ms,
        min(maximum_ms, elapsed_ms),
    )
    result_type = (
        DirectBranchResultV1
        if kind is HybridBranchKind.DIRECT
        else ExtendedBranchResultV1
    )
    return BranchEnvelopeV1(
        kind,
        BranchStatusV1.SUCCEEDED,
        result_type(candidates, provenance),
        None,
        BranchSafeDiagnosticsV1(
            len(seeds),
            len(snapshot.identity_keys),
            len(snapshot.relation_groups),
            len(candidates),
            provenance.elapsed_ms,
        ),
    )


__all__ = [
    "ProductionSharedScopeV1",
    "graph_candidates",
    "ppr_failure_envelope",
    "ppr_config",
    "success_envelope",
    "topology_caps",
]
