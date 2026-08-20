"""Private PostgreSQL parity topology loader; never a production fallback."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..projected_types import ProjectedAuthorizedGraphSnapshotV1
from .contracts import (
    HybridBranchKind,
    ProjectedSeedV1,
    ReadyGenerationBundleV1,
    TopologyCapsV1,
    TopologyFailureReason,
)
from .failures import TopologyLoadError


@dataclass(frozen=True, slots=True, repr=False)
class PostgresParityCapability:
    _source: object = field(repr=False)

    def __repr__(self) -> str:
        return "<PostgresParityCapability redacted>"


def _make_test_postgres_parity_capability(*, source: object):
    if not callable(getattr(source, "load", None)):
        raise TypeError("source must expose the private parity load seam")
    return PostgresParityCapability(source)


class PostgresProjectedTopologyLoader:
    def __init__(self, source: object, capability: PostgresParityCapability):
        if (
            type(capability) is not PostgresParityCapability
            or capability._source is not source
        ):
            raise ValueError("postgres parity requires the exact private capability")
        if not callable(getattr(source, "load", None)):
            raise TypeError("source must expose the private parity load seam")
        self.source = source
        self._capability = capability

    def load(
        self,
        *,
        capability: PostgresParityCapability,
        ready: ReadyGenerationBundleV1,
        seeds: tuple[ProjectedSeedV1, ...],
        caps: TopologyCapsV1,
        deadline: float,
    ) -> ProjectedAuthorizedGraphSnapshotV1:
        if capability is not self._capability:
            raise ValueError("postgres parity requires the exact private capability")
        invalid = (
            TopologyFailureReason.DIRECT_TOPOLOGY_INVALID
            if caps.branch_kind is HybridBranchKind.DIRECT
            else TopologyFailureReason.EXTENDED_TOPOLOGY_INVALID
        )
        timeout = (
            TopologyFailureReason.DIRECT_TOPOLOGY_TIMEOUT
            if caps.branch_kind is HybridBranchKind.DIRECT
            else TopologyFailureReason.EXTENDED_TOPOLOGY_TIMEOUT
        )
        try:
            snapshot = self.source.load(
                ready=ready, seeds=seeds, caps=caps, deadline=deadline
            )
        except TopologyLoadError:
            raise
        except TimeoutError as error:
            raise TopologyLoadError(timeout) from error
        except Exception as error:
            raise TopologyLoadError(
                TopologyFailureReason.BACKEND_UNAVAILABLE
            ) from error
        if type(snapshot) is not ProjectedAuthorizedGraphSnapshotV1:
            raise TopologyLoadError(invalid)
        documents = tuple(
            sorted(row.document_key for row in ready.authorized_documents)
        )
        collections = tuple(row.collection_key for row in ready.selected_generations)
        if (
            snapshot.allowed_scope.document_keys != documents
            or snapshot.allowed_scope.collection_keys != collections
            or len(snapshot.identity_keys) > caps.max_nodes
            or len(snapshot.relation_groups) > caps.max_edges
            or snapshot.load_max_hops > caps.max_depth
        ):
            raise TopologyLoadError(invalid)
        return snapshot


__all__ = ["PostgresParityCapability", "PostgresProjectedTopologyLoader"]
