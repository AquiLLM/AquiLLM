# ruff: noqa: E501
# fmt: off
"""Provider-neutral contracts for bounded authorized graph topology loading."""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from hashlib import sha256
from math import fsum, isclose, isfinite
from typing import Protocol, final, runtime_checkable

from apps.knowledge_graph.projection.serialization import _count, _key
from apps.knowledge_graph.projection.serialization import _token as _projection_token
from apps.knowledge_graph.retrieval.projected_types import (
    ProjectedAuthorizedGraphSnapshotV1,
    projected_snapshot_checksum,
)

PROJECTED_SEED_MASS_ABS_TOLERANCE, PROJECTED_SEED_MASS_POLICY = 1e-12, "projected-seed-mass-v1"
class HybridBranchKind(StrEnum):
    DIRECT = "direct"
    EXTENDED = "extended"
class TopologyQueryName(StrEnum):
    GENERATION_MANIFESTS = "generation_manifests"
    AUTOMATIC_MEMBERSHIPS = "automatic_memberships"
    RELATION_TOPOLOGY = "relation_topology"
    EVIDENCE_MENTIONS = "evidence_mentions"
class TopologyFailureReason(StrEnum):
    READINESS_MISMATCH = "readiness_mismatch"
    AUTHORIZATION_CONTEXT_INVALID = "authorization_context_invalid"
    BACKEND_AUTHENTICATION = "backend_authentication"
    BACKEND_UNAVAILABLE = "backend_unavailable"
    BACKEND_PROVENANCE_MISMATCH = "backend_provenance_mismatch"
    BACKEND_SCHEMA_MISMATCH = "backend_schema_mismatch"
    OVERALL_DEADLINE = "overall_deadline"
    DIRECT_TOPOLOGY_TIMEOUT = "direct_topology_timeout"
    DIRECT_TOPOLOGY_INVALID = "direct_topology_invalid"
    EXTENDED_TOPOLOGY_TIMEOUT = "extended_topology_timeout"
    EXTENDED_TOPOLOGY_INVALID = "extended_topology_invalid"
_SHARED_TOPOLOGY_FAILURES = frozenset((TopologyFailureReason.READINESS_MISMATCH, TopologyFailureReason.AUTHORIZATION_CONTEXT_INVALID, TopologyFailureReason.BACKEND_AUTHENTICATION, TopologyFailureReason.BACKEND_UNAVAILABLE, TopologyFailureReason.BACKEND_PROVENANCE_MISMATCH, TopologyFailureReason.BACKEND_SCHEMA_MISMATCH, TopologyFailureReason.OVERALL_DEADLINE))
def _token(value: object, name: str) -> None:
    _projection_token(value, name, maximum=128)
def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
@final
@dataclass(frozen=True, slots=True)
class SelectedCollectionGenerationV1:
    collection_key: str
    generation_key: str
    active_artifact_key: str
    projection_key: str
    graph_checksum: str
    schema_version: str
    projection_version: str
    identifier_key_version: str
    membership_epoch: int
    membership_checksum: str
    resolver_version: str
    resolution_config_checksum: str
    ontology_checksum: str
    embedding_model_signature: str
    def __post_init__(self) -> None:
        for name in (
            "collection_key",
            "generation_key",
            "active_artifact_key",
            "projection_key",
            "graph_checksum",
            "membership_checksum",
            "resolution_config_checksum",
            "ontology_checksum",
        ):
            _key(getattr(self, name), name)
        for name in (
            "schema_version",
            "projection_version",
            "identifier_key_version",
            "resolver_version",
            "embedding_model_signature",
        ):
            _token(getattr(self, name), name)
        _count(self.membership_epoch, "membership_epoch", 0, 2**63 - 1)
@final
@dataclass(frozen=True, slots=True)
class AuthorizedProjectedDocumentV1:
    document_key: str
    collection_key: str
    generation_key: str
    def __post_init__(self) -> None:
        _key(self.document_key, "document_key")
        _key(self.collection_key, "collection_key")
        _key(self.generation_key, "generation_key")
def ready_generation_bundle_checksum(
    selected_generations: tuple[SelectedCollectionGenerationV1, ...],
    authorized_documents: tuple[AuthorizedProjectedDocumentV1, ...],
    authorization_context_signature: str,
) -> str:
    _rows(selected_generations, SelectedCollectionGenerationV1, "selected_generations")
    _rows(authorized_documents, AuthorizedProjectedDocumentV1, "authorized_documents")
    if len(selected_generations) > 128 or len(authorized_documents) > 10_000:
        raise ValueError("ready bundle checksum input exceeds its hard cap")
    _key(authorization_context_signature, "authorization_context_signature")
    payload = {
        "authorization_context_signature": authorization_context_signature,
        "documents": [asdict(row) for row in authorized_documents],
        "selected_generations": [asdict(row) for row in selected_generations],
    }
    return sha256(_canonical(payload)).hexdigest()
@final
@dataclass(frozen=True, slots=True)
class ReadyGenerationBundleV1:
    selected_generations: tuple[SelectedCollectionGenerationV1, ...]
    authorized_documents: tuple[AuthorizedProjectedDocumentV1, ...]
    authorization_context_signature: str
    bundle_checksum: str
    def __post_init__(self) -> None:
        _rows(self.selected_generations, SelectedCollectionGenerationV1, "selected_generations")
        _rows(self.authorized_documents, AuthorizedProjectedDocumentV1, "authorized_documents")
        if not self.selected_generations:
            raise ValueError("selected_generations must not be empty")
        if len(self.selected_generations) > 128:
            raise ValueError("selected_generations exceed the hard cap")
        if len(self.authorized_documents) > 10_000:
            raise ValueError("authorized_documents exceed the hard cap")
        generation_order = tuple(
            row.collection_key for row in self.selected_generations
        )
        _ordered_unique(generation_order, "selected_generations")
        for name in ("generation_key", "projection_key", "active_artifact_key"):
            values = tuple(getattr(row, name) for row in self.selected_generations)
            if len(set(values)) != len(values):
                raise ValueError(f"selected {name} values must be unique")
        signatures = {
            (row.schema_version, row.projection_version, row.identifier_key_version, row.resolver_version, row.resolution_config_checksum)
            for row in self.selected_generations
        }
        if len(signatures) != 1:
            raise ValueError("selected generations disagree on projection/resolver signature")
        document_order = tuple(
            (row.collection_key, row.document_key) for row in self.authorized_documents
        )
        _ordered_unique(document_order, "authorized_documents")
        generations = {
            (row.collection_key, row.generation_key)
            for row in self.selected_generations
        }
        if any(
            (row.collection_key, row.generation_key) not in generations
            for row in self.authorized_documents
        ):
            raise ValueError("authorized document generation/scope closure is broken")
        if len({row.document_key for row in self.authorized_documents}) != len(
            self.authorized_documents
        ):
            raise ValueError("authorized document keys must be unique")
        _key(self.authorization_context_signature, "authorization_context_signature")
        _key(self.bundle_checksum, "bundle_checksum")
        expected = ready_generation_bundle_checksum(
            self.selected_generations,
            self.authorized_documents,
            self.authorization_context_signature,
        )
        if self.bundle_checksum != expected:
            raise ValueError("bundle_checksum does not bind the exact ready bundle")
@final
@dataclass(frozen=True, slots=True)
class ProjectedSeedV1:
    identity_key: str
    mass: float
    def __post_init__(self) -> None:
        _key(self.identity_key, "identity_key")
        if type(self.mass) is not float:
            raise TypeError("mass must be an exact float")
        if not isfinite(self.mass) or not 0.0 < self.mass <= 1.0:
            raise ValueError("mass must be finite and in (0, 1]")
def projected_seed_checksum(seeds: tuple[ProjectedSeedV1, ...]) -> str:
    _rows(seeds, ProjectedSeedV1, "seeds")
    if len(seeds) > 64:
        raise ValueError("seeds exceed the hard cap")
    return sha256(_canonical([{"identity_key": row.identity_key, "mass": row.mass.hex()} for row in seeds])).hexdigest()
def validate_projected_seed_sequence(seeds: tuple[ProjectedSeedV1, ...], *, maximum: int, expected_checksum: str) -> None:
    _count(maximum, "maximum", 1, 64)
    _rows(seeds, ProjectedSeedV1, "seeds")
    if not seeds or len(seeds) > maximum:
        raise ValueError("seeds are empty or exceed branch caps")
    _ordered_unique(tuple(row.identity_key for row in seeds), "seeds")
    if not isclose(fsum(row.mass for row in seeds), 1.0, rel_tol=0.0, abs_tol=PROJECTED_SEED_MASS_ABS_TOLERANCE):
        raise ValueError(f"seed mass must normalize to one under {PROJECTED_SEED_MASS_POLICY}")
    _key(expected_checksum, "expected_checksum")
    if expected_checksum != projected_seed_checksum(seeds):
        raise ValueError("seed checksum does not bind the exact sequence")
@final
@dataclass(frozen=True, slots=True)
class TopologyCapsV1:
    branch_kind: HybridBranchKind
    max_seeds: int
    max_depth: int
    max_nodes: int
    max_edges: int
    max_results: int
    def __post_init__(self) -> None:
        if type(self.branch_kind) is not HybridBranchKind:
            raise TypeError("branch_kind must be exact")
        for name, maximum in (
            ("max_seeds", 64),
            ("max_depth", 2),
            ("max_nodes", 200),
            ("max_edges", 1000),
            ("max_results", 20),
        ):
            _count(getattr(self, name), name, 1, maximum)
@final
@dataclass(frozen=True, slots=True)
class TopologyDeadlineV1:
    branch_kind: HybridBranchKind
    overall_deadline: float
    branch_deadline: float
    def __post_init__(self) -> None:
        if type(self.branch_kind) is not HybridBranchKind:
            raise TypeError("branch_kind must be exact")
        for name in ("overall_deadline", "branch_deadline"):
            value = getattr(self, name)
            if type(value) is not float:
                raise TypeError(f"{name} must be an exact monotonic float")
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.branch_deadline > self.overall_deadline:
            raise ValueError("branch_deadline exceeds overall_deadline")
@final
@dataclass(frozen=True, slots=True)
class ProjectedTopologyRequestV1:
    ready: ReadyGenerationBundleV1
    seeds: tuple[ProjectedSeedV1, ...]
    seed_checksum: str
    caps: TopologyCapsV1
    deadline: TopologyDeadlineV1
    def __post_init__(self) -> None:
        if type(self.ready) is not ReadyGenerationBundleV1:
            raise TypeError("ready must be exact")
        if type(self.caps) is not TopologyCapsV1 or type(self.deadline) is not TopologyDeadlineV1:
            raise TypeError("caps and deadline must be exact")
        validate_projected_seed_sequence(seeds=self.seeds, maximum=self.caps.max_seeds, expected_checksum=self.seed_checksum)
        if self.caps.branch_kind is not self.deadline.branch_kind:
            raise ValueError("branch kinds disagree across caps and deadline")
type TopologyScalar = str | int | float | bool | None
@runtime_checkable
class ProjectedTopologyQueryDriver(Protocol):
    def execute_read(self, *, query: TopologyQueryName, parameters: Mapping[str, TopologyScalar], deadline: float, max_records: int) -> tuple[Mapping[str, TopologyScalar], ...]: ...
@runtime_checkable
class ProjectedTopologyLoader(Protocol):
    def load(self, *, ready: ReadyGenerationBundleV1, seeds: tuple[ProjectedSeedV1, ...], caps: TopologyCapsV1, deadline: float) -> ProjectedAuthorizedGraphSnapshotV1: ...
@final
@dataclass(frozen=True, slots=True)
class TopologyLoadResultV1:
    branch_kind: HybridBranchKind
    ready_bundle_checksum: str
    seed_checksum: str
    snapshot_checksum: str | None
    node_count: int
    edge_count: int
    elapsed_ms: int
    snapshot: ProjectedAuthorizedGraphSnapshotV1 | None
    failure_reason: TopologyFailureReason | None
    def __post_init__(self) -> None:
        if type(self.branch_kind) is not HybridBranchKind:
            raise TypeError("branch_kind must be exact")
        _key(self.ready_bundle_checksum, "ready_bundle_checksum")
        _key(self.seed_checksum, "seed_checksum")
        _count(self.node_count, "node_count", 0, 200)
        _count(self.edge_count, "edge_count", 0, 1000)
        _count(self.elapsed_ms, "elapsed_ms", 0, 5000)
        if self.failure_reason is None:
            if type(self.snapshot) is not ProjectedAuthorizedGraphSnapshotV1:
                raise TypeError("successful result requires an exact snapshot")
            _key(self.snapshot_checksum, "snapshot_checksum")
            if self.snapshot_checksum != projected_snapshot_checksum(self.snapshot):
                raise ValueError("snapshot_checksum does not bind the snapshot")
            if (self.node_count, self.edge_count) != (
                len(self.snapshot.identity_keys),
                len(self.snapshot.relation_groups),
            ):
                raise ValueError("result counts disagree with snapshot")
        else:
            if type(self.failure_reason) is not TopologyFailureReason:
                raise TypeError("failure_reason must be exact")
            local = {TopologyFailureReason.DIRECT_TOPOLOGY_TIMEOUT, TopologyFailureReason.DIRECT_TOPOLOGY_INVALID} if self.branch_kind is HybridBranchKind.DIRECT else {TopologyFailureReason.EXTENDED_TOPOLOGY_TIMEOUT, TopologyFailureReason.EXTENDED_TOPOLOGY_INVALID}
            if self.failure_reason not in _SHARED_TOPOLOGY_FAILURES | local:
                raise ValueError("failure_reason is not compatible with branch kind")
            if self.snapshot is not None or self.snapshot_checksum is not None:
                raise ValueError("failed result must not expose a snapshot")
            if self.node_count or self.edge_count:
                raise ValueError("failed result counts must be zero")
def _rows(value: object, kind: type, name: str) -> None:
    if type(value) is not tuple or any(type(row) is not kind for row in value):
        raise TypeError(f"{name} must contain exact {kind.__name__} rows")
def _ordered_unique(keys: tuple, name: str) -> None:
    if len(set(keys)) != len(keys) or keys != tuple(sorted(keys)):
        raise ValueError(f"{name} must be unique and canonically sorted")
