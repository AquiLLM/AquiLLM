"""Immutable compatible groups retaining the original authorization signature."""

import json
from dataclasses import asdict, dataclass
from hashlib import sha256

from apps.documents.services.representation_contracts import RepresentationSnapshotV1
from apps.documents.services.representation_store import representation_snapshot_digest

from .graph_handoff import GraphCoverageV2
from .ready_scope import SelectedReadyScopeV1, _checksum, _encoded
from .topology.contracts import (
    ReadyGenerationBundleV1,
    ready_generation_bundle_checksum,
)


def _digest(payload):
    return sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def graph_authority_group_checksum_v2(
    *, scope, ontology_checksum, representation_snapshot
):
    return _digest(
        {
            "version": "graph-authority-group-v2",
            "authorization_context_signature": (
                scope.ready.authorization_context_signature
            ),
            "ontology_checksum": ontology_checksum,
            "ready_bundle_checksum": scope.ready.bundle_checksum,
            "representation_digest": representation_snapshot_digest(
                representation_snapshot
            ),
            "selected_document_ids": sorted(
                str(d) for d in scope.selected_document_ids
            ),
            "generation_keys": sorted(
                g.generation_key for g in scope.ready.selected_generations
            ),
        }
    )


def compatible_signature(authority):
    return tuple(
        getattr(authority, field)
        for field in (
            "ontology_version",
            "ontology_checksum",
            "resolver_version",
            "resolution_config_checksum",
            "embedding_model_signature",
            "schema_version",
            "projection_version",
            "identifier_key_version",
        )
    )


@dataclass(frozen=True, slots=True)
class GraphAuthorityGroupV2:
    scope: SelectedReadyScopeV1
    ontology_checksum: str
    group_checksum: str

    def __post_init__(self):
        if type(self.scope) is not SelectedReadyScopeV1 or not self.scope.projections:
            raise TypeError("scope must be an exact nonempty ready scope")
        _checksum(self.ontology_checksum, "ontology_checksum")
        _checksum(self.group_checksum, "group_checksum")
        if len({compatible_signature(p) for p in self.scope.projections}) != 1:
            raise ValueError("group contains incompatible generation signatures")
        if any(
            p.ontology_checksum != self.ontology_checksum
            for p in self.scope.projections
        ):
            raise ValueError("group ontology mismatch")
        documents = tuple(
            sorted((d for a in self.scope.projections for d, _ in a.documents), key=str)
        )
        mapping = self.scope.generation_keys_by_projection
        if (
            documents != self.scope.selected_document_ids
            or len(set(documents)) != len(documents)
            or {key for _, key in mapping}
            != {g.generation_key for g in self.scope.ready.selected_generations}
            or len(mapping) != len(self.scope.ready.selected_generations)
            or len(documents) != len(self.scope.ready.authorized_documents)
        ):
            raise ValueError("group ready bundle and authority membership differ")


@dataclass(frozen=True, slots=True)
class SelectedGraphScopeV2:
    groups: tuple[GraphAuthorityGroupV2, ...]
    authorization_context_signature: str
    representation_snapshot: RepresentationSnapshotV1
    coverage: GraphCoverageV2

    def __post_init__(self):
        if type(self.groups) is not tuple or any(
            type(g) is not GraphAuthorityGroupV2 for g in self.groups
        ):
            raise TypeError("groups must contain exact graph authority groups")
        _checksum(
            self.authorization_context_signature, "authorization_context_signature"
        )
        representation_snapshot_digest(self.representation_snapshot)
        if type(self.coverage) is not GraphCoverageV2:
            raise TypeError("coverage must be exact")
        if self.groups != tuple(sorted(self.groups, key=lambda g: g.group_checksum)):
            raise ValueError("groups must be canonically ordered")
        documents, generations = [], []
        for group in self.groups:
            scope = group.scope
            if (
                scope.ready.authorization_context_signature
                != self.authorization_context_signature
            ):
                raise ValueError("group replaced the original authorization")
            if group.group_checksum != graph_authority_group_checksum_v2(
                scope=scope,
                ontology_checksum=group.ontology_checksum,
                representation_snapshot=self.representation_snapshot,
            ):
                raise ValueError("group checksum mismatch")
            documents.extend(scope.selected_document_ids)
            generations.extend(
                g.generation_key for g in scope.ready.selected_generations
            )
        selected = {s.document.id for s in self.representation_snapshot.selections}
        if (
            len(set(documents)) != len(documents)
            or len(set(generations)) != len(generations)
            or not set(documents) <= selected
            or self.coverage.ready_documents != len(documents)
            or self.coverage.selected_documents != len(selected)
        ):
            raise ValueError("group membership or coverage mismatch")


def graph_selection_digest_v2(selected: SelectedGraphScopeV2) -> str:
    if type(selected) is not SelectedGraphScopeV2:
        raise TypeError("selection must be exact")
    return _digest(
        {
            "version": "graph-selection-v2",
            "authorization_context_signature": selected.authorization_context_signature,
            "representation_digest": representation_snapshot_digest(
                selected.representation_snapshot
            ),
            "group_checksums": [g.group_checksum for g in selected.groups],
            "coverage": asdict(selected.coverage),
        }
    )


def assemble_group_v2(*, authorities, authorization, codec, representation_snapshot):
    """Encode a subset directly; never manufacture a narrowed authorization."""
    authorities = tuple(sorted(authorities, key=lambda a: a.collection_id))
    encoded = tuple(_encoded(row, codec) for row in authorities)
    generations = tuple(sorted((r[0] for r in encoded), key=lambda g: g.collection_key))
    documents = tuple(
        sorted(
            (d for _, docs in encoded for d in docs),
            key=lambda d: (d.collection_key, d.document_key),
        )
    )
    signature = authorization.authorization_context_signature
    ready = ReadyGenerationBundleV1(
        generations,
        documents,
        signature,
        ready_generation_bundle_checksum(generations, documents, signature),
    )
    scope = SelectedReadyScopeV1(
        ready,
        authorities,
        tuple(sorted((d for a in authorities for d, _ in a.documents), key=str)),
        tuple(
            (a.projection_id, e[0].generation_key)
            for a, e in zip(authorities, encoded, strict=True)
        ),
    )
    ontology = authorities[0].ontology_checksum
    return GraphAuthorityGroupV2(
        scope,
        ontology,
        graph_authority_group_checksum_v2(
            scope=scope,
            ontology_checksum=ontology,
            representation_snapshot=representation_snapshot,
        ),
    )
