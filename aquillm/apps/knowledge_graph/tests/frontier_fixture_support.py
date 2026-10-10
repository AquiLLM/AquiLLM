"""Synthetic two-generation bridge with original A-only seeds."""

from dataclasses import fields, replace
from hashlib import sha256
from time import monotonic
from uuid import UUID

from apps.documents.services.representation_contracts import (
    DocumentRefV1,
    RepresentationSelectionV1,
    RepresentationSnapshotV1,
)
from apps.knowledge_graph.projection.identifiers import (
    HmacSha256ProjectionIdentifierCodec,
)
from apps.knowledge_graph.retrieval.graph_handoff import GraphCoverageV2
from apps.knowledge_graph.retrieval.graph_query_contracts import GraphRelationRequestV2
from apps.knowledge_graph.retrieval.ready_scope import SelectedReadyScopeV1, _encoded
from apps.knowledge_graph.retrieval.ready_scope_v2 import (
    GraphAuthorityGroupV2,
    SelectedGraphScopeV2,
    graph_authority_group_checksum_v2,
)
from apps.knowledge_graph.retrieval.topology.contracts import (
    ProjectedSeedV1,
    ReadyGenerationBundleV1,
    ready_generation_bundle_checksum,
)
from apps.knowledge_graph.tests.test_projected_topology_adapter import (
    _FAMILIES,
    _caps,
    _ready,
)
from apps.knowledge_graph.tests.test_projection_records import _bundle
from apps.knowledge_graph.tests.test_ready_scope import _authority

from .frontier_legacy_fixture import (
    LegacyNeighborhoodDriver as LegacyNeighborhoodDriver,
)


def key(text):
    return sha256(text.encode()).hexdigest()


def bridge_bundles():
    first = _bundle()

    def rekey(row):
        return replace(
            row,
            **{
                f.name: key("bridge-b:" + value)
                for f in fields(row)
                if f.name.endswith("_key")
                and (value := getattr(row, f.name)) is not None
            },
        )

    families = {}
    for _label, family, _identity in _FAMILIES:
        families[family] = tuple(rekey(row) for row in getattr(first, family))
    shared = first.automatic_memberships[1].automatic_membership_key
    memberships = list(families["automatic_memberships"])
    memberships[0] = replace(memberships[0], automatic_membership_key=shared)
    memberships[1] = replace(memberships[1], automatic_membership_key=key("answer-c"))
    families["automatic_memberships"] = tuple(memberships)
    orders = {
        "entities": lambda r: r.entity_key,
        "automatic_memberships": lambda r: r.entity_key,
        "documents": lambda r: r.document_key,
        "chunks": lambda r: (r.document_key, r.chunk_number, r.chunk_key),
        "relation_semantics": lambda r: (
            r.artifact_key,
            r.relation_type,
            r.semantics_key,
        ),
        "relations": lambda r: r.relation_key,
        "evidence": lambda r: r.evidence_key,
        "entity_mentions": lambda r: (r.entity_key, r.provenance_key, r.mention_key),
        "artifact_provenance": lambda r: (r.scope_type, r.scope_key, r.artifact_key),
    }
    second = replace(
        first,
        generation=rekey(first.generation),
        **{
            name: tuple(sorted(rows, key=orders[name]))
            for name, rows in families.items()
        },
    )
    return first, second


class FrontierFixture:
    def __init__(self, bundles=None):
        self.codec = HmacSha256ProjectionIdentifierCodec(
            b"synthetic-frontier-test", key_version="key-v1"
        )
        self.bundles = tuple(
            self._encoded_bundle(b, i)
            for i, b in enumerate(bundles or bridge_bundles(), 1)
        )
        self.deadline = monotonic() + 15.0
        self.shared_identity = (
            _bundle().automatic_memberships[1].automatic_membership_key
        )
        self.answer_identity = key("answer-c")
        self.generation_b = self.bundles[1].generation.generation_key
        parts = tuple(_ready(b) for b in self.bundles)
        generations = tuple(
            sorted(
                (p.selected_generations[0] for p in parts),
                key=lambda r: r.collection_key,
            )
        )
        documents = tuple(
            sorted(
                (p.authorized_documents[0] for p in parts),
                key=lambda r: (r.collection_key, r.document_key),
            )
        )
        signature = parts[0].authorization_context_signature
        ready = ReadyGenerationBundleV1(
            generations,
            documents,
            signature,
            ready_generation_bundle_checksum(generations, documents, signature),
        )
        authorities, mapping, selections = [], [], []
        for index, bundle in enumerate(self.bundles, 1):
            selected = _ready(bundle).selected_generations[0]
            did = UUID(int=index)
            authority = _authority(index, did, str(index))
            authority = replace(
                authority,
                **{
                    name: getattr(selected, name)
                    for name in (
                        "schema_version",
                        "projection_version",
                        "identifier_key_version",
                        "membership_epoch",
                        "membership_checksum",
                        "graph_checksum",
                        "resolver_version",
                        "resolution_config_checksum",
                        "ontology_checksum",
                        "embedding_model_signature",
                    )
                },
            )
            authorities.append(authority)
            mapping.append((authority.projection_id, selected.generation_key))
            selections.append(
                RepresentationSelectionV1(
                    DocumentRefV1(
                        "apps_documents.rawtextdocument",
                        index,
                        did,
                        index,
                        key(str(index)),
                    ),
                    None,
                    key(str(index)),
                    None,
                )
            )
        scope = SelectedReadyScopeV1(
            ready,
            tuple(authorities),
            tuple(s.document.id for s in selections),
            tuple(mapping),
        )
        snapshot = RepresentationSnapshotV1(tuple(selections))
        ontology = generations[0].ontology_checksum
        self.group = GraphAuthorityGroupV2(
            scope,
            ontology,
            graph_authority_group_checksum_v2(
                scope=scope,
                ontology_checksum=ontology,
                representation_snapshot=snapshot,
            ),
        )
        self.selection = SelectedGraphScopeV2(
            (self.group,),
            signature,
            snapshot,
            GraphCoverageV2(2, 2, "complete", (), False),
        )
        self.seeds = (
            ProjectedSeedV1(
                next(
                    m.entity_key
                    for m in self.bundles[0].automatic_memberships
                    if m.automatic_membership_key is None
                ),
                1.0,
            ),
        )
        self.request = GraphRelationRequestV2(
            ontology, (("knows", 1.0),), "relation-request-v2", key("query")
        )
        self.caps = _caps()

    def _encoded_bundle(self, bundle, index):
        authority = _authority(index, UUID(int=index), str(index))
        generation, documents = _encoded(authority, self.codec)
        mapping = {
            bundle.generation.generation_key: generation.generation_key,
            bundle.generation.collection_key: generation.collection_key,
            bundle.generation.artifact_key: generation.active_artifact_key,
            bundle.generation.projection_key: generation.projection_key,
            bundle.documents[0].document_key: documents[0].document_key,
        }

        def convert(row):
            return replace(
                row,
                **{
                    f.name: mapping[value]
                    for f in fields(row)
                    if f.name.endswith("_key")
                    and (value := getattr(row, f.name)) in mapping
                },
            )

        return replace(
            bundle,
            generation=convert(bundle.generation),
            **{
                family: tuple(convert(row) for row in getattr(bundle, family))
                for _, family, _ in _FAMILIES
            },
        )

    def budget(self, **limits):
        from apps.knowledge_graph.retrieval.topology.frontier_contracts_v2 import (
            FrontierBudgetV2,
        )
        from lib.retrieval.turn_budget import TurnBudget, TurnLimits

        return FrontierBudgetV2(
            turn_budget=TurnBudget(TurnLimits()),
            deadline=self.deadline,
            clock=monotonic,
            **{
                **dict(
                    max_identities=200,
                    max_edges=1000,
                    max_source_rows=5000,
                    max_queries=64,
                    max_wire_bytes=1048576,
                ),
                **limits,
            },
        )
