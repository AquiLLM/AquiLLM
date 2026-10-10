"""Actual frontier assembly over validated synthetic source pages."""

from dataclasses import replace
from types import SimpleNamespace

from apps.knowledge_graph.projection.frontier_adapter_v2 import page_from_rows
from apps.knowledge_graph.projection.frontier_index_build_v2 import quality_key
from apps.knowledge_graph.projection.frontier_snapshot_v2 import FrontierAssemblyV2
from apps.knowledge_graph.projection.frontier_validation_v2 import (
    validate_relation_entry,
    validate_source_page,
)
from apps.knowledge_graph.retrieval.frontier_ranking_validation_v2 import (
    FrontierRankingContextV2,
)
from apps.knowledge_graph.retrieval.graph_evidence_selection_v2 import (
    GraphEvidenceSelectionEnvelopeV2,
    GraphEvidenceSelectionInputV2,
    select_graph_evidence_v2,
)
from apps.knowledge_graph.retrieval.graph_handoff import GraphCoverageV2
from apps.knowledge_graph.retrieval.graph_query_contracts import (
    BoundedGraphTopologyV2,
    GraphGroupRelationRequestV2,
)
from apps.knowledge_graph.retrieval.relation_request import parse_relation_request_v2
from apps.knowledge_graph.tests.frontier_fixture_support import key
from apps.knowledge_graph.tests.test_projected_topology_adapter import _caps, _ready
from apps.knowledge_graph.tests.test_projection_records import _bundle


class AssociationFixture:
    def __init__(self, labels, query="Show conflicting Atlas accuracy evidence"):
        self.bundle, self.query, self.texts = _bundle(), query, {}
        self.pages = []
        b = self.bundle
        self.documents = {
            label: replace(b.documents[0], document_key=key(label + "doc"))
            for label in labels
        }
        self.sources = {
            label: replace(
                b.artifact_provenance[1],
                scope_key=self.documents[label].document_key,
                artifact_key=key(label + "artifact"),
            )
            for label in labels
        }
        self.generation = b.generation.generation_key
        self.request = parse_relation_request_v2(
            query,
            ontology_checksum=b.artifact_provenance[0].ontology_checksum,
            allowed_relations=frozenset({"supports", "contradicts", "knows"}),
        )

    def representation(self, generation, document):
        assert generation == self.generation
        assert document in {r.document_key for r in self.documents.values()}
        return None

    def base_rows(self, label, names):
        b = self.bundle
        chunk = replace(
            b.chunks[0],
            chunk_key=key(label),
            document_key=self.documents[label].document_key,
        )
        rows = {
            ("ProjectedDocument", self.documents[label].document_key): self.documents[
                label
            ],
            ("ProjectedChunk", chunk.chunk_key): chunk,
        }
        for r in (b.artifact_provenance[0], self.sources[label]):
            rows["ArtifactProvenance", r.scope_key] = r
        for name in names:
            entity = replace(b.entities[0], entity_key=key(name))
            member = replace(b.automatic_memberships[0], entity_key=key(name))
            rows["ProjectedEntity", entity.entity_key] = entity
            rows["AutomaticMembership", member.entity_key] = member
        return rows, chunk

    def add(
        self,
        label,
        *,
        relations=(),
        mentions=(),
        text="Atlas accuracy",
        reverse=False,
        direction="directed",
    ):
        self.texts[key(label)] = (text, self.documents[label].document_key)
        b = self.bundle
        for source, predicate, target in relations:
            rows, chunk = self.base_rows(label, (source, target))
            relation = replace(
                b.relations[0],
                relation_key=key(source + predicate + target),
                source_entity_key=key(source),
                target_entity_key=key(target),
                relation_type=predicate,
            )
            semantic = replace(
                b.relation_semantics[0],
                semantics_key=key(predicate),
                relation_type=predicate,
                direction=direction,
            )
            evidence = replace(
                b.evidence[0],
                evidence_key=key(label + source + predicate + target),
                relation_key=relation.relation_key,
                relation_type=predicate,
                chunk_key=chunk.chunk_key,
                document_key=chunk.document_key,
                source_document_key=chunk.document_key,
                artifact_key=self.sources[label].artifact_key,
                head_mapping_key=key(source),
                tail_mapping_key=key(target),
            )
            rows.update(
                {
                    ("ProjectedRelation", relation.relation_key): relation,
                    ("ProjectedRelationSemantics", semantic.semantics_key): semantic,
                    ("ProjectedEvidence", evidence.evidence_key): evidence,
                }
            )
            page = page_from_rows(rows)
            origin, destination = (target, source) if reverse else (source, target)
            traversal = (
                "undirected"
                if direction == "undirected"
                else "reverse_directed"
                if reverse
                else "forward"
            )
            entry = {
                "order": [
                    quality_key(1.0),
                    relation.relation_key,
                    key(origin),
                    key(destination),
                    traversal,
                ]
            }
            validate_relation_entry(page, entry)
            self.pages.append((page, entry, key(origin)))
        for name in mentions:
            rows, chunk = self.base_rows(label, (name,))
            mention = replace(
                b.entity_mentions[0],
                mention_key=key(label + name + "mention"),
                provenance_key=key(label + name + "provenance"),
                entity_key=key(name),
                chunk_key=chunk.chunk_key,
                document_key=chunk.document_key,
            )
            rows["ProjectedEntityMention", mention.mention_key] = mention
            page = page_from_rows(rows)
            self.pages.append((page, None, key(name)))

    def topology(self):
        from .graph_association_bundle_fixture_v2 import bundle_from_pages

        bundle = bundle_from_pages(self.bundle, self.pages)
        self.ready = _ready(bundle)
        self.selected = self.ready.selected_generations[0]
        scope_rows = {
            ("ArtifactProvenance", r.scope_key): r for r in bundle.artifact_provenance
        }
        self.assembly = FrontierAssemblyV2(
            SimpleNamespace(ready=self.ready, scope_rows={self.generation: scope_rows}),
            SimpleNamespace(representation=self.representation),
            _caps(),
            FrontierRankingContextV2(key("group"), 200, 1000, 5000, 64, 1048576),
        )
        for page, entry, origin in self.pages:
            assert validate_source_page(page, selected=self.selected, ready=self.ready)
            for member in page.memberships:
                assert self.assembly.add_copy(
                    member.entity_key,
                    0,
                    {("AutomaticMembership", member.entity_key): member},
                )
            if entry is None:
                self.assembly.add_mention(page, self.generation, origin)
            else:
                assert self.assembly.add_relation(
                    page, entry, self.generation, origin, 0
                )
        snapshot, refs = self.assembly.build(self.request)
        return BoundedGraphTopologyV2(
            snapshot,
            refs,
            GraphCoverageV2(
                len(self.documents), len(self.documents), "complete", (), False
            ),
            key("request"),
            key("group"),
        )

    def select(self, maximum=20):
        topology = self.topology()
        inputs = tuple(
            GraphEvidenceSelectionInputV2(
                r,
                self.texts[r.chunk_key][0],
                key(self.texts[r.chunk_key][0]),
                self.texts[r.chunk_key][1],
            )
            for r in topology.evidence_candidates
        )
        envelope = GraphEvidenceSelectionEnvelopeV2(
            inputs, topology.coverage, key("algorithm"), ((key("group"), key("ready")),)
        )
        return select_graph_evidence_v2(
            query=self.query,
            requests=(GraphGroupRelationRequestV2(key("group"), self.request),),
            envelope=envelope,
            identity_scores=tuple((i, 1.0) for i in topology.snapshot.identity_keys),
            max_candidates=maximum,
            max_mentions_per_identity=2,
            max_evidence_per_transition=3,
        )
