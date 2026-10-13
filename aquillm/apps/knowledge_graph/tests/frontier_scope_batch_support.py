"""Synthetic encoded component fixture; persisted four-mode tests stay separate."""

from copy import deepcopy
from dataclasses import replace
from uuid import UUID

from apps.documents.services.representation_contracts import RepresentationSnapshotV1
from apps.knowledge_graph.projection.frontier_adapter_v2 import FrontierSessionV2
from apps.knowledge_graph.projection.frontier_reads_v2 import QUERIES
from apps.knowledge_graph.retrieval.graph_handoff import GraphCoverageV2
from apps.knowledge_graph.retrieval.ready_scope import SelectedReadyScopeV1, _encoded
from apps.knowledge_graph.retrieval.ready_scope_v2 import (
    GraphAuthorityGroupV2, SelectedGraphScopeV2, graph_authority_group_checksum_v2,
)
from apps.knowledge_graph.tests.frontier_fixture_support import FrontierFixture, key
from apps.knowledge_graph.tests.frontier_index_fixture_driver import FrontierIndexFixtureDriver
from apps.knowledge_graph.tests.test_projected_topology_adapter import _ready


class BatchDriver(FrontierIndexFixtureDriver):
    """Real encoded records behind fixed-query test transport, with physical counters."""

    def __init__(self, bundles):
        super().__init__(bundles)
        self.transform = None
        self.scanned = 0
        self.bounds = []

    def execute_read(self, query, parameters, *, timeout_seconds, max_records):
        if query == QUERIES.get("index_batch"):
            self.calls.append((query, parameters.copy()))
            rows = tuple(
                {"lookup_key": k, "record": deepcopy(self.index[k])}
                for k in parameters["keys_csv"].split(",") if k in self.index
            )
        else:
            rows = super().execute_read(
                query, parameters, timeout_seconds=timeout_seconds, max_records=max_records,
            )
            rows = deepcopy(rows)
        if self.transform is not None:
            rows = self.transform(query, parameters, rows)
        self.bounds.append((query, max_records, timeout_seconds))
        self.scanned += len(rows)
        return rows


class ScopeBatchFixture:
    def __init__(self, count=32):
        base = FrontierFixture()
        original = base.bundles[0]
        ids = tuple(UUID(int=i) for i in range(1, count + 1))
        authority = replace(
            base.group.scope.projections[0],
            documents=tuple((did, 200 + i) for i, did in enumerate(ids, 1)),
        )
        _, encoded = _encoded(authority, base.codec)
        first = original.documents[0]
        provenance = next(p for p in original.artifact_provenance if p.scope_type == "document")
        additions = tuple(d for d in encoded if d.document_key != first.document_key)
        bundle = replace(
            original,
            documents=tuple(sorted(
                (first, *(replace(first, document_key=d.document_key) for d in additions)),
                key=lambda r: r.document_key,
            )),
            artifact_provenance=tuple(sorted(
                (*original.artifact_provenance, *(replace(
                    provenance, scope_key=d.document_key, artifact_key=key("artifact:" + d.document_key),
                ) for d in additions)),
                key=lambda r: (r.scope_type, r.scope_key, r.artifact_key),
            )),
            counts=replace(original.counts, document_count=count, artifact_provenance_count=count + 1),
        )
        ready = _ready(bundle)
        authority = replace(authority, graph_checksum=ready.selected_generations[0].graph_checksum)
        scope = SelectedReadyScopeV1(
            ready, (authority,), ids,
            ((authority.projection_id, bundle.generation.generation_key),),
        )
        original_selection = base.selection.representation_snapshot.selections[0]
        snapshot = RepresentationSnapshotV1(tuple(
            replace(original_selection, document=replace(
                original_selection.document, pkid=i, id=did,
            )) for i, did in enumerate(ids, 1)
        ))
        self.group = GraphAuthorityGroupV2(
            scope, base.group.ontology_checksum,
            graph_authority_group_checksum_v2(
                scope=scope, ontology_checksum=base.group.ontology_checksum,
                representation_snapshot=snapshot,
            ),
        )
        self.selection = SelectedGraphScopeV2(
            (self.group,), ready.authorization_context_signature, snapshot,
            GraphCoverageV2(count, count, "complete", (), False),
        )
        self.codec, self.seeds, self.request, self.caps = base.codec, base.seeds, base.request, base.caps
        self.bundle, self.driver = bundle, BatchDriver((bundle,))
        self.budget, self.deadline = base.budget(), base.deadline
        self.session = FrontierSessionV2(
            self.driver, group=self.group, request=base.request,
            budget=self.budget, deadline=self.deadline,
        )

    def reader(self):
        selected = self.group.scope.ready.selected_generations[0]
        self.session.reads.attest(selected)
        return self.session.reads, selected.generation_key


def batch_indices(reads, generation, keys):
    method = getattr(reads, "indices", None)
    assert callable(method), "bounded authenticated index batch reader is missing"
    return method(generation, keys)
