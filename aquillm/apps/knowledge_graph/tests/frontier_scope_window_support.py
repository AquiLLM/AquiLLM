"""Mixed, builder-produced 31/530 fixture for the physical feasibility gate."""

from dataclasses import fields, replace
from time import monotonic
from types import SimpleNamespace

from apps.knowledge_graph.projection.frontier_adapter_v2 import FrontierSessionV2
from apps.knowledge_graph.projection.frontier_reads_v2 import QUERIES
from apps.knowledge_graph.projection.frontier_snapshot_v2 import FrontierAssemblyV2
from apps.knowledge_graph.retrieval.topology.contracts import (
    ProjectedSeedV1,
    ReadyGenerationBundleV1,
    ready_generation_bundle_checksum,
)
from apps.knowledge_graph.retrieval.topology.frontier_contracts_v2 import (
    FrontierBudgetV2,
)
from apps.knowledge_graph.tests.frontier_generation_batch_support import (
    GenerationDriver,
    GenerationFixture,
    key,
)
from apps.knowledge_graph.tests.test_projected_topology_adapter import _caps, _ready
from apps.knowledge_graph.tests.test_projection_records import _bundle
from lib.retrieval.turn_budget import TurnBudget, TurnLimits


def mixed_bundle(scope, copies):
    """Keep every authorized scope; add real copy streams and one evidence edge."""
    template = _bundle()
    marker = scope.generation
    document = scope.documents[0].document_key
    provenance = next(p for p in scope.artifact_provenance if p.scope_key == document)
    mapping = {
        template.generation.generation_key: marker.generation_key,
        template.generation.collection_key: marker.collection_key,
        template.generation.artifact_key: marker.artifact_key,
        template.generation.projection_key: marker.projection_key,
        template.documents[0].document_key: document,
        template.evidence[0].artifact_key: provenance.artifact_key,
    }

    def convert(row):
        return replace(
            row,
            **{
                f.name: mapping.get(value, key(marker.generation_key + value))
                for f in fields(row)
                if f.name.endswith("_key")
                and (value := getattr(row, f.name)) is not None
            },
        )

    entities = [convert(row) for row in template.entities]
    members = [convert(row) for row in template.automatic_memberships]
    for i in range(copies):
        entity = key(marker.generation_key + ":unrelated:" + str(i))
        entities.append(replace(entities[0], entity_key=entity))
        members.append(replace(members[0], entity_key=entity))
    return replace(
        scope,
        entities=tuple(sorted(entities, key=lambda r: r.entity_key)),
        automatic_memberships=tuple(sorted(members, key=lambda r: r.entity_key)),
        chunks=tuple(convert(r) for r in template.chunks),
        relation_semantics=tuple(convert(r) for r in template.relation_semantics),
        relations=tuple(convert(r) for r in template.relations),
        evidence=tuple(convert(r) for r in template.evidence),
        entity_mentions=tuple(convert(r) for r in template.entity_mentions),
        counts=replace(
            scope.counts,
            entity_count=copies + 2,
            automatic_membership_count=copies + 2,
            chunk_count=1,
            relation_semantics_count=1,
            relation_count=1,
            evidence_count=1,
            entity_mention_count=1,
        ),
    )


class CountingGenerationDriver(GenerationDriver):
    def __init__(self, bundles):
        super().__init__(bundles)
        self.transcript = []
        self.phase = "initialize"

    def execute_read(self, query, parameters, *, timeout_seconds, max_records):
        rows = super().execute_read(
            query, parameters, timeout_seconds=timeout_seconds, max_records=max_records
        )
        if "lookups_csv" in parameters:
            pairs = tuple(
                tuple(p.split(":")) for p in parameters["lookups_csv"].split(",")
            )
        elif "keys_csv" in parameters:
            pairs = tuple(
                (parameters["generation_key"], k)
                for k in parameters["keys_csv"].split(",")
            )
        else:
            index_key = parameters["index_key"]
            pairs = ((self.index[index_key]["generation_key"], index_key),)
        self.transcript.append(
            {
                "phase": self.phase,
                "kind": next(k for k, q in QUERIES.items() if q == query),
                "pairs": pairs,
                "rows": len(rows),
                "maximum": max_records,
            }
        )
        return rows


class MixedScopeFixture:
    def __init__(self, *, copies=256):
        scopes = GenerationFixture().bundles
        self.bundles = tuple(mixed_bundle(scope, copies) for scope in scopes)
        parts = tuple(_ready(bundle) for bundle in self.bundles)
        selected = tuple(
            sorted(
                (p.selected_generations[0] for p in parts),
                key=lambda r: r.collection_key,
            )
        )
        documents = tuple(
            sorted(
                (d for p in parts for d in p.authorized_documents),
                key=lambda r: (r.collection_key, r.document_key),
            )
        )
        signature = parts[0].authorization_context_signature
        self.ready = ReadyGenerationBundleV1(
            selected,
            documents,
            signature,
            ready_generation_bundle_checksum(selected, documents, signature),
        )
        self.driver = CountingGenerationDriver(self.bundles)
        # Start the unchanged invocation budget after offline fixture construction.
        self.budget = FrontierBudgetV2(
            turn_budget=TurnBudget(TurnLimits()),
            deadline=monotonic() + 5.0,
            max_identities=200,
            max_edges=1000,
            max_source_rows=5000,
            max_queries=64,
            max_wire_bytes=1048576,
            clock=monotonic,
        )
        self.group = SimpleNamespace(
            scope=SimpleNamespace(ready=self.ready), group_checksum=key("mixed-group")
        )
        self.request = SimpleNamespace(
            query_digest=key("query"),
            ontology_checksum=key("ontology"),
            predicate_weights=(("knows", 1.0),),
            policy_version="test",
        )
        self.session = FrontierSessionV2(
            self.driver,
            group=self.group,
            request=self.request,
            budget=self.budget,
            deadline=self.budget.deadline,
        )
        self.caps = _caps()
        # Put useful content in the first generation searched, not behind misses.
        first = min(self.bundles, key=lambda b: b.generation.generation_key)
        self.seed = ProjectedSeedV1(first.relations[0].source_entity_key, 1.0)
        self.seed_generation = first.generation.generation_key
        # Direct-document fixture has no alternate representation generation.
        binding = SimpleNamespace(representation=lambda _generation, _document: None)
        context = SimpleNamespace(group_checksum=self.group.group_checksum)
        self.assembly = FrontierAssemblyV2(self.session, binding, self.caps, context)
