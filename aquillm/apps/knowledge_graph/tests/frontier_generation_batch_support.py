"""Scope-only encoded fixture; no Django or live services required."""

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from time import monotonic
from types import SimpleNamespace

from apps.knowledge_graph.projection.frontier_adapter_v2 import FrontierSessionV2
from apps.knowledge_graph.projection.frontier_reads_v2 import QUERIES
from apps.knowledge_graph.retrieval.topology.contracts import (
    ReadyGenerationBundleV1,
    ready_generation_bundle_checksum,
)
from apps.knowledge_graph.retrieval.topology.frontier_contracts_v2 import (
    FrontierBudgetV2,
)
from apps.knowledge_graph.tests.frontier_index_fixture_driver import (
    FrontierIndexFixtureDriver,
)
from apps.knowledge_graph.tests.test_projected_topology_adapter import _ready
from apps.knowledge_graph.tests.test_projection_records import _bundle
from lib.retrieval.turn_budget import TurnBudget, TurnLimits


def key(value):
    return sha256(value.encode()).hexdigest()


class GenerationDriver(FrontierIndexFixtureDriver):
    def __init__(self, bundles):
        super().__init__(bundles)
        self.scanned, self.transform, self.bounds = 0, None, []

    def execute_read(self, query, parameters, *, timeout_seconds, max_records):
        if "lookups_csv" in parameters:
            self.calls.append((query, parameters.copy()))
            rows = []
            for pair in parameters["lookups_csv"].split(","):
                generation, lookup = pair.split(":")
                identity = {"generation_key": generation, "lookup_key": lookup}
                if query == QUERIES.get("manifest_batch"):
                    if generation in self.markers:
                        rows.append({**identity, "record": self.markers[generation]})
                elif query == QUERIES.get("index_batch_multi"):
                    if lookup in self.index:
                        rows.append({**identity, "record": self.index[lookup]})
                elif query == QUERIES.get("raw_batch_multi"):
                    rows.extend(
                        {**r, **identity}
                        for r in self.raw.get((generation, lookup), ())[:10]
                    )
                else:
                    raise AssertionError("unexpected fixed batch query")
            rows = tuple(rows)
        else:
            rows = super().execute_read(
                query,
                parameters,
                timeout_seconds=timeout_seconds,
                max_records=max_records,
            )
        rows = deepcopy(rows)
        if self.transform:
            rows = self.transform(query, parameters, rows)
        self.scanned += len(rows)
        self.bounds.append(max_records)
        return rows


class GenerationFixture:
    def __init__(self, generations=31, documents=530, *, budget=None):
        original = _bundle()
        bundles = []
        for i in range(generations):
            marker = replace(
                original.generation,
                **{
                    name: key(f"{i}:{name}")
                    for name in (
                        "generation_key",
                        "collection_key",
                        "artifact_key",
                        "projection_key",
                    )
                },
            )
            count = documents // generations + (i < documents % generations)
            docs = tuple(
                sorted(
                    (
                        replace(
                            original.documents[0],
                            document_key=key(f"{i}:doc:{j}"),
                            generation_key=marker.generation_key,
                        )
                        for j in range(count)
                    ),
                    key=lambda r: r.document_key,
                )
            )
            provenance = (
                replace(
                    original.artifact_provenance[0],
                    scope_key=marker.collection_key,
                    collection_key=marker.collection_key,
                    artifact_key=marker.artifact_key,
                ),
                *(
                    replace(
                        original.artifact_provenance[1],
                        scope_key=d.document_key,
                        collection_key=marker.collection_key,
                        artifact_key=key("artifact:" + d.document_key),
                    )
                    for d in docs
                ),
            )
            bundles.append(
                replace(
                    original,
                    generation=marker,
                    entities=(),
                    automatic_memberships=(),
                    documents=docs,
                    chunks=(),
                    relation_semantics=(),
                    relations=(),
                    evidence=(),
                    entity_mentions=(),
                    artifact_provenance=tuple(
                        sorted(
                            provenance,
                            key=lambda r: (r.scope_type, r.scope_key, r.artifact_key),
                        )
                    ),
                    counts=type(original.counts)(0, 0, count, 0, 0, 0, 0, 0, count + 1),
                )
            )
        self.bundles = tuple(bundles)
        parts = tuple(_ready(b) for b in bundles)
        selected = tuple(
            sorted(
                (p.selected_generations[0] for p in parts),
                key=lambda r: r.collection_key,
            )
        )
        docs = tuple(
            sorted(
                (d for p in parts for d in p.authorized_documents),
                key=lambda r: (r.collection_key, r.document_key),
            )
        )
        signature = parts[0].authorization_context_signature
        self.ready = ReadyGenerationBundleV1(
            selected,
            docs,
            signature,
            ready_generation_bundle_checksum(selected, docs, signature),
        )
        self.group = SimpleNamespace(
            scope=SimpleNamespace(ready=self.ready), group_checksum=key("group")
        )
        request = SimpleNamespace(
            query_digest=key("query"),
            ontology_checksum=key("ontology"),
            predicate_weights=(),
            policy_version="test",
        )
        self.driver = GenerationDriver(bundles)
        self.budget = budget or FrontierBudgetV2(
            turn_budget=TurnBudget(TurnLimits()),
            deadline=monotonic() + 15,
            max_identities=200,
            max_edges=1000,
            max_source_rows=5000,
            max_queries=64,
            max_wire_bytes=1048576,
            clock=monotonic,
        )
        self.session = FrontierSessionV2(
            self.driver,
            group=self.group,
            request=request,
            budget=self.budget,
            deadline=self.budget.deadline,
        )
