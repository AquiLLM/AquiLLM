"""Exact equality lookups only; never returns an unrequested neighborhood."""

from dataclasses import asdict

from apps.knowledge_graph.projection.frontier_cypher_v2 import (
    INDEX_BATCH_LOOKUP,
    INDEX_LOOKUP,
    MANIFEST_LOOKUP,
    RAW_LOOKUP,
)
from apps.knowledge_graph.projection.frontier_index_build_v2 import build_frontier_index
from apps.knowledge_graph.projection.serialization import projection_checksum
from apps.knowledge_graph.tests.test_projected_topology_adapter import _FAMILIES


class FrontierIndexFixtureDriver:
    def __init__(self, bundles):
        self.markers, self.index, self.raw, self.calls = {}, {}, {}, []
        for bundle in bundles:
            generation = bundle.generation.generation_key
            index = build_frontier_index(bundle)
            self.markers[generation] = {
                **asdict(bundle.generation),
                **index.marker,
                "graph_checksum": projection_checksum(bundle),
                "state": "ready",
            }
            self.index.update((r["index_key"], r) for r in index.records)
            for label, family, identity in _FAMILIES:
                for row in getattr(bundle, family):
                    key = getattr(row, identity)
                    self.raw.setdefault((generation, key), []).append(
                        {
                            "lookup_key": key,
                            "record": {
                                **asdict(row),
                                "generation_key": generation,
                                "opaque_key": key,
                            },
                            "labels": ["ProjectedRecord", label],
                        }
                    )

    def execute_read(self, query, parameters, *, timeout_seconds, max_records):
        self.calls.append((query, parameters.copy()))
        if query == MANIFEST_LOOKUP:
            return ({"record": self.markers[parameters["generation_key"]].copy()},)
        if query == INDEX_LOOKUP:
            row = self.index.get(parameters["index_key"])
            return ({"record": row},) if row else ()
        if query == INDEX_BATCH_LOOKUP:
            return tuple(
                {"lookup_key": key, "record": self.index[key]}
                for key in parameters["keys_csv"].split(",")
                if key in self.index
            )
        if query == RAW_LOOKUP:
            return tuple(
                row
                for key in parameters["keys_csv"].split(",")
                for row in self.raw.get((parameters["generation_key"], key), ())[:10]
            )
        raise AssertionError("unexpected non-equality frontier query")
