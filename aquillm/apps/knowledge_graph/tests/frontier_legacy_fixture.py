"""Legacy requested-neighborhood fixture demonstrating the missed bridge."""

from dataclasses import asdict

from apps.knowledge_graph.tests.test_projected_topology_adapter import _FAMILIES, _ready


class LegacyNeighborhoodDriver:
    """Return only physical neighborhoods requested by the legacy adapter."""

    def __init__(self, bundles):
        self.bundles = {b.generation.generation_key: b for b in bundles}

    def execute_read(self, query, parameters, *, timeout_seconds, max_records):
        if "AS collection_key" in query:
            return tuple(
                {
                    "collection_key": s.collection_key,
                    "generation_key": s.generation_key,
                    "projection_key": s.projection_key,
                    "active_artifact_key": s.active_artifact_key,
                    "graph_checksum": s.graph_checksum,
                    "membership_checksum": s.membership_checksum,
                    "state": "ready",
                }
                for generation in parameters["generation_keys_csv"].split(",")
                for s in _ready(self.bundles[generation]).selected_generations
            )
        bundle = self.bundles[parameters["generation_key"]]
        if "CollectionGeneration" in query:
            return ({"record": asdict(bundle.generation)},)
        seeds = set(parameters["seed_keys_csv"].split(","))
        reachable = {
            m.entity_key
            for m in bundle.automatic_memberships
            if m.entity_key in seeds or m.automatic_membership_key in seeds
        }
        for _ in range(parameters["max_depth"]):
            reachable |= {
                endpoint
                for r in bundle.relations
                if r.source_entity_key in reachable or r.target_entity_key in reachable
                for endpoint in (r.source_entity_key, r.target_entity_key)
            }
        if "frontier_count" in query:
            return ({"frontier_count": int(bool(reachable))},)
        for label, family, identity in _FAMILIES:
            if f"(n:{label} " in query:
                rows = getattr(bundle, family)
                if not reachable and family not in {"documents", "artifact_provenance"}:
                    rows = ()
                return tuple(
                    {
                        "record": {**asdict(row), "opaque_key": getattr(row, identity)},
                        "cursor_id": index,
                    }
                    for index, row in enumerate(rows)
                )
        raise AssertionError("unexpected legacy query")
