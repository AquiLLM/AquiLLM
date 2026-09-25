"""Compatibility fixtures for projected PPR behavior tests."""

from apps.knowledge_graph.evals.projected_replay_snapshot import (
    key,
    projected_snapshot,
    provenance,
)
from apps.knowledge_graph.retrieval.expansion import AuthorizedArtifactProvenance

__all__ = ["key", "projected_snapshot", "provenance", "legacy_provenance"]


def legacy_provenance(scope_type: str, scope_id: str, collection_id: int):
    is_collection = scope_type == "collection"
    return AuthorizedArtifactProvenance(
        10 if is_collection else 20,
        scope_type,
        scope_id,
        collection_id,
        None,
        False,
        key(f"legacy-build-{scope_type}"),
        1,
        1,
        key(f"legacy-source-{scope_type}"),
        "ontology-v1",
        key("legacy-ontology"),
        "extractor-v1",
        "resolver-v1",
        key("legacy-resolution"),
        "filter-v1",
        key("legacy-filter"),
        "embed-v1" if is_collection else "",
        "assembly-v1",
        key("legacy-assembly"),
    )
