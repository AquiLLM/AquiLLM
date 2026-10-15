"""Existing private31 fixture ported for product SQL differential measurements."""

from dataclasses import replace
from uuid import uuid4


def many(fixture):
    from apps.collections.models import Collection
    from apps.knowledge_graph.models import (
        CollectionGraphMembershipState,
        GraphArtifact,
    )

    scope, states, entities, assignments = fixture
    states = list(states)
    artifact = GraphArtifact.objects.get(pk=scope.selected_artifact_ids[0])
    for i in range(30):
        collection = Collection.objects.create(name="metadata-plan-" + str(i))
        selected = GraphArtifact.objects.create(
            scope_type="collection",
            scope_id=collection.pk,
            status="building",
            source_hash="a" * 64,
            ontology_version=artifact.ontology_version,
            ontology_checksum=artifact.ontology_checksum,
            extractor_version=artifact.extractor_version,
            filter_policy_version=artifact.filter_policy_version,
            resolver_version=artifact.resolver_version,
            embedding_model_signature=artifact.embedding_model_signature,
        )
        GraphArtifact.objects.filter(pk=selected.pk).update(status="active")
        row = {
            **states[0],
            "collection_id": collection.pk,
            "active_artifact_id": selected.pk,
        }
        CollectionGraphMembershipState.objects.create(
            collection_id=collection.pk,
            **{k: v for k, v in row.items() if k != "collection_id"},
        )
        states.append(row)
    scope = replace(
        scope,
        selected_collection_ids=tuple(sorted(r["collection_id"] for r in states)),
        selected_artifact_ids=tuple(sorted(r["active_artifact_id"] for r in states)),
        generation_keys_by_artifact=tuple(
            sorted(
                (r["active_artifact_id"], f"{i + 100:064x}")
                for i, r in enumerate(states)
            )
        ),
        generation_ids_by_artifact=tuple(
            sorted((r["active_artifact_id"], uuid4()) for r in states)
        ),
    )
    return scope, tuple(states), entities, assignments


def subtree(node):
    return {
        "type": node["Node Type"],
        "rows": node.get("Actual Rows"),
        "loops": node.get("Actual Loops"),
        "children": [subtree(n) for n in node.get("Plans", [])],
    }


def find(node, name):
    if node.get("Subplan Name") == name:
        return node
    return next(
        (got for child in node.get("Plans", []) if (got := find(child, name))), None
    )


def first_limit(node):
    if node["Node Type"] == "Limit":
        return node
    return next(
        (got for child in node.get("Plans", []) if (got := first_limit(child))), None
    )
