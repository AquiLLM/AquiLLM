"""Load one ontology compatible with every selected collection artifact."""

from __future__ import annotations

from dataclasses import dataclass

from apps.knowledge_graph.retrieval.direct_seed_contracts import DirectFailureReason
from apps.knowledge_graph.services.ontology import (
    OntologyDefinition,
    OntologyValidationError,
    load_ontology_yaml,
)


@dataclass(frozen=True, slots=True)
class QueryOntologyOutcomeV1:
    ontology: OntologyDefinition | None
    failure_reason: DirectFailureReason | None
    selected_artifact_count: int

    def __post_init__(self) -> None:
        if type(self.selected_artifact_count) is not int:
            raise TypeError("selected_artifact_count must be an exact int")
        if not 0 <= self.selected_artifact_count <= 128:
            raise ValueError("selected_artifact_count is outside its hard cap")
        if self.failure_reason is None:
            if type(self.ontology) is not OntologyDefinition:
                raise TypeError(
                    "successful ontology outcome requires an exact ontology"
                )
        elif (
            self.failure_reason is not DirectFailureReason.MIXED_ONTOLOGY
            or self.ontology is not None
        ):
            raise ValueError("query ontology failure must be mixed_ontology")


def _load_selected_artifact_rows(
    *, selected_artifact_ids: tuple[int, ...], using: str
) -> tuple[dict[str, object], ...]:
    from apps.knowledge_graph.models import GraphArtifact

    rows = (
        GraphArtifact.objects.using(using)
        .filter(
            id__in=selected_artifact_ids,
            scope_type=GraphArtifact.ScopeType.COLLECTION,
            status=GraphArtifact.Status.ACTIVE,
            evaluation_only=False,
        )
        .order_by("id")
        .values("id", "ontology_version", "ontology_checksum")
    )
    return tuple(rows)


def _load_active_ontology_row(*, using: str) -> dict[str, object] | None:
    from apps.knowledge_graph.models import OntologyVersion

    rows = tuple(
        OntologyVersion.objects.using(using)
        .filter(
            kind=OntologyVersion.Kind.GRAPH,
            status=OntologyVersion.Status.ACTIVE,
        )
        .order_by("id")
        .values("version", "checksum", "metadata")[:2]
    )
    return rows[0] if len(rows) == 1 else None


def _mixed(count: int) -> QueryOntologyOutcomeV1:
    return QueryOntologyOutcomeV1(None, DirectFailureReason.MIXED_ONTOLOGY, count)


def load_query_ontology(
    *, selected_artifact_ids: tuple[int, ...], using: str
) -> QueryOntologyOutcomeV1:
    if type(selected_artifact_ids) is not tuple:
        raise TypeError("selected_artifact_ids must be an exact tuple")
    if (
        not selected_artifact_ids
        or len(selected_artifact_ids) > 128
        or any(type(value) is not int or value <= 0 for value in selected_artifact_ids)
        or selected_artifact_ids != tuple(sorted(set(selected_artifact_ids)))
    ):
        raise ValueError("selected_artifact_ids must be bounded, sorted, and unique")
    if type(using) is not str or not using or using != using.strip():
        raise ValueError("using must be a nonempty exact database alias")
    count = len(selected_artifact_ids)
    rows = _load_selected_artifact_rows(
        selected_artifact_ids=selected_artifact_ids, using=using
    )
    if tuple(row.get("id") for row in rows) != selected_artifact_ids:
        return _mixed(count)
    identities = {
        (row.get("ontology_version"), row.get("ontology_checksum")) for row in rows
    }
    if len(identities) != 1:
        return _mixed(count)
    version, checksum = next(iter(identities))
    active = _load_active_ontology_row(using=using)
    if (
        active is None
        or active.get("version") != version
        or active.get("checksum") != checksum
    ):
        return _mixed(count)
    metadata = active.get("metadata")
    if type(metadata) is not dict or type(metadata.get("yaml")) is not str:
        return _mixed(count)
    try:
        ontology = load_ontology_yaml(metadata["yaml"])
    except OntologyValidationError:
        return _mixed(count)
    if ontology.version != version or ontology.checksum != checksum:
        return _mixed(count)
    return QueryOntologyOutcomeV1(ontology, None, count)


__all__ = ["QueryOntologyOutcomeV1", "load_query_ontology"]
