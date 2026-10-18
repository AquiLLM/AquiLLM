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
    *, selected_artifact_ids: tuple[int, ...], using: str, read_budget=None
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
    if read_budget is None:
        return tuple(rows)
    from .graph_seed_reads_v2 import seed_query_rows

    return seed_query_rows(rows, maximum=len(selected_artifact_ids), budget=read_budget)


def _load_active_ontology_row(
    *, using: str, version: str, checksum: str, read_budget=None
) -> dict[str, object] | None:
    from apps.knowledge_graph.models import OntologyVersion

    query = (
        OntologyVersion.objects.using(using)
        .filter(
            kind=OntologyVersion.Kind.GRAPH,
            status=OntologyVersion.Status.ACTIVE,
            version=version,
            checksum=checksum,
        )
        .order_by("id")
        .values("version", "checksum", "metadata")[:2]
    )
    if read_budget is None:
        rows = tuple(query)
    else:
        from .graph_seed_reads_v2 import seed_query_rows

        rows = seed_query_rows(query, maximum=2, budget=read_budget)
    return rows[0] if len(rows) == 1 else None


def _mixed(count: int) -> QueryOntologyOutcomeV1:
    return QueryOntologyOutcomeV1(None, DirectFailureReason.MIXED_ONTOLOGY, count)


def load_query_ontology(
    *, selected_artifact_ids: tuple[int, ...], using: str, read_budget=None
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
    optional = {} if read_budget is None else {"read_budget": read_budget}
    rows = _load_selected_artifact_rows(
        selected_artifact_ids=selected_artifact_ids, using=using, **optional
    )
    if tuple(row.get("id") for row in rows) != selected_artifact_ids:
        return _mixed(count)
    identities = {
        (row.get("ontology_version"), row.get("ontology_checksum")) for row in rows
    }
    if len(identities) != 1:
        return _mixed(count)
    version, checksum = next(iter(identities))
    return _complete_query_ontology(version=version, checksum=checksum, count=count, using=using, read_budget=read_budget)


def _complete_query_ontology(*, version, checksum, count, using, read_budget=None):
    optional = {} if read_budget is None else {"read_budget": read_budget}
    active = _load_active_ontology_row(
        using=using, version=version, checksum=checksum, **optional
    )
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

_NATIVE_ONTOLOGY_METHODS = (load_query_ontology, _load_selected_artifact_rows, _load_active_ontology_row, _complete_query_ontology, load_ontology_yaml)
