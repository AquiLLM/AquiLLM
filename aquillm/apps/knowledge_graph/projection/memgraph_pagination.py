"""Fixed keyset pagination for projection records."""

from __future__ import annotations

PAGE_SIZE = 1_000

FAMILY_ORDER_FIELDS = {
    "ProjectedEntity": ("entity_key",),
    "AutomaticMembership": ("entity_key",),
    "ProjectedDocument": ("document_key",),
    "ProjectedChunk": ("document_key", "chunk_number", "chunk_key"),
    "ProjectedRelationSemantics": ("artifact_key", "relation_type", "semantics_key"),
    "ProjectedRelation": ("relation_key",),
    "ProjectedEvidence": ("evidence_key",),
    "ProjectedEntityMention": ("entity_key", "provenance_key", "mention_key"),
    "ArtifactProvenance": ("scope_type", "scope_key", "artifact_key"),
}

FAMILY_IDENTITY_FIELDS = {
    "ProjectedEntity": "entity_key",
    "AutomaticMembership": "entity_key",
    "ProjectedDocument": "document_key",
    "ProjectedChunk": "chunk_key",
    "ProjectedRelationSemantics": "semantics_key",
    "ProjectedRelation": "relation_key",
    "ProjectedEvidence": "evidence_key",
    "ProjectedEntityMention": "mention_key",
    "ArtifactProvenance": "scope_key",
}


def _fields(label: str) -> tuple[str, ...]:
    try:
        return FAMILY_ORDER_FIELDS[label]
    except KeyError:
        raise ValueError("unknown Memgraph projection family") from None


def cursor_predicate(label: str, *, alias: str = "n") -> str:
    fields = _fields(label)
    branches = []
    for index, field in enumerate(fields):
        equal = " AND ".join(
            f"{alias}.{prior} = $cursor_{prior}" for prior in fields[:index]
        )
        greater = f"{alias}.{field} > $cursor_{field}"
        branches.append(f"({equal} AND {greater})" if equal else f"({greater})")
    equal = " AND ".join(
        f"{alias}.{field} = $cursor_{field}" for field in fields
    )
    branches.append(f"({equal} AND id({alias}) > $cursor_id)")
    return "NOT $has_cursor OR " + " OR ".join(branches)


def page_return(label: str, *, alias: str = "n") -> str:
    order = ", ".join(f"{alias}.{field}" for field in _fields(label))
    return (
        f"RETURN {alias} AS record, id({alias}) AS cursor_id "
        f"ORDER BY {order}, id({alias}) LIMIT $page_limit"
    )


def full_family_page_query(label: str) -> str:
    return (
        f"MATCH (n:{label} {{generation_key:$generation_key}}) "
        f"WHERE {cursor_predicate(label)} {page_return(label)}"
    )


def initial_cursor_parameters(label: str) -> dict[str, object]:
    result: dict[str, object] = {"has_cursor": False, "cursor_id": -1}
    for field in _fields(label):
        result[f"cursor_{field}"] = 0 if field == "chunk_number" else ""
    return result


def advance_cursor_parameters(
    label: str, properties: dict[str, object], cursor_id: object
) -> dict[str, object]:
    if type(cursor_id) is not int or cursor_id < 0:
        raise ValueError("Memgraph page cursor id is invalid")
    result: dict[str, object] = {"has_cursor": True, "cursor_id": cursor_id}
    for field in _fields(label):
        try:
            result[f"cursor_{field}"] = properties[field]
        except KeyError:
            raise ValueError("Memgraph page cursor record is incomplete") from None
    return result


def canonical_record_key(label: str, record: object) -> tuple[object, ...]:
    return tuple(getattr(record, field) for field in _fields(label))


__all__ = [
    "FAMILY_IDENTITY_FIELDS",
    "PAGE_SIZE",
    "advance_cursor_parameters",
    "canonical_record_key",
    "cursor_predicate",
    "full_family_page_query",
    "initial_cursor_parameters",
    "page_return",
]
