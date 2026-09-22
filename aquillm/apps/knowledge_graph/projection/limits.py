"""Complete-generation ceilings, distinct from database/driver page sizes."""

from types import MappingProxyType

MAX_PAGE_ROWS = 5_000
MAX_ENTITY_ROWS = 50_000
MAX_DOCUMENT_ROWS = 10_000
MAX_DETAIL_ROWS = 250_000
MAX_ARTIFACT_ROWS = MAX_DOCUMENT_ROWS + 1

FAMILY_LIMITS = MappingProxyType(
    {
        "ProjectedEntity": MAX_ENTITY_ROWS,
        "AutomaticMembership": MAX_ENTITY_ROWS,
        "ProjectedDocument": MAX_DOCUMENT_ROWS,
        "ProjectedChunk": MAX_DETAIL_ROWS,
        "ProjectedRelationSemantics": 10_000,
        "ProjectedRelation": MAX_DETAIL_ROWS,
        "ProjectedEvidence": MAX_DETAIL_ROWS,
        "ProjectedEntityMention": MAX_DETAIL_ROWS,
        "ArtifactProvenance": MAX_ARTIFACT_ROWS,
    }
)


def bounded_projection_rows(
    query,
    fields: tuple[str, ...],
    batch_size: int,
    *,
    maximum: int = MAX_DETAIL_ROWS,
    order: tuple[str, ...] = ("pk",),
) -> tuple[dict, ...]:
    """Fetch one SQL snapshot in bounded cursor pages, including an overflow sentinel.

    A single cursor also preserves joined rows with repeated parent primary keys;
    keyset paging on just the parent key would silently skip their later mentions.
    """
    if type(batch_size) is not int or not 1 <= batch_size <= MAX_PAGE_ROWS:
        raise ValueError("batch_size must be an integer in 1..5000")
    if type(maximum) is not int or not 1 <= maximum <= MAX_DETAIL_ROWS:
        raise ValueError("projection family maximum is invalid")
    rows = tuple(
        query.order_by(*order)
        .values(*fields)[: maximum + 1]
        .iterator(chunk_size=batch_size)
    )
    if len(rows) > maximum:
        raise ValueError("projection row family exceeds its hard cap")
    return rows
