"""Auditable resolution of historical extraction with non-entity label noise."""

from collections.abc import Mapping
from itertools import islice

from .normalization import normalize_entity_label

EXCLUSION_FIELDS = frozenset(
    {"excluded_mention_ids", "excluded_relation_ids", "exclusion_reason"}
)
EXCLUSION_REASON = "unresolvable_entity_label"


def _field(record, name):
    return record.get(name) if isinstance(record, Mapping) else getattr(record, name)


def _bounded_mentions(mentions):
    from .coreference import MAX_DOCUMENT_MENTIONS

    bounded = tuple(islice(iter(mentions), MAX_DOCUMENT_MENTIONS + 1))
    if len(bounded) > MAX_DOCUMENT_MENTIONS:
        raise ValueError(f"document mention cap exceeded ({MAX_DOCUMENT_MENTIONS})")
    return bounded


def resolvable_mentions(mentions):
    """Keep all valid labels; only the meaningless-label category is excludable."""
    accepted = []
    for mention in _bounded_mentions(mentions):
        try:
            normalize_entity_label(_field(mention, "raw_text"))
        except ValueError as exc:
            if str(exc) != "entity label must contain meaningful characters":
                raise
        else:
            accepted.append(mention)
    return tuple(accepted)


def source_exclusion_audit(mentions, relations):
    """Bind explicit exclusions to retained, immutable mention/relation row IDs."""
    mentions = _bounded_mentions(mentions)
    accepted_ids = {_field(row, "id") for row in resolvable_mentions(mentions)}
    excluded_ids = sorted(
        _field(row, "id") for row in mentions if _field(row, "id") not in accepted_ids
    )
    if not excluded_ids:
        return {}
    excluded = set(excluded_ids)
    from ..extraction.pipeline import DOCUMENT_EXTRACTION_V1_MAX_RELATIONS

    bounded = tuple(islice(iter(relations), DOCUMENT_EXTRACTION_V1_MAX_RELATIONS + 1))
    if len(bounded) > DOCUMENT_EXTRACTION_V1_MAX_RELATIONS:
        raise ValueError("document relation cap exceeded")
    return {
        "excluded_mention_ids": excluded_ids,
        "excluded_relation_ids": sorted(
            _field(row, "id")
            for row in bounded
            if _field(row, "head_id") in excluded or _field(row, "tail_id") in excluded
        ),
        "exclusion_reason": EXCLUSION_REASON,
    }


def exclusion_marker_is_valid(marker, audit):
    if not audit:
        return marker.get("version") == 1 and not EXCLUSION_FIELDS.intersection(marker)
    return bool(
        marker.get("version") == 2
        and marker.get("exclusion_reason") == EXCLUSION_REASON
        and all(marker.get(key) == value for key, value in audit.items())
        and set(audit) == EXCLUSION_FIELDS
        and audit["excluded_mention_ids"]
        and all(
            type(record.get(field)) is list
            and all(type(value) is int and value > 0 for value in record[field])
            and record[field] == sorted(set(record[field]))
            for record in (audit, marker)
            for field in ("excluded_mention_ids", "excluded_relation_ids")
        )
    )
