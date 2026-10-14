"""Exact fresh marker and raw-record validation, independent of read scheduling."""

from dataclasses import fields

from apps.knowledge_graph.retrieval.topology.contracts import TopologyFailureReason
from apps.knowledge_graph.retrieval.topology.failures import TopologyLoadError

from .frontier_index_contracts_v2 import marker_present, record_digest
from .memgraph_pagination import FAMILY_IDENTITY_FIELDS
from .memgraph_records import FAMILIES, _dto

KINDS = dict(FAMILIES)


def schema_error(message="frontier source schema mismatch"):
    return TopologyLoadError(TopologyFailureReason.BACKEND_SCHEMA_MISMATCH)


def validate_marker(selected, marker, *, initial, final):
    if type(marker) is not dict:
        raise schema_error()
    names = (
        "collection_key",
        "generation_key",
        "projection_key",
        "graph_checksum",
        "schema_version",
        "projection_version",
        "identifier_key_version",
        "membership_epoch",
        "membership_checksum",
    )
    if any(marker.get(n) != getattr(selected, n) for n in names) or (
        marker.get("state") != "ready"
        or marker.get("artifact_key") != selected.active_artifact_key
    ):
        raise TopologyLoadError(TopologyFailureReason.READINESS_MISMATCH)
    try:
        indexed = marker_present(marker)
    except (ValueError, TypeError):
        raise schema_error() from None
    if final and initial != marker:
        raise TopologyLoadError(TopologyFailureReason.READINESS_MISMATCH)
    return indexed


def validate_raw(generation, expected, keys, rows):
    actual, occupancies = {}, {}
    for row in rows:
        if type(row) is not dict or set(row) != {"lookup_key", "record", "labels"}:
            raise schema_error()
        key, properties, labels = row["lookup_key"], row["record"], row["labels"]
        if key not in keys or type(properties) is not dict or type(labels) is not list:
            raise schema_error()
        occupancies[key] = occupancies.get(key, 0) + 1
        families = set(labels) & set(KINDS)
        if (
            occupancies[key] > 9
            or len(families) != 1
            or set(labels) != families | {"ProjectedRecord"}
        ):
            raise schema_error()
        label = next(iter(families))
        if (
            (label, key) in actual
            or properties.get("generation_key") != generation
            or properties.get("opaque_key") != key
        ):
            raise schema_error()
        try:
            names = {f.name for f in fields(KINDS[label])} | {
                "opaque_key",
                "generation_key",
            }
            if set(properties) - names:
                raise ValueError
            value = _dto(KINDS[label], properties)
            if getattr(value, FAMILY_IDENTITY_FIELDS[label]) != key:
                raise ValueError
        except (KeyError, TypeError, ValueError):
            raise schema_error() from None
        actual[label, key] = value
    if any(
        k not in actual or record_digest(actual[k]) != v for k, v in expected.items()
    ):
        raise schema_error()
    return {k: actual[k] for k in expected}
