# Versioned document resolution commit contracts.

from collections.abc import Mapping

from .coreference_types import _is_count, _is_hash
from .source_exclusions import EXCLUSION_FIELDS, exclusion_marker_is_valid

_COMMIT_FIELDS = frozenset(
    (
        "version",
        "resolver_version",
        "ontology_checksum",
        "assembly_version",
        "assembly_config_checksum",
        "source_mention_count",
        "source_mention_fingerprint",
        "document_entity_count",
        "membership_count",
        "result_checksum",
    )
)


def resolution_commit_is_valid(
    marker: object,
    *,
    resolver_version: str,
    ontology_checksum: str,
    assembly_version: str,
    assembly_config_checksum: str,
    source_mention_count: int,
    source_mention_fingerprint: str,
    document_entity_count: int,
    membership_count: int,
    result_checksum: str,
    exclusion_audit: dict | None = None,
) -> bool:
    """Validate the complete resolution commit marker against persisted state."""

    audit = exclusion_audit or {}
    return bool(
        isinstance(marker, Mapping)
        and frozenset(marker) == _COMMIT_FIELDS | (EXCLUSION_FIELDS if audit else set())
        and type(resolver_version) is str
        and _is_hash(ontology_checksum)
        and type(assembly_version) is str
        and assembly_version
        and _is_hash(assembly_config_checksum)
        and _is_count(source_mention_count)
        and _is_hash(source_mention_fingerprint)
        and _is_count(document_entity_count)
        and _is_count(membership_count)
        and _is_hash(result_checksum)
        and type(marker.get("version")) is int
        and exclusion_marker_is_valid(marker, audit)
        and type(marker.get("resolver_version")) is str
        and marker.get("resolver_version") == resolver_version
        and _is_hash(marker.get("ontology_checksum"))
        and marker.get("ontology_checksum") == ontology_checksum
        and type(marker.get("assembly_version")) is str
        and marker.get("assembly_version") == assembly_version
        and _is_hash(marker.get("assembly_config_checksum"))
        and marker.get("assembly_config_checksum") == assembly_config_checksum
        and _is_count(marker.get("source_mention_count"))
        and marker.get("source_mention_count") == source_mention_count
        and _is_hash(marker.get("source_mention_fingerprint"))
        and marker.get("source_mention_fingerprint") == source_mention_fingerprint
        and _is_count(marker.get("document_entity_count"))
        and marker.get("document_entity_count") == document_entity_count
        and _is_count(marker.get("membership_count"))
        and marker.get("membership_count") == membership_count
        and membership_count + len(audit.get("excluded_mention_ids", ()))
        == source_mention_count
        and _is_hash(marker.get("result_checksum"))
        and marker.get("result_checksum") == result_checksum
    )
