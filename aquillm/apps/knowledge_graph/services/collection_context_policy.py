"""Bounded collection admission and safe preflight failure classification."""

from .failure_codes import CorruptBuildError, StaleBuildError


class CollectionCapacityError(CorruptBuildError):
    """An admitted collection input exceeds a finite configured boundary."""

    def __init__(self, dimension: str):
        if dimension not in {"document", "entity"}:
            raise ValueError("unknown collection capacity dimension")
        self.error_code = f"collection_{dimension}_limit"
        super().__init__(f"collection context {dimension} cap exceeded")


def context_failure_code(error: Exception) -> str:
    if isinstance(error, CollectionCapacityError):
        return error.error_code
    if isinstance(error, StaleBuildError):
        return "collection_manifest_changed"
    if isinstance(error, CorruptBuildError):
        return "collection_context_invalid"
    return "collection_context_failed"


def select_capacity_configs(entity_count, resolution_config=None, assembly_config=None):
    """Keep defaults stable; choose one bounded bucket for larger snapshots.

    Explicit caller configurations remain authoritative, including smaller caps.
    The selected configs participate in the existing immutable build checksum.
    """
    from apps.knowledge_graph.graph.assembly import AssemblyConfig
    from apps.knowledge_graph.resolution.collection import (
        DEFAULT_COLLECTION_ENTITIES,
        MAX_COLLECTION_ENTITIES,
        CollectionResolutionConfig,
    )

    if type(entity_count) is not int or entity_count < 0:
        raise CorruptBuildError("collection entity count must be nonnegative integer")
    capacity = (
        MAX_COLLECTION_ENTITIES
        if entity_count > DEFAULT_COLLECTION_ENTITIES
        else DEFAULT_COLLECTION_ENTITIES
    )
    if resolution_config is None:
        resolution_config = CollectionResolutionConfig(max_entities=capacity)
    if assembly_config is None:
        assembly_config = AssemblyConfig(
            max_entities=capacity, max_orphan_entities=capacity
        )
    return resolution_config, assembly_config


def validate_collection_context_caps(
    *,
    document_count,
    entity_count,
    resolution_config,
    assembly_config,
) -> None:
    """Reject oversized graph inputs independently from raw source volume."""
    if any(
        type(value) is not int or value < 0 for value in (document_count, entity_count)
    ):
        raise CorruptBuildError(
            "collection context counts must be nonnegative integers"
        )
    for name, value, cap in (
        (
            "document",
            document_count,
            min(
                resolution_config.max_document_inputs,
                assembly_config.max_document_inputs,
            ),
        ),
        (
            "entity",
            entity_count,
            min(resolution_config.max_entities, assembly_config.max_entities),
        ),
    ):
        if value > cap:
            raise CollectionCapacityError(name)
