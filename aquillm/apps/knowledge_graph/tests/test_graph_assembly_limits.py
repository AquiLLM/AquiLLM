from __future__ import annotations

import inspect
from dataclasses import replace
from types import SimpleNamespace

import pytest

from apps.knowledge_graph.graph import assembly


def test_assembly_config_addresses_a_hard_v1_link_cap():
    config = assembly.AssemblyConfig()

    assert hasattr(assembly, "ASSEMBLY_V1_MAX_LINKS")
    assert config.max_links == assembly.ASSEMBLY_V1_MAX_LINKS
    default_checksum = assembly.assembly_config_checksum(config)
    lower_cap_checksum = assembly.assembly_config_checksum(
        replace(config, max_links=config.max_links - 1)
    )
    assert default_checksum != lower_cap_checksum
    with pytest.raises(ValueError, match="max_links.*bounded range"):
        replace(config, max_links=assembly.ASSEMBLY_V1_MAX_LINKS + 1)


def test_task9_links_are_counted_before_they_are_materialized():
    source = inspect.getsource(assembly._load_locked_task9_rows)

    assert "_bounded_query_rows(" in source
    assert "link_query" in source
    assert "config.max_links" in source
    assert "tuple(link_query)" not in source


def test_task9_entities_are_counted_before_they_are_materialized():
    source = inspect.getsource(assembly._load_locked_task9_rows)

    assert "_bounded_query_rows(" in source
    assert "entity_query" in source
    assert "config.max_entities" in source
    assert "tuple(entity_query)" not in source


def test_persisted_assembly_rows_are_bounded_before_materialization():
    loader = getattr(assembly, "_load_locked_assembly_rows", None)

    assert callable(loader)
    source = inspect.getsource(loader)
    compact = " ".join(source.split())
    assert "_bounded_query_rows( relation_query" in compact
    assert "_bounded_query_rows( evidence_query" in compact
    assert "config.max_relations" in source
    assert "config.max_evidence" in source
    assert "tuple(relation_query)" not in source
    assert "tuple(evidence_query)" not in source
    assert "_load_locked_assembly_rows(" in inspect.getsource(
        assembly.assemble_collection_graph
    )
    assert "_load_locked_assembly_rows(" in inspect.getsource(
        assembly._validate_locked_complete_artifact
    )


def test_collection_paths_lock_contributors_before_the_ontology_row():
    projection_source = inspect.getsource(assembly._locked_projection)
    active_source = inspect.getsource(
        assembly.validate_locked_active_collection_snapshot
    )

    assert projection_source.index("_validate_locked_manifest(") < (
        projection_source.index("_resolve_ontology(")
    )
    assert active_source.index("_validate_locked_manifest(") < active_source.index(
        "_resolve_ontology("
    )


def test_filter_lineage_depth_is_hard_bounded_and_checksum_addressed():
    hard_cap = getattr(assembly, "ASSEMBLY_V1_MAX_FILTER_LINEAGE_DEPTH", None)

    assert type(hard_cap) is int and 1 <= hard_cap <= 128
    config = assembly.AssemblyConfig()
    assert config.max_filter_lineage_depth == hard_cap
    assert assembly.assembly_config_checksum(
        config
    ) != assembly.assembly_config_checksum(
        replace(config, max_filter_lineage_depth=hard_cap - 1)
    )
    with pytest.raises(ValueError, match="max_filter_lineage_depth.*bounded range"):
        replace(config, max_filter_lineage_depth=hard_cap + 1)


def test_filter_lineage_cycle_is_rejected_before_reloading_the_seen_source():
    walker = getattr(assembly, "_walk_task9_lineage", None)
    assert callable(walker)
    loaded = []

    def context(artifact_id):
        return SimpleNamespace(artifact=SimpleNamespace(pk=artifact_id))

    def validate_node(current):
        source_id = {1: 2, 2: 1}[current.artifact.pk]
        return "filter_rerun", {"source_artifact_id": source_id}, source_id

    def load_source(_current, marker):
        loaded.append(marker["source_artifact_id"])
        return context(marker["source_artifact_id"])

    with pytest.raises(assembly.CollectionGraphAssemblyError, match="cycle"):
        walker(
            context(1),
            max_filter_lineage_depth=4,
            validate_node=validate_node,
            load_source=load_source,
        )
    assert loaded == [2]


def test_filter_lineage_depth_is_rejected_before_loading_the_over_cap_hop():
    walker = getattr(assembly, "_walk_task9_lineage", None)
    assert callable(walker)
    loaded = []

    def context(artifact_id):
        return SimpleNamespace(artifact=SimpleNamespace(pk=artifact_id))

    def validate_node(current):
        source_id = current.artifact.pk + 1
        return "filter_rerun", {"source_artifact_id": source_id}, source_id

    def load_source(_current, marker):
        loaded.append(marker["source_artifact_id"])
        return context(marker["source_artifact_id"])

    with pytest.raises(assembly.CollectionGraphAssemblyError, match="depth cap"):
        walker(
            context(1),
            max_filter_lineage_depth=2,
            validate_node=validate_node,
            load_source=load_source,
        )
    assert loaded == [2, 3]


def test_filter_lineage_rows_are_counted_before_materialization_without_recursion():
    loader = getattr(assembly, "_load_filter_source_lineage", None)
    assert callable(loader)
    source = inspect.getsource(loader)

    compact = " ".join(source.split())
    assert "_bounded_query_rows( source_manifest_query" in compact
    assert "_bounded_query_rows( source_entity_query" in compact
    assert "_bounded_query_rows( source_link_query" in compact
    assert "tuple(source_manifest_query)" not in source
    assert "tuple(source_entity_query)" not in source
    assert "tuple(source_link_query)" not in source
    assert "_validate_task9_lineage(" not in source


def test_contributor_chunk_validation_reuses_task7_bounded_loader():
    source = inspect.getsource(assembly._lock_current_contributors)

    assert "_ordered_chunks(" in source
    assert "_bounded_query_rows(" in source
    assert 'values_list("id", flat=True)' in source
    assert "TextChunk.objects.select_for_update" not in source


def test_endpoint_ids_are_split_into_bounded_deterministic_query_batches():
    batcher = getattr(assembly, "_id_batches", None)
    batch_size = getattr(assembly, "_ENDPOINT_ID_BATCH_SIZE", None)

    assert callable(batcher)
    assert type(batch_size) is int and 1 <= batch_size <= 10_000
    endpoint_ids = tuple(range(1, 20_003))
    batches = tuple(batcher(reversed(endpoint_ids)))

    assert all(1 <= len(batch) <= batch_size for batch in batches)
    assert tuple(value for batch in batches for value in batch) == endpoint_ids


def test_membership_loading_never_uses_the_full_endpoint_set_in_one_in_predicate():
    source = inspect.getsource(assembly._load_assembly_evidence)

    assert "_id_batches(endpoint_ids)" in source
    assert "mention_id__in=endpoint_batch" in source
    assert "mention_id__in=endpoint_ids" not in source


def test_provenance_memberships_batch_document_entity_predicates_too():
    source = inspect.getsource(assembly._active_entity_provenance)

    assert "_id_batches(document_entity_ids)" in source
    assert "document_entity_id__in=document_entity_batch" in source
    assert "document_entity_id__in=document_entity_ids" not in source
    assert 'select_related("mention", "document_entity")' in source
