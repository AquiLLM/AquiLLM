from __future__ import annotations

import inspect
from dataclasses import replace

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

    assert "link_query.count()" in source
    assert "config.max_links" in source
    assert source.index("link_query.count()") < source.index("tuple(link_query)")


def test_task9_entities_are_counted_before_they_are_materialized():
    source = inspect.getsource(assembly._load_locked_task9_rows)

    assert "entity_query.count()" in source
    assert "config.max_entities" in source
    assert source.index("entity_query.count()") < source.index("tuple(entity_query)")


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
