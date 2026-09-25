"""Large collection admission keeps ordinary build identities unchanged."""

import pytest

from apps.knowledge_graph.graph.assembly import AssemblyConfig, assembly_config_checksum
from apps.knowledge_graph.resolution.collection import (
    CollectionResolutionConfig,
    resolution_config_checksum,
)
from apps.knowledge_graph.services import collection_context_policy as policy


@pytest.mark.parametrize("count", [0, 49_999, 50_000])
def test_small_collection_keeps_default_identity(count):
    resolver, assembly = policy.select_capacity_configs(count)
    assert resolution_config_checksum(resolver) == resolution_config_checksum(
        CollectionResolutionConfig()
    )
    assert assembly_config_checksum(assembly) == assembly_config_checksum(
        AssemblyConfig()
    )


@pytest.mark.parametrize("count", [50_001, 89_052, 100_000])
def test_large_collection_gets_bounded_bucket(count):
    resolver, assembly = policy.select_capacity_configs(count)
    assert resolver.max_entities == assembly.max_entities == 100_000
    assert assembly.max_orphan_entities == 100_000
    policy.validate_collection_context_caps(
        document_count=80,
        entity_count=count,
        resolution_config=resolver,
        assembly_config=assembly,
    )


def test_explicit_configs_are_respected_and_capacity_is_typed():
    original = CollectionResolutionConfig(max_entities=30_000)
    resolver, assembly = policy.select_capacity_configs(89_052, original)
    assert resolver is original
    with pytest.raises(policy.CollectionCapacityError) as caught:
        policy.validate_collection_context_caps(
            document_count=80,
            entity_count=89_052,
            resolution_config=resolver,
            assembly_config=assembly,
        )
    assert caught.value.error_code == "collection_entity_limit"


def test_above_hard_cap_is_rejected_without_unbounded_config():
    resolver, assembly = policy.select_capacity_configs(100_001)
    with pytest.raises(policy.CollectionCapacityError):
        policy.validate_collection_context_caps(
            document_count=80,
            entity_count=100_001,
            resolution_config=resolver,
            assembly_config=assembly,
        )


def test_recovery_reports_capacity_without_publishing(monkeypatch):
    from unittest.mock import Mock

    from apps.knowledge_graph.graph import recovery
    from apps.knowledge_graph.services import builds

    monkeypatch.setattr(
        builds,
        "_collection_context",
        Mock(side_effect=policy.CollectionCapacityError("entity")),
    )
    publish = Mock()
    monkeypatch.setattr(builds, "enqueue_collection_refresh", publish)
    assert recovery._recover_collection(1) == recovery.RecoveryOutcome.CAPACITY_BLOCKED
    publish.assert_not_called()
