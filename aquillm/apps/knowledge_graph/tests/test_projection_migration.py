from __future__ import annotations

import importlib

from django.db import migrations


def test_projection_migration_uses_reserved_number_and_exact_dependency() -> None:
    migration = importlib.import_module(
        "apps.knowledge_graph.migrations.0007_memgraph_projection_authority"
    )

    assert migration.Migration.dependencies == [
        ("apps_knowledge_graph", "0006_graph_rebuild_live_indexes")
    ]


def test_projection_migration_contains_only_four_authority_tables() -> None:
    migration = importlib.import_module(
        "apps.knowledge_graph.migrations.0007_memgraph_projection_authority"
    )

    assert all(
        isinstance(operation, migrations.CreateModel)
        for operation in migration.Migration.operations
    )
    assert {operation.name for operation in migration.Migration.operations} == {
        "CollectionGraphMembershipState",
        "CollectionGraphProjection",
        "ProjectionChunkReference",
        "GraphProjectionOutbox",
    }


def test_projection_migration_carries_required_constraints_and_indexes() -> None:
    migration = importlib.import_module(
        "apps.knowledge_graph.migrations.0007_memgraph_projection_authority"
    )
    constraints = {
        constraint.name
        for operation in migration.Migration.operations
        for constraint in operation.options.get("constraints", ())
    }
    indexes = {
        index.name
        for operation in migration.Migration.operations
        for index in operation.options.get("indexes", ())
    }

    assert {
        "kg_membership_one_collection",
        "kg_projection_generation_unique",
        "kg_projection_active_identity_unique",
        "kg_projection_nonnegative_counts",
        "kg_projection_lease_pair",
        "kg_projection_chunk_key_unique",
        "kg_projection_chunk_coordinate_unique",
        "kg_projection_outbox_operation_unique",
    } <= constraints
    assert {
        "kg_proj_state_updated_idx",
        "kg_projection_lease_idx",
        "kg_projection_outbox_due_idx",
    } <= indexes
