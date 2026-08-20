from __future__ import annotations

import os

import pytest

pytestmark = [
    pytest.mark.django_db(transaction=True),
    pytest.mark.skipif(
        os.environ.get("KG_REQUIRE_POSTGRES_TESTS") != "1",
        reason="set KG_REQUIRE_POSTGRES_TESTS=1 for forced PostgreSQL race tests",
    ),
]


def test_projection_ready_and_membership_mutation_share_collection_first_lock_order():
    from django.db import connection

    if connection.vendor != "postgresql":
        pytest.fail("KG_REQUIRE_POSTGRES_TESTS requires PostgreSQL")
    from apps.knowledge_graph.projection.lifecycle import _READY_LOCK_ORDER

    assert _READY_LOCK_ORDER == (
        "collection",
        "active_artifact",
        "membership_state",
        "projection",
    )
