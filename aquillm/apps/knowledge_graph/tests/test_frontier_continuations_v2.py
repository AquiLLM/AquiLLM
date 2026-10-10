"""Private continuation bindings reject cross-request and cross-round pages."""

from dataclasses import replace

import pytest

from apps.knowledge_graph.projection.frontier_adapter_v2 import FrontierSessionV2
from apps.knowledge_graph.retrieval.topology.failures import TopologyLoadError
from apps.knowledge_graph.retrieval.topology.frontier_contracts_v2 import (
    FrontierCursorV2,
)
from apps.knowledge_graph.tests.frontier_fixture_support import FrontierFixture
from apps.knowledge_graph.tests.frontier_index_fixture_driver import (
    FrontierIndexFixtureDriver,
)


@pytest.mark.parametrize(
    "field,value",
    [
        ("request_digest", "a" * 64),
        ("generation_key", "b" * 64),
        ("depth", 1),
    ],
)
def test_private_cursor_rejects_wrong_request_generation_or_depth(field, value):
    f = FrontierFixture()
    session = FrontierSessionV2(
        FrontierIndexFixtureDriver(f.bundles),
        group=f.group,
        request=f.request,
        budget=f.budget(),
        deadline=f.deadline,
    )
    generation = f.bundles[0].generation.generation_key
    cursor = FrontierCursorV2(session.request_digest, generation, 0, ("a",))
    session.require_cursor(cursor, generation=generation, depth=0)
    with pytest.raises(TopologyLoadError):
        session.require_cursor(
            replace(cursor, **{field: value}), generation=generation, depth=0
        )
