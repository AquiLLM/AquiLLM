from __future__ import annotations

import uuid

import pytest

from apps.knowledge_graph.projection.django_projection_source import (
    DjangoProjectionRowSource,
)
from apps.knowledge_graph.projection.records import CollectionGraphProjectionBundleV1
from apps.knowledge_graph.tests.test_django_projection_source import _Loader, _snapshot


@pytest.mark.parametrize(
    ("state", "purpose"),
    (("ready", "audit"), ("superseded", "prune"), ("failed", "prune")),
)
def test_source_loads_immutable_bundle_for_explicit_lifecycle_purpose(
    state, purpose
) -> None:
    projection_id = uuid.uuid4()
    snapshot = _snapshot(projection_id, uuid.uuid4())
    snapshot["projection"]["state"] = state
    loader = _Loader(snapshot)
    source = DjangoProjectionRowSource(
        using="graph_reader",
        loader=loader,
        identifier_key=b"secret-a",
        identifier_key_version="key-v1",
    )

    rows = source.load_projection_rows(
        projection_id=projection_id,
        batch_size=37,
        purpose=purpose,
    )

    assert (
        type(CollectionGraphProjectionBundleV1(**rows))
        is CollectionGraphProjectionBundleV1
    )
    assert loader.calls == [(projection_id, 37, purpose)]


def test_source_rejects_ready_bundle_for_building_worker() -> None:
    projection_id = uuid.uuid4()
    snapshot = _snapshot(projection_id, uuid.uuid4())
    snapshot["projection"]["state"] = "ready"
    source = DjangoProjectionRowSource(
        using="graph_reader",
        loader=_Loader(snapshot),
        identifier_key=b"secret-a",
        identifier_key_version="key-v1",
    )

    with pytest.raises(ValueError, match="stale"):
        source.load_projection_rows(projection_id=projection_id, batch_size=37)
