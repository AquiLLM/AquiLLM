"""Regression tests for function-backed projection state ordering."""

from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import UUID

import pytest

from apps.knowledge_graph.projection import state_repository


def test_stale_ready_validation_never_reads_source_or_calls_state(monkeypatch) -> None:
    class ForbiddenQuery:
        def using(self, _alias):
            raise AssertionError("stale validation opened projection source")

    monkeypatch.setattr(
        state_repository.CollectionGraphProjection,
        "objects",
        ForbiddenQuery(),
    )
    repository = state_repository.FunctionProjectionStateRepository()
    monkeypatch.setattr(
        repository,
        "_one",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("stale validation called projection state")
        ),
    )
    validation = SimpleNamespace(
        generation_key="b" * 64,
        validation_checksum="c" * 64,
        valid=True,
        counts=SimpleNamespace(),
    )

    with pytest.raises(ValueError, match="stale"):
        repository.ready(
            projection_id=UUID("11111111-1111-4111-8111-111111111111"),
            owner="worker-a",
            validation=validation,
            expected_generation_key="d" * 64,
            expected_graph_checksum="c" * 64,
            expected_private_mapping_checksum="e" * 64,
            now=datetime(2026, 8, 20, tzinfo=UTC),
            versions=("collection-graph-v1", "projection-v1", "task21-key-v1"),
        )
