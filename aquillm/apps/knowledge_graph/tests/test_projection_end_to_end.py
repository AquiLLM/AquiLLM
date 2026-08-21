"""Projection lifecycle and live least-privilege acceptance proofs."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import UUID

import pytest

from apps.knowledge_graph.projection.state_repository import (
    FunctionProjectionStateRepository,
)
from apps.knowledge_graph.retrieval.direct_seed_contracts import DirectResolutionTier
from apps.knowledge_graph.retrieval.direct_seed_resolution import (
    resolve_direct_seed_components,
)
from apps.knowledge_graph.tests.test_direct_seed_resolution import (
    Repository,
    _match,
    _ready,
    _settings,
)
from lib.knowledge_graph.query_extractor.contracts import QueryEntitySpanV1

P1 = UUID("11111111-1111-4111-8111-111111111111")
P2 = UUID("22222222-2222-4222-8222-222222222222")
NOW = datetime(2026, 8, 20, tzinfo=UTC)
VERSIONS = ("collection-graph-v1", "projection-v1", "task21-key-v1")


def test_function_state_replay_creates_two_generations_and_rejects_stale_source(
    monkeypatch,
) -> None:
    repository = FunctionProjectionStateRepository()
    calls: list[tuple[str, tuple[object, ...]]] = []
    responses = iter(
        (
            {"projection_id": P1},
            {"projection_id": P2},
            None,
        )
    )

    def one(operation, parameters):
        calls.append((operation, parameters))
        return next(responses)

    monkeypatch.setattr(repository, "_one", one)
    first = repository.replay(
        projection_id=None,
        collection_id=7,
        artifact_id=11,
        versions=VERSIONS,
        now=NOW,
    )
    second = repository.replay(
        projection_id=first,
        collection_id=7,
        artifact_id=12,
        versions=VERSIONS,
        now=NOW,
    )

    assert (first, second) == (P1, P2)
    assert [call[0] for call in calls] == ["replay", "replay"]
    assert calls[0][1][0] is None and calls[1][1][0] == P1
    assert calls[0][1][2:7] == (7, 11, *VERSIONS)
    assert calls[1][1][2:7] == (7, 12, *VERSIONS)
    with pytest.raises(RuntimeError, match="stale"):
        repository.replay(
            projection_id=second,
            collection_id=7,
            artifact_id=13,
            versions=VERSIONS,
            now=NOW,
        )


def test_stale_validation_is_rejected_before_any_state_function(monkeypatch) -> None:
    from apps.knowledge_graph.projection import state_repository

    class Query:
        def using(self, _alias):
            return self

        def values(self, *_fields):
            return self

        def get(self, **_kwargs):
            return {
                "generation_key": P1,
                "collection_id": 7,
                "artifact_id": 11,
                "membership_epoch": 2,
                "membership_checksum": "a" * 64,
            }

    monkeypatch.setattr(state_repository.CollectionGraphProjection, "objects", Query())
    repository = FunctionProjectionStateRepository()
    calls = []
    monkeypatch.setattr(
        repository,
        "_one",
        lambda *args: calls.append(args) or {
            "published": True,
            "state": "ready",
            "failure_code": None,
        },
    )
    validation = SimpleNamespace(
        generation_key="b" * 64,
        validation_checksum="c" * 64,
        valid=True,
        counts=SimpleNamespace(
            entity_count=2,
            relation_semantics_count=1,
            relation_count=1,
            evidence_count=1,
            entity_mention_count=1,
            chunk_count=1,
        ),
    )

    with pytest.raises(ValueError, match="stale"):
        repository.ready(
            projection_id=P1,
            owner="worker-a",
            validation=validation,
            expected_generation_key="d" * 64,
            expected_graph_checksum="c" * 64,
            expected_private_mapping_checksum="e" * 64,
            now=NOW,
            versions=VERSIONS,
        )
    assert calls == []


def test_indexed_alias_fixture_resolves_to_the_opaque_automatic_component() -> None:
    span = QueryEntitySpanV1("person", 0, 5, 1.0)
    repository = Repository(
        {
            ("alias", 0): (
                _match(
                    span=0,
                    entity="3" * 64,
                    component="4" * 64,
                    tier=DirectResolutionTier.ALIAS,
                ),
            )
        }
    )

    outcome = resolve_direct_seed_components(
        spans=(span,),
        repository=repository,
        ready=_ready(),
        settings=_settings(),
        deadline=10.0,
    )

    assert outcome.matches[0].tier is DirectResolutionTier.ALIAS
    assert outcome.seeds[0].component_key == "4" * 64
    assert repository.calls == [("identifier", 0), ("name", 0), ("alias", 0)]


@pytest.mark.container
@pytest.mark.skipif(
    os.environ.get("KG_REQUIRE_CONTAINER_TESTS") != "1",
    reason="Task22 live PostgreSQL role proof runs only in the host harness",
)
def test_live_postgres_roles_are_source_read_only_and_state_function_only() -> None:
    import psycopg

    source_dsn = os.environ["KG_PROJECTION_POSTGRES_SOURCE_DSN"]
    state_dsn = os.environ["KG_PROJECTION_POSTGRES_STATE_DSN"]
    with psycopg.connect(source_dsn, autocommit=True) as source:
        source.execute("SELECT 1 FROM public.aquillm_collection LIMIT 1")
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            source.execute("UPDATE public.aquillm_collection SET id=id WHERE false")
    with psycopg.connect(state_dsn, autocommit=True) as state:
        assert (
            state.execute(
                "SELECT * FROM public.kg_projection_claim(%s,%s,CURRENT_TIMESTAMP,%s)",
                (P1, "task22-live", 30),
            ).fetchone()
            is None
        )
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            state.execute(
                "UPDATE public.apps_knowledge_graph_collectiongraphprojection "
                "SET state=state WHERE false"
            )
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            state.execute(
                "SELECT 1 FROM public.apps_knowledge_graph_collectiongraphprojection"
            )
