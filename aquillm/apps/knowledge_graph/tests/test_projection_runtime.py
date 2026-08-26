from __future__ import annotations

from types import SimpleNamespace

import pytest

from apps.knowledge_graph.projection import runtime


def _projection_environment() -> dict[str, str]:
    return {
        "KG_MEMGRAPH_PROJECTION_ENABLED": "1",
        "KG_MEMGRAPH_URI": "bolt://graph.internal:7687",
        "KG_MEMGRAPH_DATABASE": "projection",
        "KG_MEMGRAPH_PROJECTION_USERNAME": "writer",
        "KG_MEMGRAPH_PROJECTION_PASSWORD": "writer-secret",
        "KG_PROJECTION_POSTGRES_SOURCE_DSN": (
            "postgresql://aquillm_projection_source@source.internal/source"
        ),
        "KG_PROJECTION_POSTGRES_STATE_DSN": (
            "postgresql://aquillm_projection_state@state.internal/state"
        ),
        "KG_PROJECTION_IDENTIFIER_HMAC_KEY": "identifier-secret",
        "KG_PROJECTION_IDENTIFIER_KEY_VERSION": "key-v7",
        "KG_PROJECTION_SCHEMA_VERSION": "collection-graph-v1",
        "KG_PROJECTION_FORMAT_VERSION": "projection-v1",
        "KG_PROJECTION_QUEUE": "projection-control",
        "KG_PROJECTION_BATCH_SIZE": "37",
        "KG_PROJECTION_LEASE_SECONDS": "41",
        "KG_PROJECTION_MAX_ATTEMPTS": "7",
        "KG_PROJECTION_RETENTION": "3",
    }


def _projection_hook_environment() -> dict[str, str]:
    return {
        "KG_MEMGRAPH_PROJECTION_ENABLED": "0",
        "KG_MEMGRAPH_PROJECTION_HOOK_ENABLED": "1",
        "KG_PROJECTION_IDENTIFIER_HMAC_KEY": "identifier-secret",
        "KG_PROJECTION_IDENTIFIER_KEY_VERSION": "key-v7",
        "KG_PROJECTION_SCHEMA_VERSION": "collection-graph-v1",
        "KG_PROJECTION_FORMAT_VERSION": "projection-v1",
    }


def test_runtime_consumes_only_frozen_projection_configuration_names() -> None:
    source = {**_projection_environment(), "KG_BUILD_ENABLED": "1"}

    settings = runtime.load_projection_runtime_settings(source)

    assert settings.projection_schema_version == "collection-graph-v1"
    assert settings.projection_format_version == "projection-v1"
    assert settings.projection_identifier_key_version == "key-v7"
    assert settings.projection_batch_size == 37
    assert settings.projection_lease_seconds == 41
    assert settings.projection_queue == "projection-control"


def test_memgraph_factory_uses_projection_credentials_and_fixed_database(
    monkeypatch,
) -> None:
    settings = runtime.load_projection_runtime_settings(_projection_environment())
    observed: dict[str, object] = {}

    def driver(uri, username, password, *, database):
        observed.update(
            uri=uri,
            username=username,
            password=password,
            database=database,
        )
        return SimpleNamespace(execute_read=lambda *_a, **_k: ())

    monkeypatch.setattr(runtime, "Neo4jMemgraphDriver", driver)
    monkeypatch.setattr(runtime, "MemgraphProjectionRepository", lambda value: value)

    runtime.memgraph_projection_repository(settings)

    assert observed == {
        "uri": "bolt://graph.internal:7687",
        "username": "writer",
        "password": "writer-secret",
        "database": "projection",
    }


def test_postgres_factory_rejects_missing_function_state_repository_before_io(
    monkeypatch,
) -> None:
    settings = runtime.load_projection_runtime_settings(_projection_environment())
    constructed = []

    def forbidden(*_args, **_kwargs):
        constructed.append("direct-store")
        raise AssertionError("direct projection-state ORM store was constructed")

    monkeypatch.setattr(
        runtime, "DjangoChunkReferenceStore", forbidden, raising=False
    )
    monkeypatch.setattr(runtime, "DjangoProjectionRowSource", forbidden)

    with pytest.raises(RuntimeError, match="function state repository is required"):
        runtime.postgres_projection_repository(settings)

    assert constructed == []


def test_postgres_factory_rejects_nonexact_state_repository_before_io(
    monkeypatch,
) -> None:
    settings = runtime.load_projection_runtime_settings(_projection_environment())

    def forbidden_source(*_args, **_kwargs):
        pytest.fail("source constructed before authority check")

    monkeypatch.setattr(
        runtime,
        "DjangoProjectionRowSource",
        forbidden_source,
    )

    with pytest.raises(TypeError, match="exact function state repository"):
        runtime.postgres_projection_repository(settings, state_repository=object())


def test_postgres_factory_injects_frozen_hmac_versions_and_function_repository(
    monkeypatch,
) -> None:
    from apps.knowledge_graph.projection.state_repository import (
        FunctionProjectionStateRepository,
    )

    settings = runtime.load_projection_runtime_settings(_projection_environment())
    aliases = runtime.ProjectionDatabaseAliases(
        source="projection_source",
        state="projection_state",
    )
    observed: dict[str, object] = {}

    def source(using, **kwargs):
        observed["source"] = (using, kwargs)
        return "source"

    def repository(**kwargs):
        observed["repository"] = kwargs
        return "repository"

    monkeypatch.setattr(runtime, "DjangoProjectionRowSource", source, raising=False)
    monkeypatch.setattr(
        runtime, "PostgresProjectionRepository", repository, raising=False
    )
    function_repository = FunctionProjectionStateRepository(owner="worker-1")

    result = runtime.postgres_projection_repository(
        settings,
        aliases=aliases,
        state_repository=function_repository,
    )

    assert result == "repository"
    assert observed == {
        "source": (
            "projection_source",
            {
                "state_using": "projection_source",
                "identifier_key": b"identifier-secret",
                "identifier_key_version": "key-v7",
                "schema_version": "collection-graph-v1",
                "projection_version": "projection-v1",
            },
        ),
        "repository": {
            "using": "projection_source",
            "source": "source",
            "chunk_store": function_repository,
        },
    }


def test_activation_projection_hook_is_default_off_and_fail_open(monkeypatch) -> None:
    called = []
    monkeypatch.setattr(
        runtime,
        "enqueue_collection_projection_locked",
        lambda **kwargs: called.append(kwargs),
        raising=False,
    )

    assert runtime.enqueue_activated_collection_projection(7, 9, source={}) is False
    assert called == []


def test_enabled_activation_injects_frozen_membership_hmac_on_web_alias(
    monkeypatch,
) -> None:
    from apps.knowledge_graph.projection import lifecycle

    observed = {}
    monkeypatch.setattr(
        lifecycle,
        "enqueue_collection_projection_locked",
        lambda **kwargs: observed.update(kwargs),
    )
    monkeypatch.setattr(
        runtime.transaction,
        "on_commit",
        lambda _callback, **_kwargs: None,
    )

    enabled = runtime.enqueue_activated_collection_projection(
        7,
        9,
        source=_projection_hook_environment(),
    )

    assert enabled is True
    assert observed["using"] == "default"
    assert observed["codec"].key_version == "key-v7"


def test_enabled_activation_dispatches_projection_outbox_after_commit(
    monkeypatch,
) -> None:
    from apps.knowledge_graph.projection import lifecycle, tasks

    callbacks = []
    dispatched = []
    monkeypatch.setattr(
        lifecycle,
        "enqueue_collection_projection_locked",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        runtime.transaction,
        "on_commit",
        lambda callback, **_kwargs: callbacks.append(callback),
    )
    monkeypatch.setattr(
        tasks.reconcile_knowledge_graph_projections,
        "delay",
        lambda **kwargs: dispatched.append(kwargs),
    )

    assert runtime.enqueue_activated_collection_projection(
        7,
        9,
        source=_projection_hook_environment(),
    )
    assert dispatched == []
    assert len(callbacks) == 1

    callbacks[0]()

    assert dispatched == [{"collection_id": 7}]


def test_worker_postgres_factory_uses_the_live_configured_repository(
    monkeypatch,
) -> None:
    from apps.knowledge_graph.projection import worker

    settings = runtime.load_projection_runtime_settings(_projection_environment())
    repository = object()
    observed = []
    monkeypatch.setattr(worker, "_projection_settings", lambda: settings)
    state_repository = object()

    def factory(value, *, state_repository):
        observed.append((value, state_repository))
        return repository

    monkeypatch.setattr(worker, "postgres_projection_repository", factory)
    token = worker._STATE_REPOSITORY.set(state_repository)
    try:
        assert worker._postgres_repository() is repository
    finally:
        worker._STATE_REPOSITORY.reset(token)
    assert observed == [(settings, state_repository)]
    assert worker._state_using() == "projection_state"


def test_identifier_codec_factory_can_frame_a_persisted_key_version() -> None:
    environment = {
        **_projection_environment(),
        "KG_PROJECTION_IDENTIFIER_KEY_VERSION": "key-v2",
    }
    settings = runtime.load_projection_runtime_settings(environment)

    codec = runtime.projection_identifier_codec(settings, key_version="key-v1")

    assert codec.key_version == "key-v1"
    assert runtime.projection_identifier_codec(settings).key_version == "key-v2"


@pytest.mark.parametrize(
    ("value", "error"),
    ((1, TypeError), ("", ValueError), (" key-v1", ValueError)),
)
def test_identifier_codec_factory_rejects_noncanonical_version_without_secret(
    value, error
) -> None:
    settings = runtime.load_projection_runtime_settings(_projection_environment())

    with pytest.raises(error) as captured:
        runtime.projection_identifier_codec(settings, key_version=value)

    assert "identifier-secret" not in repr(captured.value)
