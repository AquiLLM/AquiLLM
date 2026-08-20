from __future__ import annotations

from types import SimpleNamespace

from apps.knowledge_graph.projection import runtime


def _projection_environment() -> dict[str, str]:
    return {
        "KG_MEMGRAPH_PROJECTION_ENABLED": "1",
        "KG_MEMGRAPH_URI": "bolt://graph.internal:7687",
        "KG_MEMGRAPH_DATABASE": "projection",
        "KG_MEMGRAPH_PROJECTION_USERNAME": "writer",
        "KG_MEMGRAPH_PROJECTION_PASSWORD": "writer-secret",
        "KG_PROJECTION_POSTGRES_SOURCE_DSN": (
            "postgresql://source_reader@source.internal/source"
        ),
        "KG_PROJECTION_POSTGRES_STATE_DSN": (
            "postgresql://state_writer@state.internal/state"
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
