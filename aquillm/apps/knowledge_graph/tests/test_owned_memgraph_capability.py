"""Owned-service gates and fixture isolation without a host Docker dependency."""

from dataclasses import replace
import os

import pytest

from scripts.rag_remediation_isolation import test_environment as environment_values
from scripts.run_rag_remediation_tests import IsolationError
from apps.knowledge_graph.projection.runtime import load_projection_runtime_settings
from apps.knowledge_graph.tests import owned_memgraph_test_support as owned


def test_valid_owned_settings_and_legacy_opt_in_are_distinct():
    environment = environment_values()
    settings = load_projection_runtime_settings(environment)
    assert owned.owned_target(environment, settings) == {
        "uri": "bolt://memgraph:7687", "database": "memgraph",
    }
    assert owned.owned_target({"KG_REQUIRE_MEMGRAPH_TESTS": "1"}, object()) is None


@pytest.mark.parametrize("change", [
    {"RAG_REMEDIATION_TEST": "0"}, {"POSTGRES_HOST": "foreign"},
    {"KG_MEMGRAPH_URI": "bolt://foreign:7687"},
    {"KG_MEMGRAPH_IMAGE": "memgraph/memgraph-mage:latest"},
    {"UNEXPECTED_URL": "bolt://foreign:7687"},
])
def test_environment_mismatch_cannot_create_a_client(monkeypatch, change):
    environment = {**environment_values(), **change}
    settings = load_projection_runtime_settings(environment_values())
    monkeypatch.setattr(owned, "_current_target", lambda: owned.owned_target(environment, settings))
    calls = []
    monkeypatch.setattr(owned, "_connect", lambda target: calls.append(target))
    with pytest.raises(IsolationError):
        with owned.owned_memgraph():
            pytest.fail("unsafe target admitted")
    assert calls == []


@pytest.mark.parametrize("field,value", [
    ("memgraph_uri", "bolt://foreign:7687"), ("memgraph_database", "foreign"),
    ("memgraph_image", "memgraph/memgraph-mage:latest"),
    ("memgraph_projection_username", "foreign"),
])
def test_effective_setting_mismatch_is_rejected(field, value):
    environment = environment_values()
    settings = replace(load_projection_runtime_settings(environment), **{field: value})
    with pytest.raises(IsolationError):
        owned.owned_target(environment, settings)


def test_missing_marker_and_changed_password_are_rejected():
    environment = environment_values()
    settings = load_projection_runtime_settings(environment)
    del environment["RAG_REMEDIATION_TEST"]
    with pytest.raises(IsolationError):
        owned.owned_target(environment, settings)
    environment = environment_values()
    settings = load_projection_runtime_settings({
        **environment, "KG_MEMGRAPH_PROJECTION_PASSWORD": "changed",
    })
    with pytest.raises(IsolationError):
        owned.owned_target(environment, settings)


class Rows(list):
    def consume(self):
        return None


class Client:
    def __init__(self):
        self.closed, self.calls = 0, []
        self.dirty = {"constraints", "indexes", "nodes"}
        self.version = [{"version": "3.8.1"}]

    def session(self, **kwargs):
        assert kwargs == {"database": "memgraph"}
        return self

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def close(self):
        self.closed += 1

    def run(self, statement):
        self.calls.append(statement)
        if statement == "SHOW VERSION":
            return Rows(self.version)
        drops = {"DROP ALL CONSTRAINTS": "constraints", "DROP ALL INDEXES": "indexes",
                 "MATCH (n) DETACH DELETE n": "nodes"}
        probes = {"SHOW CONSTRAINT INFO": "constraints", "SHOW INDEX INFO": "indexes",
                  "MATCH (n) RETURN n LIMIT 1": "nodes"}
        if statement in drops:
            self.dirty.discard(drops[statement])
            return Rows()
        assert statement in probes
        return Rows([{"remaining": 1}] if probes[statement] in self.dirty else [])


@pytest.fixture
def client(monkeypatch):
    value = Client()
    monkeypatch.setattr(owned, "_current_target", lambda: {
        "uri": "bolt://memgraph:7687", "database": "memgraph",
    })
    monkeypatch.setattr(owned, "_connect", lambda target: value)
    monkeypatch.setattr(owned, "_run", lambda session, statement: session.run(statement))
    return value


@pytest.mark.parametrize("body_fails", [False, True])
def test_context_resets_data_and_schema_before_after_and_closes(client, body_fails):
    class BodyFailure(Exception):
        pass

    try:
        with owned.owned_memgraph() as target:
            assert target["uri"] == "bolt://memgraph:7687"
            assert client.dirty == set()
            client.dirty.update(("constraints", "indexes", "nodes"))
            if body_fails:
                raise BodyFailure()
    except BodyFailure:
        assert body_fails
    assert client.dirty == set()
    assert client.closed == 1
    assert client.calls.count("DROP ALL INDEXES") == 2
    assert not owned._ACTIVE.locked()


@pytest.mark.parametrize("version", [[], [{"version": "3.8.0"}], [{"version": "3.8.1"}] * 2])
def test_version_mismatch_closes_before_any_reset(client, version):
    client.version = version
    with pytest.raises(IsolationError):
        with owned.owned_memgraph():
            pytest.fail("wrong server admitted")
    assert client.calls == ["SHOW VERSION"]
    assert client.closed == 1
    assert not owned._ACTIVE.locked()


def test_context_does_not_redirect_cleanup_or_admit_overlap(client, monkeypatch):
    with pytest.raises(IsolationError, match="changed before cleanup"):
        with owned.owned_memgraph():
            with pytest.raises(IsolationError, match="must not overlap"):
                with owned.owned_memgraph():
                    pytest.fail("overlapping reset")
            monkeypatch.setattr(owned, "_current_target", lambda: None)
    assert client.closed == 1
    assert client.calls.count("DROP ALL INDEXES") == 1
    assert not owned._ACTIVE.locked()


@pytest.mark.parametrize("failure_call", [1, 2])
def test_reset_failure_still_closes_and_releases_fixture_lock(client, monkeypatch, failure_call):
    original = owned._reset
    calls = []

    def failing_reset(value):
        calls.append(value)
        if len(calls) == failure_call:
            raise RuntimeError("reset failed")
        original(value)

    monkeypatch.setattr(owned, "_reset", failing_reset)
    with pytest.raises(RuntimeError, match="reset failed"):
        with owned.owned_memgraph():
            pass
    assert len(calls) == failure_call
    assert client.closed == 1
    assert not owned._ACTIVE.locked()


def test_historical_settings_are_real_parser_local_and_require_admitted_context(client, monkeypatch):
    from types import SimpleNamespace

    environment = environment_values()
    before = environment.copy()
    monkeypatch.setattr(owned, "os", SimpleNamespace(environ=environment))
    target = {"uri": "bolt://memgraph:7687", "database": "memgraph"}
    with pytest.raises(IsolationError):
        owned.owned_projection_settings(target)
    with owned.owned_memgraph() as admitted:
        settings = owned.owned_projection_settings(admitted)
        assert settings.memgraph_projection_enabled is True
        assert settings.memgraph_uri == target["uri"]
        assert settings.memgraph_database == target["database"]
        assert settings.memgraph_projection_username == "rag_projection"
        assert settings.memgraph_projection_password.get_secret_value() == "rag-synthetic-password"
    assert environment == before


def test_live_owned_contexts_do_not_share_graph_indexes_or_constraints():
    if os.environ.get("RAG_REMEDIATION_TEST") != "1":
        pytest.skip("requires sanctioned owned Memgraph service")
    with owned.owned_memgraph() as target:
        with owned._connect(target) as client, client.session(database="memgraph") as session:
            for statement in (
                "CREATE INDEX ON :OwnedResetProbe(key)",
                "CREATE CONSTRAINT ON (n:OwnedResetProbe) ASSERT n.key IS UNIQUE",
                "CREATE (:OwnedResetProbe {key: 'first-context'})",
            ):
                owned._run(session, statement).consume()
    with owned.owned_memgraph() as target:
        with owned._connect(target) as client, client.session(database="memgraph") as session:
            for statement in ("SHOW INDEX INFO", "SHOW CONSTRAINT INFO",
                              "MATCH (n) RETURN n LIMIT 1"):
                assert next(iter(owned._run(session, statement)), None) is None
