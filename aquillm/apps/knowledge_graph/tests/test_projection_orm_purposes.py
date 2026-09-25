from __future__ import annotations

from uuid import uuid4

from apps.knowledge_graph.projection import django_projection_rows as rows


class _Query:
    def __init__(self, result, calls, family):
        self._result = result
        self._calls = calls
        self._family = family

    def using(self, alias):
        self._calls.append((self._family, "using", alias))
        return self

    def filter(self, **kwargs):
        self._calls.append((self._family, "filter", kwargs))
        return self

    def values(self, *_fields):
        return self

    def get(self):
        value = self._result.pop(0) if type(self._result) is list else self._result
        return dict(value)


def _projection(projection_id, state):
    return {
        "id": projection_id,
        "generation_key": uuid4(),
        "collection_id": 7,
        "collection_pk_snapshot": 7,
        "artifact_id": 11,
        "artifact_pk_snapshot": 11,
        "schema_version": "collection-graph-v1",
        "projection_version": "projection-v1",
        "identifier_key_version": "key-v7",
        "membership_epoch": 3,
        "membership_checksum": "a" * 64,
        "state": state,
    }


def test_default_orm_loader_routes_ready_audit_and_terminal_prune(monkeypatch):
    projection_id = uuid4()
    calls = []
    projection_rows = [
        _projection(projection_id, "ready"),
        _projection(projection_id, "superseded"),
    ]
    monkeypatch.setattr(
        rows.CollectionGraphProjection,
        "objects",
        _Query(projection_rows, calls, "projection"),
    )
    monkeypatch.setattr(
        rows.GraphArtifact,
        "objects",
        _Query({"scope_id": "7"}, calls, "artifact"),
    )
    monkeypatch.setattr(
        rows.CollectionGraphMembershipState,
        "objects",
        _Query(
            {
                "active_artifact_id": 11,
                "registry_epoch": 3,
                "membership_checksum": "a" * 64,
            },
            calls,
            "membership",
        ),
    )
    loader = rows.DjangoProjectionOrmLoader(
        "projection_source",
        state_using="projection_state",
    )

    ready = loader._projection(projection_id, "audit")
    terminal = loader._projection(projection_id, "prune")

    assert ready["state"] == "ready"
    assert terminal["state"] == "superseded"
    assert [call for call in calls if call[1] == "using"] == [
        ("projection", "using", "projection_state"),
        ("artifact", "using", "projection_source"),
        ("membership", "using", "projection_state"),
        ("projection", "using", "projection_state"),
        ("artifact", "using", "projection_source"),
    ]
    state_filters = [
        call[2]["state__in"]
        for call in calls
        if call[0] == "projection" and call[1] == "filter"
    ]
    assert state_filters == [("ready",), ("failed", "superseded")]
    artifact_filters = [
        call[2]["status__in"]
        for call in calls
        if call[0] == "artifact" and call[1] == "filter"
    ]
    assert artifact_filters == [("active",), ("active", "superseded")]
