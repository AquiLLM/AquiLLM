from contextlib import contextmanager
from io import StringIO
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import UUID, uuid4

import pytest
from django.core.management.base import CommandError

from apps.knowledge_graph.management.commands import (
    certify_knowledge_graph_membership as command,
)
from apps.knowledge_graph.projection import tasks
from apps.knowledge_graph.projection.operation_context import current_operation


@pytest.fixture(autouse=True)
def issuance_enabled(monkeypatch):
    monkeypatch.setattr(
        command,
        "load_projection_runtime_settings",
        lambda: SimpleNamespace(
            membership_certificate_enabled=True,
        ),
        raising=False,
    )


@pytest.mark.parametrize("ready", [False, True])
def test_only_ready_dispatches_after_operation_has_closed(monkeypatch, ready):
    identifier = uuid4()
    monkeypatch.setattr(
        tasks,
        "project_generation",
        lambda **kwargs: SimpleNamespace(ready=ready, failure_code=None),
    )

    def dispatch(value):
        assert value == identifier
        assert current_operation() is None
        raise TimeoutError("broker unavailable")

    dispatch = Mock(side_effect=dispatch)
    monkeypatch.setattr(tasks, "dispatch_membership_certificate", dispatch)
    assert tasks.project_knowledge_graph_projection.run(str(identifier)) == {
        "ready": ready,
        "failure_code": None,
    }
    assert dispatch.call_count == int(ready)


@pytest.mark.parametrize("limit", [0, 129, True, -1])
def test_command_rejects_unbounded_limit(limit):
    with pytest.raises(CommandError):
        command.Command().handle(limit=limit)


def test_command_bounds_selection_and_redacts_failure(monkeypatch):
    identifiers = [uuid4(), uuid4(), uuid4()]
    query = Mock()
    query.using.return_value = query
    query.filter.return_value = query
    query.order_by.return_value = query
    query.values_list.return_value = identifiers
    monkeypatch.setattr(
        command, "CollectionGraphProjection", SimpleNamespace(objects=query)
    )

    @contextmanager
    def database(*args, **kwargs):
        yield

    monkeypatch.setattr(command, "projection_database", database)
    certify = Mock(side_effect=[True, RuntimeError("raw source secret")])
    monkeypatch.setattr(command, "certify_ready_membership", certify)
    output = StringIO()
    command.Command(stdout=output).handle(limit=2)
    assert certify.call_count == 2
    assert "examined=2 certified=1 failed=1" in output.getvalue()
    assert "raw source" not in output.getvalue()


def test_disabled_command_reports_skip_without_query(monkeypatch):
    monkeypatch.setattr(
        command, "load_projection_runtime_settings", lambda: SimpleNamespace()
    )
    monkeypatch.setattr(command, "CollectionGraphProjection", None)
    output = StringIO()
    command.Command(stdout=output).handle(limit=2)
    assert "failure_code=certificate_disabled" in output.getvalue()
    assert "examined=0 certified=0 failed=0" in output.getvalue()


@pytest.mark.parametrize(
    "cursor", ["bad", "00000000-0000-0000-0000-0000000000AB", True]
)
def test_command_rejects_noncanonical_cursor(cursor):
    with pytest.raises(CommandError):
        command.Command().handle(limit=2, after_projection_id=cursor)


def test_command_cursor_traverses_past_first_bounded_page(monkeypatch):
    identifiers = [UUID(int=i) for i in range(1, 6)]

    class Query:
        def using(self, alias):
            return self

        def filter(self, **kwargs):
            self.after = kwargs.get("pk__gt")
            return self

        def order_by(self, *args):
            return self

        def values_list(self, *args, **kwargs):
            return [i for i in identifiers if self.after is None or i > self.after]

    monkeypatch.setattr(
        command, "CollectionGraphProjection", SimpleNamespace(objects=Query())
    )

    @contextmanager
    def database(*args, **kwargs):
        yield

    monkeypatch.setattr(command, "projection_database", database)
    visited = []

    def certify(**kwargs):
        visited.append(kwargs["projection_id"])
        return False

    monkeypatch.setattr(command, "certify_ready_membership", certify)
    first, second = StringIO(), StringIO()
    command.Command(stdout=first).handle(limit=2)
    assert f"next_after_projection_id={identifiers[1]}" in first.getvalue()
    command.Command(stdout=second).handle(
        limit=2, after_projection_id=str(identifiers[1])
    )
    assert visited == identifiers[:4]
    assert "not_certified=2" in second.getvalue()
    assert f"next_after_projection_id={identifiers[3]}" in second.getvalue()
