from contextlib import contextmanager
from io import StringIO
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest
from django.core.management.base import CommandError

from apps.knowledge_graph.management.commands import certify_knowledge_graph_membership as command
from apps.knowledge_graph.projection import tasks
from apps.knowledge_graph.projection.operation_context import current_operation


@pytest.mark.parametrize("ready", [False, True])
def test_only_ready_dispatches_after_operation_has_closed(monkeypatch, ready):
    identifier = uuid4()
    monkeypatch.setattr(tasks, "project_generation", lambda **kwargs: SimpleNamespace(ready=ready, failure_code=None))
    def dispatch(value):
        assert value == identifier
        assert current_operation() is None
        raise TimeoutError("broker unavailable")
    dispatch = Mock(side_effect=dispatch)
    monkeypatch.setattr(tasks, "dispatch_membership_certificate", dispatch)
    assert tasks.project_knowledge_graph_projection.run(str(identifier)) == {"ready": ready, "failure_code": None}
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
    monkeypatch.setattr(command, "CollectionGraphProjection", SimpleNamespace(objects=query))
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
