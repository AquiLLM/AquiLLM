from __future__ import annotations

import json
from io import StringIO
from types import SimpleNamespace

import pytest
from django.core.management.base import CommandError


def test_reconcile_command_is_dry_run_bounded_and_outputs_counts_only(monkeypatch):
    from apps.knowledge_graph.management.commands import (
        reconcile_knowledge_graph_projection as command_module,
    )

    monkeypatch.setattr(
        command_module,
        "reconcile_graph_projections",
        lambda **kwargs: SimpleNamespace(
            examined_count=4,
            enqueued_count=0 if kwargs["dry_run"] else 4,
        ),
    )
    command = command_module.Command(stdout=StringIO())

    command.handle(page_size=25, dry_run=True)

    payload = json.loads(command.stdout.getvalue())
    assert payload == {"examined_count": 4, "enqueued_count": 0}


def test_project_command_requires_exactly_collection_or_all(monkeypatch):
    from apps.knowledge_graph.management.commands import project_knowledge_graph

    command = project_knowledge_graph.Command(stdout=StringIO())
    with pytest.raises(CommandError):
        command.handle(collection=None, all=False, dry_run=True, page_size=10)
    with pytest.raises(CommandError):
        command.handle(collection=1, all=True, dry_run=True, page_size=10)


def test_inspect_and_prune_commands_never_emit_private_backend_values(monkeypatch):
    from apps.knowledge_graph.management.commands import (
        inspect_knowledge_graph_projection as inspect_module,
    )
    from apps.knowledge_graph.management.commands import (
        prune_knowledge_graph_projection as prune_module,
    )

    monkeypatch.setattr(
        inspect_module,
        "inspect_projection_authority",
        lambda **_kwargs: {"ready_count": 2, "drift_count": 1},
    )
    monkeypatch.setattr(
        prune_module,
        "prune_graph_projection_generations",
        lambda **_kwargs: SimpleNamespace(candidate_count=3, deleted_count=0),
    )
    inspect = inspect_module.Command(stdout=StringIO())
    prune = prune_module.Command(stdout=StringIO())

    inspect.handle(collection=None, all=True, page_size=10)
    prune.handle(page_size=10, retain=2, dry_run=True)

    output = inspect.stdout.getvalue() + prune.stdout.getvalue()
    assert "bolt://" not in output and "password" not in output
    assert "ready_count" in output and "candidate_count" in output
