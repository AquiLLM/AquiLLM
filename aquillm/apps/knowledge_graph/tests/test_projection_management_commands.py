from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from io import StringIO
from types import SimpleNamespace

import pytest
from django.core.management.base import CommandError


def test_reconcile_command_is_dry_run_bounded_and_outputs_counts_only(monkeypatch):
    from apps.knowledge_graph.management.commands import (
        reconcile_knowledge_graph_projection as command_module,
    )

    observed = {}
    monkeypatch.setattr(
        command_module,
        "reconcile_graph_projections",
        lambda **kwargs: (
            observed.update(kwargs)
            or SimpleNamespace(
                examined_count=4,
                enqueued_count=0 if kwargs["dry_run"] else 4,
                drift_count=1,
                orphan_count=2,
                replayed_count=3,
            )
        ),
    )
    command = command_module.Command(stdout=StringIO())

    command.handle(collection=7, all=False, page_size=25, dry_run=True)

    payload = json.loads(command.stdout.getvalue())
    assert payload == {
        "drift_count": 1,
        "enqueued_count": 0,
        "examined_count": 4,
        "orphan_count": 2,
        "replayed_count": 3,
    }
    assert observed["collection_id"] == 7


def test_project_command_requires_exactly_collection_or_all(monkeypatch):
    from apps.knowledge_graph.management.commands import (
        inspect_knowledge_graph_projection,
        project_knowledge_graph,
    )

    command = project_knowledge_graph.Command(stdout=StringIO())
    inspect = inspect_knowledge_graph_projection.Command(stdout=StringIO())
    with pytest.raises(CommandError):
        command.handle(collection=None, all=False, dry_run=True, page_size=10)
    with pytest.raises(CommandError):
        command.handle(collection=1, all=True, dry_run=True, page_size=10)
    with pytest.raises(CommandError):
        command.handle(collection=0, all=False, dry_run=True, page_size=10)
    with pytest.raises(CommandError):
        inspect.handle(collection=0, all=False, page_size=10)


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
    observed = {}
    monkeypatch.setattr(
        prune_module,
        "prune_graph_projection_generations",
        lambda **kwargs: (
            observed.update(kwargs)
            or SimpleNamespace(candidate_count=3, deleted_count=0)
        ),
    )
    inspect = inspect_module.Command(stdout=StringIO())
    prune = prune_module.Command(stdout=StringIO())

    inspect.handle(collection=None, all=True, page_size=10)
    prune.handle(
        projection=None,
        collection=11,
        all=False,
        page_size=10,
        retain=2,
        dry_run=True,
    )

    output = inspect.stdout.getvalue() + prune.stdout.getvalue()
    assert "bolt://" not in output and "password" not in output
    assert "ready_count" in output and "candidate_count" in output
    assert observed["collection_id"] == 11


def test_reconcile_and_prune_commands_reject_invalid_selectors():
    from apps.knowledge_graph.management.commands import (
        prune_knowledge_graph_projection as prune_module,
    )
    from apps.knowledge_graph.management.commands import (
        reconcile_knowledge_graph_projection as reconcile_module,
    )

    reconcile = reconcile_module.Command(stdout=StringIO())
    prune = prune_module.Command(stdout=StringIO())

    with pytest.raises(CommandError):
        reconcile.handle(collection=None, all=False, page_size=10, dry_run=True)
    with pytest.raises(CommandError):
        reconcile.handle(collection=0, all=False, page_size=10, dry_run=True)
    with pytest.raises(CommandError):
        prune.handle(
            projection="NOT-A-UUID",
            collection=None,
            all=False,
            page_size=10,
            retain=2,
            dry_run=True,
        )


def test_command_page_and_retention_defaults_use_frozen_configuration(monkeypatch):
    from apps.knowledge_graph.management.commands import (
        inspect_knowledge_graph_projection as inspect_module,
    )
    from apps.knowledge_graph.management.commands import (
        project_knowledge_graph as project_module,
    )
    from apps.knowledge_graph.management.commands import (
        prune_knowledge_graph_projection as prune_module,
    )
    from apps.knowledge_graph.management.commands import (
        reconcile_knowledge_graph_projection as reconcile_module,
    )

    settings = SimpleNamespace(projection_batch_size=17, projection_retention=9)
    modules = (inspect_module, project_module, prune_module, reconcile_module)
    parsed = []
    for module in modules:
        monkeypatch.setattr(
            module, "load_projection_runtime_settings", lambda: settings, raising=False
        )
        parser = argparse.ArgumentParser()
        module.Command().add_arguments(parser)
        parsed.append(parser.parse_args([]))

    assert [value.page_size for value in parsed] == [17, 17, 17, 17]
    assert parsed[2].retain == 9


def test_project_command_routes_source_state_and_injects_configured_codec(monkeypatch):
    from apps.knowledge_graph.management.commands import (
        project_knowledge_graph as command_module,
    )

    observed = []

    class Query:
        def using(self, alias):
            observed.append(("source", alias))
            return self

        def filter(self, **_kwargs):
            return self

        def values_list(self, *_args, **_kwargs):
            return self

        def first(self):
            return 11

    settings = SimpleNamespace()
    codec = object()
    monkeypatch.setattr(command_module.GraphArtifact, "objects", Query())
    monkeypatch.setattr(
        command_module,
        "ProjectionDatabaseAliases",
        lambda: SimpleNamespace(source="projection_source", state="projection_state"),
    )
    monkeypatch.setattr(
        command_module, "load_projection_runtime_settings", lambda: settings
    )
    monkeypatch.setattr(
        command_module, "projection_identifier_codec", lambda value: codec
    )
    monkeypatch.setattr(
        command_module.transaction,
        "atomic",
        lambda *, using: observed.append(("state", using)) or nullcontext(),
    )
    monkeypatch.setattr(
        command_module,
        "enqueue_collection_projection_locked",
        lambda **kwargs: observed.append(("enqueue", kwargs)),
    )

    command_module.Command(stdout=StringIO()).handle(
        collection=7,
        all=False,
        dry_run=False,
        page_size=10,
    )

    assert observed == [
        ("source", "projection_source"),
        ("state", "projection_state"),
        (
            "enqueue",
            {
                "collection_id": 7,
                "artifact_id": 11,
                "using": "projection_state",
                "codec": codec,
            },
        ),
    ]
