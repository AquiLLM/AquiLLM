from __future__ import annotations

import importlib
import inspect
import json
import os
import socket
import subprocess
import sys
import textwrap
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from io import StringIO
from threading import Barrier, Event
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from django.core.management import CommandError, call_command
from django.test import override_settings

REQUEST_ID = uuid.UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
DOCUMENT_ID = uuid.UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")


def test_pytest_runner_is_an_explicit_test_environment() -> None:
    from django.conf import settings

    assert settings.TESTING is True


@pytest.mark.parametrize(
    ("configured", "expected"),
    ((None, "5432"), ("55432", "55432"), ("0", "5432"), ("invalid", "5432")),
)
def test_postgres_port_setting_has_a_bounded_environment_override(
    configured: str | None,
    expected: str,
) -> None:
    repository_root = __import__("pathlib").Path(__file__).resolve().parents[4]
    environment = os.environ.copy()
    if configured is None:
        environment.pop("POSTGRES_PORT", None)
    else:
        environment["POSTGRES_PORT"] = configured
    environment["DJANGO_DEBUG"] = "1"
    environment["PYTHONPATH"] = str(repository_root / "aquillm")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from aquillm import settings; "
                "print(settings.DATABASES['default']['PORT'])"
            ),
        ],
        cwd=repository_root / "aquillm",
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == expected


def _database_is_reachable() -> bool:
    from django.conf import settings

    database = settings.DATABASES["default"]
    try:
        with socket.create_connection(
            (database["HOST"], int(database.get("PORT") or 5432)), timeout=0.2
        ):
            return True
    except OSError:
        return False


database_required = pytest.mark.skipif(
    not _database_is_reachable() and os.environ.get("KG_REQUIRE_POSTGRES_TESTS") != "1",
    reason="configured PostgreSQL database is not reachable",
)


@pytest.mark.django_db(transaction=True)
@database_required
def test_concurrent_same_uuid_creation_resumes_one_durable_request(
    monkeypatch,
) -> None:
    from django.db import close_old_connections

    from apps.collections.models import Collection
    from apps.knowledge_graph.models import GraphRebuildRequest
    from apps.knowledge_graph.services import builds

    collection = Collection.objects.create(name=f"same-id-{uuid.uuid4().hex}")
    barrier = Barrier(2)
    monkeypatch.setattr(
        builds,
        "resume_rebuild_request",
        lambda request_id: GraphRebuildRequest.objects.get(pk=request_id),
    )

    def create_once() -> uuid.UUID:
        close_old_connections()
        try:
            barrier.wait(timeout=10)
            return builds.create_rebuild_request(
                scope_type="collection",
                scope_id=collection.pk,
                request_id=REQUEST_ID,
            ).pk
        finally:
            close_old_connections()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = tuple(executor.submit(create_once) for _ in range(2))
        results = tuple(future.result(timeout=20) for future in futures)

    assert results == (REQUEST_ID, REQUEST_ID)
    assert GraphRebuildRequest.objects.filter(pk=REQUEST_ID).count() == 1


def test_graph_rebuild_request_exposes_durable_correlation_contract() -> None:
    from apps.knowledge_graph.models import GraphRebuildRequest

    fields = {field.name for field in GraphRebuildRequest._meta.fields}
    assert {
        "id",
        "parent_request",
        "scope_type",
        "scope_id",
        "requested_documents",
        "expected_aggregate_signature",
        "status",
        "evaluation_only",
        "document_count",
        "completed_document_count",
        "terminal_failure_count",
        "created_at",
        "started_at",
        "completed_at",
    } <= fields
    assert "activated_artifact" not in fields


@pytest.mark.parametrize(
    "arguments",
    [
        (),
        ("--document", str(DOCUMENT_ID), "--collection", "7"),
        ("--document", str(DOCUMENT_ID), "--all"),
        ("--collection", "7", "--all"),
    ],
)
def test_rebuild_requires_exactly_one_scope(arguments: tuple[str, ...]) -> None:
    with pytest.raises(CommandError):
        call_command("rebuild_knowledge_graph", *arguments)


def test_rebuild_all_requires_confirmation_unless_dry_run() -> None:
    with pytest.raises(CommandError, match="--yes"):
        call_command("rebuild_knowledge_graph", "--all")


def test_rebuild_dry_run_is_non_mutating_and_prints_bounded_counts() -> None:
    output = StringIO()
    with (
        patch(
            "apps.knowledge_graph.management.commands.rebuild_knowledge_graph.preview_rebuild",
            return_value={"document_count": 4, "collection_count": 2},
        ) as preview,
        patch(
            "apps.knowledge_graph.management.commands.rebuild_knowledge_graph.create_rebuild_request"
        ) as create,
    ):
        call_command(
            "rebuild_knowledge_graph",
            "--collection",
            "7",
            "--dry-run",
            stdout=output,
        )
    preview.assert_called_once()
    create.assert_not_called()
    assert "documents=4" in output.getvalue()
    assert "collections=2" in output.getvalue()


@override_settings(DEBUG=False, TESTING=False, KG_EVAL_BYPASS_ALLOWED=True)
def test_eval_only_is_rejected_outside_debug_or_test() -> None:
    with pytest.raises(CommandError, match="evaluation"):
        call_command(
            "rebuild_knowledge_graph",
            "--collection",
            "7",
            "--eval-only",
        )


@pytest.mark.parametrize(
    "arguments",
    [
        ("--document", str(DOCUMENT_ID), "--eval-only"),
        ("--all", "--yes", "--eval-only"),
    ],
)
@override_settings(DEBUG=True, TESTING=False, KG_EVAL_BYPASS_ALLOWED=True)
def test_eval_only_requires_one_concrete_collection(arguments: tuple[str, ...]) -> None:
    with pytest.raises(CommandError, match="collection"):
        call_command("rebuild_knowledge_graph", *arguments)


@override_settings(DEBUG=True, TESTING=False, KG_EVAL_BYPASS_ALLOWED=True)
def test_rebuild_prints_caller_supplied_request_uuid_and_enqueues() -> None:
    output = StringIO()
    with (
        patch(
            "apps.knowledge_graph.management.commands.rebuild_knowledge_graph.create_rebuild_request",
            return_value=type("Request", (), {"pk": REQUEST_ID})(),
        ) as create,
        patch(
            "apps.knowledge_graph.management.commands.rebuild_knowledge_graph._extraction_worker_available",
            return_value=True,
        ),
    ):
        call_command(
            "rebuild_knowledge_graph",
            "--collection",
            "7",
            "--request-id",
            str(REQUEST_ID),
            "--eval-only",
            stdout=output,
        )
    assert str(REQUEST_ID) in output.getvalue()
    assert create.call_args.kwargs["request_id"] == REQUEST_ID
    assert create.call_args.kwargs["evaluation_only"] is True


def test_rebuild_requires_extraction_queue_worker_before_request_mutation() -> None:
    with (
        patch(
            "apps.knowledge_graph.management.commands.rebuild_knowledge_graph.get_build_enabled",
            return_value=True,
        ),
        patch(
            "apps.knowledge_graph.management.commands.rebuild_knowledge_graph._extraction_worker_available",
            return_value=False,
        ),
        patch(
            "apps.knowledge_graph.management.commands.rebuild_knowledge_graph.create_rebuild_request"
        ) as create,
    ):
        with pytest.raises(CommandError, match="worker is unavailable"):
            call_command("rebuild_knowledge_graph", "--collection", "7")
    create.assert_not_called()


@override_settings(KG_EXTRACTION_QUEUE="isolated-graph-test")
def test_worker_probe_uses_the_configured_extraction_queue() -> None:
    command = importlib.import_module(
        "apps.knowledge_graph.management.commands.rebuild_knowledge_graph"
    )
    inspector = SimpleNamespace(
        active_queues=lambda: {
            "worker": [{"name": "isolated-graph-test"}],
        }
    )

    with patch.object(command.current_app.control, "inspect", return_value=inspector):
        assert command._extraction_worker_available() is True

    inspector.active_queues = lambda: {
        "worker": [{"name": "knowledge-graph-extraction"}],
    }
    with patch.object(command.current_app.control, "inspect", return_value=inspector):
        assert command._extraction_worker_available() is False


@override_settings(
    KG_EXTRACTION_QUEUE="invalid-knowledge-graph-extraction",
    KG_EXTRACTION_QUEUE_VALID=False,
)
def test_worker_probe_fails_closed_before_inspection_for_invalid_queue() -> None:
    command = importlib.import_module(
        "apps.knowledge_graph.management.commands.rebuild_knowledge_graph"
    )

    with patch.object(command.current_app.control, "inspect") as inspect:
        assert command._extraction_worker_available() is False

    inspect.assert_not_called()


def test_generated_request_uuid_is_printed_on_publication_failure() -> None:
    from apps.knowledge_graph.services.builds import RebuildPublicationError

    output = StringIO()
    observed: list[uuid.UUID] = []

    def fail_publication(**kwargs):
        observed.append(kwargs["request_id"])
        raise RebuildPublicationError(
            kwargs["request_id"],
            "document_task_publish_failed",
        )

    with (
        patch(
            "apps.knowledge_graph.management.commands.rebuild_knowledge_graph.get_build_enabled",
            return_value=True,
        ),
        patch(
            "apps.knowledge_graph.management.commands.rebuild_knowledge_graph._extraction_worker_available",
            return_value=True,
        ),
        patch(
            "apps.knowledge_graph.management.commands.rebuild_knowledge_graph.create_rebuild_request",
            side_effect=fail_publication,
        ),
    ):
        with pytest.raises(CommandError, match="document_task_publish_failed"):
            call_command(
                "rebuild_knowledge_graph",
                "--collection",
                "7",
                stdout=output,
            )
    assert len(observed) == 1
    assert output.getvalue().strip() == str(observed[0])


def test_inspection_command_delegates_to_bounded_service() -> None:
    output = StringIO()
    report = {
        "request_id": str(REQUEST_ID),
        "status": "succeeded",
        "request_error_code": "",
        "artifact_count": 1,
        "build_count": 1,
        "stale_count": 0,
        "active_evidence_count": 3,
        "failure_count": 0,
    }
    with patch(
        "apps.knowledge_graph.management.commands.inspect_knowledge_graph.inspect_graph_state",
        return_value=report,
    ) as inspect_graph:
        call_command(
            "inspect_knowledge_graph",
            "--request-id",
            str(REQUEST_ID),
            stdout=output,
        )
    inspect_graph.assert_called_once()
    assert "succeeded" in output.getvalue()
    assert "request_id" in output.getvalue()
    assert json.loads(output.getvalue())["request_error_code"] == ""


@pytest.mark.parametrize(
    ("error_code", "expected"),
    (
        ("", ""),
        ("resnapshot_pending", "resnapshot_pending"),
        ("resnapshot_churn", "resnapshot_churn"),
        ("scope_deleted", "scope_deleted"),
        ("scope_ineligible", "scope_ineligible"),
        ("unsafe error text", "invalid"),
        ("x" * 129, "invalid"),
    ),
)
def test_request_inspection_exposes_only_one_bounded_private_code(
    error_code: str,
    expected: str,
) -> None:
    from apps.knowledge_graph.services.inspection import _request_summary

    request = SimpleNamespace(status="partial", error_code=error_code)

    assert _request_summary(request) == {
        "status": "partial",
        "request_error_code": expected,
    }
    assert _request_summary(None) == {
        "status": None,
        "request_error_code": "",
    }


@pytest.mark.django_db(transaction=True)
@database_required
def test_inspection_waits_for_the_exact_request_transaction_commit(
    monkeypatch,
) -> None:
    from django.db import close_old_connections, transaction
    from django.utils import timezone

    from apps.knowledge_graph.models import GraphArtifact, GraphRebuildRequest
    from apps.knowledge_graph.services import inspection

    now = timezone.now()
    older = GraphArtifact.objects.create(
        scope_type=GraphArtifact.ScopeType.DOCUMENT,
        scope_id=str(uuid.uuid4()),
        status=GraphArtifact.Status.ACTIVE,
        source_hash="a" * 64,
        ontology_version="ontology-v1",
        extractor_version="extractor-v1",
        resolver_version="resolver-v1",
        filter_policy_version="filter-v1",
        activated_at=now,
        completed_at=now,
    )
    first_poll = Event()
    real_close = inspection.close_old_connections

    def observed_close() -> None:
        first_poll.set()
        real_close()

    monkeypatch.setattr(inspection, "close_old_connections", observed_close)

    def wait_for_request() -> uuid.UUID:
        close_old_connections()
        try:
            return inspection._wait_for_request(REQUEST_ID, 5).pk
        finally:
            close_old_connections()

    with ThreadPoolExecutor(max_workers=1) as executor:
        waiting = executor.submit(wait_for_request)
        assert first_poll.wait(timeout=5)
        assert not waiting.done()
        with transaction.atomic():
            GraphRebuildRequest.objects.create(
                id=REQUEST_ID,
                scope_type=GraphRebuildRequest.ScopeType.ALL,
                scope_id="",
                requested_documents=[],
                document_count=0,
                collection_count=0,
                status=GraphRebuildRequest.Status.SUCCEEDED,
                completed_at=now,
                enumeration_high_water=0,
                enumeration_complete=True,
                document_publication_state=(
                    GraphRebuildRequest.PublicationState.NOT_APPLICABLE
                ),
                collection_publication_state=(
                    GraphRebuildRequest.PublicationState.NOT_APPLICABLE
                ),
            )
            assert not waiting.done()
        assert waiting.result(timeout=10) == REQUEST_ID

    older.refresh_from_db()
    assert older.status == GraphArtifact.Status.ACTIVE
    assert older.rebuild_request_id is None


@pytest.mark.django_db(transaction=True)
@database_required
def test_inspection_follows_exact_successor_activation_and_fails_closed() -> None:
    from django.utils import timezone

    from apps.documents.models import RawTextDocument
    from apps.knowledge_graph.models import (
        GraphArtifact,
        GraphBuildRun,
        GraphRebuildRequest,
    )
    from apps.knowledge_graph.models.artifacts import _activation_audit_values
    from apps.knowledge_graph.services import inspection

    document_id = uuid.uuid4()
    source_hash = "c" * 64
    now = timezone.now()
    snapshot = [
        {
            "document_id": str(document_id),
            "document_pkid": 1,
            "model_label": RawTextDocument._meta.label_lower,
            "collection_id": 1,
            "source_hash": source_hash,
        }
    ]
    root = GraphRebuildRequest.objects.create(
        id=REQUEST_ID,
        scope_type=GraphRebuildRequest.ScopeType.DOCUMENT,
        scope_id=str(document_id),
        requested_documents=snapshot,
        document_count=1,
        status=GraphRebuildRequest.Status.PARTIAL,
        error_code="request_snapshot_changed",
        started_at=now,
        completed_at=now,
        document_publication_state=GraphRebuildRequest.PublicationState.PUBLISHED,
    )
    successor = GraphRebuildRequest.objects.create(
        id=uuid.uuid5(root.pk, "successor"),
        predecessor_request=root,
        lineage_root=root,
        scope_type=GraphRebuildRequest.ScopeType.DOCUMENT,
        scope_id=str(document_id),
        requested_documents=snapshot,
        document_count=1,
        status=GraphRebuildRequest.Status.RUNNING,
        started_at=now,
        document_publication_state=GraphRebuildRequest.PublicationState.PUBLISHED,
    )
    artifact = GraphArtifact.objects.create(
        scope_type=GraphArtifact.ScopeType.DOCUMENT,
        scope_id=str(document_id),
        status=GraphArtifact.Status.ACTIVE,
        source_hash=source_hash,
        ontology_version="ontology-v1",
        extractor_version="extractor-v1",
        resolver_version="resolver-v1",
        filter_policy_version="filter-v1",
        build_key="d" * 64,
        build_generation=1,
        orchestration_version=GraphArtifact.OrchestrationVersion.SCOPED_V1,
        rebuild_request=successor,
        activated_at=now,
        completed_at=now,
    )
    run = GraphBuildRun.objects.create(
        artifact=artifact,
        rebuild_request=successor,
        stage=GraphBuildRun.Stage.ACTIVE,
        status=GraphBuildRun.Status.SUCCEEDED,
        attempt=1,
        finished_at=now,
    )
    successor.status = GraphRebuildRequest.Status.SUCCEEDED
    successor.completed_document_count = 1
    successor.completed_at = now
    for field, value in _activation_audit_values(artifact, run).items():
        setattr(successor, field, value)
    successor.save()

    effective = inspection._wait_for_request(root.pk, 1)
    assert effective.pk == successor.pk
    assert effective.activated_artifact_pk == artifact.pk
    assert effective.activated_run_pk == run.pk

    failed_id = uuid.uuid4()
    GraphRebuildRequest.objects.create(
        id=failed_id,
        scope_type=GraphRebuildRequest.ScopeType.ALL,
        scope_id="",
        requested_documents=[],
        document_count=0,
        collection_count=0,
        status=GraphRebuildRequest.Status.FAILED,
        error_code="worker_unavailable",
        completed_at=now,
        enumeration_high_water=0,
        enumeration_complete=True,
        document_publication_state=(
            GraphRebuildRequest.PublicationState.NOT_APPLICABLE
        ),
        collection_publication_state=(
            GraphRebuildRequest.PublicationState.NOT_APPLICABLE
        ),
    )
    with pytest.raises(RuntimeError, match="failed"):
        inspection._wait_for_request(failed_id, 1)
    with pytest.raises(TimeoutError, match="not visible"):
        inspection._wait_for_request(uuid.uuid4(), 0.05)


def test_non_extractor_command_imports_do_not_load_optional_ml(monkeypatch) -> None:
    blocked = {"gliner2", "torch", "transformers", "peft", "huggingface_hub"}
    before = set(importlib.sys.modules)
    for module_name in (
        "apps.knowledge_graph.management.commands.rebuild_knowledge_graph",
        "apps.knowledge_graph.management.commands.inspect_knowledge_graph",
        "apps.knowledge_graph.management.commands.prune_knowledge_graph",
    ):
        importlib.import_module(module_name)
    newly_loaded = set(importlib.sys.modules) - before
    assert not {name.partition(".")[0] for name in newly_loaded}.intersection(blocked)


def test_non_extractor_commands_import_in_process_without_optional_ml() -> None:
    repository_root = __import__("pathlib").Path(__file__).resolve().parents[4]
    script = textwrap.dedent(
        f"""
        import importlib
        import importlib.abc
        import os
        import sys

        blocked = {{'gliner2', 'torch', 'transformers', 'peft', 'huggingface_hub'}}
        sys.path.insert(0, {str(repository_root / "aquillm")!r})

        class BlockOptional(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.partition('.')[0] in blocked:
                    raise AssertionError(fullname)
                return None

        sys.meta_path.insert(0, BlockOptional())
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'aquillm.settings')
        for name in (
            'apps.knowledge_graph.management.commands.rebuild_knowledge_graph',
            'apps.knowledge_graph.management.commands.inspect_knowledge_graph',
            'apps.knowledge_graph.management.commands.prune_knowledge_graph',
        ):
            importlib.import_module(name)
        assert not {{name.partition('.')[0] for name in sys.modules}} & blocked
        """
    )
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(repository_root / "aquillm"),
            "SECRET_KEY": "test-only",
            "GOOGLE_OAUTH2_CLIENT_ID": "test",
            "GOOGLE_OAUTH2_CLIENT_SECRET": "test",
            "OPENAI_API_KEY": "test",
            "ANTHROPIC_API_KEY": "test",
            "GEMINI_API_KEY": "test",
        }
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_explicit_extractor_check_validates_fixture_and_prints_identity() -> None:
    from lib.knowledge_graph.types import (
        EntityCandidate,
        ExtractionBatchResult,
        RelationCandidate,
    )

    fixture = "The Atlas model uses the Northstar dataset for retrieval training."
    head, tail = "Atlas model", "Northstar dataset"
    relation = ("uses_dataset", head, tail, 4, 15, 25, 42, 0.9)

    class Backend:
        def extract_batch(self, texts, *, ontology):
            assert texts == (fixture,)
            return (
                ExtractionBatchResult(
                    entities=(
                        EntityCandidate("model", head, 4, 15, 0.9),
                        EntityCandidate("dataset", tail, 25, 42, 0.9),
                    ),
                    relations=(RelationCandidate(*relation),),
                    diagnostics=(),
                ),
            )

    output = StringIO()
    with (
        patch(
            "apps.knowledge_graph.services.builds._active_ontology",
            return_value=object(),
        ),
        patch(
            "lib.knowledge_graph.extractors.get_extraction_backend",
            return_value=Backend(),
        ),
    ):
        call_command("check_knowledge_graph_extractor", stdout=output)

    rendered = output.getvalue()
    assert "package=gliner2==1.3.2" in rendered
    assert "model=fastino/gliner2-base-v1" in rendered
    assert "revision=" in rendered


def test_task_and_build_entrypoints_carry_request_uuid_and_eval_marker() -> None:
    tasks = importlib.import_module("apps.knowledge_graph.tasks")
    builds = importlib.import_module("apps.knowledge_graph.services.builds")
    document_task_parameters = tuple(
        inspect.signature(tasks.build_document_graph_task.run).parameters
    )
    collection_task_parameters = tuple(
        inspect.signature(tasks.refresh_collection_graph_task.run).parameters
    )
    assert document_task_parameters[-2:] == (
        "request_id",
        "eval_only",
    )
    assert collection_task_parameters[-2:] == (
        "request_id",
        "eval_only",
    )
    assert tuple(inspect.signature(builds.build_document_graph).parameters)[-2:] == (
        "request_id",
        "eval_only",
    )
    collection_build_parameters = tuple(
        inspect.signature(builds.refresh_collection_graph).parameters
    )
    assert collection_build_parameters[-2:] == (
        "request_id",
        "eval_only",
    )


@override_settings(DEBUG=False, TESTING=False, KG_EVAL_BYPASS_ALLOWED=True)
def test_task_rejects_forged_eval_bypass_before_build_import(monkeypatch) -> None:
    from apps.knowledge_graph import tasks

    monkeypatch.setattr(tasks, "get_build_enabled", lambda: False)
    with pytest.raises(PermissionError, match="evaluation"):
        tasks.build_document_graph_task.run(
            str(DOCUMENT_ID),
            "a" * 64,
            "b" * 64,
            str(REQUEST_ID),
            True,
        )


@override_settings(DEBUG=False, TESTING=False, KG_EVAL_BYPASS_ALLOWED=True)
def test_build_service_rejects_forged_eval_before_content_read(monkeypatch) -> None:
    from apps.knowledge_graph.services import builds

    def forbidden_content_read(*args, **kwargs):
        raise AssertionError("document content was read before authorization")

    monkeypatch.setattr(builds, "_document_context", forbidden_content_read)
    with pytest.raises(PermissionError, match="evaluation"):
        builds.build_document_graph(
            DOCUMENT_ID,
            "a" * 64,
            "b" * 64,
            str(REQUEST_ID),
            True,
        )


@pytest.mark.parametrize("service", ("document", "collection"))
@override_settings(DEBUG=False, TESTING=False, KG_EVAL_BYPASS_ALLOWED=True)
def test_eval_request_with_false_payload_is_rejected_before_content_read(
    monkeypatch,
    service: str,
) -> None:
    from apps.knowledge_graph.models import GraphRebuildRequest
    from apps.knowledge_graph.services import builds

    request = SimpleNamespace(
        pk=REQUEST_ID,
        evaluation_only=True,
    )

    def forbidden_content_read(*args, **kwargs):
        raise AssertionError("content was read before the stored marker was authorized")

    monkeypatch.setattr(builds, "_document_context", forbidden_content_read)
    monkeypatch.setattr(
        builds,
        "_collection_context_for_request",
        forbidden_content_read,
    )
    with patch.object(GraphRebuildRequest.objects, "filter") as filter_request:
        filter_request.return_value.first.return_value = request
        with pytest.raises(PermissionError, match="evaluation"):
            if service == "document":
                builds.build_document_graph(
                    DOCUMENT_ID,
                    "a" * 64,
                    "b" * 64,
                    REQUEST_ID,
                    False,
                )
            else:
                builds.refresh_collection_graph(
                    1,
                    "a" * 64,
                    "b" * 64,
                    REQUEST_ID,
                    False,
                )


@pytest.mark.parametrize("service", ("document", "collection"))
@override_settings(DEBUG=True, TESTING=True, KG_EVAL_BYPASS_ALLOWED=True)
def test_eval_request_cross_scope_is_rejected_before_content_read(
    monkeypatch,
    service: str,
) -> None:
    from apps.knowledge_graph.models import GraphRebuildRequest
    from apps.knowledge_graph.services import builds

    request = SimpleNamespace(
        pk=REQUEST_ID,
        evaluation_only=True,
        status=GraphRebuildRequest.Status.RUNNING,
        scope_type=GraphRebuildRequest.ScopeType.COLLECTION,
        scope_id="1",
        expected_aggregate_signature="a" * 64,
        requested_documents=(
            {
                "document_id": str(DOCUMENT_ID),
                "collection_id": 1,
                "source_hash": "a" * 64,
            },
        ),
    )

    def forbidden_content_read(*args, **kwargs):
        raise AssertionError("cross-scope content was read before request validation")

    monkeypatch.setattr(builds, "_document_context", forbidden_content_read)
    monkeypatch.setattr(
        builds,
        "_collection_context_for_request",
        forbidden_content_read,
    )
    with patch.object(GraphRebuildRequest.objects, "filter") as filter_request:
        filter_request.return_value.first.return_value = request
        with pytest.raises(builds.StaleBuildError, match="outside"):
            if service == "document":
                builds.build_document_graph(
                    uuid.UUID("cccccccc-cccc-4ccc-8ccc-cccccccccccc"),
                    "a" * 64,
                    "b" * 64,
                    REQUEST_ID,
                    True,
                )
            else:
                builds.refresh_collection_graph(
                    2,
                    "a" * 64,
                    "b" * 64,
                    REQUEST_ID,
                    True,
                )


def test_empty_collection_persists_document_publication_before_advancing(
    monkeypatch,
) -> None:
    from apps.knowledge_graph.models import GraphRebuildRequest
    from apps.knowledge_graph.services import builds

    request = SimpleNamespace(
        pk=REQUEST_ID,
        status=GraphRebuildRequest.Status.RUNNING,
        requested_documents=(),
        document_count=0,
        document_publish_cursor=0,
        document_publication_state=GraphRebuildRequest.PublicationState.PENDING,
        error_code="",
        save=Mock(),
    )

    def fail_after_publication(request_id) -> None:
        assert request_id == REQUEST_ID
        assert request.document_publication_state == (
            GraphRebuildRequest.PublicationState.PUBLISHED
        )
        raise builds.RebuildPublicationError(
            REQUEST_ID,
            "collection_refresh_publish_failed",
        )

    monkeypatch.setattr(builds, "advance_rebuild_request", fail_after_publication)
    monkeypatch.setattr(builds, "_effective_rebuild_request", lambda row: row)
    monkeypatch.setattr(
        builds,
        "_lock_rebuild_request_prefix",
        lambda request_id: (None, request),
    )
    monkeypatch.setattr(builds.transaction, "atomic", nullcontext)
    with patch.object(GraphRebuildRequest.objects, "filter") as filter_request:
        filter_request.return_value.first.return_value = request
        with pytest.raises(
            builds.RebuildPublicationError,
            match="collection_refresh_publish_failed",
        ):
            builds._publish_rebuild_document_tasks(REQUEST_ID)
    request.save.assert_called_once()


@pytest.mark.parametrize("build_kind", ("document", "collection"))
def test_correlated_enqueue_raises_if_builds_disable_after_request_commit(
    monkeypatch,
    build_kind: str,
) -> None:
    from apps.knowledge_graph.services import builds

    monkeypatch.setenv("KG_BUILD_ENABLED", "0")
    with pytest.raises(RuntimeError, match="correlated.*disabled"):
        if build_kind == "document":
            builds.enqueue_document_build(
                DOCUMENT_ID,
                "a" * 64,
                request_id=REQUEST_ID,
            )
        else:
            builds.enqueue_collection_refresh(
                1,
                "a" * 64,
                "b" * 64,
                request_id=REQUEST_ID,
            )


def test_correlated_document_completion_does_not_swallow_refresh_failure(
    monkeypatch,
) -> None:
    from apps.knowledge_graph.services import builds

    callbacks: list[tuple[object, bool]] = []

    def capture(callback, *, robust=False) -> None:
        callbacks.append((callback, robust))

    monkeypatch.setattr(builds.transaction, "on_commit", capture)
    run = type("Run", (), {"rebuild_request_id": REQUEST_ID})()

    builds._register_document_refresh_callbacks(object(), run)

    assert len(callbacks) == 1
    assert callbacks[0][1] is False


def test_fast_document_consumer_is_reconciled_after_cursor_publication(
    monkeypatch,
) -> None:
    from apps.knowledge_graph.models import GraphRebuildRequest
    from apps.knowledge_graph.services import builds

    request = SimpleNamespace(
        pk=REQUEST_ID,
        status=GraphRebuildRequest.Status.RUNNING,
        requested_documents=(
            {
                "document_id": str(DOCUMENT_ID),
                "source_hash": "a" * 64,
            },
        ),
        document_count=1,
        document_publish_cursor=0,
        document_publication_state=GraphRebuildRequest.PublicationState.PENDING,
        error_code="",
        evaluation_only=False,
        save=Mock(),
    )
    observed_states: list[str] = []

    def advance(request_id) -> None:
        assert request_id == REQUEST_ID
        observed_states.append(request.document_publication_state)

    monkeypatch.setattr(
        builds,
        "advance_rebuild_request",
        advance,
    )
    monkeypatch.setattr(
        builds,
        "enqueue_document_build",
        lambda *_args, **_kwargs: advance(REQUEST_ID),
    )
    monkeypatch.setattr(builds, "_effective_rebuild_request", lambda row: row)
    monkeypatch.setattr(
        builds,
        "_lock_rebuild_request_prefix",
        lambda request_id: (None, request),
    )
    monkeypatch.setattr(builds.transaction, "atomic", nullcontext)
    with patch.object(GraphRebuildRequest.objects, "filter") as filter_request:
        filter_request.return_value.first.return_value = request
        assert builds._publish_rebuild_document_tasks(REQUEST_ID) is True

    assert observed_states == [
        GraphRebuildRequest.PublicationState.PENDING,
        GraphRebuildRequest.PublicationState.PUBLISHED,
    ]
    assert request.document_publish_cursor == 1


def test_fast_collection_consumer_is_reconciled_after_publication_state(
    monkeypatch,
) -> None:
    from apps.knowledge_graph.models import GraphArtifact, GraphRebuildRequest
    from apps.knowledge_graph.services import builds

    artifact = SimpleNamespace(pk=17)
    request = SimpleNamespace(
        pk=REQUEST_ID,
        status=GraphRebuildRequest.Status.RUNNING,
        activated_artifact_pk=None,
        collection_publication_state=GraphRebuildRequest.PublicationState.PENDING,
        collection_refresh_published_at=None,
        error_code="",
        save=Mock(),
    )
    observed_states: list[str] = []

    def complete(request_id, completed_artifact) -> None:
        assert request_id == REQUEST_ID
        assert completed_artifact is artifact
        observed_states.append(request.collection_publication_state)

    monkeypatch.setattr(builds, "complete_collection_rebuild", complete)
    monkeypatch.setattr(
        builds,
        "enqueue_collection_refresh",
        lambda *_args, **_kwargs: complete(REQUEST_ID, artifact),
    )
    monkeypatch.setattr(
        builds,
        "_lock_rebuild_request_prefix",
        lambda request_id: (None, request),
    )
    monkeypatch.setattr(builds.transaction, "atomic", nullcontext)
    with patch.object(GraphArtifact.objects, "filter") as filter_artifact:
        filter_artifact.return_value.order_by.return_value.first.return_value = artifact
        builds._publish_correlated_collection_refresh(
            REQUEST_ID,
            7,
            "a" * 64,
            "b" * 64,
            False,
        )

    assert observed_states == [
        GraphRebuildRequest.PublicationState.PENDING,
        GraphRebuildRequest.PublicationState.PUBLISHED,
    ]


def test_all_rebuild_wraps_child_publish_failure_with_operator_root_uuid(
    monkeypatch,
) -> None:
    from apps.knowledge_graph.models import GraphRebuildRequest
    from apps.knowledge_graph.services import builds

    child_id = uuid.UUID("cccccccc-cccc-4ccc-8ccc-cccccccccccc")
    request = SimpleNamespace(
        pk=REQUEST_ID,
        scope_type=GraphRebuildRequest.ScopeType.ALL,
        status=GraphRebuildRequest.Status.RUNNING,
        error_code="",
    )
    monkeypatch.setattr(builds, "_effective_rebuild_request", lambda row: row)
    monkeypatch.setattr(
        builds,
        "_publish_operator_rebuild_children",
        lambda _request_id: (_ for _ in ()).throw(
            builds.RebuildPublicationError(
                child_id,
                "document_task_publish_failed",
            )
        ),
    )
    with patch.object(GraphRebuildRequest.objects, "filter") as filter_request:
        filter_request.return_value.first.return_value = request
        with pytest.raises(builds.RebuildPublicationError) as exc:
            builds.resume_rebuild_request(REQUEST_ID)
    assert exc.value.request_id == REQUEST_ID
    assert exc.value.error_code == "document_task_publish_failed"


@pytest.mark.django_db(transaction=True)
@database_required
def test_collection_request_snapshots_exact_members_before_publish(
    admin_user,
    monkeypatch,
) -> None:
    from apps.collections.models import Collection
    from apps.documents.models import RawTextDocument
    from apps.knowledge_graph.models import GraphRebuildRequest
    from apps.knowledge_graph.services import builds

    monkeypatch.setattr(
        builds,
        "enqueue_document_build",
        lambda *_args, **_kwargs: None,
    )

    collection = Collection.objects.create(name=f"rebuild-{uuid.uuid4()}")
    text = "request snapshots immutable source identity"
    document = RawTextDocument(
        id=DOCUMENT_ID,
        title="snapshot",
        full_text=text,
        full_text_hash=RawTextDocument.hash_fn(text),
        collection=collection,
        ingested_by=admin_user,
        ingestion_complete=True,
    )
    RawTextDocument.objects.bulk_create([document])

    request = builds.create_rebuild_request(
        scope_type="collection",
        scope_id=collection.pk,
        request_id=REQUEST_ID,
    )

    request.refresh_from_db()
    assert request.pk == REQUEST_ID
    assert request.status == GraphRebuildRequest.Status.RUNNING
    assert request.requested_documents == [
        {
            "document_id": str(DOCUMENT_ID),
            "document_pkid": document.pkid,
            "model_label": RawTextDocument._meta.label_lower,
            "collection_id": collection.pk,
            "source_hash": RawTextDocument.hash_fn(text),
        }
    ]


@pytest.mark.django_db(transaction=True)
@database_required
def test_request_drift_is_partial_and_resnapshotted_without_activation(
    admin_user,
    monkeypatch,
) -> None:
    from apps.collections.models import Collection
    from apps.documents.models import RawTextDocument
    from apps.knowledge_graph.models import GraphRebuildRequest
    from apps.knowledge_graph.services import builds

    monkeypatch.setattr(
        builds,
        "enqueue_document_build",
        lambda *_args, **_kwargs: None,
    )

    collection = Collection.objects.create(name=f"drift-{uuid.uuid4()}")
    text = "original immutable source"
    document = RawTextDocument(
        title="snapshot drift",
        full_text=text,
        full_text_hash=RawTextDocument.hash_fn(text),
        collection=collection,
        ingested_by=admin_user,
        ingestion_complete=True,
    )
    RawTextDocument.objects.bulk_create([document])
    request = builds.create_rebuild_request(
        scope_type="collection", scope_id=collection.pk, request_id=REQUEST_ID
    )
    RawTextDocument.objects.filter(pk=document.pk).update(full_text_hash="f" * 64)

    builds.advance_rebuild_request(request.pk)

    request.refresh_from_db()
    assert request.status == GraphRebuildRequest.Status.PARTIAL
    assert request.activated_artifact_pk is None
    replacement = GraphRebuildRequest.objects.get(predecessor_request=request)
    assert replacement.requested_documents[0]["source_hash"] == "f" * 64


@pytest.mark.django_db(transaction=True)
@database_required
def test_same_request_id_repairs_a_dropped_resnapshot_callback(
    admin_user,
    monkeypatch,
) -> None:
    from apps.collections.models import Collection
    from apps.documents.models import RawTextDocument
    from apps.knowledge_graph.models import GraphRebuildRequest
    from apps.knowledge_graph.services import builds

    monkeypatch.setattr(
        builds,
        "enqueue_document_build",
        lambda *_args, **_kwargs: None,
    )
    collection = Collection.objects.create(name=f"resume-{uuid.uuid4().hex}")
    text = "durable successor reconciliation survives a dropped callback"
    document = RawTextDocument(
        title="resume dropped callback",
        full_text=text,
        full_text_hash=RawTextDocument.hash_fn(text),
        collection=collection,
        ingested_by=admin_user,
        ingestion_complete=True,
    )
    RawTextDocument.objects.bulk_create([document])
    request = builds.create_rebuild_request(
        scope_type=GraphRebuildRequest.ScopeType.DOCUMENT,
        scope_id=document.id,
        request_id=REQUEST_ID,
    )
    real_on_commit = builds.transaction.on_commit
    monkeypatch.setattr(
        builds.transaction,
        "on_commit",
        lambda *_args, **_kwargs: None,
    )

    builds.record_rebuild_failure(
        request.pk,
        error_code="request_snapshot_changed",
        resnapshot=True,
    )

    request.refresh_from_db()
    assert request.status == GraphRebuildRequest.Status.PARTIAL
    assert request.error_code == "resnapshot_pending"
    assert not GraphRebuildRequest.objects.filter(
        predecessor_request_id=request.pk
    ).exists()

    monkeypatch.setattr(builds.transaction, "on_commit", real_on_commit)
    builds.resume_rebuild_request(request.pk)

    successor = GraphRebuildRequest.objects.get(predecessor_request_id=request.pk)
    assert successor.status == GraphRebuildRequest.Status.RUNNING
    assert successor.requested_documents == request.requested_documents


@pytest.mark.django_db(transaction=True)
@database_required
@pytest.mark.parametrize("scope_type", ("document", "collection"))
def test_deleted_scope_terminalizes_without_an_unbuildable_successor(
    admin_user,
    monkeypatch,
    scope_type: str,
) -> None:
    from apps.collections.models import Collection
    from apps.documents.models import RawTextDocument
    from apps.knowledge_graph.models import GraphRebuildRequest
    from apps.knowledge_graph.services import builds

    monkeypatch.setattr(
        builds,
        "enqueue_document_build",
        lambda *_args, **_kwargs: None,
    )
    collection = Collection.objects.create(name=f"deleted-{uuid.uuid4().hex}")
    text = "scope is deleted after its immutable request snapshot"
    document = RawTextDocument(
        title="deleted scope",
        full_text=text,
        full_text_hash=RawTextDocument.hash_fn(text),
        collection=collection,
        ingested_by=admin_user,
        ingestion_complete=True,
    )
    RawTextDocument.objects.bulk_create([document])
    scope_id = document.id if scope_type == "document" else collection.pk
    request = builds.create_rebuild_request(
        scope_type=scope_type,
        scope_id=scope_id,
        request_id=REQUEST_ID,
    )

    if scope_type == "document":
        RawTextDocument.objects.get(pkid=document.pkid).delete()
    else:
        Collection.objects.get(pk=collection.pk).delete()
    builds.advance_rebuild_request(request.pk)

    request.refresh_from_db()
    assert request.status == GraphRebuildRequest.Status.PARTIAL
    assert request.error_code == "scope_deleted"
    assert not GraphRebuildRequest.objects.filter(predecessor_request=request).exists()


@pytest.mark.django_db(transaction=True)
@database_required
def test_deleted_operator_child_terminalizes_its_parent_without_a_successor(
    admin_user,
    monkeypatch,
) -> None:
    from apps.collections.models import Collection
    from apps.documents.models import RawTextDocument
    from apps.knowledge_graph.models import GraphRebuildRequest
    from apps.knowledge_graph.services import builds

    monkeypatch.setattr(
        builds,
        "enqueue_document_build",
        lambda *_args, **_kwargs: None,
    )
    collection = Collection.objects.create(name=f"deleted-all-{uuid.uuid4().hex}")
    text = "operator child scope is deleted after enumeration"
    RawTextDocument.objects.bulk_create(
        [
            RawTextDocument(
                title="deleted all child",
                full_text=text,
                full_text_hash=RawTextDocument.hash_fn(text),
                collection=collection,
                ingested_by=admin_user,
                ingestion_complete=True,
            )
        ]
    )
    parent = builds.create_rebuild_request(
        scope_type="all",
        scope_id=None,
        request_id=REQUEST_ID,
    )
    child = GraphRebuildRequest.objects.get(parent_request=parent)

    Collection.objects.get(pk=collection.pk).delete()
    builds.advance_rebuild_request(child.pk)

    child.refresh_from_db()
    parent.refresh_from_db()
    assert child.status == GraphRebuildRequest.Status.PARTIAL
    assert child.error_code == "scope_deleted"
    assert not GraphRebuildRequest.objects.filter(predecessor_request=child).exists()
    assert parent.status == GraphRebuildRequest.Status.FAILED
    assert parent.failed_collection_count == 1


@pytest.mark.django_db(transaction=True)
@database_required
def test_operator_rebuild_pages_resume_deterministically_from_one_root(
    monkeypatch,
) -> None:
    from apps.collections.models import Collection
    from apps.knowledge_graph.models import GraphRebuildRequest
    from apps.knowledge_graph.services import builds

    collections = Collection.objects.bulk_create(
        [
            Collection(name=f"operator-page-{index:03d}-{uuid.uuid4().hex}")
            for index in range(101)
        ]
    )
    collection_ids = tuple(row.pk for row in collections)
    original_resume = builds.resume_rebuild_request
    original_enumerate = builds._enumerate_operator_rebuild_page
    monkeypatch.setattr(
        builds,
        "resume_rebuild_request",
        lambda request_id: GraphRebuildRequest.objects.get(pk=request_id),
    )
    parent = builds.create_rebuild_request(
        scope_type="all",
        scope_id=None,
        request_id=REQUEST_ID,
    )
    assert parent.enumeration_high_water == collection_ids[-1]

    first_page = original_enumerate(parent.pk)
    assert len(first_page) == builds._ALL_REBUILD_PAGE_SIZE == 100
    parent.refresh_from_db()
    assert parent.enumeration_cursor == collection_ids[99]
    assert parent.enumeration_complete is False

    inserted_later = Collection.objects.create(
        name=f"operator-page-later-{uuid.uuid4().hex}"
    )
    page_sizes = [len(first_page)]

    def tracked_enumeration(parent_id):
        page = original_enumerate(parent_id)
        page_sizes.append(len(page))
        return page

    publication_attempts: list[uuid.UUID] = []
    fail_once = True

    def child_resume(request_id):
        nonlocal fail_once
        request_uuid = uuid.UUID(str(request_id))
        publication_attempts.append(request_uuid)
        if fail_once:
            fail_once = False
            raise builds.RebuildPublicationError(
                request_uuid,
                "document_task_publish_failed",
            )
        return GraphRebuildRequest.objects.get(pk=request_uuid)

    monkeypatch.setattr(builds, "_enumerate_operator_rebuild_page", tracked_enumeration)
    monkeypatch.setattr(builds, "resume_rebuild_request", child_resume)
    with pytest.raises(builds.RebuildPublicationError) as raised:
        original_resume(parent.pk)
    assert raised.value.request_id == parent.pk

    original_resume(parent.pk)
    parent.refresh_from_db()
    children = tuple(
        GraphRebuildRequest.objects.filter(parent_request=parent).order_by(
            "scope_id", "pk"
        )
    )
    child_by_scope = {int(row.scope_id): row.pk for row in children}
    assert page_sizes == [100, 1, 0]
    assert parent.enumeration_cursor == parent.enumeration_high_water
    assert parent.enumeration_complete is True
    assert parent.expected_child_count == parent.collection_count == 101
    assert len(children) == len(child_by_scope) == 101
    assert set(child_by_scope) == set(collection_ids)
    assert inserted_later.pk not in child_by_scope
    assert all(
        child_by_scope[collection_id]
        == uuid.uuid5(parent.pk, f"collection:{collection_id}")
        for collection_id in collection_ids
    )
    assert publication_attempts[0] in set(first_page)
