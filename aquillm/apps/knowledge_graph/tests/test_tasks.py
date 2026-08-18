from __future__ import annotations

import importlib
import inspect
import os
import subprocess
import sys
import textwrap
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from celery.exceptions import Retry
from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import IntegrityError, OperationalError

DOCUMENT_ID = "12345678-1234-4234-9234-123456789abc"
SOURCE_HASH = "a" * 64
DOCUMENT_BUILD_KEY = "b" * 64
COLLECTION_ID = 17
AGGREGATE_SOURCE_SIGNATURE = "c" * 64
COLLECTION_BUILD_KEY = "d" * 64


def _tasks_module():
    return importlib.import_module("apps.knowledge_graph.tasks")


def _builds_module():
    return importlib.import_module("apps.knowledge_graph.services.builds")


def _synthetic_exception(module: str, name: str, **attributes: object) -> Exception:
    exception_type = type(name, (Exception,), {"__module__": module})
    exception = exception_type(name)
    for attribute, value in attributes.items():
        setattr(exception, attribute, value)
    return exception


def test_document_task_passes_only_the_exact_scalar_build_identity(monkeypatch) -> None:
    monkeypatch.setenv("KG_BUILD_ENABLED", "1")
    tasks = _tasks_module()
    builds = _builds_module()
    calls: list[tuple[object, ...]] = []

    def fake_build(*args):
        calls.append(args)
        return SimpleNamespace(pk=41)

    monkeypatch.setattr(builds, "build_document_graph", fake_build)

    result = tasks.build_document_graph_task.run(
        DOCUMENT_ID,
        SOURCE_HASH,
        DOCUMENT_BUILD_KEY,
    )

    assert result == 41
    assert calls == [
        (uuid.UUID(DOCUMENT_ID), SOURCE_HASH, DOCUMENT_BUILD_KEY),
    ]


def test_collection_task_passes_only_the_exact_scalar_build_identity(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KG_BUILD_ENABLED", "1")
    tasks = _tasks_module()
    builds = _builds_module()
    calls: list[tuple[object, ...]] = []

    def fake_refresh(*args):
        calls.append(args)
        return SimpleNamespace(pk=42)

    monkeypatch.setattr(builds, "refresh_collection_graph", fake_refresh)

    result = tasks.refresh_collection_graph_task.run(
        COLLECTION_ID,
        AGGREGATE_SOURCE_SIGNATURE,
        COLLECTION_BUILD_KEY,
    )

    assert result == 42
    assert calls == [
        (COLLECTION_ID, AGGREGATE_SOURCE_SIGNATURE, COLLECTION_BUILD_KEY),
    ]


def test_task_entry_points_expose_json_scalar_signatures_only() -> None:
    tasks = _tasks_module()

    document_parameters = inspect.signature(
        tasks.build_document_graph_task.run
    ).parameters
    collection_parameters = inspect.signature(
        tasks.refresh_collection_graph_task.run
    ).parameters

    assert tuple(document_parameters) == (
        "document_id",
        "expected_source_hash",
        "document_build_key",
    )
    assert tuple(collection_parameters) == (
        "collection_id",
        "aggregate_source_signature",
        "collection_build_key",
    )
    assert "gliner" not in repr(document_parameters).lower()
    assert "gliner" not in repr(collection_parameters).lower()


def test_extraction_tasks_are_routed_to_the_isolated_queue() -> None:
    tasks = _tasks_module()
    expected_queue = "knowledge-graph-extraction"

    for task in (
        tasks.build_document_graph_task,
        tasks.refresh_collection_graph_task,
    ):
        assert task.queue == expected_queue
        assert settings.CELERY_TASK_ROUTES[task.name] == {"queue": expected_queue}

    assert settings.CELERY_TASK_ROUTES[tasks.prune_graph_artifacts_task.name] == {
        "queue": expected_queue,
        "priority": 9,
    }
    assert settings.CELERY_TASK_PUBLISH_RETRY is True
    assert 0 < settings.CELERY_TASK_PUBLISH_RETRY_POLICY["max_retries"] <= 5


def test_extraction_tasks_use_crash_safe_delivery_options() -> None:
    tasks = _tasks_module()

    for task in (
        tasks.build_document_graph_task,
        tasks.refresh_collection_graph_task,
    ):
        assert task.acks_late is True
        assert task.reject_on_worker_lost is True
        assert task.ignore_result is True
        assert task.serializer == "json"


def test_disabled_build_short_circuits_before_service_import(monkeypatch) -> None:
    tasks = _tasks_module()
    monkeypatch.setenv("KG_BUILD_ENABLED", "0")
    sys.modules.pop("apps.knowledge_graph.services.builds", None)

    result = tasks.build_document_graph_task.run(
        DOCUMENT_ID,
        SOURCE_HASH,
        DOCUMENT_BUILD_KEY,
    )

    assert result is None
    assert "apps.knowledge_graph.services.builds" not in sys.modules


def test_disabled_collection_refresh_short_circuits_before_service_import(
    monkeypatch,
) -> None:
    tasks = _tasks_module()
    monkeypatch.setenv("KG_BUILD_ENABLED", "0")
    sys.modules.pop("apps.knowledge_graph.services.builds", None)

    result = tasks.refresh_collection_graph_task.run(
        COLLECTION_ID,
        AGGREGATE_SOURCE_SIGNATURE,
        COLLECTION_BUILD_KEY,
    )

    assert result is None
    assert "apps.knowledge_graph.services.builds" not in sys.modules


@pytest.mark.parametrize(
    ("task_name", "args"),
    (
        (
            "build_document_graph_task",
            ("forged\n" * 2_000, SOURCE_HASH, DOCUMENT_BUILD_KEY),
        ),
        (
            "refresh_collection_graph_task",
            ("forged\n" * 2_000, AGGREGATE_SOURCE_SIGNATURE, COLLECTION_BUILD_KEY),
        ),
    ),
)
def test_malformed_scope_is_rejected_without_raw_value_in_task_logs(
    monkeypatch,
    task_name: str,
    args: tuple[object, ...],
) -> None:
    monkeypatch.setenv("KG_BUILD_ENABLED", "1")
    tasks = _tasks_module()
    logger = Mock()
    monkeypatch.setattr(tasks, "logger", logger)

    with pytest.raises(ValueError):
        getattr(tasks, task_name).run(*args)

    logger.error.assert_called_once()
    task_log = logger.error.call_args
    assert task_log.kwargs["scope_id"] == "invalid"
    assert task_log.kwargs["terminal"] is True
    assert str(args[0]) not in repr(task_log)


@pytest.mark.parametrize(
    ("task_name", "service_name", "args", "invalid_value"),
    (
        (
            "build_document_graph_task",
            "build_document_graph",
            (DOCUMENT_ID, "A" * 64, DOCUMENT_BUILD_KEY),
            "A" * 64,
        ),
        (
            "build_document_graph_task",
            "build_document_graph",
            (DOCUMENT_ID, SOURCE_HASH, "b" * 63),
            "b" * 63,
        ),
        (
            "refresh_collection_graph_task",
            "refresh_collection_graph",
            (COLLECTION_ID, "C" * 64, COLLECTION_BUILD_KEY),
            "C" * 64,
        ),
        (
            "refresh_collection_graph_task",
            "refresh_collection_graph",
            (COLLECTION_ID, AGGREGATE_SOURCE_SIGNATURE, "d" * 63 + "\n"),
            "d" * 63 + "\n",
        ),
    ),
)
def test_invalid_task_hashes_are_rejected_privately_before_service_import(
    monkeypatch,
    task_name: str,
    service_name: str,
    args: tuple[object, ...],
    invalid_value: str,
) -> None:
    monkeypatch.setenv("KG_BUILD_ENABLED", "1")
    tasks = _tasks_module()
    builds = _builds_module()
    service = Mock(side_effect=AssertionError("invalid payload reached build service"))
    logger = Mock()
    monkeypatch.setattr(builds, service_name, service)
    monkeypatch.setattr(tasks, "logger", logger)
    sys.modules.pop("apps.knowledge_graph.services.builds", None)

    with pytest.raises(ValueError) as captured:
        getattr(tasks, task_name).run(*args)

    service.assert_not_called()
    assert "apps.knowledge_graph.services.builds" not in sys.modules
    assert invalid_value not in str(captured.value)
    logger.error.assert_called_once()
    assert invalid_value not in repr(logger.error.call_args)


def test_invalid_enabled_config_is_terminally_logged_without_config_value(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KG_BUILD_ENABLED", "1")
    invalid_revision = "floating-revision\nforged-log-field"
    monkeypatch.setenv("KG_GLINER2_REVISION", invalid_revision)
    tasks = _tasks_module()
    logger = Mock()
    monkeypatch.setattr(tasks, "logger", logger)

    with pytest.raises(ValueError):
        tasks.build_document_graph_task.run(
            DOCUMENT_ID,
            SOURCE_HASH,
            DOCUMENT_BUILD_KEY,
        )

    logger.error.assert_called_once()
    task_log = logger.error.call_args
    assert task_log.kwargs["scope_id"] == "invalid"
    assert task_log.kwargs["terminal"] is True
    assert invalid_revision not in repr(task_log)


@pytest.mark.parametrize("build_enabled", ("0", "1"))
def test_task_autodiscovery_does_not_import_optional_ml_runtime(
    build_enabled: str,
) -> None:
    repository_root = Path(__file__).resolve().parents[4]
    blocked_modules = {
        "gliner2",
        "torch",
        "transformers",
        "peft",
        "huggingface_hub",
    }
    script = textwrap.dedent(
        f"""
        import importlib.abc
        import os
        import sys

        BLOCKED = {blocked_modules!r}
        sys.path.insert(0, {str(repository_root / "aquillm")!r})

        class BlockOptionalRuntime(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.partition('.')[0] in BLOCKED:
                    raise AssertionError(f'optional ML import attempted: {{fullname}}')
                return None

        sys.meta_path.insert(0, BlockOptionalRuntime())
        os.environ['KG_BUILD_ENABLED'] = {build_enabled!r}

        import lib.knowledge_graph.config
        import lib.knowledge_graph.extractors
        import aquillm.settings
        import aquillm.asgi
        import apps.knowledge_graph.models
        from aquillm.celery import app

        app.autodiscover_tasks(['apps.knowledge_graph'], force=True)

        loaded = sorted(
            name for name in sys.modules if name.partition('.')[0] in BLOCKED
        )
        assert loaded == [], loaded
        """
    )
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(repository_root / "aquillm"),
            "DJANGO_DEBUG": "0",
            "SECRET_KEY": "test-only-secret-key",
            "GOOGLE_OAUTH2_CLIENT_ID": "test-client-id",
            "GOOGLE_OAUTH2_CLIENT_SECRET": "test-client-secret",
            "OPENAI_API_KEY": "test-openai-key",
            "ANTHROPIC_API_KEY": "test-anthropic-key",
            "GEMINI_API_KEY": "test-gemini-key",
            "KG_BUILD_ENABLED": build_enabled,
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


def test_transient_build_failure_requests_a_bounded_retry(monkeypatch) -> None:
    monkeypatch.setenv("KG_BUILD_ENABLED", "1")
    tasks = _tasks_module()
    builds = _builds_module()
    failure = OperationalError("database temporarily unavailable")
    retry_calls: list[dict[str, object]] = []

    def fail_build(*_args):
        raise failure

    def fake_retry(**kwargs):
        retry_calls.append(kwargs)
        raise Retry(exc=kwargs["exc"])

    monkeypatch.setattr(builds, "build_document_graph", fail_build)
    monkeypatch.setattr(tasks.build_document_graph_task, "retry", fake_retry)

    with pytest.raises(Retry):
        tasks.build_document_graph_task.run(
            DOCUMENT_ID,
            SOURCE_HASH,
            DOCUMENT_BUILD_KEY,
        )

    assert tasks.build_document_graph_task.max_retries == 3
    assert len(retry_calls) == 1
    assert retry_calls[0]["exc"] is failure
    assert 0 < retry_calls[0]["countdown"] <= 60


def test_live_build_lease_retries_after_the_lease_expiry_window(monkeypatch) -> None:
    monkeypatch.setenv("KG_BUILD_ENABLED", "1")
    tasks = _tasks_module()
    builds = _builds_module()
    failure = builds.BuildInProgressError("another worker owns the live lease")
    retry_calls: list[dict[str, object]] = []

    def fail_build(*_args):
        raise failure

    def fake_retry(**kwargs):
        retry_calls.append(kwargs)
        raise Retry(exc=kwargs["exc"])

    monkeypatch.setattr(builds, "build_document_graph", fail_build)
    monkeypatch.setattr(tasks.build_document_graph_task, "retry", fake_retry)

    with pytest.raises(Retry):
        tasks.build_document_graph_task.run(
            DOCUMENT_ID,
            SOURCE_HASH,
            DOCUMENT_BUILD_KEY,
        )

    assert retry_calls[0]["countdown"] == builds.BUILD_LEASE_RETRY_SECONDS


def test_lost_build_lease_uses_a_short_bounded_retry(monkeypatch) -> None:
    monkeypatch.setenv("KG_BUILD_ENABLED", "1")
    tasks = _tasks_module()
    builds = _builds_module()
    failure = builds.BuildLeaseLostError("lease token was rotated")
    retry_calls: list[dict[str, object]] = []

    def fail_build(*_args):
        raise failure

    def fake_retry(**kwargs):
        retry_calls.append(kwargs)
        raise Retry(exc=kwargs["exc"])

    monkeypatch.setattr(builds, "build_document_graph", fail_build)
    monkeypatch.setattr(tasks.build_document_graph_task, "retry", fake_retry)

    with pytest.raises(Retry):
        tasks.build_document_graph_task.run(
            DOCUMENT_ID,
            SOURCE_HASH,
            DOCUMENT_BUILD_KEY,
        )

    assert 0 < retry_calls[0]["countdown"] <= 60


def test_stale_build_is_acknowledged_without_retry_or_terminal_task_failure(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KG_BUILD_ENABLED", "1")
    tasks = _tasks_module()
    builds = _builds_module()
    failure = builds.StaleBuildError("immutable request is no longer current")
    failure.__cause__ = TimeoutError("stale cause must not change classification")
    logger = Mock()

    def fail_build(*_args):
        raise failure

    monkeypatch.setattr(builds, "build_document_graph", fail_build)
    monkeypatch.setattr(tasks, "logger", logger)
    monkeypatch.setattr(
        tasks.build_document_graph_task,
        "retry",
        Mock(side_effect=AssertionError("stale builds must not retry")),
    )

    result = tasks.build_document_graph_task.run(
        DOCUMENT_ID,
        SOURCE_HASH,
        DOCUMENT_BUILD_KEY,
    )

    assert result is None
    logger.error.assert_not_called()


@pytest.mark.parametrize(
    "provider_failure",
    (
        _synthetic_exception("openai", "APIConnectionError"),
        _synthetic_exception("openai", "APITimeoutError"),
        _synthetic_exception("openai", "RateLimitError"),
        _synthetic_exception("openai", "APIStatusError", status_code=503),
        _synthetic_exception("botocore.exceptions", "EndpointConnectionError"),
        _synthetic_exception(
            "botocore.exceptions",
            "ClientError",
            response={"ResponseMetadata": {"HTTPStatusCode": 503}},
        ),
        _synthetic_exception(
            "botocore.exceptions",
            "ClientError",
            response={"ResponseMetadata": {"HTTPStatusCode": 429}},
        ),
        _synthetic_exception(
            "botocore.exceptions",
            "ClientError",
            response={"Error": {"Code": "ThrottlingException"}},
        ),
        *(
            _synthetic_exception(
                "botocore.exceptions",
                "ClientError",
                response={"Error": {"Code": error_code}},
            )
            for error_code in (
                "RequestTimeout",
                "RequestTimeoutException",
                "ServiceUnavailable",
                "InternalError",
            )
        ),
    ),
    ids=lambda exc: f"{type(exc).__module__}.{type(exc).__name__}",
)
def test_known_provider_availability_failures_retry_without_provider_imports(
    monkeypatch,
    provider_failure: Exception,
) -> None:
    monkeypatch.setenv("KG_BUILD_ENABLED", "1")
    tasks = _tasks_module()
    builds = _builds_module()
    retry_calls: list[dict[str, object]] = []

    def fail_build(*_args):
        raise provider_failure

    def fake_retry(**kwargs):
        retry_calls.append(kwargs)
        raise Retry(exc=kwargs["exc"])

    monkeypatch.setattr(builds, "build_document_graph", fail_build)
    monkeypatch.setattr(tasks.build_document_graph_task, "retry", fake_retry)

    with pytest.raises(Retry):
        tasks.build_document_graph_task.run(
            DOCUMENT_ID,
            SOURCE_HASH,
            DOCUMENT_BUILD_KEY,
        )

    assert retry_calls[0]["exc"] is provider_failure
    assert 0 < retry_calls[0]["countdown"] <= 60


def test_transient_extraction_wrapper_preserves_cause_classification(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KG_BUILD_ENABLED", "1")
    tasks = _tasks_module()
    builds = _builds_module()
    from lib.knowledge_graph.extractors.gliner2_local import ExtractionBackendError

    provider_failure = _synthetic_exception("openai", "APITimeoutError")
    failure = ExtractionBackendError("provider inference failed")
    failure.__cause__ = provider_failure
    retry_calls: list[dict[str, object]] = []

    def fail_build(*_args):
        raise failure

    def fake_retry(**kwargs):
        retry_calls.append(kwargs)
        raise Retry(exc=kwargs["exc"])

    monkeypatch.setattr(builds, "build_document_graph", fail_build)
    monkeypatch.setattr(tasks.build_document_graph_task, "retry", fake_retry)

    with pytest.raises(Retry):
        tasks.build_document_graph_task.run(
            DOCUMENT_ID,
            SOURCE_HASH,
            DOCUMENT_BUILD_KEY,
        )

    assert retry_calls[0]["exc"] is failure


def test_bare_extraction_backend_failure_retries_without_provider_imports(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KG_BUILD_ENABLED", "1")
    tasks = _tasks_module()
    builds = _builds_module()
    from lib.knowledge_graph.extractors.gliner2_local import ExtractionBackendError

    failure = ExtractionBackendError("model load failed")
    retry_calls: list[dict[str, object]] = []

    def fail_build(*_args):
        raise failure

    def fake_retry(**kwargs):
        retry_calls.append(kwargs)
        raise Retry(exc=kwargs["exc"])

    monkeypatch.setattr(builds, "build_document_graph", fail_build)
    monkeypatch.setattr(tasks.build_document_graph_task, "retry", fake_retry)

    with pytest.raises(Retry):
        tasks.build_document_graph_task.run(
            DOCUMENT_ID,
            SOURCE_HASH,
            DOCUMENT_BUILD_KEY,
        )

    assert retry_calls[0]["exc"] is failure
    assert 0 < retry_calls[0]["countdown"] <= 60


def test_transient_cause_walker_is_depth_bounded_and_cycle_safe() -> None:
    tasks = _tasks_module()
    builds = _builds_module()
    head = RuntimeError("head")
    current = head
    for index in range(64):
        nested = RuntimeError(f"nested-{index}")
        current.__cause__ = nested
        current = nested
    current.__cause__ = TimeoutError("too deep to trust")

    assert tasks._retry_countdown(head, builds=builds, retry_count=0) is None

    cycle = RuntimeError("cycle")
    cycle.__cause__ = cycle
    assert tasks._retry_countdown(cycle, builds=builds, retry_count=0) is None


def test_suppressed_transient_context_does_not_retry_permanent_failure() -> None:
    tasks = _tasks_module()
    builds = _builds_module()

    try:
        raise TimeoutError("transient context")
    except TimeoutError:
        try:
            raise RuntimeError("permanent failure") from None
        except RuntimeError as captured:
            failure = captured

    assert failure.__suppress_context__ is True
    assert isinstance(failure.__context__, TimeoutError)
    assert tasks._retry_countdown(failure, builds=builds, retry_count=0) is None


def test_permanent_build_failure_is_recorded_without_retry(monkeypatch) -> None:
    monkeypatch.setenv("KG_BUILD_ENABLED", "1")
    tasks = _tasks_module()
    builds = _builds_module()
    failure = ValueError("invalid immutable build key")
    logger = Mock()

    def fail_build(*_args):
        raise failure

    monkeypatch.setattr(builds, "build_document_graph", fail_build)
    monkeypatch.setattr(tasks, "logger", logger)
    monkeypatch.setattr(
        tasks.build_document_graph_task,
        "retry",
        Mock(side_effect=AssertionError("permanent failures must not retry")),
    )

    with pytest.raises(ValueError, match="invalid immutable build key"):
        tasks.build_document_graph_task.run(
            DOCUMENT_ID,
            SOURCE_HASH,
            DOCUMENT_BUILD_KEY,
        )

    logger.error.assert_called_once()
    (event,) = logger.error.call_args.args
    assert event == "obs.kg.task_failed"
    assert logger.error.call_args.kwargs["terminal"] is True


@pytest.mark.parametrize(
    "failure_kind",
    ("corrupt", "validation", "integrity", "runtime"),
)
def test_nonretryable_failures_are_terminal(
    monkeypatch,
    failure_kind: str,
) -> None:
    monkeypatch.setenv("KG_BUILD_ENABLED", "1")
    tasks = _tasks_module()
    builds = _builds_module()
    failures = {
        "corrupt": builds.CorruptBuildError("corrupt durable state"),
        "validation": ValidationError("invalid build request"),
        "integrity": IntegrityError("constraint violation"),
        "runtime": RuntimeError("unclassified provider failure"),
    }
    failure = failures[failure_kind]
    logger = Mock()

    def fail_build(*_args):
        raise failure

    monkeypatch.setattr(builds, "build_document_graph", fail_build)
    monkeypatch.setattr(tasks, "logger", logger)
    monkeypatch.setattr(
        tasks.build_document_graph_task,
        "retry",
        Mock(side_effect=AssertionError("nonretryable failure retried")),
    )

    with pytest.raises(type(failure)):
        tasks.build_document_graph_task.run(
            DOCUMENT_ID,
            SOURCE_HASH,
            DOCUMENT_BUILD_KEY,
        )

    logger.error.assert_called_once()
    assert logger.error.call_args.kwargs["terminal"] is True


def test_transient_failure_at_retry_limit_is_recorded_as_terminal(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KG_BUILD_ENABLED", "1")
    tasks = _tasks_module()
    builds = _builds_module()
    failure = TimeoutError("provider timed out")
    logger = Mock()

    def fail_build(*_args):
        raise failure

    monkeypatch.setattr(builds, "build_document_graph", fail_build)
    monkeypatch.setattr(tasks, "logger", logger)
    monkeypatch.setattr(
        tasks.build_document_graph_task,
        "retry",
        Mock(side_effect=AssertionError("retry limit must be bounded")),
    )

    tasks.build_document_graph_task.push_request(
        retries=tasks.build_document_graph_task.max_retries
    )
    try:
        with pytest.raises(TimeoutError, match="provider timed out"):
            tasks.build_document_graph_task.run(
                DOCUMENT_ID,
                SOURCE_HASH,
                DOCUMENT_BUILD_KEY,
            )
    finally:
        tasks.build_document_graph_task.pop_request()

    logger.error.assert_called_once()
    assert logger.error.call_args.kwargs["terminal"] is True
    assert logger.error.call_args.kwargs["retry_count"] == 3


def test_document_enqueue_routes_only_canonical_json_scalars(monkeypatch) -> None:
    monkeypatch.setenv("KG_BUILD_ENABLED", "1")
    tasks = _tasks_module()
    builds = _builds_module()
    apply_async = Mock()
    derive_key = Mock(return_value=DOCUMENT_BUILD_KEY)
    monkeypatch.setattr(tasks.build_document_graph_task, "apply_async", apply_async)
    monkeypatch.setattr(builds, "derive_current_document_build_key", derive_key)

    builds.enqueue_document_build(uuid.UUID(DOCUMENT_ID), SOURCE_HASH)

    derive_key.assert_called_once_with(uuid.UUID(DOCUMENT_ID), SOURCE_HASH)
    apply_async.assert_called_once()
    options = apply_async.call_args.kwargs
    assert options["kwargs"] == {
        "document_id": DOCUMENT_ID,
        "expected_source_hash": SOURCE_HASH,
        "document_build_key": DOCUMENT_BUILD_KEY,
    }
    assert options["retry"] is True
    assert 0 < options["retry_policy"]["max_retries"] <= 5


def test_disabled_document_producer_does_not_derive_or_publish(monkeypatch) -> None:
    monkeypatch.setenv("KG_BUILD_ENABLED", "0")
    tasks = _tasks_module()
    builds = _builds_module()
    derive_key = Mock(side_effect=AssertionError("disabled producer derived a key"))
    apply_async = Mock(side_effect=AssertionError("disabled producer published"))
    monkeypatch.setattr(builds, "derive_current_document_build_key", derive_key)
    monkeypatch.setattr(tasks.build_document_graph_task, "apply_async", apply_async)

    result = builds.enqueue_document_build(uuid.UUID(DOCUMENT_ID), SOURCE_HASH)

    assert result is None
    derive_key.assert_not_called()
    apply_async.assert_not_called()


def test_collection_refresh_enqueue_routes_the_post_commit_snapshot(
    monkeypatch,
) -> None:
    """Task 11's on-commit seam must become a real Task 12 delivery."""

    monkeypatch.setenv("KG_BUILD_ENABLED", "1")
    tasks = _tasks_module()
    builds = _builds_module()
    apply_async = Mock()
    monkeypatch.setattr(tasks.refresh_collection_graph_task, "apply_async", apply_async)

    builds.enqueue_collection_refresh(
        COLLECTION_ID,
        AGGREGATE_SOURCE_SIGNATURE,
        COLLECTION_BUILD_KEY,
    )

    apply_async.assert_called_once()
    options = apply_async.call_args.kwargs
    assert options["kwargs"] == {
        "collection_id": COLLECTION_ID,
        "aggregate_source_signature": AGGREGATE_SOURCE_SIGNATURE,
        "collection_build_key": COLLECTION_BUILD_KEY,
    }
    assert options["retry"] is True
    assert 0 < options["retry_policy"]["max_retries"] <= 5


@pytest.mark.parametrize(
    ("producer_name", "task_name", "args", "build_kind", "scope_id"),
    (
        (
            "enqueue_document_build",
            "build_document_graph_task",
            (uuid.UUID(DOCUMENT_ID), SOURCE_HASH),
            "document",
            DOCUMENT_ID,
        ),
        (
            "enqueue_collection_refresh",
            "refresh_collection_graph_task",
            (COLLECTION_ID, AGGREGATE_SOURCE_SIGNATURE, COLLECTION_BUILD_KEY),
            "collection",
            str(COLLECTION_ID),
        ),
    ),
)
def test_exhausted_task_publish_failure_is_logged_without_payload(
    monkeypatch,
    producer_name: str,
    task_name: str,
    args: tuple[object, ...],
    build_kind: str,
    scope_id: str,
) -> None:
    monkeypatch.setenv("KG_BUILD_ENABLED", "1")
    tasks = _tasks_module()
    builds = _builds_module()
    secret = "broker-secret-from-exception"
    publish_failure = ConnectionError(secret)
    task = getattr(tasks, task_name)
    logger = Mock()
    monkeypatch.setattr(task, "apply_async", Mock(side_effect=publish_failure))
    monkeypatch.setattr(builds, "logger", logger)
    monkeypatch.setattr(
        builds,
        "derive_current_document_build_key",
        Mock(return_value=DOCUMENT_BUILD_KEY),
    )

    with pytest.raises(ConnectionError) as captured:
        getattr(builds, producer_name)(*args)

    assert captured.value is publish_failure
    logger.error.assert_called_once()
    publish_log = logger.error.call_args
    assert publish_log.args == ("obs.kg.task_publish_failed",)
    assert publish_log.kwargs["task_name"] == task.name
    assert publish_log.kwargs["build_kind"] == build_kind
    assert publish_log.kwargs["scope_id"] == scope_id
    assert publish_log.kwargs["error_type"] == "ConnectionError"
    assert publish_log.kwargs["publish_retry_exhausted"] is True
    assert secret not in repr(publish_log)
    assert SOURCE_HASH not in repr(publish_log)
    assert DOCUMENT_BUILD_KEY not in repr(publish_log)
    assert AGGREGATE_SOURCE_SIGNATURE not in repr(publish_log)
    assert COLLECTION_BUILD_KEY not in repr(publish_log)


def test_disabled_collection_producer_does_not_publish(monkeypatch) -> None:
    monkeypatch.setenv("KG_BUILD_ENABLED", "0")
    tasks = _tasks_module()
    builds = _builds_module()
    apply_async = Mock(side_effect=AssertionError("disabled producer published"))
    monkeypatch.setattr(tasks.refresh_collection_graph_task, "apply_async", apply_async)

    result = builds.enqueue_collection_refresh(
        COLLECTION_ID,
        AGGREGATE_SOURCE_SIGNATURE,
        COLLECTION_BUILD_KEY,
    )

    assert result is None
    apply_async.assert_not_called()


def test_task_retry_contract_uses_public_build_service_types() -> None:
    builds = _builds_module()

    assert builds.BUILD_LEASE_RETRY_SECONDS > 30 * 60
    for name in (
        "BuildInProgressError",
        "StaleBuildError",
        "CorruptBuildError",
        "derive_current_document_build_key",
        "enqueue_document_build",
        "enqueue_collection_refresh",
    ):
        assert name in builds.__all__


def test_pruning_task_is_low_priority_lazy_and_unscheduled() -> None:
    sys.modules.pop("apps.knowledge_graph.services.pruning", None)
    tasks = _tasks_module()

    assert tasks.prune_graph_artifacts_task.queue == "knowledge-graph-extraction"
    assert tasks.prune_graph_artifacts_task.priority == 9
    assert tasks.prune_graph_artifacts_task.acks_late is True
    assert tasks.prune_graph_artifacts_task.reject_on_worker_lost is True
    assert tasks.prune_graph_artifacts_task.ignore_result is True
    assert tasks.prune_graph_artifacts_task.serializer == "json"
    assert "apps.knowledge_graph.services.pruning" not in sys.modules
    schedules = getattr(settings, "CELERY_BEAT_SCHEDULE", {})
    assert all(
        entry.get("task") != tasks.prune_graph_artifacts_task.name
        for entry in schedules.values()
    )


def test_disabled_build_short_circuits_before_pruning_service_import(
    monkeypatch,
) -> None:
    tasks = _tasks_module()
    monkeypatch.setenv("KG_BUILD_ENABLED", "0")
    sys.modules.pop("apps.knowledge_graph.services.pruning", None)

    result = tasks.prune_graph_artifacts_task.run()

    assert result is None
    assert "apps.knowledge_graph.services.pruning" not in sys.modules
