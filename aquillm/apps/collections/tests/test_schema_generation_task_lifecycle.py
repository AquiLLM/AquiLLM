from __future__ import annotations

from contextlib import nullcontext
from datetime import timedelta
import sys
from types import SimpleNamespace
import uuid

import pytest


def test_claim_resumes_a_running_run_after_retry_or_worker_redelivery(monkeypatch):
    """An expired late-ack delivery can recover without changing its first start time."""

    from apps.collections.tasks import schema_generation as tasks
    from apps.collections import models

    run = SimpleNamespace(
        status="running", started_at="original", error_code="", lease_token=uuid.UUID(int=1),
        lease_expires_at=tasks.timezone.now() - timedelta(seconds=1), save=lambda **kwargs: None,
    )
    manager = SimpleNamespace(
        select_for_update=lambda: manager,
        select_related=lambda *args: pytest.fail("claiming a run must not lock a joined Collection row"),
        filter=lambda **kwargs: manager,
        first=lambda: run,
    )
    monkeypatch.setattr(models, "CollectionSchemaGenerationRun", SimpleNamespace(objects=manager), raising=False)
    monkeypatch.setattr(tasks.transaction, "atomic", nullcontext)

    claim = tasks._claim_run("run-id")
    assert claim.run is run
    assert claim.lease_token != uuid.UUID(int=1)
    assert run.started_at == "original"


def test_claim_excludes_live_duplicate_delivery_and_recovers_only_an_expired_lease(monkeypatch):
    """Two concurrent deliveries must never execute the same durable run."""

    from apps.collections.tasks import schema_generation as tasks
    from apps.collections import models

    now = tasks.timezone.now()
    run = SimpleNamespace(
        status="queued", started_at=None, error_code="", lease_token=None,
        lease_expires_at=None, save=lambda **kwargs: None,
    )
    manager = SimpleNamespace(
        select_for_update=lambda: manager,
        select_related=lambda *args: pytest.fail("claiming a run must not lock a joined Collection row"),
        filter=lambda **kwargs: manager,
        first=lambda: run,
    )
    monkeypatch.setattr(models, "CollectionSchemaGenerationRun", SimpleNamespace(objects=manager), raising=False)
    monkeypatch.setattr(tasks.transaction, "atomic", nullcontext)

    first = tasks._claim_run("run-id")
    assert first is not None
    assert first.lease_token == run.lease_token
    assert tasks._claim_run("run-id") is None

    run.lease_expires_at = now - timedelta(seconds=1)
    recovered = tasks._claim_run("run-id")
    assert recovered is not None
    assert recovered.lease_token != first.lease_token


def test_retry_releases_only_its_own_lease_before_scheduling(monkeypatch):
    """A stale worker may not release a newer worker's durable lease."""

    from apps.collections.tasks import schema_generation as tasks

    released = []
    retry = SimpleNamespace(
        request=SimpleNamespace(retries=0), max_retries=3,
        retry=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("retry-scheduled")),
    )
    token = uuid.uuid4()
    monkeypatch.setattr(
        tasks,
        "_release_lease_for_retry",
        lambda run_id, lease_token: released.append((run_id, lease_token)) or True,
    )

    with pytest.raises(RuntimeError, match="retry-scheduled"):
        tasks._retry_or_fail(retry, uuid.UUID(int=1), token, ConnectionError("local unavailable"))
    assert released == [(uuid.UUID(int=1), token)]


def test_retry_keeps_the_durable_run_resumable_and_marks_only_exhaustion(monkeypatch):
    """Retrying without durable recovery leaves the next delivery unable to claim."""

    from apps.collections.tasks import schema_generation as tasks

    failures = []
    retry = SimpleNamespace(
        request=SimpleNamespace(retries=0),
        max_retries=3,
        retry=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("retry-scheduled")),
    )
    token = uuid.uuid4()
    monkeypatch.setattr(tasks, "_release_lease_for_retry", lambda run_id, lease_token: True)
    monkeypatch.setattr(tasks, "_fail_run", lambda run_id, code, lease_token: failures.append((run_id, code, lease_token)))

    with pytest.raises(RuntimeError, match="retry-scheduled"):
        tasks._retry_or_fail(retry, uuid.uuid4(), token, ConnectionError("local unavailable"))
    assert failures == []

    exhausted = SimpleNamespace(request=SimpleNamespace(retries=3), max_retries=3)
    tasks._retry_or_fail(exhausted, uuid.UUID(int=1), token, ConnectionError("local unavailable"))
    assert failures == [(uuid.UUID(int=1), "local_inference_failed", token)]


def test_final_source_fence_locks_source_before_task_one_draft_write(monkeypatch):
    """Writing after an unlocked signature check permits a stale source draft."""

    from apps.collections.tasks import schema_generation as tasks
    from apps.collections import models
    import sys
    from types import ModuleType

    calls = []
    schema = ModuleType("apps.collections.services.schema")
    schema.canonicalize_definitions = lambda definitions: {"canonical": definitions}
    schema.write_generated_draft = lambda *args: calls.append(args) or "draft"
    monkeypatch.setitem(sys.modules, "apps.collections.services.schema", schema)
    monkeypatch.setattr(tasks.transaction, "atomic", nullcontext)
    now = tasks.timezone.now()
    monkeypatch.setattr(tasks.timezone, "now", lambda: now)
    monkeypatch.setattr(tasks, "_locked_collection_source_signature", lambda collection_id: "source")
    collection_locks = []
    collection_manager = SimpleNamespace(
        select_for_update=lambda: collection_manager,
        get=lambda **kwargs: collection_locks.append(kwargs),
    )
    monkeypatch.setattr(models, "Collection", SimpleNamespace(objects=collection_manager))
    lease_run = SimpleNamespace(lease_token=uuid.UUID(int=3), lease_expires_at=None, save=lambda **kwargs: None)
    run_filters = []
    run_manager = SimpleNamespace(
        select_for_update=lambda: run_manager,
        filter=lambda **kwargs: run_filters.append(kwargs) or run_manager,
        first=lambda: lease_run,
    )
    monkeypatch.setattr(
        models,
        "CollectionSchemaGenerationRun",
        SimpleNamespace(objects=run_manager),
        raising=False,
    )

    assert tasks._write_draft_with_source_fence(uuid.UUID(int=2), 1, "source", uuid.UUID(int=3), {"entities": []}, {"counts": {}}) == "draft"
    assert collection_locks == [{"pk": 1}]
    assert run_filters == [{"id": uuid.UUID(int=2), "status": "running", "lease_token": uuid.UUID(int=3), "lease_expires_at__gt": now}]
    assert calls == [(uuid.UUID(int=2), {"canonical": {"entities": []}}, {"counts": {}})]

    monkeypatch.setattr(tasks, "_locked_collection_source_signature", lambda collection_id: "changed")
    with pytest.raises(tasks._SourceChanged):
        tasks._write_draft_with_source_fence(uuid.UUID(int=2), 1, "source", uuid.UUID(int=3), {}, {})


def test_terminal_failure_is_conditioned_on_the_owned_lease(monkeypatch):
    """An expired worker must not fail a run that a newer worker reclaimed."""

    from apps.collections.tasks import schema_generation as tasks
    from apps.collections import models

    filters, updates = [], []
    manager = SimpleNamespace(
        filter=lambda **kwargs: filters.append(kwargs) or SimpleNamespace(
            update=lambda **values: updates.append(values)
        )
    )
    monkeypatch.setattr(models, "CollectionSchemaGenerationRun", SimpleNamespace(objects=manager), raising=False)
    now = tasks.timezone.now()
    monkeypatch.setattr(tasks.timezone, "now", lambda: now)
    token = uuid.uuid4()

    tasks._fail_run(uuid.UUID(int=4), "invalid_candidate", token)

    assert filters == [{"id": uuid.UUID(int=4), "status__in": ("queued", "running"), "lease_token": token, "lease_expires_at__gt": now}]
    assert updates[0]["status"] == "failed"
    assert updates[0]["lease_token"] is None


@pytest.mark.parametrize(
    ("source", "samples", "candidate_error", "write_error", "expected"),
    (
        ("changed", ["sample"], None, None, "source_changed"),
        ("source", [], None, None, "no_collection_text"),
        ("source", ["sample"], ValueError("invalid"), None, "invalid_candidate"),
        ("source", ["sample"], None, "draft_conflict", "draft_conflict"),
        ("source", ["sample"], None, None, None),
    ),
)
def test_task_fake_lifecycle_maps_terminal_outcomes_without_payload_logs(
    monkeypatch, source, samples, candidate_error, write_error, expected
):
    """Changing a terminal mapping must not retry, strand, or log source data."""

    from apps.collections.tasks import schema_generation as tasks
    from apps.collections.services.schema_generation import InvalidSchemaCandidate

    run = SimpleNamespace(collection_id=1, source_signature="source")
    lease_token = uuid.uuid4()
    failures, writes, logs = [], [], []
    monkeypatch.setenv("KG_SCHEMA_GENERATION_ENABLED", "1")
    monkeypatch.setattr(tasks, "_claim_run", lambda run_id: tasks._RunClaim(run, lease_token))
    monkeypatch.setattr(tasks, "_fail_run", lambda run_id, code, token: failures.append(code))
    monkeypatch.setattr(tasks, "collection_source_signature", lambda collection_id: source)
    monkeypatch.setattr(tasks, "load_schema_generation_config", lambda: SimpleNamespace(max_chunks=2, max_characters=20))
    monkeypatch.setattr(tasks, "sample_collection_chunks", lambda *args: samples)
    monkeypatch.setattr(tasks, "_safe_log_failure", lambda code, exc=None: logs.append((code, type(exc).__name__ if exc else None)))
    if candidate_error:
        monkeypatch.setattr(tasks, "generate_schema_candidate", lambda samples: (_ for _ in ()).throw(InvalidSchemaCandidate("SENTINEL TEXT")))
    else:
        monkeypatch.setattr(tasks, "generate_schema_candidate", lambda samples: {"candidate": True})
    monkeypatch.setattr(tasks, "collect_candidate_evidence", lambda candidate, samples: ({"definitions": True}, {"counts": {}}))
    if write_error == "draft_conflict":
        conflict = type("SchemaGenerationDraftConflict", (Exception,), {})
        schema = SimpleNamespace(SchemaGenerationDraftConflict=conflict)
        monkeypatch.setitem(sys.modules, "apps.collections.services.schema", schema)
        monkeypatch.setattr(tasks, "_write_draft_with_source_fence", lambda *args: (_ for _ in ()).throw(conflict()))
    else:
        monkeypatch.setattr(tasks, "_write_draft_with_source_fence", lambda *args: writes.append(args))

    tasks.generate_collection_schema_task.run(str(uuid.UUID(int=9)))

    assert failures == ([] if expected is None else [expected])
    assert bool(writes) is (expected is None)
    assert all("SENTINEL TEXT" not in repr(log) for log in logs)


def test_safe_failure_logger_excludes_exception_message_and_credentials(monkeypatch):
    """Logging exception text would expose provider or collection secrets."""

    from apps.collections.tasks import schema_generation as tasks

    observed = []
    monkeypatch.setattr(tasks, "logger", SimpleNamespace(error=lambda event, **fields: observed.append((event, fields))))

    tasks._safe_log_failure("invalid_candidate", ValueError("SENTINEL API KEY AND COLLECTION TEXT"))

    assert observed == [("obs.collections.schema_generation_failed", {"error_code": "invalid_candidate", "error_type": "ValueError"})]
