from __future__ import annotations

from contextlib import nullcontext
import sys
from types import SimpleNamespace
import uuid

import pytest


def test_expired_final_write_raises_lease_loss_without_writing(monkeypatch):
    """An expired owner must leave the run retriable rather than writing stale output."""

    from apps.collections.tasks import schema_generation as tasks
    from apps.collections import models
    from types import ModuleType

    schema = ModuleType("apps.collections.services.schema")
    writes = []
    schema.canonicalize_definitions = lambda definitions: definitions
    schema.write_generated_draft = lambda *args: writes.append(args)
    monkeypatch.setitem(sys.modules, "apps.collections.services.schema", schema)
    monkeypatch.setattr(tasks.transaction, "atomic", nullcontext)
    monkeypatch.setattr(tasks, "_locked_collection_source_signature", lambda collection_id: "source")
    collection_manager = SimpleNamespace(select_for_update=lambda: collection_manager, get=lambda **kwargs: None)
    run_manager = SimpleNamespace(select_for_update=lambda: run_manager, filter=lambda **kwargs: run_manager, first=lambda: None)
    monkeypatch.setattr(models, "Collection", SimpleNamespace(objects=collection_manager))
    monkeypatch.setattr(models, "CollectionSchemaGenerationRun", SimpleNamespace(objects=run_manager), raising=False)

    with pytest.raises(tasks._LeaseLost):
        tasks._write_draft_with_source_fence(uuid.UUID(int=2), 1, "source", uuid.UUID(int=3), {}, {})
    assert writes == []


def test_terminal_failure_is_conditioned_on_the_owned_lease(monkeypatch):
    """An expired worker must not fail a run that a newer worker reclaimed."""

    from apps.collections.tasks import schema_generation as tasks
    from apps.collections import models

    filters, updates = [], []
    manager = SimpleNamespace(filter=lambda **kwargs: filters.append(kwargs) or SimpleNamespace(update=lambda **values: updates.append(values) or 1))
    monkeypatch.setattr(models, "CollectionSchemaGenerationRun", SimpleNamespace(objects=manager), raising=False)
    now = tasks.timezone.now()
    monkeypatch.setattr(tasks.timezone, "now", lambda: now)
    token = uuid.uuid4()

    assert tasks._fail_run(uuid.UUID(int=4), "invalid_candidate", token) is True

    assert filters == [{"id": uuid.UUID(int=4), "status__in": ("queued", "running"), "lease_token": token, "lease_expires_at__gt": now}]
    assert updates[0]["status"] == "failed"
    assert updates[0]["lease_token"] is None


def test_exhausted_lease_failure_targets_only_the_same_expired_owner(monkeypatch):
    """The exhaustion fallback must not write a newer unexpired owner's run."""

    from apps.collections.tasks import schema_generation as tasks
    from apps.collections import models

    filters, updates = [], []
    manager = SimpleNamespace(filter=lambda **kwargs: filters.append(kwargs) or SimpleNamespace(update=lambda **values: updates.append(values) or 1))
    monkeypatch.setattr(models, "CollectionSchemaGenerationRun", SimpleNamespace(objects=manager), raising=False)
    now = tasks.timezone.now()
    monkeypatch.setattr(tasks.timezone, "now", lambda: now)
    token = uuid.uuid4()

    assert tasks._fail_expired_run(uuid.UUID(int=9), "source_changed", token) is True

    assert filters == [{"id": uuid.UUID(int=9), "status": "running", "lease_token": token, "lease_expires_at__lte": now}]
    assert updates[0]["error_code"] == "source_changed"


def test_expired_terminal_failure_reschedules_the_delivery(monkeypatch):
    """A false terminal update must not ACK the only message for an expired lease."""

    from apps.collections.tasks import schema_generation as tasks

    retry = SimpleNamespace(request=SimpleNamespace(retries=0), max_retries=3, retry=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("retry-scheduled")))
    monkeypatch.setattr(tasks, "_fail_run", lambda *args: False)

    with pytest.raises(RuntimeError, match="retry-scheduled"):
        tasks._fail_or_retry(retry, uuid.UUID(int=5), uuid.uuid4(), "invalid_candidate", ValueError("stale"))


def test_expired_retry_release_reschedules_the_delivery(monkeypatch):
    """A stale retry attempt must remain deliverable until lease recovery can claim it."""

    from apps.collections.tasks import schema_generation as tasks

    retry = SimpleNamespace(request=SimpleNamespace(retries=0), max_retries=3, retry=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("retry-scheduled")))
    monkeypatch.setattr(tasks, "_release_lease_for_retry", lambda *args: False)

    with pytest.raises(RuntimeError, match="retry-scheduled"):
        tasks._retry_or_fail(retry, uuid.UUID(int=6), uuid.uuid4(), ConnectionError("stale"))


def test_lease_lost_final_write_reschedules_the_task(monkeypatch):
    """The task must not ACK when the final fence discovers an expired lease."""

    from apps.collections.tasks import schema_generation as tasks

    run = SimpleNamespace(collection_id=1, source_signature="source")
    token = uuid.uuid4()
    monkeypatch.setenv("KG_SCHEMA_GENERATION_ENABLED", "1")
    monkeypatch.setattr(tasks, "_claim_run", lambda run_id: tasks._RunClaim(run, token))
    monkeypatch.setattr(tasks, "collection_source_signature", lambda collection_id: "source")
    monkeypatch.setattr(tasks, "load_schema_generation_config", lambda: SimpleNamespace(max_chunks=1, max_characters=1))
    monkeypatch.setattr(tasks, "sample_collection_chunks", lambda *args: ["sample"])
    monkeypatch.setattr(tasks, "generate_schema_candidate", lambda samples: {"candidate": True})
    monkeypatch.setattr(tasks, "collect_candidate_evidence", lambda candidate, samples: ({}, {}))
    monkeypatch.setattr(tasks, "_write_draft_with_source_fence", lambda *args: (_ for _ in ()).throw(tasks._LeaseLost()))
    monkeypatch.setattr(tasks.generate_collection_schema_task, "retry", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("retry-scheduled")))

    with pytest.raises(RuntimeError, match="retry-scheduled"):
        tasks.generate_collection_schema_task.run(str(uuid.UUID(int=8)))


@pytest.mark.parametrize("expired_failure", (False, True))
def test_retry_exhaustion_only_fails_the_same_expired_owner(monkeypatch, expired_failure):
    """Exhaustion may fail an expired owner, but must not overwrite a newer lease."""

    from apps.collections.tasks import schema_generation as tasks

    exhausted = SimpleNamespace(request=SimpleNamespace(retries=3), max_retries=3)
    expired_calls, logs = [], []
    monkeypatch.setattr(tasks, "_fail_run", lambda *args: False)
    monkeypatch.setattr(tasks, "_fail_expired_run", lambda run_id, error_code, lease_token: expired_calls.append((run_id, error_code, lease_token)) or expired_failure)
    monkeypatch.setattr(tasks, "_safe_log_failure", lambda code, exc=None: logs.append(code))
    token = uuid.uuid4()

    tasks._retry_or_fail(exhausted, uuid.UUID(int=7), token, ConnectionError("stale"))

    assert expired_calls == [(uuid.UUID(int=7), "local_inference_failed", token)]
    assert logs == (["local_inference_failed"] if expired_failure else [])
