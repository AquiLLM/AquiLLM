from __future__ import annotations

from contextlib import nullcontext
import sys
from types import SimpleNamespace
import uuid

import pytest


def test_claim_resumes_a_running_run_after_retry_or_worker_redelivery(monkeypatch):
    """Treating running as terminal strands Celery retry and late-ack redelivery."""

    from apps.collections.tasks import schema_generation as tasks
    from apps.collections import models

    run = SimpleNamespace(status="running", started_at="original", error_code="", save=lambda **kwargs: None)
    manager = SimpleNamespace(
        select_for_update=lambda: manager,
        select_related=lambda *args: manager,
        filter=lambda **kwargs: manager,
        first=lambda: run,
    )
    monkeypatch.setattr(models, "CollectionSchemaGenerationRun", SimpleNamespace(objects=manager), raising=False)
    monkeypatch.setattr(tasks.transaction, "atomic", nullcontext)

    assert tasks._claim_run("run-id") is run
    assert run.started_at == "original"


def test_retry_keeps_the_durable_run_resumable_and_marks_only_exhaustion(monkeypatch):
    """Retrying without durable recovery leaves the next delivery unable to claim."""

    from apps.collections.tasks import schema_generation as tasks

    failures = []
    retry = SimpleNamespace(
        request=SimpleNamespace(retries=0),
        max_retries=3,
        retry=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("retry-scheduled")),
    )
    monkeypatch.setattr(tasks, "_fail_run", lambda run_id, code: failures.append((run_id, code)))

    with pytest.raises(RuntimeError, match="retry-scheduled"):
        tasks._retry_or_fail(retry, uuid.uuid4(), ConnectionError("local unavailable"))
    assert failures == []

    exhausted = SimpleNamespace(request=SimpleNamespace(retries=3), max_retries=3)
    tasks._retry_or_fail(exhausted, uuid.UUID(int=1), ConnectionError("local unavailable"))
    assert failures == [(uuid.UUID(int=1), "local_inference_failed")]


def test_final_source_fence_locks_source_before_task_one_draft_write(monkeypatch):
    """Writing after an unlocked signature check permits a stale source draft."""

    from apps.collections.tasks import schema_generation as tasks
    import sys
    from types import ModuleType

    calls = []
    schema = ModuleType("apps.collections.services.schema")
    schema.canonicalize_definitions = lambda definitions: {"canonical": definitions}
    schema.write_generated_draft = lambda *args: calls.append(args) or "draft"
    monkeypatch.setitem(sys.modules, "apps.collections.services.schema", schema)
    monkeypatch.setattr(tasks.transaction, "atomic", nullcontext)
    monkeypatch.setattr(tasks, "_locked_collection_source_signature", lambda collection_id: "source")

    assert tasks._write_draft_with_source_fence(uuid.UUID(int=2), 1, "source", {"entities": []}, {"counts": {}}) == "draft"
    assert calls == [(uuid.UUID(int=2), {"canonical": {"entities": []}}, {"counts": {}})]

    monkeypatch.setattr(tasks, "_locked_collection_source_signature", lambda collection_id: "changed")
    with pytest.raises(tasks._SourceChanged):
        tasks._write_draft_with_source_fence(uuid.UUID(int=2), 1, "source", {}, {})


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
    failures, writes, logs = [], [], []
    monkeypatch.setenv("KG_SCHEMA_GENERATION_ENABLED", "1")
    monkeypatch.setattr(tasks, "_claim_run", lambda run_id: run)
    monkeypatch.setattr(tasks, "_fail_run", lambda run_id, code: failures.append(code))
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
