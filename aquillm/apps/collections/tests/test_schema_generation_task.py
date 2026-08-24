from __future__ import annotations

import uuid

import pytest


def test_enqueue_uses_the_knowledge_graph_queue(monkeypatch):
    """Moving generation to the default queue would run it without GLiNER2."""

    from apps.collections.tasks.schema_generation import enqueue_schema_generation
    from apps.collections.tasks.schema_generation import generate_collection_schema_task

    observed = {}

    def delay(run_id):
        observed["run_id"] = run_id

    monkeypatch.setattr(generate_collection_schema_task, "delay", delay)
    run_id = uuid.uuid4()
    enqueue_schema_generation(run_id)

    assert observed == {"run_id": str(run_id)}
    assert generate_collection_schema_task.queue


@pytest.mark.django_db
def test_task_marks_disabled_run_failed_without_text_or_secrets(monkeypatch, caplog):
    """A disabled local pipeline must terminate safely without logging payload data."""

    from apps.collections.models import CollectionSchemaGenerationRun
    from apps.collections.models import Collection
    from apps.collections.tasks.schema_generation import generate_collection_schema_task

    collection = Collection.objects.create(name="Schema task collection")
    run = CollectionSchemaGenerationRun.objects.create(
        collection=collection,
        source_signature="a" * 64,
        status="queued",
        statistics={},
    )
    monkeypatch.setenv("KG_SCHEMA_GENERATION_ENABLED", "0")

    generate_collection_schema_task.run(str(run.id))
    run.refresh_from_db()

    assert run.status == "failed"
    assert run.error_code == "disabled"
    assert "a" * 64 not in caplog.text
    assert "VLLM_API_KEY" not in caplog.text
