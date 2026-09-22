"""Bulk uploads settle before inference without relaxing source or editor fences."""

from datetime import timedelta

import pytest
from celery.exceptions import Retry
from django.contrib.auth.models import User

from apps.collections.models import (
    Collection,
    CollectionSchemaDraft,
    CollectionSchemaGenerationRun,
)
from apps.collections.services.schema_generation import collection_source_signature
from apps.collections.tasks import schema_generation as tasks
from apps.documents.models import RawTextDocument
from apps.ingestion.models import IngestionBatch, IngestionBatchItem


@pytest.fixture
def generation(monkeypatch):
    monkeypatch.setenv("KG_SCHEMA_GENERATION_ENABLED", "1")
    user = User.objects.create_user(username="settling-owner")
    collection = Collection.objects.create(name="Bulk collection")
    draft = CollectionSchemaDraft.objects.create(
        collection=collection,
        last_editor=user,
        revision=3,
        definitions={"entities": [], "relations": []},
    )
    run = CollectionSchemaGenerationRun.objects.create(
        collection=collection,
        requested_by=user,
        source_signature=collection_source_signature(collection.pk),
        base_draft_id=draft.pk,
        base_draft_revision=draft.revision,
    )
    return user, collection, draft, run


def add_document(user, collection, *, complete=True):
    # bulk_create bypasses document ingestion scheduling; only source rows matter here.
    return RawTextDocument.objects.bulk_create(
        [
            RawTextDocument(
                collection=collection,
                ingested_by=user,
                title="New source",
                full_text="new text",
                full_text_hash="a" * 64,
                ingestion_complete=complete,
            )
        ]
    )[0]


def deliver(run, *, retries=0):
    task = tasks.generate_collection_schema_task
    task.push_request(retries=retries, called_directly=False, is_eager=True)
    try:
        task.run(str(run.pk))
    finally:
        task.pop_request()


@pytest.mark.django_db
def test_queued_upload_changes_refresh_source_without_changing_editor_fence(
    generation, monkeypatch
):
    user, collection, draft, run = generation
    add_document(user, collection)
    expected = collection_source_signature(collection.pk)
    monkeypatch.setattr(tasks, "sample_collection_chunks", lambda *args: ["sample"])
    monkeypatch.setattr(tasks, "generate_schema_candidate", lambda samples: {})
    monkeypatch.setattr(
        tasks,
        "collect_candidate_evidence",
        lambda *args: ({"entities": [], "relations": []}, {}),
    )

    deliver(run)

    run.refresh_from_db()
    draft.refresh_from_db()
    assert run.status == "succeeded"
    assert run.source_signature == expected
    assert (run.base_draft_id, run.base_draft_revision) == (draft.pk, 3)
    assert draft.revision == 4


@pytest.mark.django_db
@pytest.mark.parametrize("pending_kind", ["batch", "document"])
def test_incomplete_uploads_defer_without_using_inference(
    generation, monkeypatch, pending_kind
):
    user, collection, draft, run = generation
    if pending_kind == "batch":
        batch = IngestionBatch.objects.create(user=user, collection=collection)
        IngestionBatchItem.objects.create(
            batch=batch, original_filename="pending.pdf", source_file="pending.pdf"
        )
    else:
        add_document(user, collection, complete=False)
    monkeypatch.setattr(
        tasks,
        "generate_schema_candidate",
        lambda _: pytest.fail("inference ran before uploads settled"),
    )

    with pytest.raises(Retry) as retry:
        deliver(run)

    run.refresh_from_db()
    assert run.status == "queued"
    assert run.lease_token is None
    assert run.statistics["source_deferrals"] == 1
    assert retry.value.when == 30
    first_start = run.started_at
    with pytest.raises(Retry):
        deliver(run, retries=1)
    run.refresh_from_db()
    assert run.started_at == first_start
    assert run.statistics["source_deferrals"] == 2


@pytest.mark.django_db
@pytest.mark.parametrize("exhaustion", ["elapsed", "attempts"])
def test_continuous_uploads_terminate_at_a_durable_bound(generation, exhaustion):
    user, collection, draft, run = generation
    add_document(user, collection, complete=False)
    run.started_at = tasks.timezone.now() - timedelta(
        minutes=11 if exhaustion == "elapsed" else 1
    )
    run.statistics = {"source_deferrals": 20 if exhaustion == "attempts" else 1}
    run.save(update_fields=["started_at", "statistics"])

    deliver(run)

    run.refresh_from_db()
    assert (run.status, run.error_code) == ("failed", "source_changed")
    assert run.lease_token is None


@pytest.mark.django_db
@pytest.mark.parametrize("change", ["revision", "identity"])
def test_source_refresh_cannot_rebase_over_editor_changes(
    generation, monkeypatch, change
):
    user, collection, draft, run = generation
    add_document(user, collection)
    if change == "revision":
        draft.revision += 1
        draft.save(update_fields=["revision"])
    else:
        draft.delete()
        draft = CollectionSchemaDraft.objects.create(
            collection=collection,
            last_editor=user,
            revision=3,
            definitions={"entities": [], "relations": []},
        )
    monkeypatch.setattr(
        tasks,
        "generate_schema_candidate",
        lambda _: pytest.fail("inference ran after editor conflict"),
    )

    deliver(run)

    run.refresh_from_db()
    assert (run.status, run.error_code) == ("failed", "draft_conflict")
    assert run.source_signature != collection_source_signature(collection.pk)


@pytest.mark.django_db
def test_source_change_during_inference_discards_candidate_and_requeues(
    generation, monkeypatch
):
    user, collection, draft, run = generation
    monkeypatch.setattr(tasks, "sample_collection_chunks", lambda *args: ["sample"])
    monkeypatch.setattr(
        tasks, "generate_schema_candidate", lambda _: add_document(user, collection)
    )
    monkeypatch.setattr(
        tasks,
        "collect_candidate_evidence",
        lambda *args: ({"entities": [], "relations": []}, {}),
    )

    with pytest.raises(Retry):
        deliver(run)

    run.refresh_from_db()
    draft.refresh_from_db()
    assert run.status == "queued"
    assert run.statistics["source_deferrals"] == 1
    assert draft.revision == 3


@pytest.mark.django_db
def test_source_deferrals_do_not_consume_the_inference_retry_budget(
    generation, monkeypatch
):
    user, collection, draft, run = generation
    run.statistics = {"source_deferrals": 8}
    run.save(update_fields=["statistics"])
    monkeypatch.setattr(tasks, "sample_collection_chunks", lambda *args: ["sample"])
    monkeypatch.setattr(
        tasks,
        "generate_schema_candidate",
        lambda _: (_ for _ in ()).throw(ConnectionError("offline")),
    )

    for attempt in range(3):
        with pytest.raises(Retry):
            deliver(run, retries=8 + attempt)
        run.refresh_from_db()
        assert run.status == "queued"
        assert run.statistics["inference_retries"] == attempt + 1
        assert run.statistics["source_deferrals"] == 8
    deliver(run, retries=11)
    run.refresh_from_db()
    assert (run.status, run.error_code) == ("failed", "local_inference_failed")


@pytest.mark.django_db
def test_stale_owner_cannot_refresh_sources_or_release_newer_lease(generation):
    _, collection, _, run = generation
    old_claim = tasks._claim_run(run.pk)
    CollectionSchemaGenerationRun.objects.filter(pk=run.pk).update(
        lease_expires_at=tasks.timezone.now() - timedelta(seconds=1),
    )
    new_claim = tasks._claim_run(run.pk)
    with pytest.raises(tasks._LeaseLost):
        tasks._prepare_run_source(old_claim.run, old_claim.lease_token)
    task = tasks.generate_collection_schema_task
    task.push_request(retries=0, called_directly=False, is_eager=True)
    try:
        with pytest.raises(Retry):
            tasks._defer_source(task, run.pk, old_claim.lease_token)
    finally:
        task.pop_request()
    run.refresh_from_db()
    assert run.status == "running"
    assert run.lease_token == new_claim.lease_token
    assert run.statistics == {}
