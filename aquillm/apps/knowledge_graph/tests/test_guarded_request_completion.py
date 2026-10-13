"""Real request orchestration with a narrowly simulated future I5 snapshot owner."""

from copy import deepcopy
from dataclasses import replace
from uuid import uuid4

import pytest
from django.db.models import QuerySet
from django.utils import timezone

from apps.documents.services.representation_contracts import (
    RepresentationSnapshotV1,
    SourceScopeUnavailable,
)
from apps.documents.services.representation_store import (
    freeze_active_representations,
    validate_representation_snapshot,
)
from apps.knowledge_graph.models import GraphBuildRun, GraphRebuildRequest
from apps.knowledge_graph.services import builds
from apps.knowledge_graph.services import representation_request_guard as guard
from apps.knowledge_graph.tests.test_models import _artifact
from apps.knowledge_graph.tests.test_representation_request_guard import (
    create_guarded_request,
)

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.parametrize("swap", [False, True])
def test_guarded_publication_advances_and_completion_fences_same_source_swap(
    monkeypatch, swap
):
    fixture, request, key = create_guarded_request(monkeypatch)
    document = fixture.documents["A"]
    QuerySet.update(
        type(document).objects.filter(pk=document.pk), ingestion_complete=True
    )
    expected = deepcopy(request.requested_documents)
    current_key, captures, published = [key], [], []

    def future_store(refs, *, using):
        # Only this acquisition boundary simulates I5. Real I4 still validates
        # concrete owner and source; no production selector/validator is weakened.
        legacy = freeze_active_representations(refs, using=using)
        captures.append((refs, using))
        return RepresentationSnapshotV1(
            tuple(
                replace(selection, generation_key=current_key[0])
                for selection in legacy.selections
            )
        )

    monkeypatch.setattr(
        guard, "freeze_active_representations", future_store, raising=False
    )
    monkeypatch.setattr(
        builds,
        "enqueue_document_build",
        lambda *args, **kwargs: published.append(kwargs),
    )
    builds.resume_rebuild_request(request.pk)
    request.refresh_from_db()
    assert request.status == "running" and request.error_code == ""
    assert request.document_publication_state == "published"
    assert (
        request.document_publish_cursor == 1 and request.completed_document_count == 0
    )
    assert published[0]["representation_request"] == expected[0]
    assert captures and request.requested_documents == expected

    now = timezone.now()
    # I4 cannot serialize a non-null captured binding either. This correlated
    # occurrence tests completion orchestration, not I5's future build store.
    artifact = _artifact(
        scope_id=document.id,
        source_hash=document.full_text_hash,
        rebuild_request=request,
        build_generation=2,
        status="active",
        activated_at=now,
        completed_at=now,
    )
    artifact.save()
    run = GraphBuildRun.objects.create(
        artifact=artifact,
        stage="active",
        status="succeeded",
        finished_at=now,
    )
    assert artifact.rebuild_request_id == run.rebuild_request_id == request.pk
    assert artifact.scope_id == run.scope_id == str(document.id)
    assert artifact.source_hash == run.source_hash == expected[0]["source_hash"]
    assert artifact.build_key == run.build_key and run.build_generation == 2
    assert builds._run_identity_matches_artifact(artifact, run, build_kind="document")
    if swap:
        current_key[0] = uuid4()
    before = len(captures)
    builds.advance_rebuild_request(request.pk)
    request.refresh_from_db()
    assert len(captures) == before + 1
    assert request.requested_documents == expected
    assert request.status == ("partial" if swap else "succeeded")
    assert request.error_code == ("representation_changed" if swap else "")
    assert request.activated_artifact_pk == (None if swap else artifact.pk)
    assert request.activated_run_pk == (None if swap else run.pk)
    assert not GraphRebuildRequest.objects.filter(predecessor_request=request).exists()
    # The test double is not evidence for a successful non-null I4 store.
    with pytest.raises(SourceScopeUnavailable):
        validate_representation_snapshot(
            future_store((fixture.snapshot.selections[0].document,), using="default"),
            using="default",
        )
