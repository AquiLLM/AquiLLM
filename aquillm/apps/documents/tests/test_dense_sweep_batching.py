"""Fresh bounded dense metadata sweeps retain original authority and membership."""
import pytest
from django.db import IntegrityError, transaction

from apps.collections.models import Collection, CollectionPermission
from apps.documents.models import DocumentEmbeddingState, EmbeddingGeneration, ImageUploadDocument
from apps.documents.services import embedding_index_query as query
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.services.source_deadline import source_preparation_scope
from apps.documents.services.source_loading import SourcePreparationLimited
from lib.embeddings.tests.test_document_provider import transport as transport
from lib.retrieval.turn_budget import TurnBudget, TurnLimits
from .dense_sweep_support import corpus, freeze, build, original_scope, dense_reads, assert_scalar_reads

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.parametrize("count,queries", [(32, 1), (65, 2)])
def test_empty_dense_snapshot_uses_bounded_fresh_scalar_sweeps(count, queries, transport, monkeypatch, settings):
    case = corpus(count, transport[0], monkeypatch, settings)
    assert not EmbeddingGeneration.objects.exists()
    with dense_reads() as reads:
        captured = freeze(case)
    assert captured.selections == captured.generation_keys == ()
    assert_scalar_reads(reads, queries)
    with dense_reads() as reads:
        assert query.validate_dense_snapshot(case.snapshot, captured, using="default") == ()
    assert_scalar_reads(reads, queries)
    assert freeze(case) == captured


def test_each_original_scope_boundary_keeps_fresh_authority_with_empty_dense(transport, monkeypatch, settings):
    case = corpus(32, transport[0], monkeypatch, settings)
    scope = original_scope(case)
    original, budget = scope.runtime._frozen_retrieval_scope, scope.runtime.budget
    try:
        for _ in range(2):
            with scope.database(), dense_reads() as reads:
                assert scope.validate() == ()
            assert_scalar_reads(reads, 1)
        CollectionPermission.objects.filter(user=case.owner, collection=case.collection).delete()
        with scope.database(), dense_reads() as reads:
            with pytest.raises(SourceScopeUnavailable, match="^source_scope_unavailable$"):
                scope.validate()
        assert reads == [] and budget.can_publish()
        assert scope.runtime._frozen_retrieval_scope is original
        assert scope.representation is original.representation and scope.runtime.budget is budget
    finally:
        budget.close("dense_sweep_finished")


def test_empty_capture_never_adopts_later_dense_generation(transport, monkeypatch, settings):
    case = corpus(2, transport[0], monkeypatch, settings)
    empty = freeze(case)
    actual = build(case)
    assert query.validate_dense_snapshot(case.snapshot, empty, using="default") == ()
    current = freeze(case)
    assert current.selections == (actual,) and current.manifest_sha256 != empty.manifest_sha256


def test_final_batch_ready_owner_is_retained_in_capture_and_revalidation(transport, monkeypatch, settings):
    from apps.documents.services.representation_store import ordered_selections
    case = corpus(65, transport[0], monkeypatch, settings)
    assert ordered_selections(case.snapshot)[-1].document.id == case.docs[-1].id
    tail = build(case, 64)
    with dense_reads() as reads:
        captured = freeze(case)
    assert captured.selections == (tail,)
    assert_scalar_reads(reads, 2)
    with dense_reads() as reads:
        assert query.validate_dense_snapshot(case.snapshot, captured, using="default") == (tail.generation_key,)
    assert_scalar_reads(reads, 2)


def test_replacement_pointer_cannot_reenter_original_dense_snapshot(transport, monkeypatch, settings):
    case = corpus(2, transport[0], monkeypatch, settings)
    first, other = build(case, 0), build(case, 1)
    captured = freeze(case)
    replacement = build(case, 0)
    assert replacement.generation_key != first.generation_key
    assert query.validate_dense_snapshot(case.snapshot, captured, using="default") == (other.generation_key,)
    fresh = freeze(case)
    assert {s.generation_key for s in fresh.selections} == {replacement.generation_key, other.generation_key}
    assert tuple(s.document_id for s in fresh.selections) == tuple(s.document.id for s in case.snapshot.selections)


@pytest.mark.parametrize("change", ["building", "failed", "failure_kind", "source_hash", "canonical_sha256", "foreign_owner"])
def test_invalid_active_generation_is_omitted_without_weakening_other_owner(change, transport, monkeypatch, settings):
    case = corpus(2, transport[0], monkeypatch, settings)
    first, other = build(case, 0), build(case, 1)
    captured = freeze(case)
    if change == "foreign_owner":
        assert DocumentEmbeddingState.objects.filter(active_generation_id=first.generation_key).update(
            active_generation_id=other.generation_key) == 1
    else:
        values = ({"state": change} if change in {"building", "failed"}
                  else {change: "upstream" if change == "failure_kind" else "f" * 64})
        assert EmbeddingGeneration.objects.filter(pk=first.generation_key).update(**values) == 1
    assert query.validate_dense_snapshot(case.snapshot, captured, using="default") == (other.generation_key,)
    assert freeze(case).selections == (other,)


@pytest.mark.parametrize("field", ["model_label", "document_pkid", "document_id", "collection_id", "representation_key", "space_id"])
def test_foreign_state_tuple_is_not_an_original_owner(field, transport, monkeypatch, settings):
    from uuid import uuid4
    from .representation_cutover_support import distinct_verified_space
    from apps.documents.services.embedding_index_store import persist_embedding_space
    case = corpus(2, transport[0], monkeypatch, settings)
    first = build(case, 0)
    captured = freeze(case)
    values = {"model_label": "apps_documents.pdfdocument", "document_pkid": case.docs[1].pkid,
              "document_id": uuid4(), "collection_id": Collection.objects.create(name="Other").pk,
              "representation_key": uuid4()}
    if field == "space_id":
        foreign = distinct_verified_space()
        persist_embedding_space(foreign, using="default")
        value = foreign.space_key
    else:
        value = values[field]
    assert DocumentEmbeddingState.objects.filter(active_generation_id=first.generation_key).update(**{field: value}) == 1
    assert query.validate_dense_snapshot(case.snapshot, captured, using="default") == ()
    assert freeze(case).selections == ()


def test_invalid_selected_manifest_still_raises_and_duplicate_owner_stays_constrained(transport, monkeypatch, settings):
    case = corpus(1, transport[0], monkeypatch, settings)
    actual = build(case)
    state = DocumentEmbeddingState.objects.get(active_generation_id=actual.generation_key)
    values = {name: getattr(state, name) for name in (
        "model_label", "document_pkid", "document_id", "collection_id", "representation_key", "space_id")}
    with pytest.raises(IntegrityError), transaction.atomic():
        DocumentEmbeddingState.objects.create(**values)
    assert EmbeddingGeneration.objects.filter(pk=actual.generation_key).update(manifest_sha256="invalid") == 1
    with pytest.raises(SourceScopeUnavailable, match="^source_scope_unavailable$"):
        freeze(case)


def test_same_uuid_concrete_owner_collision_still_fails_before_dense_read(transport, monkeypatch, settings):
    case = corpus(1, transport[0], monkeypatch, settings)
    original = case.docs[0]
    foreign = ImageUploadDocument(id=original.id, title="Collision", full_text=original.full_text,
        collection=Collection.objects.create(name="Private collision"), ingested_by=case.owner)
    foreign.save(dont_rechunk=True)
    with dense_reads() as reads, pytest.raises(SourceScopeUnavailable, match="^source_scope_unavailable$"):
        freeze(case)
    assert reads == []


def test_expired_original_source_deadline_prevents_second_batch(transport, monkeypatch, settings):
    case = corpus(65, transport[0], monkeypatch, settings)
    now = [0.0]
    budget = TurnBudget(TurnLimits(), clock=lambda: now[0])
    deadline = budget.remaining_ms() / 1000
    def expire(count):
        if count == 1:
            now[0] = deadline + 0.001
    try:
        with dense_reads(after=expire) as reads:
            with pytest.raises(SourcePreparationLimited, match="source preparation deadline"):
                with source_preparation_scope(deadline, lambda: now[0], budget):
                    freeze(case)
        assert len(reads) == 1
    finally:
        budget.close("dense_sweep_finished")
