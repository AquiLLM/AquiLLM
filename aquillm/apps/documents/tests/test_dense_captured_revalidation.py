"""Dense revalidation reads captured owners but validates every source owner."""

from dataclasses import replace

import pytest

from apps.documents.models import EmbeddingGeneration, RawTextDocument
from apps.documents.services import embedding_dense_rows
from apps.documents.services import embedding_index_query as query
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.services.source_deadline import source_preparation_scope
from apps.documents.services.source_loading import SourcePreparationLimited
from apps.documents.tests.dense_sweep_support import (
    assert_scalar_reads,
    build,
    corpus,
    dense_reads,
    freeze,
)
from lib.embeddings.tests.test_document_provider import transport as transport
from lib.retrieval.turn_budget import TurnBudget, TurnLimits

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.parametrize("captured_count", [0, 3])
def test_revalidation_dense_work_tracks_only_captured_owners(
    captured_count, transport, monkeypatch, settings
):
    case = corpus(65, transport[0], monkeypatch, settings)
    for number in range(65 - captured_count, 65):
        build(case, number)
    snapshot = freeze(case)
    assert len(snapshot.selections) == captured_count
    real_rows, scanned = embedding_dense_rows._batch_rows, []
    real_digest, digests = query.dense_generation_snapshot_digest, []
    real_representation, representations = query.validate_representation_snapshot, []

    def rows(identities, *, using):
        result = real_rows(identities, using=using)
        scanned.extend(identities)
        return result

    def digest(*args, **kwargs):
        result = real_digest(*args, **kwargs)
        digests.append(result)
        return result

    def validate(representation, *, using):
        result = real_representation(representation, using=using)
        representations.append(representation)
        return result

    monkeypatch.setattr(embedding_dense_rows, "_batch_rows", rows)
    monkeypatch.setattr(query, "dense_generation_snapshot_digest", digest)
    monkeypatch.setattr(query, "validate_representation_snapshot", validate)
    with dense_reads() as reads:
        keys = query.validate_dense_snapshot(case.snapshot, snapshot, using="default")
    assert keys == tuple(s.generation_key for s in snapshot.selections)
    assert representations == [case.snapshot]
    assert len(representations[0].selections) == 65
    assert len(scanned) == captured_count
    assert {row[2] for row in scanned} == {
        s.document_id for s in snapshot.selections
    }
    assert_scalar_reads(reads, int(bool(captured_count)))
    assert digests == [snapshot.manifest_sha256]


def test_uncaptured_source_owner_still_requires_fresh_valid_representation(
    transport, monkeypatch, settings
):
    case = corpus(2, transport[0], monkeypatch, settings)
    build(case)
    snapshot = freeze(case)
    assert len(snapshot.selections) == 1
    assert RawTextDocument.objects.filter(pk=case.docs[1].pk).update(
        full_text="Changed uncaptured owner without updating source hash"
    ) == 1
    with dense_reads() as reads:
        with pytest.raises(SourceScopeUnavailable):
            query.validate_dense_snapshot(case.snapshot, snapshot, using="default")
    assert reads == []


@pytest.mark.parametrize("change", ["manifest", "keys", "duplicate", "owner"])
def test_original_dense_invariants_are_checked_before_current_reads(
    change, transport, monkeypatch, settings
):
    case = corpus(2, transport[0], monkeypatch, settings)
    build(case)
    snapshot = freeze(case)
    selection = snapshot.selections[0]
    if change == "manifest":
        snapshot = replace(snapshot, manifest_sha256="f" * 64)
    elif change == "keys":
        snapshot = replace(snapshot, generation_keys=())
    elif change == "duplicate":
        snapshot = replace(snapshot, selections=(selection, selection))
    else:
        crossed = replace(selection, document_pkid=case.docs[1].pkid)
        snapshot = replace(snapshot, selections=(crossed,))
    with dense_reads() as reads:
        with pytest.raises(SourceScopeUnavailable):
            query.validate_dense_snapshot(case.snapshot, snapshot, using="default")
    assert reads == []


def test_malformed_current_captured_manifest_remains_a_hard_failure(
    transport, monkeypatch, settings
):
    case = corpus(1, transport[0], monkeypatch, settings)
    selected = build(case)
    snapshot = freeze(case)
    assert EmbeddingGeneration.objects.filter(pk=selected.generation_key).update(
        manifest_sha256="invalid"
    ) == 1
    with pytest.raises(SourceScopeUnavailable):
        query.validate_dense_snapshot(case.snapshot, snapshot, using="default")


def test_expiry_after_real_captured_owner_read_prevents_return(
    transport, monkeypatch, settings
):
    case = corpus(2, transport[0], monkeypatch, settings)
    build(case)
    snapshot = freeze(case)
    now = [0.0]
    budget = TurnBudget(TurnLimits(), clock=lambda: now[0])
    deadline = budget.remaining_ms() / 1000

    def expire(count):
        assert count == 1
        now[0] = deadline + 0.001

    returned = []
    try:
        with dense_reads(after=expire) as reads:
            with pytest.raises(SourcePreparationLimited):
                with source_preparation_scope(deadline, lambda: now[0], budget):
                    returned.append(query.validate_dense_snapshot(
                        case.snapshot, snapshot, using="default"
                    ))
        assert len(reads) == 1
        assert returned == []
    finally:
        budget.close("captured_dense_test_finished")


def test_empty_dense_capture_retains_deadline_after_full_representation_check(
    transport, monkeypatch, settings
):
    case = corpus(1, transport[0], monkeypatch, settings)
    snapshot = freeze(case)
    assert snapshot.selections == ()
    now = [0.0]
    budget = TurnBudget(TurnLimits(), clock=lambda: now[0])
    deadline = budget.remaining_ms() / 1000
    real_validate, checked = query.validate_representation_snapshot, []

    def validate(representation, *, using):
        result = real_validate(representation, using=using)
        checked.append(representation)
        now[0] = deadline + 0.001
        return result

    monkeypatch.setattr(query, "validate_representation_snapshot", validate)
    returned = []
    try:
        with pytest.raises(SourcePreparationLimited):
            with source_preparation_scope(deadline, lambda: now[0], budget):
                returned.append(query.validate_dense_snapshot(
                    case.snapshot, snapshot, using="default"
                ))
        assert checked == [case.snapshot]
        assert returned == []
    finally:
        budget.close("captured_dense_test_finished")
