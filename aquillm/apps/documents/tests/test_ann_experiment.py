"""Tiny actual-DB correctness controls; no latency or plan measurements."""

from dataclasses import replace
from copy import deepcopy

import pytest
from django.db import DatabaseError, transaction

from apps.collections.models import CollectionPermission
from apps.documents.models import TextChunk
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.services.source_loading import source_runtime_scope
from apps.documents.tests.ann_experiment_support import ann_case as ann_case
from apps.ingestion.tests.output_support import upload as upload

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.parametrize("arm", ["production_exact", "experimental_direct_distance"])
def test_rank_excludes_hidden_retired_and_incompatible_nearest_rows(ann_case, arm):
    from apps.chat.evals.ann_experiment import rank_experiment

    case = ann_case
    with source_runtime_scope(case.runtime):
        result = rank_experiment(case.scope, case.observation, limit=3, arm=arm)
    assert tuple(row.chunk_id for row in result) == tuple(row.pk for row in case.chunks)
    assert {row.embedding_generation_key for row in result} == {
        case.current.generation_key
    }
    assert case.old.generation_key != case.current.generation_key
    assert all(selection.generation_key is not None for selection in case.scope.representation.selections)
    assert all(row.representation_key == case.chunks[0].representation_generation_id for row in result)
    assert len(case.native_sources) == 3 and all(record['artifact_bytes'] for record in case.native_sources)
    assert case.observation.compatibility_key == case.space.space_key
    assert case.observation.receipt["observed_identity"]["model"] == case.space.model


@pytest.mark.parametrize("arm", ["production_exact", "experimental_direct_distance"])
def test_current_permission_revocation_blocks_both_arms(ann_case, arm):
    from apps.chat.evals.ann_experiment import rank_experiment

    CollectionPermission.objects.filter(
        user=ann_case.actor, collection=ann_case.public
    ).delete()
    with source_runtime_scope(ann_case.runtime), pytest.raises(SourceScopeUnavailable):
        rank_experiment(ann_case.scope, ann_case.observation, limit=3, arm=arm)


def test_experimental_arm_cannot_borrow_another_turn_or_raise_candidate_cap(ann_case):
    from apps.chat.evals.ann_experiment import rank_experiment

    with pytest.raises(ValueError):
        rank_experiment(
            ann_case.scope,
            ann_case.observation,
            limit=3,
            arm="experimental_direct_distance",
        )
    with source_runtime_scope(ann_case.runtime), pytest.raises(ValueError):
        rank_experiment(
            ann_case.scope,
            ann_case.observation,
            limit=46,
            arm="experimental_direct_distance",
        )


@pytest.mark.parametrize("arm", ["production_exact", "experimental_direct_distance"])
def test_unknown_query_identity_never_returns_candidates(ann_case, arm):
    from apps.chat.evals.ann_experiment import rank_experiment

    unknown = replace(ann_case.observation, compatibility_key=None)
    with source_runtime_scope(ann_case.runtime):
        assert rank_experiment(ann_case.scope, unknown, limit=3, arm=arm) == ()


@pytest.mark.parametrize("arm", ["production_exact", "experimental_direct_distance"])
def test_current_native_chunk_is_immutable_and_original_positive_remains(ann_case, arm):
    from apps.chat.evals.ann_experiment import rank_experiment

    with pytest.raises(DatabaseError, match='immutable_citation_chunk'), transaction.atomic():
        TextChunk.objects.filter(pk=ann_case.chunks[0].pk).update(content="changed source")
    with source_runtime_scope(ann_case.runtime):
        result = rank_experiment(ann_case.scope, ann_case.observation, limit=3, arm=arm)
    assert tuple(row.chunk_id for row in result) == tuple(
        row.pk for row in ann_case.chunks
    )


@pytest.mark.parametrize('arm', ['production_exact', 'experimental_direct_distance'])
def test_actual_canonical_owner_edit_invalidates_original_snapshot(ann_case, arm):
    from apps.chat.evals.ann_experiment import rank_experiment
    doc = ann_case.docs[0]
    doc.full_text += ' A new canonical source edit.'
    doc.save(dont_rechunk=True)
    with source_runtime_scope(ann_case.runtime), pytest.raises(SourceScopeUnavailable):
        rank_experiment(ann_case.scope, ann_case.observation, limit=3, arm=arm)


@pytest.mark.parametrize('media', [['malformed'], 'malformed', 1])
def test_experiment_uses_actual_bounded_payload_and_rejects_malformed_media(ann_case, monkeypatch, media):
    from apps.chat.evals import ann_experiment
    from apps.chat.evals.ann_candidate_validation import validate_candidate
    original, captured = ann_experiment.bounded_dense_payloads, []

    def payloads(rows, *, budget):
        assert budget is ann_case.runtime.budget
        assert rows.query.high_mark == 3 and rows.query.low_mark == 0
        result = original(rows, budget=budget)
        captured.extend(result)
        return result

    monkeypatch.setattr(ann_experiment, 'bounded_dense_payloads', payloads)
    with source_runtime_scope(ann_case.runtime):
        ranked = ann_experiment.rank_experiment(ann_case.scope, ann_case.observation,
            limit=3, arm='experimental_direct_distance')
        assert len(ranked) == len(captured) == 3
        representations = {s.generation_key: s.representation_key for s in ann_case.scope.dense.selections}
        valid = captured[0]
        assert validate_candidate(valid, space=ann_case.space, representation_keys=representations) == ranked[0]
        malformed = deepcopy(valid)  # Detached negative; never mutate guarded persisted provenance.
        malformed['matched_manifest_entry']['media_artifact'] = media
        assert validate_candidate(malformed, space=ann_case.space, representation_keys=representations) is None
