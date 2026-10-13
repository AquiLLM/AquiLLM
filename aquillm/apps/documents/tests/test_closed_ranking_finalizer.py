"""Native production ranking finalization closes one freshly checked union."""

from dataclasses import replace

import pytest
from django.db import connection
from django.test.utils import CaptureQueriesContext

from apps.collections.models import CollectionPermission
from apps.documents.models import RawTextDocument, TextChunk
from apps.documents.services import chunk_search, chunk_search_authorization
from apps.documents.services import chunk_search_scope as scopes
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from lib.embeddings.tests.test_document_provider import transport as transport

from .test_ranking_final_union_authorization import handoff as handoff

pytestmark = pytest.mark.django_db(transaction=True)


def production(handoff, monkeypatch, settings, *, empty=False, after_score=None):
    case, scope, _, snapshot, ranking = handoff
    settings.RAG_FULL_TEXT_ENABLED = False
    settings.RAG_RRF_ENABLED = False
    settings.KG_OVERLAY_ENABLED = False
    if empty:
        ranking = replace(ranking, ranked_results=())
    monkeypatch.setattr(chunk_search, "load_query_embedding", lambda _: None)
    monkeypatch.setattr(
        chunk_search, "collect_hybrid_candidate_snapshot", lambda *a, **k: snapshot
    )

    def score(*args, **kwargs):
        if after_score:
            after_score()
        return ranking

    monkeypatch.setattr(chunk_search, "materialize_and_rerank_candidates", score)
    return lambda: chunk_search.text_chunk_search(
        TextChunk, "query", 2, case.docs, authorization_context=scope.authorization
    )


@pytest.mark.parametrize("empty", [False, True])
def test_actual_native_final_union_uses_two_fresh_owner_sweeps(
    handoff, monkeypatch, settings, empty
):
    _, _, rows, _, ranking = handoff
    run = production(handoff, monkeypatch, settings, empty=empty)
    real_final, real_validate = (
        chunk_search.finalize_ranking,
        scopes.FrozenRetrievalScope.validate,
    )
    checks, outputs = [], []

    def validate(current):
        result = real_validate(current)
        checks.append(current)
        return result

    def final(*args, **kwargs):
        checks.clear()
        with CaptureQueriesContext(connection) as statements:
            result = real_final(*args, **kwargs)
        outputs.append(result)
        owners = [
            s for s in statements if "UNION ALL" in s["sql"] and "sha256" in s["sql"]
        ]
        assert len(checks) == len(owners) == 2
        return result

    monkeypatch.setattr(scopes.FrozenRetrievalScope, "validate", validate)
    monkeypatch.setattr(chunk_search, "finalize_ranking", final)
    run()
    snapshot, channels, scores = outputs[0]
    assert channels[1] == rows and snapshot.baseline_candidates == rows
    assert tuple(channels[2]) == (() if empty else ranking.ranked_results)
    assert scores.candidate_order == (() if empty else (rows[0].pk, rows[2].pk))
    assert tuple(s.chunk_pk for s in scores.scores) == scores.candidate_order


def test_original_witness_replacement_during_inference_cannot_be_adopted(
    handoff, monkeypatch, settings
):
    _, scope, _, _, _ = handoff
    runtime, original = scope.runtime, scope.runtime._frozen_retrieval_scope

    def replace_witness():
        runtime._frozen_retrieval_scope = replace(original)

    run = production(handoff, monkeypatch, settings, after_score=replace_witness)
    with pytest.raises(SourceScopeUnavailable):
        run()


def test_captured_native_mode_cannot_switch_to_full_text_during_inference(
    handoff, monkeypatch, settings
):
    run = production(
        handoff,
        monkeypatch,
        settings,
        after_score=lambda: setattr(settings, "RAG_FULL_TEXT_ENABLED", True),
    )
    with pytest.raises(SourceScopeUnavailable):
        run()


def test_changed_hydrated_body_cannot_keep_its_score_in_native_handoff(
    handoff, monkeypatch, settings
):
    _, _, rows, _, _ = handoff
    run = production(
        handoff,
        monkeypatch,
        settings,
        after_score=lambda: setattr(rows[2], "content", "Changed hydrated body"),
    )
    real_final, outputs = chunk_search.finalize_ranking, []

    def final(*args, **kwargs):
        result = real_final(*args, **kwargs)
        outputs.append(result)
        return result

    monkeypatch.setattr(chunk_search, "finalize_ranking", final)
    run()
    snapshot, channels, scores = outputs[0]
    assert channels[2] == [rows[0]]
    assert snapshot.trigram_chunk_ids == (rows[0].pk, rows[1].pk)
    assert scores.candidate_order == (rows[0].pk,)
    assert tuple(s.chunk_pk for s in scores.scores) == scores.candidate_order


@pytest.mark.parametrize("change", ["source", "permission", "empty_permission"])
def test_native_union_rechecks_original_authority_after_incoming_rows(
    handoff, monkeypatch, settings, change
):
    case, _, _, _, _ = handoff
    scoring, changes, published = [], [], []
    real_incoming = chunk_search_authorization.reauthorized_baseline

    def incoming(*args, **kwargs):
        result = real_incoming(*args, **kwargs)
        if scoring and not changes:
            if change == "source":
                assert (
                    RawTextDocument.objects.filter(pk=case.docs[1].pk).update(
                        full_text="Changed source with stale stored hash"
                    )
                    == 1
                )
            else:
                CollectionPermission.objects.filter(
                    user=case.owner, collection=case.collection
                ).delete()
            changes.append(change)
        return result

    monkeypatch.setattr(chunk_search_authorization, "reauthorized_baseline", incoming)
    run = production(
        handoff,
        monkeypatch,
        settings,
        empty=change == "empty_permission",
        after_score=lambda: scoring.append(True),
    )
    with pytest.raises(SourceScopeUnavailable):
        published.append(run())
    assert changes == [change]
    assert published == []


def test_staged_completion_cannot_publish_after_original_budget_closes(
    handoff, monkeypatch, settings
):
    _, original, _, _, _ = handoff
    completed, captured = [], []
    real_finish = chunk_search_authorization._finish_search_rows

    def finish(*args, **kwargs):
        live = scopes.current_scope()
        captured.append(
            (live, live.fusion_v2, dict(live.bodies), set(live.lexical_ids))
        )
        result = real_finish(*args, **kwargs)
        original.runtime.budget.close("postcompletion_test_abort")
        completed.append(result)
        return result

    monkeypatch.setattr(chunk_search_authorization, "_finish_search_rows", finish)
    run = production(handoff, monkeypatch, settings)
    with pytest.raises(SourceScopeUnavailable):
        run()
    assert len(completed) == 1
    live, fusion, bodies, lexical = captured[0]
    assert live.fusion_v2 is fusion
    assert live.bodies == bodies and live.lexical_ids == lexical


def test_full_text_enabled_keeps_independent_native_fallback_guards(
    handoff, monkeypatch, settings
):
    from apps.documents.services.closed_search_finalizer import _bind_search_authorizer

    _, scope, rows, snapshot, ranking = handoff
    settings.RAG_FULL_TEXT_ENABLED = True
    authorizer = _bind_search_authorizer(
        chunk_search_authorization.authorized_search_rows, scope.authorization, False
    )
    assert authorizer.header is None
    with CaptureQueriesContext(connection) as statements:
        _, channels, scores = chunk_search_authorization.finalize_ranking(
            snapshot, ranking, authorize=authorizer, query="query", top_k=2
        )
    owners = [s for s in statements if "UNION ALL" in s["sql"] and "sha256" in s["sql"]]
    assert len(owners) == 4
    assert tuple(channels[1]) == rows
    assert tuple(channels[2]) == ranking.ranked_results
    assert scores.candidate_order == (rows[0].pk, rows[2].pk)
