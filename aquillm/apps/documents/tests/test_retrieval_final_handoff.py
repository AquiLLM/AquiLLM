"""Final permission traversal cannot outlive or detach the captured evidence."""

from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest
from django.db import connection
from django.test import override_settings

from apps.chat.services.retrieval_authorization import (
    resolve_document_retrieval_authorization,
)
from apps.collections.models import CollectionPermission
from apps.documents.models import TextChunk
from apps.documents.services import (
    chunk_search,
)
from apps.documents.services import (
    chunk_search_authorization as authorization,
)
from apps.documents.services import embedding_index_store as store
from apps.documents.services.chunk_rerank_results import (
    PassageScore,
    RerankScoreSet,
    fingerprint_text,
)
from apps.documents.services.chunk_search_candidates import (
    collect_hybrid_candidate_snapshot,
)
from apps.documents.services.chunk_search_lifetime import enter_search
from apps.documents.services.chunk_search_query_observation import query_embedding
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.services.source_loading import SourcePreparationLimited
from apps.documents.tests.test_embedding_generation import legacy as legacy
from apps.documents.tests.test_embedding_generation import stage
from lib.embeddings.tests.test_document_provider import transport as transport
from lib.llm.turn_context import TurnContext, bind_turn
from lib.retrieval.turn_budget import TurnBudget, TurnLimits

pytestmark = pytest.mark.django_db(transaction=True)


@contextmanager
def default_turn(legacy, transport):
    doc = legacy[0]
    CollectionPermission.objects.create(
        user=doc.ingested_by, collection=doc.collection, permission="VIEW"
    )
    now = [0.0]
    budget = TurnBudget(TurnLimits(), clock=lambda: now[0])
    with (
        override_settings(
            KG_OVERLAY_ENABLED=False,
            DOCUMENT_EMBEDDING_SPACE_KEY=transport[0].space_key,
        ),
        bind_turn(TurnContext(budget)),
    ):
        auth = resolve_document_retrieval_authorization(
            doc.ingested_by,
            SimpleNamespace(collections=[doc.collection_id]),
            [doc],
            None,
        )
        yield auth, budget, now


@pytest.mark.parametrize(
    "mutation", ["none", "expire", "cancel", "source", "body", "pointer"]
)
def test_default_final_permission_is_bounded_and_followed_by_source_checks(
    legacy, transport, monkeypatch, mutation
):
    _, first, _, _ = stage(legacy, transport)
    store.activate_embedding_generation(first, using="default")
    _, replacement, _, _ = stage(legacy, transport)
    with default_turn(legacy, transport) as (auth, budget, now):
        lifetime, scope = enter_search(TextChunk, [legacy[0]], auth, "unrelated", False)
        with lifetime:
            snapshot = collect_hybrid_candidate_snapshot(
                TextChunk,
                "unrelated",
                2,
                [legacy[0]],
                query_embedding=query_embedding("unrelated"),
            )
            assert snapshot.vector_results
            original = authorization.reauthorized_baseline
            observed = []

            def at_final_permission(rows, context):
                with connection.cursor() as cursor:
                    cursor.execute("SHOW statement_timeout")
                    observed.append((connection.in_atomic_block, cursor.fetchone()[0]))
                result = original(rows, context)
                assert scope.runtime.budget is budget
                if mutation == "expire":
                    now[0] = 16.0
                elif mutation == "cancel":
                    budget.close("cancelled")
                elif mutation == "source":
                    doc = legacy[0]
                    doc.full_text = "changed"
                    doc.save(dont_rechunk=True)
                    saved = TextChunk.objects.get(doc_id=doc.id)
                    assert saved.content == "source" and saved.legacy_source_snapshot_id is not None
                    assert saved.legacy_source_snapshot.canonical_text == "source"
                    assert saved.legacy_source_snapshot.captured_source_hash == legacy[1].source_hash
                    assert doc.full_text_hash != legacy[1].source_hash
                    assert scope.representation.selections[0].document.source_hash == legacy[1].source_hash
                elif mutation == "body":
                    TextChunk.objects.update(content="changed")
                elif mutation == "pointer":
                    store.activate_embedding_generation(replacement, using="default")
                return result

            monkeypatch.setattr(
                authorization, "reauthorized_baseline", at_final_permission
            )
            try:
                rows = authorization.authorized_search_rows(
                    snapshot.vector_results,
                    authorization_context=auth,
                    hybrid_requested=False,
                )
            except (SourceScopeUnavailable, SourcePreparationLimited):
                rows = ()
            assert bool(rows) is (mutation == "none")
            assert len(observed) == 1
            assert observed[0][0] and observed[0][1] != "0"


@pytest.mark.parametrize("mutation", ["none", "body", "pointer"])
def test_scored_search_sidecar_matches_final_rows_after_permission_mutation(
    legacy, transport, monkeypatch, mutation
):
    _, first, _, _ = stage(legacy, transport)
    store.activate_embedding_generation(first, using="default")
    _, replacement, _, _ = stage(legacy, transport)
    original_ranking = chunk_search.materialize_and_rerank_candidates
    original_authorization = authorization.reauthorized_baseline
    original_diagnostics = chunk_search.completed_search_diagnostics
    ready = []
    traversals = []
    diagnostics_rows = []

    def scored(*args, **kwargs):
        result = original_ranking(*args, **kwargs)
        row = result.ranked_results[0]
        score = PassageScore(
            row.pk,
            row.doc_id,
            row.chunk_number,
            fingerprint_text(row.content),
            "pair",
            0.9,
        )
        ready.append(True)
        return replace(
            result,
            score_set=RerankScoreSet(
                "v2",
                fingerprint_text("unrelated"),
                "scorer",
                "pool",
                "pointwise",
                "complete",
                (row.pk,),
                (score,),
            ),
        )

    def final_permission(rows, context):
        result = original_authorization(rows, context)
        if ready and rows:
            traversals.append(True)
            if len(traversals) == 2:
                if mutation == "body":
                    TextChunk.objects.update(content="changed")
                elif mutation == "pointer":
                    store.activate_embedding_generation(replacement, using="default")
        return result

    def diagnostics(*args, **kwargs):
        diagnostics_rows.append(tuple(row.pk for row in args[3]))
        return original_diagnostics(*args, **kwargs)

    monkeypatch.setattr(chunk_search, "materialize_and_rerank_candidates", scored)
    monkeypatch.setattr(authorization, "reauthorized_baseline", final_permission)
    monkeypatch.setattr(chunk_search, "completed_search_diagnostics", diagnostics)
    with default_turn(legacy, transport) as (auth, _, _):
        vectors, _, rows, details = chunk_search.text_chunk_search(
            TextChunk, "unrelated", 2, [legacy[0]], authorization_context=auth
        )
    returned = tuple(row.pk for row in rows)
    assert len(traversals) >= 2
    assert bool(returned) is (mutation == "none")
    assert diagnostics_rows == [returned]
    assert tuple(row.pk for row in vectors) == returned
    sidecar = details.get("_score_set", {})
    assert tuple(sidecar.get("candidate_order", ())) == returned
    assert tuple(score["chunk_pk"] for score in sidecar.get("scores", ())) == returned
