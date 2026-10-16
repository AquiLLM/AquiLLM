"""Native pool reaches semantic selection after fresh source/generation closure."""

import pytest

from apps.documents.models import TextChunk
from lib.embeddings.tests.test_document_provider import transport as transport

from .test_native_graph_search_dispatch import dispatch_case

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.parametrize("graph_on", [True, False])
def test_final_semantic_selection_scores_native_graph_beyond15_before_cutoff(
    monkeypatch, graph_on
):
    from time import monotonic

    from apps.chat.services.rag_evidence import build_selected_evidence_packet
    from apps.chat.services.rag_retrieval import fuse_ranked_tool_results
    from apps.chat.services.rag_selection import select_evidence
    from apps.chat.services.rag_selection_hydration import (
        revalidate_selection_candidates,
    )
    from apps.chat.services.rag_selection_scoring import prepare_selection_candidates
    from apps.chat.services.rag_selection_types import SelectionLimits, SelectionProfile
    from apps.documents.services.chunk_rerank_window_adapter import (
        WindowSelectionScorer,
    )
    from lib.tools.search.vector_search import pack_chunk_search_results

    class Provider:
        """Controlled external inference; acquisition/score/selection stay real."""

        def __init__(self, graph_text):
            self.graph_text, self.seen = graph_text, []

        def score_pair(self, pair, timeout_seconds):
            self.seen.append(pair)
            return (10.0 if pair[1] == self.graph_text else 0.0), pair

    with dispatch_case(monkeypatch, graph_on=graph_on) as case:
        rows = case.run()[2]
        budget = case.source.budget
        assert len(rows) == (44 if graph_on else 45) and case.inference == []
        graph_id = case.f.chunks["B"].pk
        if graph_on:
            assert graph_id in case.graph_events[0]
            assert next(i for i, row in enumerate(rows) if row.pk == graph_id) >= 15
        else:
            assert graph_id not in {row.pk for row in rows}
        provider = Provider(case.f.chunks["B"].content)
        scorer = WindowSelectionScorer(
            provider, budget=budget, pair_counter=lambda q, d: len(q) + len(d) + 8,
            deadline=monotonic() + 3,
        )
        payload = pack_chunk_search_results(
            rows,
            titles_by_doc_id={doc.id: doc.title for doc in case.f.documents.values()},
            docs_by_doc_id={doc.id: doc for doc in case.f.documents.values()},
            truncate=lambda value: value,
            image_modality=TextChunk.Modality.IMAGE,
            compact_items=False,
        )
        pool = fuse_ranked_tool_results([payload])
        prepared = prepare_selection_candidates(
            pool=pool, primary_query="Atlas", authorization=case.f.authorization,
            deadline=monotonic() + 3, allow_new_scores=True, scorer=scorer,
            turn_budget=budget, source_mode=True, token_ceiling=7000,
        )
        assert prepared.score_status == "model" and prepared.new_pairs > 0
        assert {candidate.chunk_id for candidate in prepared.candidates} == {
            row.pk for row in rows}
        if graph_on:
            graph_candidate = next(
                item for item in prepared.candidates if item.chunk_id == graph_id)
            assert graph_candidate.raw_score == 10.0
            assert ("Atlas", case.f.chunks["B"].content) in provider.seen
        assert budget.pairs_used["acquisition"] == 0
        assert 0 < budget.pairs_used["final"] <= 45
        assert budget.limits.retrieval_ms == 15000
        assert budget.limits.final_scoring_ms == 3000
        assert budget.sources_used <= 45
        assert budget.text_used["materialized"] <= 250000
        assert budget.text_used["tokenized"] <= 1000000
        selected = select_evidence(
            prepared.candidates, profile=SelectionProfile("test", 1, 0, "v1"),
            limits=SelectionLimits(12, 12, 7000), score_status=prepared.score_status,
        )
        assert revalidate_selection_candidates(
            selected.candidates, case.f.authorization) == selected.candidates
        packet = build_selected_evidence_packet(
            selected, query="Atlas", search_scope="selected")
        assert 0 < len(packet.chunks) <= 12
        assert packet.total_tokens <= 7000
        assert (graph_id in {row["chunk_id"] for row in packet.chunks}) is graph_on
        if graph_on:
            assert packet.chunks[0]["chunk_id"] == graph_id
        for row in packet.chunks:
            assert row["citation"] == f"[doc:{row['doc_id']} chunk:{row['chunk_id']}]"


def test_final_dense_intersection_withdraws_late_failed_generation(
    transport, monkeypatch, settings
):
    from apps.documents.models import EmbeddingGeneration
    from apps.documents.services import chunk_search_authorization as owner
    from apps.documents.services.native_graph_search_closure import (
        _try_native_search_closure,
    )

    from ._native_graph_search_closure_support import dense_case

    with dense_case(transport, monkeypatch, settings) as (scope, snapshot, authorize):
        original, dense = owner.reauthorized_baseline, scope.dense
        key = dense.generation_keys[0]

        def incoming(*args, **kwargs):
            result = original(*args, **kwargs)
            assert (
                EmbeddingGeneration.objects.filter(pk=key).update(state="failed") == 1
            )
            return result

        monkeypatch.setattr(owner, "reauthorized_baseline", incoming)
        result = _try_native_search_closure(
            snapshot,
            "source",
            15,
            authorize=authorize,
            dependencies=None,
            overlay_enabled=False,
        )
        assert result.final_rows == ((), (), []) and result.hybrid_pool == ()
        assert result.ranking.combined_candidates == () and result.score_set is None
        assert scope.dense is dense and dense.generation_keys == (key,)
