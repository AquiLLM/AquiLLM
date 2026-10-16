"""A mixed native source pool keeps one text model scale through final selection."""

from dataclasses import replace
from time import monotonic

import pytest

from apps.chat.services.rag_retrieval import fuse_ranked_tool_results
from apps.chat.services.rag_selection import select_evidence
from apps.chat.services.rag_selection_hydration import revalidate_selection_candidates
from apps.chat.services.rag_selection_scoring import prepare_selection_candidates
from apps.chat.services.rag_selection_types import SelectionLimits, SelectionProfile
from apps.documents.models import TextChunk
from apps.documents.services.chunk_rerank_results import fingerprint_text
from apps.documents.services.chunk_rerank_window_adapter import WindowSelectionScorer
from apps.documents.services.graph_source_admission_policy import charged_sources
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.services.source_loading import source_query_rows
from lib.tools.search.vector_search import pack_chunk_search_results

from .test_native_graph_search_dispatch import dispatch_case
from .test_window_caption_scoring import TextProvider, verified_counter

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.parametrize("permission_during_score", [False, True])
def test_mixed45_caption_pool_scores_and_selects_late_graph_text(
    monkeypatch, permission_during_score
):
    with dispatch_case(monkeypatch) as case:
        rows = tuple(case.run()[2])
        assert len(rows) == 44 and case.source.budget.sources_used == 45
        graph_id = case.f.chunks["B"].pk
        assert graph_id in case.graph_events[0]
        assert next(i for i, row in enumerate(rows) if row.pk == graph_id) >= 15
        ids = {row.pk for row in rows}
        extra = {pk for pk, _ in charged_sources(case.source.budget)} - ids
        assert len(extra) == 1
        # Reuse the native acquisition's already-charged supporting source. Its
        # current stored content is a caption; its text identity stays exact.
        assert graph_id not in extra
        TextChunk.objects.filter(pk__in=extra).update(modality=TextChunk.Modality.IMAGE)
        (image,) = source_query_rows(TextChunk.objects.filter(pk__in=extra))
        rows = (*rows, image)
        assert len(rows) == 45 and len({row.pk for row in rows}) == 45
        assert sum(row.modality == TextChunk.Modality.IMAGE for row in rows) == 1

        class Provider(TextProvider):
            def score_pair(self, pair, timeout_seconds):
                super().score_pair(pair, timeout_seconds)
                if permission_during_score:
                    case.f.policy.rows = ()
                return (10.0 if pair[1] == case.f.chunks["B"].content else 0.0), pair

        def pixels(_):
            pytest.fail("caption scoring inspected pixels")

        monkeypatch.setattr(TextChunk, "pixel_bytes", property(pixels), raising=False)
        budget, provider = case.source.budget, Provider()
        scorer = WindowSelectionScorer(
            provider,
            budget=budget,
            pair_counter=verified_counter(),
            deadline=monotonic() + 3,
        )
        payload = pack_chunk_search_results(
            rows,
            titles_by_doc_id={doc.id: doc.title for doc in case.f.documents.values()},
            docs_by_doc_id=case.f.documents,
            truncate=lambda value: value,
            image_modality=TextChunk.Modality.IMAGE,
            compact_items=False,
        )
        pool = fuse_ranked_tool_results([payload])
        assert len(pool.rows) == 45
        arguments = dict(
            pool=pool,
            primary_query="Atlas",
            authorization=case.f.authorization,
            deadline=monotonic() + 3,
            allow_new_scores=True,
            scorer=scorer,
            turn_budget=budget,
            source_mode=True,
            token_ceiling=7000,
        )
        if permission_during_score:
            with pytest.raises(SourceScopeUnavailable):
                prepare_selection_candidates(**arguments)
            assert len(provider.seen) == budget.pairs_used["final"] == 45
            return
        prepared = prepare_selection_candidates(**arguments)
        assert prepared.score_status == "model" and prepared.new_pairs == 45
        assert len(provider.seen) == budget.pairs_used["final"] == 45
        assert budget.pairs_used["acquisition"] == 0
        assert len(prepared.candidates) == 45
        assert ("Atlas", image.content) in provider.seen
        assert all(
            candidate.raw_score in (0.0, 10.0) for candidate in prepared.candidates
        )
        scores = scorer.score_windows("Atlas", rows)
        old_policy = fingerprint_text(
            f"window-max-v1:{provider.scorer_fingerprint}:{scorer.pair_counter.identity}"
        )
        for policy, expected in (
            (old_policy, "rank_fallback"),
            (scorer.scorer_fingerprint, "model"),
        ):
            retained = prepare_selection_candidates(
                **{
                    **arguments,
                    "pool": replace(
                        pool,
                        source_score_sets=(replace(scores, scorer_fingerprint=policy),),
                    ),
                    "allow_new_scores": False,
                }
            )
            assert retained.score_status == expected and retained.new_pairs == 0
        assert len(provider.seen) == 45
        selected = select_evidence(
            prepared.candidates,
            profile=SelectionProfile("test", 1, 0, "v1"),
            limits=SelectionLimits(12, 12, 7000),
            score_status=prepared.score_status,
        )
        assert selected.candidates[0].chunk_id == graph_id
        assert (
            revalidate_selection_candidates(selected.candidates, case.f.authorization)
            == selected.candidates
        )
        case.f.policy.rows = ()
        with pytest.raises(SourceScopeUnavailable):
            revalidate_selection_candidates(selected.candidates, case.f.authorization)
        assert budget.sources_used == 45 and budget.limits.final_pairs == 45
        assert budget.limits.final_scoring_ms == 3000
