"""Neutral preparation cannot restore a rejected scoped occurrence's rank vote."""

import pytest

from apps.chat.services import rag_recovery_preparation as preparation
from apps.chat.services.rag_preparation_order import take_preparation_order
from apps.chat.services.rag_retrieval import fuse_ranked_tool_results
from apps.chat.services.rag_selection import select_evidence
from apps.chat.services.rag_selection_types import (
    PreparedSelection, SelectionCandidate, SelectionLimits, SelectionProfile,
)
from apps.chat.tests.test_native_preparation_order import accept_native
from apps.chat.tests.test_recovery_origins import accept, frame, source
from apps.chat.tests.test_recovery_preparation_order import _invoke
from lib.llm.evidence_guard import ContextLimited


@pytest.mark.parametrize("deferred_outcome", ["none", "neutral_only", "all"])
def test_mixed_completion_retains_each_occurrences_ordinary_vote_policy(monkeypatch, deferred_outcome):
    with frame(monkeypatch) as case:
        a, b, c, d, e, f = (source(pk) for pk in range(1, 7))
        first = accept(case, (a, b))
        scoped = accept(case, (b, d, c, a), ordinary=(c,))
        neutral = accept_native(case, (e, f), (e,))
        pool = fuse_ranked_tool_results([first, scoped, neutral, first])
        assert pool.rows[0]["chunk_id"] == b.chunk_id
        order = take_preparation_order(pool, "question", case.runtime.authorization)
        assert set(order.initial_ids) == {1, 2, 3, 5} and set(order.deferred_ids) == {4, 6}
        assert order.fused_scores == pool.fused_scores
        observed = []

        def prepare(**kwargs):
            current = kwargs["pool"]
            observed.append(current)
            deferred = current.rows[0]["chunk_id"] in {4, 6}
            if deferred and deferred_outcome == "none":
                raise ContextLimited("request_tokenization_limit")
            rows = tuple(row for row in current.rows if not (
                deferred and deferred_outcome == "neutral_only" and row["chunk_id"] == 4))
            candidates = tuple(SelectionCandidate(row["chunk_id"], row["doc_id"],
                row["chunk"], row["text"], .5, rank, source(row["chunk_id"]).source_fingerprint,
                row, token_cost=1, raw_score=.5, scorer_fingerprint="fixed", scoring_kind="pointwise")
                for rank, row in enumerate(rows, 1))
            return PreparedSelection(candidates, "model", 0, 0, 0, None)

        # This unit isolates accepted occurrence votes. Native integration owns
        # current authorization, source revalidation and exact answer payment.
        monkeypatch.setattr(preparation, "revalidate_source_candidates", lambda rows, *a, **k: rows)
        result = _invoke(case, order, pool, prepare)
        selected = select_evidence(result.candidates, profile=SelectionProfile("test", 1, 0, "v1"),
            limits=SelectionLimits(1, 1, 10), score_status=result.score_status)
        if deferred_outcome == "all":
            assert selected.candidates[0].chunk_id == b.chunk_id
            assert len(result.candidates) == 6 and result.omitted_optional == ()
        else:
            assert selected.candidates[0].chunk_id == a.chunk_id, "failed scoped occurrence boosted b's tie rank"
            assert {item.chunk_id for item in result.omitted_optional} == ({4, 6} if deferred_outcome == "none" else {4})
            ranks = {1: 1, 3: 2, 5: 3, 2: 4} | ({6: 5} if deferred_outcome == "neutral_only" else {})
            assert {row.chunk_id: row.fused_rank for row in result.candidates} == ranks
            initial_votes = dict(observed[0].fused_scores)
            assert initial_votes["[doc:document chunk:1]"] == 1 / 61
            assert initial_votes["[doc:document chunk:2]"] == 1 / 62
            assert initial_votes["[doc:document chunk:5]"] == 1 / 61
