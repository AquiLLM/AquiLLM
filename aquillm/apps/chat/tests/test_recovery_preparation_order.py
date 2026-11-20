"""Rejected optional work cannot contribute ordinary rank or unbounded ancestry."""

from contextlib import contextmanager
from types import SimpleNamespace
import sys

import pytest

from apps.chat.services import rag_recovery_preparation as preparation
from apps.chat.services.rag_recovery_origins import take_recovery_order
from apps.chat.services.rag_retrieval import fuse_ranked_tool_results
from apps.chat.services.rag_selection import select_evidence
from apps.chat.services.rag_selection_types import (
    PreparedSelection, SelectionCandidate, SelectionLimits, SelectionProfile,
)
from apps.chat.tests.test_recovery_origins import accept, frame, source
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from lib.llm.evidence_guard import ContextLimited
from lib.retrieval.operation import Operation, _OPERATION


def _invoke(case, order, pool, prepare, *, clock=None):
    scorer = SimpleNamespace(provider=object(), pair_counter=object(),
        scorer_fingerprint="fixed", scoring_kind="pointwise", budget=case.budget,
        rerank_state=SimpleNamespace(can_admit=lambda: True, _failed=False))
    return preparation.prepare_recovery_selection(order, prepare=prepare, pool=pool,
        primary_query="question", authorization=case.runtime.authorization,
        deadline=3, allow_new_scores=True, scorer=scorer, chunk_loader=None,
        clock=clock or (lambda: case.now[0]), turn_budget=case.budget,
        source_mode=True, token_ceiling=7000, source_windows=None,
        # This unit isolates origin/rank and clock fences. The native answer
        # integration test exercises actual request payment and SDK validation.
        answer_preflight=lambda prepared, pin: select_evidence(prepared.candidates,
            profile=SelectionProfile("test",1,0,"v1"), limits=SelectionLimits(1,1,10),
            score_status=prepared.score_status))


@pytest.mark.parametrize("optional_outcome", ["text_limit", "revoked"])
def test_rejected_recovery_occurrence_cannot_change_ordinary_rrf_or_score_ties(monkeypatch, optional_outcome):
    with frame(monkeypatch) as case:
        a,b,c,d = (source(pk) for pk in range(1,5))
        first = accept(case, (a,b))
        second = accept(case, (b,d,c,a), ordinary=(c,))
        pool = fuse_ranked_tool_results([first,second,first])
        assert pool.rows[0]["chunk_id"] == 2, "fixture must expose the recovery/replay mixed-rank boost"
        order = take_recovery_order(pool, "question", case.runtime.authorization)
        assert order.ordinary_ids == (1,3,2) and order.optional_ids == (4,)
        observed, events = [], []
        monkeypatch.setattr(preparation, "publish", lambda event,data: events.append((event,data)))

        def prepare(**kwargs):
            current = kwargs["pool"]
            observed.append(current)
            if current.rows[0]["chunk_id"] == 4 and optional_outcome == "text_limit":
                raise ContextLimited("request_tokenization_limit")
            candidates = tuple(SelectionCandidate(row["chunk_id"], row["doc_id"],
                row["chunk"], row["text"], .5, rank, source(row["chunk_id"]).source_fingerprint,
                row, token_cost=1, raw_score=.5, scorer_fingerprint="fixed", scoring_kind="pointwise")
                for rank,row in enumerate(current.rows,1))
            return PreparedSelection(candidates, "model", 0, 0, 0, None)

        def revalidate(rows, *args, **kwargs):
            # Isolate the helper's final ranking contract: ordinary source
            # hydration/revision/authentication is exercised by the native
            # integration suite, while this seam reports its actual survivors.
            return tuple(row for row in rows if optional_outcome != "revoked" or row.chunk_id != 4)

        monkeypatch.setattr(preparation, "revalidate_source_candidates", revalidate)
        result = _invoke(case, order, pool, prepare)
        selection = select_evidence(result.candidates, profile=SelectionProfile("test",1,0,"v1"),
            limits=SelectionLimits(1,1,10), score_status=result.score_status)
        assert selection.candidates[0].chunk_id == 1, "rejected recovery boosted source 2's tie rank"
        assert {row.chunk_id: row.fused_rank for row in result.candidates} == {1:1, 3:2, 2:3}
        assert observed[0].fused_scores == (
            ("[doc:document chunk:1]", 1/61), ("[doc:document chunk:3]", 1/61),
            ("[doc:document chunk:2]", 1/62))
        assert result.acquired_count == 4 and len(result.omitted_optional) == 1
        assert events == [("selection_optional_recovery", {
            "acquired_count": 4, "ordinary_count": 3, "retained_count": 3,
            "scored_count": 3, "omitted_count": 1, "status": "partial_model_ranked"})]


@contextmanager
def _bounded_lines():
    """A deliberately corrupt ancestry must fail without an unbounded RED run."""
    prior, visits = sys.gettrace(), []

    def trace(frame, event, arg):
        if event == "line" and frame.f_code is preparation.prepare_recovery_selection.__code__:
            visits.append(True)
            if len(visits) > 1000:
                raise AssertionError("recovery ancestry traversal exceeded bounded test guard")
        return trace

    sys.settrace(trace)
    try:
        yield
    finally:
        sys.settrace(prior)


@pytest.mark.parametrize("mutation", ["cycle", "foreign", "depth"])
def test_preparation_rejects_unbounded_or_foreign_operation_ancestry(monkeypatch, mutation):
    with frame(monkeypatch) as case:
        a,b = source(1),source(2)
        pool = fuse_ranked_tool_results([accept(case,(a,b),ordinary=(a,))])
        order = take_recovery_order(pool,"question",case.runtime.authorization)
        owner = Operation(case.budget, clock=lambda: case.now[0])
        token = _OPERATION.set(owner)

        def clock():
            if mutation == "cycle":
                owner.parent = owner
            elif mutation == "foreign":
                owner.parent = SimpleNamespace(parent=None, deadline=None, clock=clock, budget=case.budget)
            else:
                cursor = owner
                for _ in range(65):
                    cursor.parent = Operation(case.budget, clock=lambda: case.now[0])
                    cursor = cursor.parent
            return case.now[0]

        try:
            with _bounded_lines(), pytest.raises(SourceScopeUnavailable, match="recovery_original_scope_unavailable"):
                _invoke(case,order,pool,lambda **kwargs: pytest.fail("invalid ancestry reached preparation"),clock=clock)
        finally:
            _OPERATION.reset(token)
