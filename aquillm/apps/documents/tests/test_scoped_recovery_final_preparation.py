"""Optional native recovery must not consume preparation owed to ordinary evidence."""

from time import monotonic

import pytest

from apps.documents.models import TextChunk
from apps.documents.tests import _native_graph_search_closure_support as support
from apps.knowledge_graph.tests import graph_dataset_fixture_v2 as dataset
from apps.knowledge_graph.tests.test_closed_graph_limits_v2 import returned_envelope
from apps.knowledge_graph.tests.test_ready_scope import _DOC_A

pytestmark = pytest.mark.django_db(transaction=True)


def _short_then_long_sources(fixture):
    """Every body is an actual slice of the same freshly captured document."""
    first = fixture.chunks["A"]
    text = fixture.documents["A"].full_text
    short = tuple(end for end in range(5, 29) if end != first.end_position)
    long = tuple(range(5000, 5019))
    assert len(short) == 23 and len(long) == 19 and len(text) > max(long)
    TextChunk.objects.bulk_create([
        TextChunk(doc_id=first.doc_id, content=text[:end], chunk_number=100 + number,
                  start_position=0, end_position=end)
        for number, end in enumerate((*short, *long))
    ])


def _all_short_sources(fixture):
    first = fixture.chunks["A"]
    text = fixture.documents["A"].full_text
    ends = tuple(end for end in range(5, 48) if end != first.end_position)
    assert len(ends) == 42 and len(text) >= max(ends)
    TextChunk.objects.bulk_create([
        TextChunk(doc_id=first.doc_id, content=text[:end], chunk_number=100 + number,
                  start_position=0, end_position=end)
        for number, end in enumerate(ends)
    ])


@pytest.mark.parametrize("recovery,cheap", [(False, False), (True, False), (True, True)],
                         ids=["ordinary-control", "optional-long-recovery", "optional-cheap-wins"])
def test_optional_scoped_recovery_preserves_ordinary_final_preparation(
        monkeypatch, recovery, cheap, *, finish=None, normal_turn=None):
    from apps.chat.services.rag_evidence import build_selected_evidence_packet
    from apps.chat.services.rag_retrieval import fuse_ranked_tool_results
    from apps.chat.services.rag_recovery_origins import (
        accept_recovery_action, recovery_action, start_recovery_origins,
    )
    from apps.chat.services.rag_selection import select_evidence
    from apps.chat.services.rag_selection_scoring import prepare_selection_candidates
    from apps.chat.services.rag_selection_types import SelectionLimits, SelectionProfile
    from apps.documents.services.chunk_rerank_window_adapter import WindowSelectionScorer
    from lib.tools.search.vector_search import pack_chunk_search_results
    from lib.retrieval.evidence import SourceEvidence, fingerprint_source

    monkeypatch.setenv("RAG_SCOPED_BASELINE_RECOVERY_ENABLED", str(int(recovery)))
    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", "source")
    # Change test corpus before its real graph/source snapshots are constructed.
    # No live body, authority, length, rank or prepared-window result is replaced.
    if not cheap:
        monkeypatch.setattr(dataset, "TEXTS", (
            "Atlas is a model", "Atlas is a model",
            "Atlas uses the Cedar dataset. " + "procedure analysis " * 450,
        ))
    monkeypatch.setattr(support, "pressure_rows", _all_short_sources if cheap else _short_then_long_sources)
    epoch = monotonic()
    with support.native_case(
        monkeypatch, fail=False, controlled_graph_clock=True, graph_epoch=epoch,
        requested=frozenset({_DOC_A}), timeout_ms=6500,
    ) as case:
        budget = case.source.budget
        ordinary = {row.pk for row in case.snapshot.baseline_candidates}
        ordinary_text = {row.content for row in case.snapshot.baseline_candidates}
        assert len(ordinary) == 25
        assert max(len(text) for text in ordinary_text) < 100
        assert len(case.snapshot.source_admission.references) == 45
        old_deadline = case.operation.deadline

        def expire_child(kwargs, envelope):
            case.now[0] = kwargs["deadline"] - epoch + .001

        returned_envelope(monkeypatch, empty=True, reasons=("deadline",), after=expire_child)
        if normal_turn is not None:
            normal_turn(case)
            return
        origins = start_recovery_origins("Atlas", budget)
        with recovery_action(origins) as action:
            closure = support.run_current_or_native(case)
            rows = closure.final_rows[2]
            sources = tuple(SourceEvidence(row.pk, str(row.doc_id), row.chunk_number,
                            fingerprint_source(row.content), row.content) for row in rows)
            payload = pack_chunk_search_results(
                rows, titles_by_doc_id={doc.id: doc.title for doc in case.f.documents.values()},
                docs_by_doc_id={doc.id: doc for doc in case.f.documents.values()},
                truncate=lambda value: value, image_modality=TextChunk.Modality.IMAGE,
                compact_items=False, source_evidence=sources,
            )
            accept_recovery_action(action, payload, sources)
        rows = closure.final_rows[2]
        assert ordinary <= {row.pk for row in rows}
        if recovery:
            assert len(rows) == 45
            assert any(len(row.content) >= 5000 for row in rows) is not cheap
            assert closure.graph_diagnostics["source_admission"]["fallback_top_up_count"] == 20
        else:
            assert len(rows) == 45  # Default-off completion adds ordinary evidence.
        assert case.operation.deadline == old_deadline and case.operation.active()
        winner = next((row for row in rows if row.pk not in ordinary
                       and row.content not in ordinary_text), None) if cheap else None
        assert not cheap or winner is not None

        class Provider:
            scorer_fingerprint = "deterministic-complete-source-control"

            def __init__(self):
                self.seen = []

            def score_pair(self, pair, timeout_seconds):
                self.seen.append(pair)
                value = 2.0 if winner is not None and pair[1] == winner.content else (
                    1.0 if pair[1] in ordinary_text else 0.0)
                return value, pair

        provider = Provider()
        clock = lambda: case.now[0]
        scorer = WindowSelectionScorer(
            provider, budget=budget, pair_counter=lambda query, text: len(query) + len(text) + 8,
            deadline=clock() + 3, clock=clock,
        )
        from .recovery_answer_support import answer_context
        case.answer = answer_context(case, monkeypatch)
        pool = fuse_ranked_tool_results([payload])
        prepared = prepare_selection_candidates(
            pool=pool, primary_query="Atlas", authorization=case.f.authorization,
            deadline=clock() + 3, allow_new_scores=True, scorer=scorer,
            turn_budget=budget, source_mode=True, token_ceiling=7000, clock=clock,
            **({"recovery_preflight": case.answer.preflight} if case.answer.preflight is not None else {}),
        )
        case.answer.prepared = prepared
        scored = {candidate.chunk_id for candidate in prepared.candidates
                  if candidate.scoring_kind == "pointwise" and candidate.raw_score is not None}
        assert ordinary <= scored, "optional recovery consumed ordinary final scoring/preparation"
        selected = prepared.prepaid_selection or select_evidence(
            prepared.candidates, profile=SelectionProfile("test", 1, 0, "v1"),
            limits=SelectionLimits(12, 12, 7000), score_status=prepared.score_status,
        )
        packet = build_selected_evidence_packet(selected, query="Atlas", search_scope="selected")
        assert packet.chunks
        if cheap:
            assert packet.chunks[0]["chunk_id"] == winner.pk, "resource protection must not force ordinary relevance"
            assert prepared.score_status == "model" and len(scored) == 45
            assert prepared.acquired_count == 45 and prepared.omitted_optional == ()
        else:
            assert {row["chunk_id"] for row in packet.chunks} <= ordinary
            if recovery:
                assert prepared.score_status == "partial_model_ranked"
                assert prepared.fallback_reason == "optional_recovery_pruned"
                assert prepared.acquired_count == 45
                assert {item.chunk_id for item in prepared.omitted_optional} == {row.pk for row in rows} - ordinary
                assert len(prepared.candidates) == len(scored) == 25
        assert budget.sources_used <= budget.limits.unique_sources == 45
        assert budget.pairs_used["final"] <= budget.limits.final_pairs == 45
        assert budget.text_used["tokenized"] <= budget.limits.tokenized_codepoints == 1_000_000
        assert budget.text_used["materialized"] <= budget.limits.materialized_codepoints == 250_000
        assert budget.limits.final_scoring_ms == 3000 and budget.limits.retrieval_ms == 15000
        if finish is not None:
            finish(case, selected, packet)
