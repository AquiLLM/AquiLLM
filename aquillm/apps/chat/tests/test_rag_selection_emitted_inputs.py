"""Final relevance describes the text actually delivered to synthesis."""

from apps.chat.consumers import utils as consumer_utils
from apps.chat.services.rag_evidence import build_selected_evidence_packet
from apps.chat.services.rag_selection import select_evidence
from apps.chat.services.rag_selection_scoring import prepare_selection_candidates
from apps.chat.services.rag_selection_types import SelectionLimits, SelectionProfile
from apps.chat.tests.test_rag_selection_scoring import (
    Policy,
    _authorization,
    _chunk,
    _pool,
    _score_set,
)
from apps.documents.services.chunk_rerank_results import (
    PassageScore,
    RerankScoreSet,
    fingerprint_pair,
    fingerprint_pool,
    fingerprint_text,
)


class ExactFinalScorer:
    scoring_kind = "pointwise"
    scorer_fingerprint = "scorer-v1"

    def __init__(self):
        self.calls = []

    def prepare_pair(self, query, chunk):
        return query, chunk.content[:1600]

    def prepare_emitted_pair(self, query, emitted_text):
        return query, emitted_text

    def score_pair(self, pair, timeout_seconds):
        self.calls.append(pair)
        return (0.9 if "threshold=7.3 ug/L" in pair[1] else 0.1), pair


def test_final_selection_does_not_borrow_support_beyond_emitted_excerpt(monkeypatch):
    monkeypatch.setattr(consumer_utils, "TOOL_CHUNK_CHAR_LIMIT", 1000)
    source = "x" * 1200 + " threshold=7.3 ug/L"
    chunks = (_chunk(1, source), _chunk(2, "threshold=7.3 ug/L"))
    acquisition_pair = ("threshold?", source[:1600])
    acquisition_score = RerankScoreSet(
        "v2",
        fingerprint_text("threshold?"),
        "scorer-v1",
        fingerprint_pool(
            ((1, fingerprint_text(source), fingerprint_pair(*acquisition_pair)),)
        ),
        "pointwise",
        "complete",
        (1,),
        (
            PassageScore(
                1,
                chunks[0].doc_id,
                chunks[0].chunk_number,
                fingerprint_text(source),
                fingerprint_pair(*acquisition_pair),
                0.99,
            ),
        ),
    )
    pool = _pool(chunks, (acquisition_score,))
    clipped = consumer_utils.truncate_tool_text(source)
    pool.rows[0]["text"] = clipped
    scorer = ExactFinalScorer()

    prepared = prepare_selection_candidates(
        pool=pool,
        primary_query="threshold?",
        authorization=_authorization(Policy()),
        deadline=100.0,
        allow_new_scores=True,
        scorer=scorer,
        chunk_loader=lambda _auth, _ids: chunks,
        clock=lambda: 0.0,
    )
    selection = select_evidence(
        prepared.candidates,
        profile=SelectionProfile("test", 1.0, 0.0, "v1"),
        limits=SelectionLimits(1, 1, 7000),
        score_status=prepared.score_status,
    )
    packet = build_selected_evidence_packet(
        selection, query="threshold?", search_scope="selected documents"
    )

    assert prepared.score_status == "model"
    assert prepared.reused_pairs == 0
    assert ("threshold?", clipped) in scorer.calls
    assert [row["chunk_id"] for row in packet.chunks] == [2]
    assert packet.chunks[0]["text"] == "threshold=7.3 ug/L"


def test_final_input_has_distinct_emitted_pair_identity():
    from apps.chat.services.rag_selection_inputs import prepare_final_input

    source = "x" * 1200 + " threshold=7.3 ug/L"
    excerpt = source[:1000]
    item = prepare_final_input(
        chunk_id=1,
        source_text=source,
        emitted_text=excerpt,
        query="threshold?",
        prepare_pair=lambda query, document: (query, document),
    )
    assert item.pair == ("threshold?", excerpt)
    assert item.pair_fingerprint != fingerprint_pair("threshold?", source[:1600])


def test_final_input_rejects_silent_document_or_query_clipping():
    from apps.chat.services.rag_selection_inputs import prepare_final_input

    for prepare_pair in (
        lambda query, document: (query, document[:3]),
        lambda query, document: (query[:1], document),
    ):
        assert (
            prepare_final_input(
                chunk_id=1,
                source_text="abcdef",
                emitted_text="abcdef",
                query="question",
                prepare_pair=prepare_pair,
            )
            is None
        )


def test_exact_short_acquisition_pair_is_reused_without_new_scoring():
    chunks = (_chunk(1, "threshold=7.3 ug/L"),)
    scorer = ExactFinalScorer()
    result = prepare_selection_candidates(
        pool=_pool(chunks, (_score_set("threshold?", chunks, (0.9,)),)),
        primary_query="threshold?",
        authorization=_authorization(Policy()),
        deadline=100.0,
        allow_new_scores=True,
        scorer=scorer,
        chunk_loader=lambda _auth, _ids: chunks,
        clock=lambda: 0.0,
    )
    assert result.score_status == "model"
    assert result.reused_pairs == 1 and result.new_pairs == 0
    assert scorer.calls == []


def test_query_clipping_falls_back_for_whole_pool_without_scoring():
    class QueryClippingScorer(ExactFinalScorer):
        def prepare_emitted_pair(self, query, emitted_text):
            return query[:3], emitted_text

    chunks = (_chunk(1, "first"), _chunk(2, "second"))
    scorer = QueryClippingScorer()
    result = prepare_selection_candidates(
        pool=_pool(chunks),
        primary_query="threshold?",
        authorization=_authorization(Policy()),
        deadline=100.0,
        allow_new_scores=True,
        scorer=scorer,
        chunk_loader=lambda _auth, _ids: chunks,
        clock=lambda: 0.0,
    )
    assert result.score_status == "rank_fallback"
    assert result.fallback_reason == "score_incompatible"
    assert [item.relevance for item in result.candidates] == [1.0, 0.0]
    assert scorer.calls == []


def test_changed_emitted_excerpt_during_inference_is_not_published(monkeypatch):
    monkeypatch.setattr(consumer_utils, "TOOL_CHUNK_CHAR_LIMIT", 1000)
    chunks = (_chunk(1, "x" * 1200), _chunk(2, "second"))
    pool = _pool(chunks)
    pool.rows[0]["text"] = consumer_utils.truncate_tool_text(chunks[0].content)

    class ChangingPolicyScorer(ExactFinalScorer):
        def score_pair(self, pair, timeout_seconds):
            monkeypatch.setattr(consumer_utils, "TOOL_CHUNK_CHAR_LIMIT", 900)
            return super().score_pair(pair, timeout_seconds)

    result = prepare_selection_candidates(
        pool=pool,
        primary_query="question",
        authorization=_authorization(Policy()),
        deadline=100.0,
        allow_new_scores=True,
        scorer=ChangingPolicyScorer(),
        chunk_loader=lambda _auth, _ids: chunks,
        clock=lambda: 0.0,
    )
    assert [item.chunk_id for item in result.candidates] == [2]
