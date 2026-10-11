"""P1 scenario 11: real relevance is not semantic support or absence proof."""
from hashlib import sha256
import json

import pytest
from channels.db import database_sync_to_async

from apps.chat.refs import CollectionsRef
from apps.documents.tests.representation_support import retained as retained, upload as upload
from lib.embeddings.tests.test_document_provider import transport as transport
from lib.llm.answer_publication import LIMITED_ANSWER

from .cross_lane_publication_support import (
    Provider, close_publication_worker as close_publication_worker,
    consumer_for, publication_source as publication_source,
)
from .cross_lane_support_observation import (
    capture, hostile_score_transport as hostile_score_transport, run_observed,
)
from .cross_lane_support_support import (
    CONTRADICTORY, DISTRACTOR, EXACT, IRRELEVANT, QUESTION, UNCERTAIN,
    citation, literal_excerpt, replace_target, support_corpus as support_corpus,
)

pytestmark = [pytest.mark.django_db(transaction=True), pytest.mark.asyncio]


def assert_scores(audit, expected):
    assert len(audit.prepared) == 1
    kwargs, prepared = audit.prepared[0]
    assert kwargs["primary_query"] == QUESTION
    assert prepared.score_status == "model" and prepared.fallback_reason is None
    assert prepared.new_pairs == len(expected) and prepared.reused_pairs == 0
    assert {c.text: (c.raw_score, c.relevance) for c in prepared.candidates} == expected
    for candidate in prepared.candidates:
        assert candidate.source_fingerprint == sha256(candidate.text.encode()).hexdigest()
        assert candidate.scorer_fingerprint
        assert candidate.scoring_kind == "pointwise"
        assert candidate.score_interpretation == "uncalibrated"
    assert len(audit.scores) == 1
    score_args, scores = audit.scores[0]
    assert score_args["query"] == QUESTION and scores.status == "complete"
    assert scores.query_fingerprint == sha256(QUESTION.encode()).hexdigest()
    assert {c.content for c in score_args["chunks"]} == set(expected)
    assert {s.value for s in scores.scores} == {value[0] for value in expected.values()}
    assert audit.dense and any(rows for _, rows in audit.dense)
    assert {row.chunk_id for _, rows in audit.dense for row in rows} == {
        candidate.chunk_id for candidate in prepared.candidates
    }


def assert_delivery(result, provider, texts):
    assert len(provider.requests) == 1 and provider.answers == []
    assert len(result.dispatches) == 1
    lease, delivered_packet, payload = result.dispatches[0]
    assert delivered_packet is result.packet and lease is result.turn.budget._synthesis
    assert result.trace.delivered_packet is result.packet
    assert len(result.trace.sdk_payloads) == 1 and result.trace.unknown == []
    assert [row.get("text", row.get("x")) for row in result.packet.chunks] == texts
    assert [row.get("text", row.get("x")) for tool in result.tool_results
            for row in tool.get("result", [])] == texts
    request = json.dumps(provider.requests[0], ensure_ascii=False)
    observed = json.dumps(payload, ensure_ascii=False)
    for text in texts:
        assert text in lease.payload and text in request and text in observed
    assert result.support.calibration_id is None and result.support.exhaustive is False
    assert result.answer._receipt is not None
    assert result.answer.evidence_digest == result.support.evidence_digest
    assert result.answer.scope_digest == result.support.scope_digest
    assert result.stored == result.answer.content == result.message.content
    assert result.events == [("stream", result.stored), ("refresh", result.stored)]
    assert "999" not in result.stored
    events = [e["event"] for e in result.trace.events]
    assert events.count("sdk_start") == events.count("sdk_end") == 1


async def exact_prerequisite(corpus, monkeypatch, calls):
    """Separate original one-collection turn; never widen a frozen scope."""
    consumer = consumer_for(corpus, QUESTION)
    provider = Provider([f"The capacity is 999 litres {citation(corpus.chunk)}."])
    before = len(calls)
    with monkeypatch.context() as delegated:
        audit = capture(delegated)
        result = await run_observed(corpus, consumer, provider)
    assert_scores(audit, {EXACT: (1000.0, 0.5)})
    assert len(calls) > before
    assert_delivery(result, provider, [EXACT])
    assert result.support.state == "complete"
    assert result.support.assessment_basis == "deterministic"
    assert result.support.requested_aspects == result.support.supported_aspects == ("capacity",)
    assert result.support.unresolved_aspects == () and result.answer.outcome == "answer"
    assert result.stored == "The selected source states:\n\n" + literal_excerpt(EXACT, citation(corpus.chunk))
    return result


@pytest.mark.parametrize("mode", ["legacy", "source"])
async def test_exact_lookup_overrides_hostile_high_score_reply(
    support_corpus, hostile_score_transport, monkeypatch, mode,
):
    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", mode)
    await exact_prerequisite(support_corpus, monkeypatch, hostile_score_transport)


@pytest.mark.parametrize("mode", ["legacy", "source"])
@pytest.mark.parametrize("text", [IRRELEVANT, CONTRADICTORY], ids=["irrelevant", "contradictory"])
async def test_highest_real_score_preserves_unknown_support_and_all_conditions(
    support_corpus, hostile_score_transport, monkeypatch, mode, text,
):
    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", mode)
    corpus = support_corpus
    positive = await exact_prerequisite(corpus, monkeypatch, hostile_score_transport)
    await database_sync_to_async(replace_target)(corpus, text)
    consumer = consumer_for(corpus, QUESTION)
    consumer.col_ref = CollectionsRef([corpus.doc.collection_id, corpus.distractor.collection_id])
    provider = Provider([f"The capacity is 999 litres {citation(corpus.chunk)}."])
    before = len(hostile_score_transport)
    with monkeypatch.context() as delegated:
        audit = capture(delegated)
        result = await run_observed(corpus, consumer, provider)
    assert result.turn is not positive.turn and result.runtime is not positive.runtime
    assert_scores(audit, {text: (1000.0, 1.0), DISTRACTOR: (-1000.0, 0.0)})
    assert len(hostile_score_transport) > before
    assert_delivery(result, provider, [text, DISTRACTOR])
    assert result.support.state == "unknown" and result.answer.outcome == "partial"
    assert result.support.assessment_basis == "unvalidated"
    assert result.support.requested_aspects == result.support.unresolved_aspects == ("capacity",)
    assert result.support.supported_aspects == ()
    expected = UNCERTAIN + "\n\n" + literal_excerpt(text, citation(corpus.chunk))
    expected += "\n\n" + literal_excerpt(DISTRACTOR, citation(corpus.distractor_chunk))
    assert result.stored == expected
    if text == CONTRADICTORY:
        for condition in ("18 litres at 20 C", "12 litres at 40 C", "neither is an accepted specification"):
            assert condition in result.stored


@pytest.mark.parametrize("mode", ["legacy", "source"])
async def test_real_authorized_empty_owner_is_limited_without_provider_or_absence_claim(
    support_corpus, hostile_score_transport, monkeypatch, mode,
):
    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", mode)
    corpus = support_corpus
    positive = await exact_prerequisite(corpus, monkeypatch, hostile_score_transport)
    consumer = consumer_for(corpus, QUESTION)
    consumer.col_ref = CollectionsRef([corpus.empty.collection_id])
    provider = Provider([])
    before = len(hostile_score_transport)
    with monkeypatch.context() as delegated:
        audit = capture(delegated)
        result = await run_observed(corpus, consumer, provider)
    assert result.turn is not positive.turn and result.runtime is not positive.runtime
    authority = result.runtime.authorization
    assert set(map(str, authority.selected_document_ids)) == {str(corpus.empty.id)}
    assert audit.searches and all(value.get("result") == [] for _, value in audit.searches)
    assert not any(rows for _, rows in audit.dense)
    assert not audit.scores and len(hostile_score_transport) == before
    assert provider.requests == [] and result.dispatches == []
    assert result.trace.sdk_payloads == []
    assert result.packet.chunks == [] and result.packet.citation_tokens == []
    assert result.tool_results and all(tool.get("result") == [] for tool in result.tool_results)
    assert result.answer.outcome == "limited" and result.stored == LIMITED_ANSWER
    assert result.events == [("stream", LIMITED_ANSWER), ("refresh", LIMITED_ANSWER)]
    assert "[doc:" not in result.stored
