"""Generated research answers survive publication without claiming entailment."""

from contextlib import asynccontextmanager

import pytest

from apps.chat.services.rag_answer_context import RagAnswerPublication
from apps.chat.services.rag_support_delivery import support_grounding
from apps.chat.tests.test_rag_answer_support import packet_for
from lib.llm.answer_publication import LIMITED_ANSWER, AnswerPublicationRejected
from lib.llm.turn_context import TurnContext, bind_turn
from lib.retrieval.turn_budget import TurnBudget, TurnLimits

SOURCE = (
    "MMAE scatter is 0.016 with 25% image masking and 0.026 without masking. "
    "Fine-tuned BCNN scatter is 0.012."
)
DRAFT = (
    "BCNN has lower reported scatter: 0.012 compared with MMAE's 0.016 "
    "with 25% image masking [doc:a chunk:1].\n\n"
    "The masking condition matters: MMAE's scatter rises to 0.026 "
    "without it [doc:a chunk:1]."
)


@asynccontextmanager
async def publication(*, allowed=True):
    packet = packet_for(SOURCE, "Compare the models and explain the conditions.")
    state = TurnContext(TurnBudget(TurnLimits()))

    async def authorize():
        return allowed

    with bind_turn(state):
        context = RagAnswerPublication(packet=packet, authorize=authorize, turn=state)
        state.answer_publication = context
        yield context, packet


@pytest.mark.asyncio
async def test_cited_model_comparison_is_published_verbatim_without_support_promotion():
    async with publication() as (context, packet):
        answer = await context.finalize(DRAFT)
        assert answer.content == DRAFT
        assert answer.outcome == "partial"
        assert packet.answer_support.state == "unknown"
        assert packet.answer_support.assessment_basis == "unvalidated"
        assert packet.answer_support.calibration_id is None
        for sink in ("stream", "refresh", "persist"):
            await context.verify(answer, content=DRAFT, sink=sink)


@pytest.mark.asyncio
@pytest.mark.parametrize("draft", ["", "An uncited answer.", "A claim [doc:foreign chunk:9]."])
async def test_unusable_model_draft_falls_back_to_complete_authorized_excerpts(draft):
    async with publication() as (context, packet):
        answer = await context.finalize(draft)
        assert answer.content != LIMITED_ANSWER
        assert "0.016" in answer.content and "25%" in answer.content
        assert "0.026" in answer.content and "without masking" in answer.content
        assert "0.012" in answer.content
        assert "uncertain" in answer.content
        assert "doc:foreign" not in answer.content
        assert answer.outcome == "partial"
        await context.verify(answer, content=answer.content, sink="persist")


@pytest.mark.asyncio
async def test_revoked_evidence_cannot_be_used_for_either_synthesis_or_excerpts():
    async with publication(allowed=False) as (context, _):
        answer = await context.finalize(DRAFT)
        assert answer.content == LIMITED_ANSWER
        assert answer.outcome == "limited"


@pytest.mark.asyncio
async def test_generated_answer_receipt_rejects_later_source_or_answer_mutation():
    async with publication() as (context, packet):
        answer = await context.finalize(DRAFT)
        with pytest.raises(AnswerPublicationRejected, match="content_changed"):
            await context.verify(answer, content=DRAFT + " An added claim.", sink="persist")
        packet.chunks[0]["text"] = "Changed source."
        with pytest.raises(AnswerPublicationRejected, match="evidence_changed"):
            await context.verify(answer, content=answer.content, sink="persist")


def test_grounding_requests_research_synthesis_without_inventing_coverage_proof():
    packet = packet_for(SOURCE, "Compare the models and explain the conditions.")
    prompt = support_grounding(packet)
    assert "synthesize" in prompt.casefold()
    assert "Unknown support permits only exact attributed excerpts" not in prompt
    assert '"state": "unknown"' in prompt
    assert '"exhaustive": false' in prompt
    assert "conditions" in prompt
    assert "Profile memory is personalization, not document evidence." in prompt
