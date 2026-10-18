"""Generated wording and independent support metadata remain separate contracts."""

import pytest

from apps.chat.services.rag_answer_context import RagAnswerPublication
from apps.chat.services.rag_evidence_handoff import prepare_evidence_handoff
from apps.chat.tests.test_rag_answer_support import packet_for
from lib.llm.turn_context import TurnContext, bind_turn
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import ToolMessage
from lib.retrieval.turn_budget import TurnBudget, TurnLimits


@pytest.mark.asyncio
async def test_unknown_keeps_cited_model_wording_without_support_promotion():
    packet = packet_for(
        "This excerpt describes telescope optics.", "Explain the retrieved topic."
    )
    state = TurnContext(TurnBudget(TurnLimits()))

    async def authorize():
        return True

    with bind_turn(state):
        context = RagAnswerPublication(packet=packet, authorize=authorize, turn=state)
        state.answer_publication = context
        draft = "The retrieved text concerns telescope optics [doc:a chunk:1]."
        answer = await context.finalize(draft)
        assert answer.content == draft
        assert answer.outcome == "partial"
        assert packet.answer_support.state == "unknown"
        assert packet.answer_support.assessment_basis == "unvalidated"
        assert packet.answer_support.calibration_id is None
        await context.verify(answer, content=answer.content, sink="stream")


@pytest.mark.asyncio
async def test_partial_keeps_model_conditions_and_unresolved_support_metadata():
    packet = packet_for(question="What is the capacity and what is the voltage?")
    state = TurnContext(TurnBudget(TurnLimits()))

    async def authorize():
        return True

    with bind_turn(state):
        context = RagAnswerPublication(packet=packet, authorize=authorize, turn=state)
        state.answer_publication = context
        draft = (
            "The reported capacity is 18 litres at 20 C [doc:a chunk:1].\n\n"
            "The retrieved evidence leaves voltage unresolved."
        )
        answer = await context.finalize(draft)
        assert answer.content == draft
        assert answer.outcome == "partial"
        assert packet.answer_support.state == "partial"
        assert packet.answer_support.supported_aspects == ("capacity",)
        assert packet.answer_support.unresolved_aspects == ("voltage",)
        assert not packet.answer_support.exhaustive


def test_grounding_gets_bounded_support_without_serializing_private_metadata():
    packet = packet_for()
    convo = Conversation(
        system="Profile memory says voltage is 12 V.",
        messages=[
            ToolMessage(
                content="evidence",
                result_dict={},
                for_whom="assistant",
                tool_call_id="t",
                tool_name="search",
            )
        ],
    )
    persisted, request = prepare_evidence_handoff(convo, packet)
    assert "answer_support" in request.system
    assert '"state": "complete"' in request.system
    assert "answer_support" not in persisted[-1].model_dump_json()
    assert "calibration_id" not in request.system


@pytest.mark.asyncio
async def test_complete_lookup_does_not_promote_generated_wording_to_semantic_proof():
    packet = packet_for()
    state = TurnContext(TurnBudget(TurnLimits()))

    async def authorize():
        return True

    with bind_turn(state):
        context = RagAnswerPublication(packet=packet, authorize=authorize, turn=state)
        state.answer_publication = context
        draft = "At 20 C, the source reports a capacity of 18 litres [doc:a chunk:1]."
        answer = await context.finalize(draft)
        assert answer.content == draft
        assert answer.outcome == "partial"
        assert packet.answer_support.state == "complete"
        assert packet.answer_support.assessment_basis == "deterministic"
        assert packet.answer_support.calibration_id is None
        assert not packet.answer_support.exhaustive


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["x" * 1201, "Unsafe source\x00 text."])
async def test_unusable_draft_with_unsafe_or_unsplittable_source_is_limited(source):
    from lib.llm.answer_publication import LIMITED_ANSWER

    packet = packet_for(source, "Explain the retrieved topic.")
    state = TurnContext(TurnBudget(TurnLimits()))

    async def authorize():
        return True

    with bind_turn(state):
        context = RagAnswerPublication(packet=packet, authorize=authorize, turn=state)
        state.answer_publication = context
        answer = await context.finalize("")
        assert answer.content == LIMITED_ANSWER
        assert answer.outcome == "limited"


@pytest.mark.asyncio
async def test_authority_failure_cannot_leave_complete_support_metadata():
    from apps.chat.services.rag_support_delivery import refresh_answer_support
    from lib.llm.answer_publication import LIMITED_ANSWER

    packet = packet_for()
    assert refresh_answer_support(packet).state == "complete"
    state = TurnContext(TurnBudget(TurnLimits()))

    async def authorize():
        return False

    with bind_turn(state):
        context = RagAnswerPublication(packet=packet, authorize=authorize, turn=state)
        state.answer_publication = context
        answer = await context.finalize("Capacity is 18 litres [doc:a chunk:1].")
        assert answer.content == LIMITED_ANSWER
        assert packet.answer_support.state == "unknown"
        assert packet.answer_support.supported_aspects == ()


def test_partial_lookup_still_requests_existing_budgeted_assessment():
    from apps.chat.services.rag_coverage import needs_coverage_assessment

    packet = packet_for(question="What is the capacity and what is the voltage?")
    assert needs_coverage_assessment(
        packet.query, [packet.source_evidence[0].source], None
    )


def test_unassessed_optional_mode_can_use_independent_exact_lookup():
    from apps.chat.services.rag_coverage import CoverageAssessment
    from apps.chat.services.rag_support_delivery import refresh_answer_support

    packet = packet_for()
    packet.coverage_assessment = CoverageAssessment()
    assert refresh_answer_support(packet).state == "complete"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "draft", ["Capacity is 18 litres.", "Capacity is 18 litres [doc:foreign chunk:9]."]
)
async def test_invalid_or_missing_citations_fall_back_with_all_source_conditions(draft):
    packet = packet_for(
        "Capacity is 18 litres. This applies only at 20 C.", "Explain the capacity."
    )
    state = TurnContext(TurnBudget(TurnLimits()))

    async def authorize():
        return True

    with bind_turn(state):
        context = RagAnswerPublication(packet=packet, authorize=authorize, turn=state)
        state.answer_publication = context
        answer = await context.finalize(draft)
        assert "uncertain" in answer.content
        assert "> Capacity is 18 litres. [doc:a chunk:1]" in answer.content
        assert "This applies only at 20 C. [doc:a chunk:1]" in answer.content
        assert answer.content.count("> ") == 1
        assert "doc:foreign" not in answer.content
        assert answer.outcome == "partial"
        await context.verify(answer, content=answer.content, sink="stream")
