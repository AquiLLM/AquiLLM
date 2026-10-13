"""Unvalidated support cannot be laundered through valid citations or memory."""

import pytest

from apps.chat.services.rag_answer_context import RagAnswerPublication
from apps.chat.services.rag_evidence_handoff import prepare_evidence_handoff
from apps.chat.tests.test_rag_answer_support import packet_for
from lib.llm.turn_context import TurnContext, bind_turn
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import ToolMessage
from lib.retrieval.turn_budget import TurnBudget, TurnLimits


@pytest.mark.asyncio
async def test_unknown_replaces_fabricated_answer_with_attributed_exact_excerpt():
    packet = packet_for(
        "This excerpt describes telescope optics.", "Does this drug improve survival?"
    )
    state = TurnContext(TurnBudget(TurnLimits()))

    async def authorize():
        return True

    with bind_turn(state):
        context = RagAnswerPublication(packet=packet, authorize=authorize, turn=state)
        state.answer_publication = context
        answer = await context.finalize("This drug improves survival [doc:a chunk:1].")
        assert "This drug improves survival" not in answer.content
        assert (
            "> This excerpt describes telescope optics. [doc:a chunk:1]"
            in answer.content
        )
        assert "uncertain" in answer.content.casefold()
        assert answer.outcome == "partial"
        await context.verify(answer, content=answer.content, sink="stream")


@pytest.mark.asyncio
async def test_partial_names_missing_aspect_without_absence_claim():
    packet = packet_for(question="What is the capacity and what is the voltage?")
    state = TurnContext(TurnBudget(TurnLimits()))

    async def authorize():
        return True

    with bind_turn(state):
        context = RagAnswerPublication(packet=packet, authorize=authorize, turn=state)
        state.answer_publication = context
        answer = await context.finalize(
            "No study exists. Voltage is 12 V [doc:a chunk:1]."
        )
        assert "no study exists" not in answer.content.casefold()
        assert "12 V" not in answer.content
        assert "voltage" in answer.content
        assert "18 litres at 20 C" in answer.content
        assert answer.outcome == "partial"


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
async def test_complete_field_cannot_publish_other_facts_or_drop_conditions():
    packet = packet_for()
    state = TurnContext(TurnBudget(TurnLimits()))

    async def authorize():
        return True

    with bind_turn(state):
        context = RagAnswerPublication(packet=packet, authorize=authorize, turn=state)
        state.answer_publication = context
        answer = await context.finalize(
            "Capacity is 18 litres and voltage is 12 V [doc:a chunk:1]."
        )
        assert "12 V" not in answer.content
        assert "18 litres at 20 C" in answer.content
        assert answer.outcome == "answer"


@pytest.mark.asyncio
async def test_unsafe_or_oversized_excerpt_produces_scoped_server_limit():
    from lib.llm.answer_publication import LIMITED_ANSWER

    packet = packet_for("x" * 1201, "Does this drug improve survival?")
    state = TurnContext(TurnBudget(TurnLimits()))

    async def authorize():
        return True

    with bind_turn(state):
        context = RagAnswerPublication(packet=packet, authorize=authorize, turn=state)
        state.answer_publication = context
        answer = await context.finalize("This drug improves survival [doc:a chunk:1].")
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
async def test_multisentence_excerpts_keep_conditions_and_local_citations():
    packet = packet_for(
        "Capacity is 18 litres. This applies only at 20 C.", "Explain the capacity."
    )
    state = TurnContext(TurnBudget(TurnLimits()))

    async def authorize():
        return True

    with bind_turn(state):
        context = RagAnswerPublication(packet=packet, authorize=authorize, turn=state)
        state.answer_publication = context
        answer = await context.finalize("Capacity is 18 litres [doc:a chunk:1].")
        assert "uncertain" in answer.content
        assert "> Capacity is 18 litres. [doc:a chunk:1]" in answer.content
        assert "This applies only at 20 C. [doc:a chunk:1]" in answer.content
        assert answer.content.count("> ") == 1
        await context.verify(answer, content=answer.content, sink="stream")
