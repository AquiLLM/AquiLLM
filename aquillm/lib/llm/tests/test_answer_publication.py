"""Actual provider output regressions for exact final answer publication."""

from contextlib import contextmanager

import pytest

from lib.llm.providers.complete_turn import complete_conversation_turn
from lib.llm.providers.request_observability import observability_scope
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import AssistantMessage, ToolMessage, UserMessage
from lib.llm.types.response import LLMResponse


@contextmanager
def synthetic_publication(
    citations=("[doc:a chunk:1]", "[doc:b chunk:2]"),
    *,
    image_urls=(),
    packet=None,
    seal=True,
):
    """Explicit local test capability; supplies no real document access."""
    from apps.chat.services.rag_answer_publication import RagAnswerPublication
    from apps.chat.services.rag_evidence import EvidencePacket
    from lib.llm.turn_context import TurnContext, bind_turn
    from lib.retrieval.synthesis_budget import seal_synthesis
    from lib.retrieval.turn_budget import TurnBudget, TurnLimits

    state = TurnContext(TurnBudget(TurnLimits()))
    packet = packet or EvidencePacket(
        [],
        list(image_urls),
        list(citations),
        "synthetic",
        "synthetic",
        "results_found",
        "",
        0,
    )

    async def authorize():
        return True

    lease = (
        seal_synthesis(
            state.budget, "synthetic", calls=5, output_tokens=20000, timeout_seconds=60
        )
        if seal
        else None
    )
    state.answer_publication = RagAnswerPublication(
        packet=packet, authorize=authorize, turn=state, lease=lease
    )
    with bind_turn(state):
        yield state.answer_publication


@pytest.mark.parametrize(
    "repair,forbidden",
    [
        ("The capacity is 999 litres.", "999"),
        ("A supported statement [doc:a chunk:1]. The capacity is 999 litres.", "999"),
        (
            "- Supported statement [doc:a chunk:1]\n- An unsupported factual assertion",
            "An unsupported factual assertion",
        ),
        ("An invented finding [doc:missing chunk:99].", "An invented finding"),
        ("", "999"),
    ],
)
@pytest.mark.parametrize("stream_setting", ["0", "1"])
@pytest.mark.parametrize("bound", [False, True])
async def test_failed_repair_never_reaches_visible_or_stored_answer(
    monkeypatch, repair, forbidden, stream_setting, bound
):
    monkeypatch.setenv("LLM_STREAM_FINAL_ANSWER_ONLY", stream_setting)
    monkeypatch.setenv("RAG_ENFORCE_CHUNK_CITATIONS", "1")
    monkeypatch.setenv("LLM_ALLOW_EXTRACTIVE_EVIDENCE_UI", "0")
    calls, events = [], []

    class Model:
        base_args = {}

        async def get_message(self, **kwargs):
            calls.append(kwargs)
            text = "The capacity is 999 litres." if len(calls) == 1 else repair
            callback = kwargs.get("stream_callback")
            if callable(callback):
                await callback(
                    {
                        "content": text,
                        "done": True,
                        "role": "assistant",
                        "message_uuid": kwargs["stream_message_uuid"],
                        "stop_reason": "stop",
                        "usage": 2,
                    }
                )
            return LLMResponse(
                text=text,
                tool_call={},
                stop_reason="stop",
                input_usage=1,
                output_usage=1,
                model="offline-fixture",
            )

    async def stream(payload):
        events.append(payload)

    convo = Conversation(
        system="Use selected evidence.",
        messages=[
            UserMessage(content="What is the capacity?"),
            AssistantMessage(
                content="",
                stop_reason="tool_use",
                tool_call_id="search",
                tool_call_name="vector_search",
                tool_call_input={},
            ),
            ToolMessage(
                content="Capacity is 18 litres.",
                tool_name="vector_search",
                for_whom="assistant",
                result_dict={
                    "result": [
                        {"doc_id": "a", "chunk_id": 1, "text": "Capacity is 18 litres."}
                    ]
                },
            ),
        ],
    )
    from contextlib import nullcontext

    with (
        synthetic_publication() if bound else nullcontext(),
        observability_scope("publication-regression", "direct_synthesis"),
    ):
        completed, _ = await complete_conversation_turn(
            Model(), convo, 1024, stream_func=stream
        )
    assert "999" not in completed[-1].content
    assert forbidden not in completed[-1].content
    assert events and all(event["content"] == completed[-1].content for event in events)
    assert completed[-1].content.strip()


@pytest.mark.parametrize(
    "text,reason",
    [
        ("No citation.", "missing_citations"),
        ("A claim [doc:missing chunk:1].", "unknown_citation"),
        ("A claim [doc:a chunk:1]. A forged ref [doc:a chunk:x].", "unknown_citation"),
        (
            "Supported statement [doc:a chunk:1].\n"
            "Another finding [doc:missing chunk:x\nContinued prose.",
            "unknown_citation",
        ),
        (
            "Supported statement [doc:a chunk:1].\r\n"
            "Another finding [doc:missing chunk:x\r\nContinued prose.",
            "unknown_citation",
        ),
        (
            "A claim [doc:a chunk:1].\n- Unsupported factual assertion",
            "uncited_factual_line",
        ),
        (
            "A claim [doc:a chunk:1]. The capacity is 999 litres.",
            "uncited_numeric_claim",
        ),
        (
            "A claim [doc:a chunk:1].\n\nThe capacity is 999 litres.",
            "uncited_numeric_claim",
        ),
    ],
)
def test_complete_predicate_reports_specific_failure(text, reason):
    from lib.llm.answer_publication import FinalCitationPolicy, validate_final_answer

    policy = FinalCitationPolicy(
        frozenset({"[doc:a chunk:1]"}), True, "evidence", "scope"
    )
    assert reason in validate_final_answer(text, policy).reasons


@pytest.mark.parametrize(
    "recovery", ["repair_error", "compact", "invalid_extract"]
)
async def test_recovery_cannot_publish_uncited_numeric_suffix(monkeypatch, recovery):
    from lib.llm.answer_publication import LIMITED_ANSWER
    from lib.llm.providers import complete_turn

    monkeypatch.setenv("LLM_STREAM_FINAL_ANSWER_ONLY", "0")
    monkeypatch.setenv("LLM_POST_TOOL_SYNTHESIS_RETRIES", "0")
    calls, events = [], []

    async def invalid_summary(*args, **kwargs):
        return "Capacity is 999 litres."

    if recovery in {"compact", "invalid_extract"}:
        monkeypatch.setattr(
            complete_turn, "_last_resort_evidence_answer", invalid_summary
        )

    class Model:
        base_args = {}

        async def get_message(self, **kwargs):
            calls.append(kwargs)
            if len(calls) > 1 and recovery == "repair_error":
                raise RuntimeError("synthetic provider failure")
            return LLMResponse(
                text=""
                if recovery in {"compact", "invalid_extract"} and len(calls) == 1
                else "Capacity is 999 litres.",
                tool_call={},
                stop_reason="stop",
                input_usage=1,
                output_usage=1,
            )

    async def stream(payload):
        events.append(payload)

    conversation = Conversation(
        system="Use evidence",
        messages=[
            UserMessage(content="Capacity?"),
            ToolMessage(
                content="Capacity is 18 litres.",
                tool_name="vector_search",
                for_whom="assistant",
                result_dict={
                    "result": [
                        {"doc_id": "a", "chunk_id": 1, "text": "Capacity is 18 litres."}
                    ]
                },
            ),
        ],
    )
    with synthetic_publication():
        result, _ = await complete_conversation_turn(
            Model(), conversation, 1024, stream_func=stream
        )
    assert result[-1].content == LIMITED_ANSWER
    assert events and all(e["content"] == LIMITED_ANSWER for e in events)
