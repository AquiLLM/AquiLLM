"""The real finalizer emits the exact bytes exercised by MessageBubble tests."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from apps.chat.services.rag_answer_context import RagAnswerPublication
from apps.chat.tests.test_rag_answer_support import packet_for
from lib.llm.turn_context import TurnContext, bind_turn
from lib.retrieval.evidence import fingerprint_prepared_evidence
from lib.retrieval.turn_budget import TurnBudget, TurnLimits

DOC = "12345678-1234-1234-1234-123456789abc"
FIXTURES = json.loads(
    (
        Path(__file__).resolve().parents[4]
        / "react/tests/fixtures/model-presentation.json"
    ).read_text(encoding="utf-8")
)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", FIXTURES, ids=lambda case: case["name"])
async def test_finalized_output_matches_the_real_renderer_fixture(case):
    packet = packet_for("The interval is 0.3 < z < 10.", "Explain the interval.")
    packet.chunks[0]["doc_id"] = DOC
    packet.citation_tokens = [f"[doc:{DOC} chunk:1]"]
    prepared = packet.source_evidence[0]
    source = replace(prepared.source, document_id=DOC)
    packet.source_evidence = (
        replace(
            prepared,
            source=source,
            evidence_fingerprint=fingerprint_prepared_evidence(source, prepared.spans),
        ),
    )
    state = TurnContext(TurnBudget(TurnLimits()))

    async def authorize():
        return True

    with bind_turn(state):
        context = RagAnswerPublication(packet=packet, authorize=authorize, turn=state)
        state.answer_publication = context
        answer = await context.finalize(case["draft"])
        assert answer.content == case["published"]
        for sink in ("stream", "refresh", "persist"):
            await context.verify(answer, content=answer.content, sink=sink)
