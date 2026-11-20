"""Outer synthesis preserves provider-issued receipts instead of finalizing twice."""

import pytest

from apps.chat.tests.test_rag_synthesis import Model, synthesize
from apps.chat.tests.test_rag_synthesis import selected as selected
from lib.llm.answer_publication import AnswerPublicationRejected, bind_answer
from lib.llm.turn_context import current_turn
from lib.llm.types.messages import AssistantMessage

pytestmark = pytest.mark.django_db(transaction=True)
SCIENTIFIC_SOURCE = "Calibration condition: use flat fields and dark frames."


class BoundModel(Model):
    def __init__(self, *, tamper=None):
        super().__init__("")
        self.tamper = tamper
        self.answer = None

    async def complete(self, convo, max_tokens, stream_func=None):
        self.complete_calls += 1
        context = current_turn().answer_publication
        # Real complete_turn finalizes and binds before returning its message.
        # The blank draft exercises trusted fallback display nodes such as wbr.
        self.answer = await context.finalize(self.reply)
        message = bind_answer(
            AssistantMessage(content=self.answer.content, stop_reason="stop"),
            self.answer,
            context,
        )
        if self.tamper == "receipt":
            message._validated_answer = None
        elif self.tamper == "context":
            message._answer_publication_context = None
        elif self.tamper == "content":
            message.content += " An injected claim."
        elif self.tamper == "all_markers":
            message._rag_answer = False
            message._validated_answer = None
            message._answer_publication_context = None
        return convo + [message], "stop"


@pytest.mark.parametrize("selected", [SCIENTIFIC_SOURCE], indirect=True)
async def test_provider_bound_fallback_keeps_exact_content_and_receipt_at_sinks(selected):
    model = BoundModel()
    result, events = await synthesize(model, selected, persist=True)

    assert model.complete_calls == 1
    assert "<wbr>" in model.answer.content
    assert result[-1].content == model.answer.content
    assert result[-1]._validated_answer is model.answer
    assert events[0]["content"] == model.answer.content
    assert result[-1]._validated_answer.outcome == "partial"


@pytest.mark.parametrize("selected", [SCIENTIFIC_SOURCE], indirect=True)
@pytest.mark.parametrize("tamper", ["receipt", "context", "content", "all_markers"])
async def test_outer_synthesis_rejects_tampered_provider_binding(selected, tamper):
    model = BoundModel(tamper=tamper)
    with pytest.raises(AnswerPublicationRejected):
        await synthesize(model, selected, persist=True)
    assert model.complete_calls == 1
