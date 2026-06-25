"""Frontend adapter applies presentation policy for assistant bubbles."""
from __future__ import annotations

from django.test import SimpleTestCase

from aquillm.llm import AssistantMessage
from aquillm.message_adapters import pydantic_message_to_frontend_dict


class AssistantDisplayPolicyTests(SimpleTestCase):
    def test_retrieving_stub_not_shown_in_frontend_dict(self):
        msg = AssistantMessage(
            content="Retrieving the paper...",
            stop_reason="stop",
            usage=10,
        )
        payload = pydantic_message_to_frontend_dict(msg)
        self.assertEqual(payload["content"], "")

    def test_real_answer_shown_in_frontend_dict(self):
        msg = AssistantMessage(
            content=(
                "The study reports calibration metrics across three datasets [doc:abc chunk:2]. "
                "Performance improves when temperature scaling is applied to the logits."
            ),
            stop_reason="stop",
            usage=10,
        )
        payload = pydantic_message_to_frontend_dict(msg)
        self.assertIn("calibration metrics", payload["content"])
        self.assertIn("[doc:abc chunk:2]", payload["content"])
