"""Canned conversation served by FakeInterface when LLM_CHOICE=FAKE.

The list below is consumed one entry per get_message() call, in order.
Plain strings are sent as end_turn text responses; tool_call_response(...)
entries emit a real tool call that the chat consumer executes against the
configured tool wiring. The provider clamps to the final entry once the
script is exhausted, so the demo never crashes if the user keeps chatting.
"""
from .llm import tool_call_response


DEMO_RESPONSES = [
    # --- User turn 1: greeting / capabilities overview, no tool call ---
    (
        "Hi! I'm a scripted demo of AquiLLM. I can pretend to search your "
        "collections, summarise documents, and answer follow-up questions. "
        "Try asking me about your documents next."
    ),
    # --- User turn 2: narrate, then call vector_search, then summarise ---
    tool_call_response(
        "vector_search",
        {"query": "demo overview"},
        text="Let me search your collections for that.",
    ),
    (
        "Here's what I found in your collections. In a real session I'd cite "
        "specific chunks, but for this demo just imagine a tidy summary of "
        "the retrieved passages with inline references."
    ),
    # --- User turn 3: closing ---
    (
        "Anything else you'd like me to dig into? I can keep going with more "
        "canned answers, but the script ends here so I'll just repeat this."
    ),
]


__all__ = ["DEMO_RESPONSES"]
