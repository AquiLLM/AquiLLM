"""Shared fakes and llm_tool stubs for chat message / LLM tests (not collected as tests)."""
from types import SimpleNamespace

from aquillm.llm import LLMInterface, llm_tool


@llm_tool(
    for_whom='assistant',
    required=[],
    param_descs={},
)
def _test_document_ids():
    """Test tool for LLMInterface.complete() retry behavior."""
    return {'result': 'ok'}


@llm_tool(
    for_whom='assistant',
    required=[],
    param_descs={},
)
def _test_image_result_tool():
    """Test tool returning image payload for serialization tests."""
    return {
        "result": {"status": "ok"},
        "_images": [
            {
                "result_index": 1,
                "title": "figure",
                "image_data_url": "data:image/jpeg;base64," + ("A" * 4096),
                "caption": "chart",
            }
        ],
        "_image_instruction": "Use markdown image syntax.",
    }


class _FakeLLMInterface(LLMInterface):
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    async def get_message(self, *args, **kwargs):
        self.calls.append(kwargs)
        return self.responses[len(self.calls) - 1]

    async def token_count(self, conversation, new_message=None):
        return 0


class _FakeTitleLLM:
    def __init__(self, text):
        self.base_args = {}
        self._text = text

    async def get_message(self, *args, **kwargs):
        return SimpleNamespace(text=self._text)
