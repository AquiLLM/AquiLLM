"""Tests for LLM tool choice serialization."""

from lib.llm import dump_tool_choice


class _ModelDumpOnly:
    def model_dump(self, *, exclude_none: bool = False):
        assert exclude_none is True
        return {"type": "auto"}


class _DictOnly:
    def dict(self, *, exclude_none: bool = False):
        assert exclude_none is True
        return {"type": "any"}


def test_dump_tool_choice_prefers_model_dump():
    assert dump_tool_choice(_ModelDumpOnly()) == {"type": "auto"}


def test_dump_tool_choice_supports_dict_fallback():
    assert dump_tool_choice(_DictOnly()) == {"type": "any"}
