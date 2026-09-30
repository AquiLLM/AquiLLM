"""Conditional document images remain valid across strict provider schemas."""

from copy import deepcopy

import pytest

from apps.chat.refs import CollectionsRef
from apps.chat.services.tool_wiring.documents import (
    search_single_document_tool,
    vector_search_tool,
    whole_document_tool,
)
from lib.llm.providers.openai_tools_format import transform_openai_tools


@pytest.fixture(params=["vector_search", "search_single_document", "whole_document"])
def document_tool(request):
    collections = CollectionsRef([])
    if request.param == "whole_document":
        return whole_document_tool(None, None, collections), ["doc_id"]
    if request.param == "search_single_document":
        return search_single_document_tool(None, collections), [
            "doc_id", "search_string", "top_k"
        ]
    return vector_search_tool(None, collections), ["search_string", "top_k"]


@pytest.mark.asyncio
async def test_strict_document_tools_require_explicit_image_choice_only_on_wire(
    document_tool, monkeypatch
):
    tool, required = document_tool
    definition = tool.llm_definition
    original = deepcopy(definition)
    monkeypatch.setenv("OPENAI_TOOL_STRICT", "1")

    wire = (await transform_openai_tools([definition]))[0]["function"]

    assert wire["strict"] is True
    assert wire["parameters"]["required"] == [*required, "include_images"]
    assert wire["parameters"]["properties"]["include_images"]["type"] == "boolean"
    assert wire["parameters"]["additionalProperties"] is False
    assert definition == original
    assert definition["input_schema"]["required"] == required


@pytest.mark.parametrize("strict_env,include_strict", [("0", True), ("1", False)])
@pytest.mark.asyncio
async def test_non_strict_document_tools_keep_image_choice_optional(
    document_tool, monkeypatch, strict_env, include_strict
):
    tool, required = document_tool
    monkeypatch.setenv("OPENAI_TOOL_STRICT", strict_env)

    wire = (await transform_openai_tools(
        [tool.llm_definition], include_strict=include_strict
    ))[0]["function"]

    assert "strict" not in wire
    assert wire["parameters"]["required"] == required
    assert wire["parameters"]["properties"]["include_images"]["type"] == "boolean"
