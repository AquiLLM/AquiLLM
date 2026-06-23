"""Collection notes reach the LLM: tool-result injection + ToolMessage render."""
from types import SimpleNamespace

import pytest

from apps.collections.models import Collection
from apps.notes.models import CollectionNote
from apps.chat.services.tool_wiring.documents import _inject_collection_notes
from aquillm.llm import ToolMessage


@pytest.mark.django_db
def test_inject_adds_notes_when_present():
    c = Collection.objects.create(name="papers")
    CollectionNote.objects.create(collection=c, body="The lead author is Ada.")
    col_ref = SimpleNamespace(collections=[c.id])

    result = _inject_collection_notes({"result": "chunks here"}, col_ref)

    assert "The lead author is Ada." in result["collection_notes"]
    assert result["_collection_notes_instruction"]  # guidance present


@pytest.mark.django_db
def test_inject_noop_when_no_notes():
    c = Collection.objects.create(name="empty")
    col_ref = SimpleNamespace(collections=[c.id])

    result = _inject_collection_notes({"result": "chunks here"}, col_ref)

    assert "collection_notes" not in result
    assert "_collection_notes_instruction" not in result


def test_tool_message_render_surfaces_notes():
    msg = ToolMessage(
        content="raw chunk text",
        tool_name="vector_search",
        for_whom="assistant",
        result_dict={
            "result": "raw chunk text",
            "collection_notes": "## Collection notes — papers\n\nThe lead author is Ada.",
            "_collection_notes_instruction": "Integrate these notes.",
        },
    )
    rendered = msg.render()
    content = rendered["content"]
    assert "The lead author is Ada." in content
    assert "Integrate these notes." in content
