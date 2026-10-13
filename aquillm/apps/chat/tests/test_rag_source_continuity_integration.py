"""Presented identities resolve only against the original current owner scope."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from channels.db import database_sync_to_async

from apps.chat.services import rag_pipeline
from apps.chat.services.rag_source_continuity import SourceAnchors, rehydrate_prior_evidence
from apps.chat.services.rag_turn import preservation_turn
from apps.chat.tests.continuity_owner_support import (
    change_owner, close_continuity_worker as close_continuity_worker,
    consumer_for, continuity_owners as continuity_owners, conversation,
    deleted_citation, historical_rows, rebuild_current,
)
from apps.documents.models import TextChunk
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from lib.llm.answer_publication import LIMITED_ANSWER
from lib.llm.types.messages import AssistantMessage

pytestmark = pytest.mark.django_db(transaction=True)


async def rehydrate(anchors, owners, runtime):
    return await database_sync_to_async(rehydrate_prior_evidence)(
        anchors, user=owners.user, selected_scope=(owners.collection.pk,),
        budget=runtime.budget,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["edit", "move", "delete", "revoke"])
async def test_original_capture_rejects_owner_transitions(continuity_owners, monkeypatch, change):
    owners = continuity_owners
    chunk = owners.chunks[0]
    rows = historical_rows(owners)
    convo = conversation(rows, "Explain the first paper")
    original = convo.model_dump()
    consumer = consumer_for(owners, convo)
    llm = SimpleNamespace(complete=AsyncMock())
    monkeypatch.setattr(rag_pipeline, "_run_vector_search", lambda *_: {"result": []})
    anchors = SourceAnchors((rows[0]["doc_id"],),
        ((chunk.pk, rows[0]["doc_id"], 0),), "explicit_citation")
    async with preservation_turn(consumer, max_func_calls=3) as runtime:
        assert [s.text for s in await rehydrate(anchors, owners, runtime)] == [chunk.content]
        frozen = runtime._frozen_retrieval_scope.representation
        await database_sync_to_async(change_owner)(owners, change)
        if change == "edit":
            with pytest.raises(SourceScopeUnavailable):
                await rehydrate(anchors, owners, runtime)
            with pytest.raises(SourceScopeUnavailable):
                await rag_pipeline.run_direct_rag_turn(consumer, llm, convo)
        else:
            assert await rehydrate(anchors, owners, runtime) == ()
            assert await rag_pipeline.run_direct_rag_turn(consumer, llm, convo) == "handled"
            assert consumer.convo[-1].content == LIMITED_ANSWER
        assert runtime._frozen_retrieval_scope.representation is frozen
        llm.complete.assert_not_called()
    assert convo.model_dump() == original
    retained = await database_sync_to_async(
        lambda: list(TextChunk.objects.filter(pk=chunk.pk).values("content", "doc_id"))
    )()
    assert retained == ([] if change == "delete" else [dict(content=chunk.content, doc_id=chunk.doc_id)])


@pytest.mark.asyncio
async def test_new_turn_after_rebuild_reads_new_coordinates_and_retains_old_text(continuity_owners):
    owners = continuity_owners
    old = owners.chunks[0]
    convo = conversation(historical_rows(owners), "Explain the first paper")
    consumer = consumer_for(owners, convo)
    old_anchor = SourceAnchors((str(old.doc_id),), ((old.pk, str(old.doc_id), 0),), "explicit_citation")
    async with preservation_turn(consumer, max_func_calls=3) as first:
        assert (await rehydrate(old_anchor, owners, first))[0].text == old.content
        frozen = first._frozen_retrieval_scope.representation
        await database_sync_to_async(change_owner)(owners, "edit")
        current = await database_sync_to_async(rebuild_current)(owners.documents[0])
        with pytest.raises(SourceScopeUnavailable):
            await rehydrate(old_anchor, owners, first)
        assert first._frozen_retrieval_scope.representation is frozen
    async with preservation_turn(consumer, max_func_calls=3) as fresh:
        assert fresh is not first and fresh.budget is not first.budget
        assert await rehydrate(old_anchor, owners, fresh) == ()
        anchors = SourceAnchors((str(current.doc_id),),
            ((current.pk, str(current.doc_id), current.chunk_number),), "explicit_citation")
        assert (await rehydrate(anchors, owners, fresh))[0].text == owners.documents[0].full_text
    assert current.pk != old.pk
    assert await database_sync_to_async(
        TextChunk.objects.filter(pk=old.pk, content=old.content, doc_id=old.doc_id).exists
    )()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,question,expected,revoke,stale", [
    ("paper", "Compare their measurements", 2, False, False),
    ("paper", "Explain the second paper", 1, False, False),
    ("report", "Compare their totals", 2, False, False),
    ("report", "Explain the second report", 1, False, False),
    ("paper", "Compare their measurements", 0, True, False),
    ("paper", "Explain Z Paper", 1, False, True),
    ("paper", "__stale_citation__", 0, False, True),
])
async def test_direct_followup_selects_current_answer_ordered_sources(
    continuity_owners, kind, question, expected, revoke, stale, monkeypatch,
):
    owners = continuity_owners
    for name, document in zip(("Z", "A"), owners.documents, strict=True):
        document.title = f"Current {name} {kind.title()}"
        await database_sync_to_async(document.save)(dont_rechunk=True, update_fields=["title"])
    rows = historical_rows(owners, kind)
    if stale:
        deleted = await database_sync_to_async(deleted_citation)(owners)
        rows.append(deleted)
        if question == "__stale_citation__":
            question = f"Check {deleted['citation']}"
    convo = conversation(rows, question, kind)
    snapshot = convo.model_dump()
    consumer = consumer_for(owners, convo)
    monkeypatch.setattr(rag_pipeline, "_run_vector_search", lambda *_: {"result": []})
    captured = []

    async def complete(request, *_args, **_kwargs):
        captured.append(request[-1].result_dict)
        return request + [AssistantMessage(
            content="Current comparison " + " ".join(
                row["citation"] for row in request[-1].result_dict["result"]),
            stop_reason="end_turn",
        )], "changed"

    llm = SimpleNamespace(complete=AsyncMock(side_effect=complete))
    if revoke:
        original_synthesize = rag_pipeline.synthesize_from_evidence

        async def revoke_at_synthesis(*args, **kwargs):
            await database_sync_to_async(owners.permission.delete)()
            return await original_synthesize(*args, **kwargs)

        monkeypatch.setattr(rag_pipeline, "synthesize_from_evidence", revoke_at_synthesis)
    assert await rag_pipeline.run_direct_rag_turn(consumer, llm, convo) == "handled"
    assert convo.model_dump() == snapshot
    if revoke or expected == 0:
        llm.complete.assert_not_called()
        assert captured == []
        assert consumer.convo[-1].content == LIMITED_ANSWER
        return
    llm.complete.assert_awaited_once()
    assert len(captured) == 1 and len(captured[0]["result"]) == expected
    indices = (0, 1) if expected == 2 else ((0,) if stale else (1,))
    selected = captured[0]["result"]
    assert [row["doc_id"] for row in selected] == [str(owners.documents[i].id) for i in indices]
    assert [row["title"] for row in selected] == [owners.documents[i].title for i in indices]
    assert [row["text"] for row in selected] == [owners.chunks[i].content for i in indices]
    assert "STALE" not in str(captured)
