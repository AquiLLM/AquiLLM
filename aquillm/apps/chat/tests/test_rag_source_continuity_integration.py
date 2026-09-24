"""Follow-up references preserve presented identities, then use current sources."""

from types import SimpleNamespace
from uuid import UUID

import pytest
from channels.db import database_sync_to_async

from apps.chat.services.rag_source_continuity import (
    SourceAnchors,
    rehydrate_prior_evidence,
)
from apps.chat.tests.test_rag_selection_scoring import Policy, _authorization
from apps.collections.services.retrieval_authorization import (
    bind_retrieval_reauthorization_capability,
    freeze_retrieval_authorization_context,
)
from apps.documents.models.chunks import TextChunk
from apps.documents.services.source_loading import SourceRuntime, source_runtime_scope
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import AssistantMessage, ToolMessage, UserMessage
from lib.retrieval.turn_budget import TurnBudget, TurnLimits


@pytest.mark.django_db(transaction=True)
def test_rehydration_uses_current_revision_and_excludes_deleted_moved_revoked():
    doc = UUID("11111111-1111-4111-8111-111111111111")
    moved = UUID("22222222-2222-4222-8222-222222222222")
    policy = Policy()
    runtime = SourceRuntime(TurnBudget(TurnLimits()), _authorization(policy))
    user = runtime.authorization.reauthorization_capability._principal
    chunk = TextChunk.objects.bulk_create(
        [
            TextChunk(
                doc_id=doc,
                chunk_number=0,
                content="Old 91 units",
                start_position=0,
                end_position=12,
            )
        ]
    )[0]
    anchors = SourceAnchors(
        (str(doc),), ((chunk.pk, str(doc), 0),), "explicit_citation"
    )
    with source_runtime_scope(runtime):
        assert (
            rehydrate_prior_evidence(
                anchors, user=user, selected_scope=(1,), budget=runtime.budget
            )[0].text
            == "Old 91 units"
        )
        TextChunk.objects.filter(pk=chunk.pk).update(content="Corrected 93 units")
        assert (
            rehydrate_prior_evidence(
                anchors, user=user, selected_scope=(1,), budget=runtime.budget
            )[0].text
            == "Corrected 93 units"
        )
        TextChunk.objects.filter(pk=chunk.pk).update(doc_id=moved)
        assert (
            rehydrate_prior_evidence(
                anchors, user=user, selected_scope=(1,), budget=runtime.budget
            )
            == ()
        )
        TextChunk.objects.filter(pk=chunk.pk).update(doc_id=doc)
        policy.allowed = ()
        assert (
            rehydrate_prior_evidence(
                anchors, user=user, selected_scope=(1,), budget=runtime.budget
            )
            == ()
        )
        policy.allowed = ((1, doc),)
        TextChunk.objects.filter(pk=chunk.pk).delete()
        assert (
            rehydrate_prior_evidence(
                anchors, user=user, selected_scope=(1,), budget=runtime.budget
            )
            == ()
        )


@pytest.mark.asyncio
@pytest.mark.django_db(transaction=True)
@pytest.mark.parametrize(
    "kind,question,expected,revoke,stale",
    [
        ("paper", "Compare their measurements", 2, False, False),
        ("paper", "Explain the second paper", 1, False, False),
        ("report", "Compare their totals", 2, False, False),
        ("report", "Explain the second report", 1, False, False),
        ("paper", "Compare their measurements", 0, True, False),
        ("paper", "Explain Z Paper", 1, False, True),
        ("paper", "__stale_citation__", 0, False, True),
    ],
)
async def test_direct_followup_selects_current_answer_ordered_sources(
    kind, question, expected, revoke, stale, monkeypatch
):
    from apps.chat.refs import CollectionsRef
    from apps.chat.services import rag_pipeline
    from apps.chat.services.tool_wiring import source_documents

    doc_z = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
    doc_a = UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
    chunks = await database_sync_to_async(TextChunk.objects.bulk_create)(
        [
            TextChunk(
                doc_id=doc_z,
                chunk_number=0,
                content="Current Z 93 units",
                start_position=0,
                end_position=18,
            ),
            TextChunk(
                doc_id=doc_a,
                chunk_number=0,
                content="Current A 12 units",
                start_position=0,
                end_position=18,
            ),
        ]
    )
    rows = [
        {
            "doc_id": str(doc_z),
            "chunk_id": chunks[0].pk,
            "chunk": 0,
            "title": f"Z {kind.title()}",
            "text": "STALE Z 91 units",
            "citation": f"[doc:{doc_z} chunk:{chunks[0].pk}]",
        },
        {
            "doc_id": str(doc_a),
            "chunk_id": chunks[1].pk,
            "chunk": 0,
            "title": f"A {kind.title()}",
            "text": "STALE A 11 units",
            "citation": f"[doc:{doc_a} chunk:{chunks[1].pk}]",
        },
    ]
    if stale:
        stale_chunk = (
            await database_sync_to_async(TextChunk.objects.bulk_create)(
                [
                    TextChunk(
                        doc_id=doc_z,
                        chunk_number=1,
                        content="Unrelated old text",
                        start_position=19,
                        end_position=37,
                    )
                ]
            )
        )[0]
        stale_row = {
            "doc_id": str(doc_z),
            "chunk_id": stale_chunk.pk,
            "chunk": 1,
            "title": "Z Paper",
            "text": "Unrelated old text",
            "citation": f"[doc:{doc_z} chunk:{stale_chunk.pk}]",
        }
        rows.append(stale_row)
        await database_sync_to_async(
            TextChunk.objects.filter(pk=stale_chunk.pk).delete
        )()
        if question == "__stale_citation__":
            question = f"Check {stale_row['citation']}"
    convo = Conversation(
        system="sys",
        messages=[
            UserMessage(content=f"Read two {kind}s"),
            ToolMessage(
                tool_name="vector_search",
                for_whom="assistant",
                content="{}",
                result_dict={
                    "result": list(reversed(rows)),
                    "retrieved_documents": [f"A {kind.title()}", f"Z {kind.title()}"],
                },
            ),
            AssistantMessage(
                content=(
                    f"Z first {rows[0]['citation']}; A second {rows[1]['citation']}."
                ),
                stop_reason="end_turn",
            ),
            UserMessage(content=question),
        ],
    )
    snapshot = convo.model_dump()
    policy = Policy()
    policy.allowed = ((1, doc_z), (1, doc_a))
    principal = object()
    authorization = freeze_retrieval_authorization_context(
        principal=principal,
        database_alias="default",
        policy=policy,
        selected_collection_ids=(1,),
        selected_document_ids=(doc_z, doc_a),
        reauthorization_capability=bind_retrieval_reauthorization_capability(
            principal=principal, policy=policy
        ),
    )
    runtime = SourceRuntime(TurnBudget(TurnLimits()), authorization)
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    monkeypatch.setenv("RAG_DIRECT_MAX_QUERIES", "1")
    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", "source")
    monkeypatch.setenv("RAG_FOLLOWUP_EVIDENCE_ENABLED", "1")
    monkeypatch.setenv("RAG_DOCUMENT_CAPACITY_MODE", "budgeted")
    monkeypatch.setenv("OPENAI_CONTEXT_LIMIT", "16000")
    monkeypatch.setattr(rag_pipeline, "_run_vector_search", lambda *_: {"result": []})
    monkeypatch.setattr(
        source_documents,
        "document_metadata",
        lambda doc: SimpleNamespace(
            title=f"Current {('Z' if str(doc) == str(doc_z) else 'A')} {kind.title()}"
        ),
    )
    captured = []

    async def complete(request, *_args, **_kw):
        captured.append(request[-1].result_dict)
        return request + [
            AssistantMessage(
                content="Current comparison "
                + " ".join(
                    row["citation"] for row in request[-1].result_dict["result"]
                ),
                stop_reason="end_turn",
            )
        ], "changed"

    if revoke:
        original_synthesize = rag_pipeline.synthesize_from_evidence

        async def revoke_at_synthesis(*args, **kwargs):
            policy.allowed = ()
            return await original_synthesize(*args, **kwargs)

        monkeypatch.setattr(
            rag_pipeline, "synthesize_from_evidence", revoke_at_synthesis
        )

    consumer = SimpleNamespace(user=principal, col_ref=CollectionsRef([1]), convo=convo)
    with source_runtime_scope(runtime):
        assert (
            await rag_pipeline.run_direct_rag_turn(
                consumer, SimpleNamespace(complete=complete), convo
            )
            == "handled"
        )
    if revoke or expected == 0:
        assert captured == []
        assert ("limits" if revoke else "can no longer access") in consumer.convo[
            -1
        ].content
        assert convo.model_dump() == snapshot
        return
    assert len(captured[0]["result"]) == expected
    assert {row["doc_id"] for row in captured[0]["result"]} == (
        {str(doc_z), str(doc_a)}
        if expected == 2
        else {str(doc_z) if stale else str(doc_a)}
    )
    assert all("Current" in row["text"] for row in captured[0]["result"])
    assert all(row["title"].startswith("Current ") for row in captured[0]["result"])
    assert "STALE" not in str(captured)
    assert convo.model_dump() == snapshot
