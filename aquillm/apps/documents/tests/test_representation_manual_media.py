"""Real human manual acquisition preserves child pixels through every final sink."""
import base64
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from asgiref.sync import async_to_sync, sync_to_async

from apps.collections.models import Collection, CollectionPermission
from apps.chat.services.manual_search_turn import run_manual_search_turn
from apps.chat.services import rag_immutable_images
from apps.documents.services import chunk_search
from apps.documents.services.representation_media_acquisition import initialize_media_runtime, media_runtime
from apps.documents.services.source_loading import current_source_runtime, source_runtime_scope
from lib.llm.turn_context import bind_turn, current_turn
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import AssistantMessage, UserMessage
from lib.llm.answer_publication import LIMITED_ANSWER
from .representation_support import upload as upload
from .test_representation_media import figure as figure
from .test_representation_media_publication import prepare

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.parametrize("source_mode", [False, True])
@pytest.mark.parametrize("selection,change", [
    ("populated", None), ("populated", "permission"), ("populated", "image_path"),
    ("none", None), ("empty", None), ("unauthorized", None),
    ("none", "permission"), ("empty", "image_path"),
    ("populated", "default_budget"),
])
def test_human_manual_image_child_survives_scope_exit_without_fresh_authority(
    figure, monkeypatch, source_mode, selection, change,
):
    from apps.chat.consumers.chat_publish import end_spin_publish
    from aquillm.message_adapters import save_conversation_to_db
    from aquillm.models import WSConversation
    from lib.llm.providers.final_stream import send_final_answer_stream
    doc, _, viewer, permission, turn, _ = prepare(figure, True)
    picker = Collection.objects.create(name="Original empty picker")
    from apps.documents.models import RawTextDocument
    if selection in ("populated", "unauthorized"):
        original_doc = RawTextDocument(title="Original picker document", full_text="Unrelated original source",
                                      collection=picker, ingested_by=viewer)
        original_doc.save(dont_rechunk=True)
    if selection != "unauthorized":
        CollectionPermission.objects.create(user=viewer, collection=picker, permission="VIEW")
    consumer = SimpleNamespace(user=viewer, col_ref=SimpleNamespace(collections=[] if selection == "none" else [picker.pk]))
    parent = initialize_media_runtime(consumer, turn)
    frozen = getattr(parent, "_frozen_retrieval_scope", None)
    snapshot = frozen.representation if frozen else None
    if selection == "populated":
        assert tuple(s.document.id for s in snapshot.selections) == (original_doc.id,)
    else:
        from apps.documents.services.source_loading import SourceRuntime
        assert not isinstance(parent, SourceRuntime) and parent.authorization is None and snapshot is None
    for key in ("RAG_RERANK_TEXT_MODE", "RAG_DOCUMENT_CAPACITY_MODE"):
        monkeypatch.setenv(key, "legacy")
    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", "source" if source_mode else "legacy")
    monkeypatch.setenv("RAG_FOLLOWUP_EVIDENCE_ENABLED", "0")
    monkeypatch.setenv("RAG_ITERATIVE_RETRIEVAL_ENABLED", "0")
    monkeypatch.setenv("LLM_TOOL_INLINE_IMAGES", "1")
    monkeypatch.setenv("OPENAI_CONTEXT_LIMIT", "65536")
    if change == "default_budget":
        monkeypatch.delenv("RAG_EVIDENCE_TOKEN_BUDGET", raising=False)
    else:
        monkeypatch.setenv("RAG_EVIDENCE_TOKEN_BUDGET", "6000")
    monkeypatch.setattr(chunk_search, "load_query_embedding", lambda *args: None)
    chunk = doc.chunks.get()
    citation = f"[doc:{doc.id} chunk:{chunk.pk}]"
    calls, events, contexts = [], [], []
    stages = []
    collect = chunk_search.collect_hybrid_candidate_snapshot
    rank = chunk_search.materialize_and_rerank_candidates
    finalize = chunk_search.finalize_ranking
    def collect_observed(*args, **kwargs):
        value = collect(*args, **kwargs)
        stages.append(("collect", [row.pk for row in value.baseline_candidates]))
        return value
    def rank_observed(*args, **kwargs):
        value = rank(*args, **kwargs)
        stages.append(("rank", [row.pk for row in value.combined_candidates], [row.pk for row in value.ranked_results]))
        return value
    def finalize_observed(*args, **kwargs):
        value = finalize(*args, **kwargs)
        stages.append(("finalize", [[row.pk for row in rows] for rows in value[1]]))
        return value
    monkeypatch.setattr(chunk_search, "collect_hybrid_candidate_snapshot", collect_observed)
    monkeypatch.setattr(chunk_search, "materialize_and_rerank_candidates", rank_observed)
    monkeypatch.setattr(chunk_search, "finalize_ranking", finalize_observed)
    class Provider:
        async def complete(self, request, *args, **kwargs):
            calls.append(request)
            context = current_turn().answer_publication
            contexts.append(context)
            assert context._media_runtime is media_runtime() and context._media_runtime is not parent
            assert context.lease is turn.budget._synthesis
            parts = request[-1].render()["content"]
            url = next(row["image_url"]["url"] for row in parts if row["type"] == "image_url")
            assert base64.b64decode(url.split(",", 1)[1]) == b"immutable-image-pixels"
            return request + [AssistantMessage(content=chunk.content + " " + citation, stop_reason="stop")], "stop"
    original = rag_immutable_images.authorize_media_dispatch
    mutated = False
    async def guard(packet, **kwargs):
        nonlocal mutated
        if not mutated:
            mutated = True
            if change == "permission":
                await sync_to_async(permission.delete)()
            elif change == "image_path":
                def replace_path():
                    doc.image_file = "unrelated-current-image.png"
                    doc.save(dont_rechunk=True, update_fields=["image_file"])
                await sync_to_async(replace_path)()
        return await original(packet, **kwargs)
    monkeypatch.setattr(rag_immutable_images, "authorize_media_dispatch", guard)
    async def stream(payload):
        events.append(payload)
    async def send(**kwargs):
        events.append(kwargs)
    conversation = Conversation(system="s", messages=[UserMessage(content=f"/search {doc.id} show the figure caption image")])
    consumer.convo = conversation
    transcript = WSConversation.objects.create(owner=viewer)
    with bind_turn(turn), source_runtime_scope(parent) if source_mode and frozen else nullcontext():
        from apps.chat.services.tool_wiring.source_documents import document_metadata
        from apps.documents.services.representation_contracts import SourceScopeUnavailable
        with pytest.raises(SourceScopeUnavailable):
            document_metadata(doc.id)
        assert async_to_sync(run_manual_search_turn)(consumer, Provider(), conversation) == "handled"
        assert media_runtime() is parent
        assert getattr(parent, "_frozen_retrieval_scope", None) is frozen
        assert (current_source_runtime() is not None) is (source_mode and frozen is not None)
        answer = consumer.convo[-1]
        if change and (change != "default_budget" or source_mode):
            assert not calls and answer.content == LIMITED_ANSWER
        else:
            assert len(calls) == 1 and citation in answer.content, stages
            assert current_turn().answer_publication is contexts[0]
        async_to_sync(send_final_answer_stream)(stream, message_uuid=str(answer.message_uuid),
            content=answer.content, stop_reason="stop", usage=0, message=answer)
        async_to_sync(end_spin_publish)(SimpleNamespace(convo=consumer.convo, _spin_epoch=0, send=send), consumer.convo)
        save_conversation_to_db(consumer.convo, transcript)
        assert all("_images" not in getattr(message, "result_dict", {}) for message in consumer.convo.messages)
    assert len(events) == 2 and transcript.db_messages.exists()
