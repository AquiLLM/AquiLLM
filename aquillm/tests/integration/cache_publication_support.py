"""STAGED, UNEXECUTED: real existing-turn handoff after an actual warm cache hit."""
from types import SimpleNamespace
from uuid import uuid4

from apps.chat.services.tool_wiring.documents import search_single_document_tool
from aquillm.models import WSConversation
from lib.llm.providers.image_context import serialize_tool_result_for_llm
from lib.llm.turn_context import current_turn, evidence_handoff
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import AssistantMessage, ToolMessage, UserMessage
from .cache_turn_support import database, useful


def publication_inputs(corpus):
    """Construct durable conversation and original consumer before measured turn."""
    consumer = corpus.consumer
    consumer.convo = Conversation(system="Use the selected sources.",
        messages=[UserMessage(content=corpus.query)])
    source = SimpleNamespace(db=WSConversation.objects.create(owner=consumer.user))
    name = search_single_document_tool(consumer.user, consumer.col_ref).name
    arguments = dict(doc_id=str(corpus.documents[0].id), search_string=corpus.query, top_k=1)
    return consumer, source, name, arguments


def warm_scoped_result(monkeypatch):
    from apps.documents.services import chunk_rerank_local_vllm as adapter
    from apps.documents.services.rag_cache_rerank import get_scoped_rerank_result
    from apps.documents.services.chunk_search_scope import current_scope
    original, events = adapter.set_scoped_rerank_result, []
    def observe(*args):
        result = original(*args)
        scope = current_scope()
        event = {"scope": scope, "arguments": args}
        events.append(event)
        try:
            assert scope.dense_rows and len(args[1]) > args[2] == 1
            event["key"] = scope.cache_key()
            event["hit"] = get_scoped_rerank_result(*args[:4])
            assert event["key"] and event["hit"] == args[4] and event["hit"]
            event["remaining"] = useful(scope.runtime.budget)
        except Exception as error:
            event["error"] = error
            raise
        return result
    monkeypatch.setattr(adapter, "set_scoped_rerank_result", observe)
    return events


def warm_numeric_result(monkeypatch):
    from apps.documents.services import chunk_rerank_local_scored as scorer
    from apps.documents.services.chunk_search_scope import current_scope
    get, set_result = scorer.reusable_scored_result, scorer.set_scored_result
    requests, events = {}, []
    def observed_get(**kwargs):
        requests[kwargs["key"]] = kwargs
        return get(**kwargs)
    def observed_set(key, scores, **kwargs):
        result = set_result(key, scores, **kwargs)
        scope = current_scope()
        event = {"scope": scope, "key": key}
        events.append(event)
        try:
            assert scope.dense_rows and scores.status == "complete"
            event["hit"] = get(**requests[key])
            assert event["hit"] is not None and event["hit"].score_set == scores
            event["remaining"] = useful(scope.runtime.budget)
        except Exception as error:
            event["error"] = error
            raise
        return result
    monkeypatch.setattr(scorer, "reusable_scored_result", observed_get)
    monkeypatch.setattr(scorer, "set_scored_result", observed_set)
    return events


async def handoff_actual_tool(consumer, provider, raw, name, arguments, runtime, mutate):
    turn, original = current_turn(), runtime._frozen_retrieval_scope
    assert turn._media_source_runtime is runtime and runtime.budget is turn.budget
    actions = turn.budget.actions_used
    assert actions == 1
    before = useful(turn.budget)
    if mutate is not None:
        await database(mutate)
    after = useful(turn.budget)
    message = ToolMessage(tool_name=name, arguments=arguments, for_whom="assistant",
        content=serialize_tool_result_for_llm(raw), result_dict=raw)
    consumer.convo = consumer.convo + [AssistantMessage(content="", stop_reason="tool_use",
        tool_call_id=str(uuid4()), tool_call_name=name, tool_call_input=arguments), message]
    handled = await evidence_handoff(provider.llm, consumer.convo, 1024, None)
    assert handled is not None and handled[1] == "changed"
    consumer.convo = handled[0]
    assert current_turn() is turn and runtime.budget is turn.budget
    assert runtime._frozen_retrieval_scope is original
    assert turn.budget.actions_used == actions
    if mutate is None:
        assert consumer.convo[-1]._validated_answer is not None
    elif consumer.convo[-1]._validated_answer is None:
        from lib.llm.answer_publication import LIMITED_ANSWER
        assert consumer.convo[-1].content == LIMITED_ANSWER
        assert turn.answer_publication is None
    turn.budget.check_active()
    if turn.budget._synthesis is not None:
        assert turn.budget._synthesis.budget is turn.budget
        assert turn.budget._synthesis.remaining_seconds() > 1
    else:
        useful(turn.budget)
    return before, after
