"""STAGED, UNEXECUTED: real prepared query-only proposal loses detached authority."""
from asgiref.sync import async_to_sync
import pytest

from apps.chat.services.rag_turn import preservation_turn
from apps.chat.services.retrieval_query_runtime import prepare_query_proposals, run_optional_proposal
from apps.chat.services.tool_wiring.documents import search_single_document_tool
from apps.collections.tests.test_vocabulary_scope import entry, publish, service
from apps.documents.services.source_loading import current_source_runtime
from lib.llm.turn_context import current_turn
from .cache_source_support import cache_corpus_factory, upload, transport
from .cache_transport_support import cache_transport
from .cache_turn_support import assert_current_result, database, modes, useful

pytestmark = pytest.mark.django_db(transaction=True)


def test_real_optional_proposal_cache_rejects_detached_original_pin(
    cache_corpus_factory, cache_transport, monkeypatch,
):
    from apps.documents.services import chunk_rerank_local_vllm as adapter
    from apps.documents.services import rag_cache_rerank as cache_service
    from apps.documents.services.chunk_search_scope import current_scope
    from apps.chat.services.rag_config import rag_preservation_config
    modes(monkeypatch, "query")
    corpus = cache_corpus_factory()
    consumer = corpus.consumer
    _, version = publish(consumer.user, corpus.documents[0].collection,
                         [entry("instrument", "calibration apparatus")])
    pin = service().attach_vocabulary(actor=consumer.user,
        collection=corpus.documents[0].collection, version=version, expected_revision=0)
    events, provider_calls, optional_calls, returned = [], [], [], []
    original = adapter.set_scoped_rerank_result
    class NoPlanner:
        base_args = {}
        async def get_message(self, **kwargs):
            provider_calls.append(kwargs)
            raise AssertionError("vocabulary-only preparation must not call a planner")

    def observe(*args):
        scope = current_scope()
        if scope.proposal_binding is None:
            return original(*args)
        event = {"scope": scope, "bound": scope.proposal_binding, "entered": []}
        events.append(event)
        try:
            assert len(args[1]) > 1 and scope.dense_rows
            original(*args)
            event["warm"] = cache_service.get_scoped_rerank_result(*args[:4])
            assert event["warm"] == args[4] and event["warm"]
            service().detach_vocabulary(actor=consumer.user,
                attachment=pin, expected_revision=pin.revision)
            event["remaining"] = useful(scope.runtime.budget)
            before = len(cache_transport.logical)
            for name, function, arguments in (
                ("set", cache_service.set_scoped_rerank_result, args),
                ("get", cache_service.get_scoped_rerank_result, args[:4]),
            ):
                event["entered"].append(name)
                try:
                    function(*arguments)
                except ValueError as error:
                    event[name] = str(error)
                else:
                    event[name] = "unexpected acceptance"
            event["after"] = cache_transport.logical[before:]
        except Exception as error:
            event["error"] = error
            raise
    monkeypatch.setattr(adapter, "set_scoped_rerank_result", observe)

    async def run():
        async with preservation_turn(consumer, max_func_calls=3) as runtime:
            assert runtime is current_source_runtime()
            assert runtime is current_turn()._media_source_runtime
            assert not rag_preservation_config().active
            original_scope = runtime._frozen_retrieval_scope
            budget = runtime.budget
            tool = search_single_document_tool(consumer.user, consumer.col_ref,
                authorization_context=runtime.authorization)
            baseline = await database(tool, doc_id=str(corpus.documents[1].id),
                search_string=corpus.query, top_k=1)
            assert_current_result(baseline, corpus, 1)
            rows = await prepare_query_proposals(corpus.query, llm=NoPlanner(), runtime=runtime)
            assert [row.kind for row in rows] == ["original", "vocabulary"]
            captured = runtime._query_vocabulary
            assert captured.version_keys == (version.version_key,)
            async def acquire(action):
                optional_calls.append(action)
                result = await database(tool, doc_id=str(corpus.documents[0].id),
                    search_string=action.query, top_k=1)
                returned.append(result)
                return result
            # This direct service probe lacks acquire_evidence's exception boundary.
            with pytest.raises(ValueError, match="pins changed"):
                await run_optional_proposal(rows[1], anchors=None, run=acquire)
            assert returned == []
            assert runtime._query_vocabulary is captured
            assert runtime._frozen_retrieval_scope is original_scope
            assert runtime.budget is budget and budget.actions_used == 2
            useful(budget)
            with pytest.raises(ValueError, match="pins changed"):
                await run_optional_proposal(rows[1], anchors=None, run=acquire)
            assert runtime._query_vocabulary is captured and budget.actions_used == 2
            assert events[0]["bound"].vocabulary is captured
        async with preservation_turn(consumer, max_func_calls=3) as fresh:
            assert fresh is not runtime
            rows = await prepare_query_proposals(corpus.query, llm=NoPlanner(), runtime=fresh)
            assert [row.kind for row in rows] == ["original"]
            assert fresh._query_vocabulary is not captured
    async_to_sync(run)()
    assert provider_calls == [] and len(optional_calls) == 1
    assert len(events) == 1 and "error" not in events[0], repr(events)
    event = events[0]
    assert event["entered"] == ["set", "get"] and not event["after"]
    assert "pins changed" in event["set"] and "pins changed" in event["get"]
    assert event["remaining"] > 1000
