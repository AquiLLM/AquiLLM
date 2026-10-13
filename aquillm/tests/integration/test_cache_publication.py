"""STAGED, UNEXECUTED: warm caches → existing handoff → SDK and three real sinks."""
import pytest

from lib.llm.answer_publication import LIMITED_ANSWER
from apps.chat.tests.rag_query_treatment_transport import selected_rows
from .cache_source_support import cache_corpus_factory, upload, transport
from .cache_transport_support import cache_transport
from .cache_turn_support import assert_current_result, modes, run_tools
from .cache_publication_support import (
    handoff_actual_tool, publication_inputs, warm_numeric_result, warm_scoped_result,
)
from .cross_lane_graph_route_transport import GraphRouteProvider
from .cross_lane_publication_support import publish, close_publication_worker

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.parametrize("mode", ["default", "query", "source"])
@pytest.mark.parametrize("changed", [False, True])
def test_warm_cache_does_not_replace_current_publication_authority(
    cache_corpus_factory, cache_transport, monkeypatch, mode, changed,
):
    modes(monkeypatch, mode)
    # Transport/conversation controls only; retain every cache/preservation switch.
    for key, value in {"OPENAI_CONTEXT_LIMIT": "32768", "OPENAI_STREAM_RESPONSES": "0",
        "RAG_APPEND_CITATION_SOURCES": "0", "CONTEXT_PACKER_ENABLED": "0",
        "TOKEN_EFFICIENCY_ENABLED": "0", "LLM_STREAM_FINAL_ANSWER_ONLY": "1"}.items():
        monkeypatch.setenv(key, value)
    corpus = cache_corpus_factory()
    mutation, new_key = corpus.replacement("source", 0)
    consumer, source, name, arguments = publication_inputs(corpus)
    original_user_message = consumer.convo[-1]
    provider = GraphRouteProvider()
    warm = (warm_numeric_result if mode == "source" else warm_scoped_result)(monkeypatch)
    observed = {}
    async def finish(runtime, results):
        assert len(results) == 1
        assert_current_result(results[0], corpus, 0)
        assert len(warm) == 1 and "error" not in warm[0], repr(warm)
        assert warm[0]["hit"] and warm[0]["remaining"] > 1000
        assert warm[0]["scope"].runtime is runtime
        snapshot = runtime._frozen_retrieval_scope.representation
        assert new_key not in {row.generation_key for row in snapshot.selections}
        observed["time"] = await handoff_actual_tool(consumer, provider, results[0],
            name, arguments, runtime, mutation if changed else None)
        assert runtime._frozen_retrieval_scope.representation is snapshot
        assert consumer.convo[0] is original_user_message
        assert runtime.budget.actions_used == 1
        if changed:
            assert provider.requests == [] and provider.answers == []
            assert consumer.convo[-1].content == LIMITED_ANSWER
        else:
            assert len(provider.requests) == len(provider.answers) == 1
            delivered = selected_rows(provider.answers[0])
            acquired = {row["chunk_id"]: row for row in results[0]["result"]}
            assert delivered
            for row in delivered:
                source_row = acquired[row["chunk_id"]]
                assert row["doc_id"] == source_row["doc_id"]
                assert row["citation"] == source_row["citation"]
                assert row["text"]
                # Source mode may hydrate more than the ordinary tool preview.
                original = next(c for c in corpus.rows[0] if c.pk == row["chunk_id"])
                assert row["text"] == source_row["text"] or row["text"] in original.content
                assert row["citation"] in consumer.convo[-1].content
            assert consumer.convo[-1].content != LIMITED_ANSWER
        stored, events = await publish(source, consumer)
        observed["stored"], observed["events"] = stored, events
        assert runtime._frozen_retrieval_scope.representation is snapshot
        assert runtime.budget.actions_used == 1
        runtime.budget.check_active()
    original, results, actions = run_tools(corpus, (0,), corpus.query, finish,
        mode=mode, consumer=consumer)
    assert actions == 1 and all(value > 1000 for value in observed["time"])
    assert provider.plans == []
    assert observed["stored"] == consumer.convo[-1].content
    assert len(observed["events"]) == 2
    prefix = "rrscore:v2:" if mode == "source" else "rrres:"
    assert any(op == "cache_set" and key.startswith(prefix) for op, key, _ in cache_transport.logical)
    assert any(op == "get" and value is not None for op, _, value in cache_transport.physical)
    if changed:
        wire = str(observed["events"]) + observed["stored"]
        for row in results[0]["result"]:
            assert row["text"] not in wire and row["citation"] not in wire
        assert observed["stored"] == LIMITED_ANSWER
