"""STAGED, UNEXECUTED: immutable computation reuse is not row-publication authority."""
from importlib import import_module

import pytest

from apps.documents.services import rag_cache
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from lib.embeddings.document_binding import restore_query_observation
from .cache_source_support import cache_corpus_factory, upload, transport
from .cache_transport_support import cache_transport
from .cache_guard_support import numeric_probe
from .cache_turn_support import assert_current_result, modes, run_tools, watch_finalization

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.parametrize("change", ["dense", "profile", "source"])
def test_source_mode_numeric_reuse_does_not_authorize_stale_ranking(
    cache_corpus_factory, cache_transport, monkeypatch, change,
):
    # Exercise the normal tool error envelope; debug mode deliberately re-raises.
    monkeypatch.setattr(import_module("lib.llm.decorators.tool"), "DEBUG", False)
    modes(monkeypatch, "source")
    corpus = cache_corpus_factory()
    mutate, replacement = corpus.replacement(change, 0)
    events = numeric_probe(monkeypatch, cache_transport, mutate)
    finalized = watch_finalization(monkeypatch)
    original, results, actions = run_tools(corpus, (0, 1), corpus.query, mode="source")
    assert actions == 2 and len(events) == 1
    event = events[0]
    assert "probe_error" not in event, repr(event.get("probe_error"))
    assert event["after"].score_set == event["before"].score_set
    assert event["remaining"] > 1000
    assert event["scope"].representation is original.representation
    assert event["scope"].space is original.space
    assert any(op == "set" and key.startswith("preservation-stage:")
               for op, key, _ in cache_transport.physical)
    assert not any(op == "cache_set" and key.startswith("rrres:")
                   for op, key, _ in cache_transport.logical)
    if change == "source":
        assert all(not result.get("result") for result in results)
        assert all(result.get("exception") == "source_scope_unavailable" for result in results)
        assert all(row.get("error") is SourceScopeUnavailable for row in finalized)
    else:
        assert finalized and not finalized[0]["value"][0].vector_chunk_ids
        for index, result in enumerate(results):
            if result.get("result"):
                assert_current_result(result, corpus, index)
    query_writes = [(key, args[0]) for op, key, args in cache_transport.logical
                    if op == "cache_set" and key.startswith("query-observation-v1:")]
    assert query_writes
    scope = event["scope"]
    key = rag_cache.stable_cache_key("query-observation-v1", original.space.space_key, scope.query_digest)
    assert {item[0] for item in query_writes} == {key}
    observation = restore_query_observation(query_writes[0][1], space=original.space)
    assert observation.compatibility_key == corpus.space.space_key
    if change == "profile":
        foreign_key = rag_cache.stable_cache_key("query-observation-v1", replacement, scope.query_digest)
        assert foreign_key != key
        assert all(written != foreign_key for written, _ in query_writes)
