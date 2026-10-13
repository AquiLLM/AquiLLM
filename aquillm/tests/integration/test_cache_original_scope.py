"""STAGED, UNEXECUTED: real tool original-scope cache composition, not ASGI."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from importlib import import_module

import pytest
from django.db import connections

from apps.documents.models import TextChunk
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from .cache_source_support import cache_corpus_factory, upload, transport
from .cache_transport_support import cache_transport
from .cache_guard_support import assert_scoped_probe, scoped_probe
from .cache_turn_support import (
    assert_current_result, modes, run_tools, watch_finalization,
)

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.parametrize("mode", ["default", "query"])
@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
@pytest.mark.parametrize("change", ["dense", "profile", "source"])
def test_original_cache_hit_then_stale_service_guard(
    cache_corpus_factory, cache_transport, monkeypatch, mode, order, change,
):
    # Exercise the normal tool error envelope; debug mode deliberately re-raises.
    monkeypatch.setattr(import_module("lib.llm.decorators.tool"), "DEBUG", False)
    modes(monkeypatch, mode)
    corpus = cache_corpus_factory()
    mutate, replacement = corpus.replacement(change, order[0])
    probe = scoped_probe(monkeypatch, cache_transport, mutate)
    finalized = watch_finalization(monkeypatch)
    original, results, actions = run_tools(corpus, order, corpus.query, mode=mode)
    event = assert_scoped_probe(probe, cache_transport, change)
    assert actions == 2
    assert event["scope"].representation is original.representation
    assert event["scope"].dense is original.dense
    assert event["scope"].requested_document_ids == {corpus.documents[order[0]].id}
    assert all(row["remaining"] > 1000 for row in finalized)
    if change == "source":
        assert all(not result.get("result") for result in results)
        assert all(result.get("exception") == "source_scope_unavailable" for result in results)
        assert all(row.get("error") is SourceScopeUnavailable for row in finalized)
    else:
        # Only lexical evidence may survive an invalidated dense contribution.
        assert finalized and "value" in finalized[0]
        assert not finalized[0]["value"][0].vector_chunk_ids
        for index, result in zip(order, results):
            if result.get("result"):
                assert_current_result(result, corpus, index)
    # A new turn is the only place permitted to capture the replacement.
    before = len(finalized)
    fresh, fresh_results, fresh_actions = run_tools(
        corpus, (order[0],), corpus.query, mode=mode)
    assert fresh_actions == 1 and fresh.runtime is not original.runtime
    assert fresh.representation is not original.representation
    assert finalized[before:] and fresh_results[0].get("result")
    if change == "source":
        selected = next(s for s in fresh.representation.selections
                        if s.document.id == corpus.documents[order[0]].id)
        assert selected.generation_key == replacement
        rows = fresh_results[0]["result"]
        old_ids = {row.pk for row in corpus.rows[order[0]]}
        assert all(row["chunk_id"] not in old_ids for row in rows)
        assert all(TextChunk.objects.get(pk=row["chunk_id"]).representation_generation_id
                   == replacement for row in rows)
    elif change == "dense":
        assert replacement in fresh.dense.generation_keys
        assert replacement not in original.dense.generation_keys
        assert_current_result(fresh_results[0], corpus, order[0])
    else:
        assert fresh.space.space_key == replacement
        assert original.space == corpus.space
        assert_current_result(fresh_results[0], corpus, order[0])


def test_concurrent_disjoint_original_turns_do_not_share_authorized_ranking(
    cache_corpus_factory, cache_transport, monkeypatch,
):
    from apps.documents.services import chunk_rerank_local_vllm as adapter
    from apps.documents.services import rag_cache_rerank as service
    from apps.documents.services.chunk_search_scope import current_scope
    modes(monkeypatch, "default")
    left, right = cache_corpus_factory(), cache_corpus_factory()
    barrier, events = Barrier(2), []
    original = adapter.set_scoped_rerank_result
    def observe(*args):
        scope = current_scope()
        event = {"scope": scope, "arguments": args}
        events.append(event)
        original(*args)
        barrier.wait(timeout=3)
        event["hit"] = service.get_scoped_rerank_result(*args[:4])
        event["key"] = scope.cache_key()
    monkeypatch.setattr(adapter, "set_scoped_rerank_result", observe)
    def run(corpus):
        try:
            return run_tools(corpus, (0,), left.query)
        finally:
            connections.close_all()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run, corpus) for corpus in (left, right)]
        outcomes = [future.result(timeout=20) for future in futures]
    assert len(events) == 2
    assert events[0]["key"] != events[1]["key"]
    assert events[0]["scope"].runtime is not events[1]["scope"].runtime
    assert events[0]["scope"].requested_document_ids.isdisjoint(events[1]["scope"].requested_document_ids)
    for event in events:
        assert event["hit"] == event["arguments"][4] and event["hit"]
    for corpus, (_, results, actions) in zip((left, right), outcomes):
        assert actions == 1
        assert_current_result(results[0], corpus, 0)
