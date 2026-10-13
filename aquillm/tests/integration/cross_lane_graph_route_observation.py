"""Delegates retain original G3, ranking and preparation objects."""
from types import SimpleNamespace

from apps.chat.services import rag_selection_coordinator
from apps.documents.services import chunk_search
from apps.knowledge_graph.retrieval import graph_runtime_execution_v2
from apps.knowledge_graph.retrieval.production_runtime import ProductionHybridBranchRuntime


def graph_route_observation(monkeypatch):
    audit = SimpleNamespace(runtime_calls=[], envelopes=[], hydration=[], rankings=[], prepared=[], snapshots=[])
    original_run = ProductionHybridBranchRuntime.run_v2
    original_load = graph_runtime_execution_v2.load_selection_inputs_v2
    original_rank = chunk_search.materialize_and_rerank_candidates
    original_prepare = rag_selection_coordinator.prepare_selection_candidates
    original_snapshot = chunk_search.collect_hybrid_candidate_snapshot

    def snapshot(*args, **kwargs):
        result = original_snapshot(*args, **kwargs)
        audit.snapshots.append((args, kwargs, result))
        return result

    def run(self, **kwargs):
        audit.runtime_calls.append((self, kwargs))
        result = original_run(self, **kwargs)
        audit.envelopes.append(result)
        return result

    def load(**kwargs):
        result = original_load(**kwargs)
        audit.hydration.append((kwargs, result))
        return result

    def rank(*args, **kwargs):
        result = original_rank(*args, **kwargs)
        audit.rankings.append((args, kwargs, result))
        return result

    def prepare(**kwargs):
        result = original_prepare(**kwargs)
        audit.prepared.append((kwargs, result))
        return result

    monkeypatch.setattr(ProductionHybridBranchRuntime, "run_v2", run)
    monkeypatch.setattr(graph_runtime_execution_v2, "load_selection_inputs_v2", load)
    monkeypatch.setattr(chunk_search, "materialize_and_rerank_candidates", rank)
    monkeypatch.setattr(chunk_search, "collect_hybrid_candidate_snapshot", snapshot)
    monkeypatch.setattr(rag_selection_coordinator, "prepare_selection_candidates", prepare)
    return audit


def assert_graph_originals(observed):
    audit, runtime, turn = observed.audit, observed.runtime, observed.turn
    assert audit.runtime_calls, "P1-GRAPH-ENTRY-1: actual nonnull direct route must invoke G3"
    assert len(audit.runtime_calls) == len(audit.envelopes) == 1
    assert audit.hydration and audit.rankings
    for _, kwargs in audit.runtime_calls:
        assert kwargs["authorization"] is runtime.authorization
        assert kwargs["representation_snapshot"] is observed.representation
        assert kwargs["budget"] is turn.budget
        assert kwargs["requested_document_ids"] == observed.document_ids
    for kwargs, rows in audit.hydration:
        assert kwargs["budget"] is turn.budget and rows
        assert kwargs["authorization"] is runtime.authorization
        assert kwargs["selected"].representation_snapshot is observed.representation
        assert all(row.ref.associations for row in rows)
    envelope = audit.envelopes[0]
    assert envelope.selection.representation_snapshot is observed.representation
    assert envelope.handoff.candidates and envelope.handoff.coverage.exhaustive is False
    return envelope
