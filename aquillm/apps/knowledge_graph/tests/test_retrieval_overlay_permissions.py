"""Permission and composition tests for the production graph overlay seam."""

from __future__ import annotations

import contextlib
import inspect
from dataclasses import replace
from uuid import UUID

import pytest

from apps.knowledge_graph.retrieval import expansion
from apps.knowledge_graph.retrieval.ppr import PPRAlgorithmConfig
from apps.knowledge_graph.retrieval.types import (
    GraphExpansionDiagnostics,
    GraphExpansionRequest,
    GraphExpansionResult,
    GraphExpansionSeed,
)

_DOC_A = UUID("11111111-1111-4111-8111-111111111111")
_DOC_B = UUID("22222222-2222-4222-8222-222222222222")
_CONFIG = PPRAlgorithmConfig(canonical_resolver_version="canonical-resolution-v1")


def _request(*, seeds: tuple[GraphExpansionSeed, ...] | None = None):
    return GraphExpansionRequest(
        seeds=seeds or (GraphExpansionSeed(10, 1, 1.0),),
        allowed_doc_ids=(_DOC_A, _DOC_B),
        allowed_collection_ids=(1, 2),
    )


def _result(request: GraphExpansionRequest, chunk_ids: tuple[int, ...] = (20,)):
    return GraphExpansionResult(
        chunk_ids=chunk_ids,
        diagnostics=GraphExpansionDiagnostics(
            status="hit" if chunk_ids else "miss",
            seed_count=len(request.seeds),
            candidate_count=len(chunk_ids),
        ),
        seed_chunk_ids=tuple(seed.chunk_id for seed in request.seeds),
    )


def test_expand_composes_one_snapshot_loader_and_ranker_with_one_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request()
    deadline = object()
    snapshot = object()
    expected = _result(request)
    calls: list[object] = []

    @contextlib.contextmanager
    def snapshot_context(*, timeout_ms: int):
        assert timeout_ms == _CONFIG.timeout_ms
        calls.append("context")
        yield deadline

    def load(observed_request, *, load_max_hops):
        assert observed_request is request
        assert load_max_hops == _CONFIG.max_hops
        calls.append("load")
        return snapshot

    def rank(observed_snapshot, observed_request, **kwargs):
        assert observed_snapshot is snapshot
        assert observed_request is request
        assert kwargs == {
            "effective_max_hops": _CONFIG.max_hops,
            "_deadline": deadline,
        }
        calls.append("rank")
        return expected

    monkeypatch.setattr(expansion, "_load_algorithm_config", lambda: _CONFIG)
    monkeypatch.setattr(expansion, "authorized_retrieval_snapshot", snapshot_context)
    monkeypatch.setattr(expansion, "load_authorized_graph_snapshot", load)
    monkeypatch.setattr(expansion, "rank_authorized_graph_snapshot", rank)

    assert expansion.expand_chunk_candidates(request) is expected
    assert calls == ["context", "load", "rank"]


@pytest.mark.parametrize(
    ("error", "expected_status"),
    ((TimeoutError(), "timeout"), (RuntimeError("storage"), "error")),
)
def test_expand_fails_open_without_scope_or_graph_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    expected_status: str,
) -> None:
    request = _request()

    @contextlib.contextmanager
    def snapshot_context(*, timeout_ms: int):
        yield object()

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(expansion, "_load_algorithm_config", lambda: _CONFIG)
    monkeypatch.setattr(expansion, "authorized_retrieval_snapshot", snapshot_context)
    monkeypatch.setattr(expansion, "load_authorized_graph_snapshot", fail)

    result = expansion.expand_chunk_candidates(request)

    assert result.chunk_ids == ()
    assert result.diagnostics.status == expected_status
    assert result.diagnostics.seed_count == 1
    assert result.diagnostics.candidate_count == 0
    assert result.diagnostics.graph_version_signature is None
    assert "document" not in repr(result.diagnostics).lower()
    assert "collection" not in repr(result.diagnostics).lower()


def test_effective_caps_are_checked_before_snapshot_or_orm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(
        seeds=(
            GraphExpansionSeed(10, 1, 1.0),
            GraphExpansionSeed(11, 2, 1.0),
        )
    )
    config = replace(_CONFIG, max_seeds=1)

    def forbidden_context(*args, **kwargs):
        raise AssertionError("snapshot opened for an oversized effective request")

    monkeypatch.setattr(expansion, "_load_algorithm_config", lambda: config)
    monkeypatch.setattr(
        expansion,
        "authorized_retrieval_snapshot",
        forbidden_context,
    )

    result = expansion.expand_chunk_candidates(request)

    assert result.chunk_ids == ()
    assert result.diagnostics.status == "miss"


def test_loader_requires_the_live_snapshot_context() -> None:
    with pytest.raises(RuntimeError, match="snapshot context"):
        expansion.load_authorized_graph_snapshot(_request(), load_max_hops=2)


def test_production_composition_has_no_network_ml_cache_or_global_graph_path() -> None:
    source = inspect.getsource(expansion)

    for forbidden in (
        "authorized_canonical_lookup",
        "requests.",
        "httpx.",
        "openai",
        "anthropic",
        "gliner",
        "django.core.cache",
        "select_for_update",
    ):
        assert forbidden not in source.lower()
    assert "allowed_doc_ids" in source
    assert "allowed_collection_ids" in source
    assert "_deadline.check()" in source
