"""Retrieval-wide runtime, source, proxy, and deployment redaction canaries."""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import pytest
from django.test import override_settings

from apps.documents.services.chunk_search import (
    HybridGraphRetrievalDependencies,
    text_chunk_search,
)
from apps.documents.tests.hybrid_graph_test_support import (
    Policy,
    authorization,
    chunk,
    hybrid_settings,
    selected_snapshot,
)
from apps.documents.tests.test_chunk_search_graph_overlay import _DOC_B, _model

REPO = Path(__file__).resolve().parents[3]
CANARY = "PRIVATE_RETRIEVAL_PAYLOAD_CANARY"
SCENARIOS = (
    "baseline",
    "direct",
    "extended",
    "combined",
    "empty",
    "timeout",
    "extractor",
    "embedding",
    "memgraph",
    "reranker",
)
TARGET_SOURCES = (
    "aquillm/apps/documents/services/chunk_search_candidates.py",
    "aquillm/apps/documents/services/chunk_search.py",
    "aquillm/apps/documents/services/chunk_rerank_local_vllm.py",
    "aquillm/apps/documents/services/chunk_rerank.py",
    "aquillm/aquillm/utils.py",
    "aquillm/lib/embeddings/local.py",
)


class _Recorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def _record(self, level, *args, **kwargs):
        self.events.append((level, args, kwargs))

    def debug(self, *args, **kwargs):
        self._record("debug", *args, **kwargs)

    def info(self, *args, **kwargs):
        self._record("info", *args, **kwargs)

    def warning(self, *args, **kwargs):
        self._record("warning", *args, **kwargs)

    def error(self, *args, **kwargs):
        self._record("error", *args, **kwargs)


class _Runtime:
    def prepare_shared(self, **_kwargs):
        return object()

    def run_direct(self, **_kwargs):
        raise AssertionError

    def prepare_extended(self, **_kwargs):
        raise AssertionError

    def run_extended(self, **_kwargs):
        raise AssertionError


def _snapshot(scenario: str):
    baseline = (chunk(1), chunk(2))
    snapshot = selected_snapshot(baseline=baseline)
    if scenario == "empty":
        baseline = ()
    return replace(
        snapshot,
        vector_results=baseline,
        vector_chunk_ids=tuple(row.pk for row in baseline),
        baseline_candidates=baseline,
        exact_terms=(f"{CANARY}:{scenario}",),
        pre_dedupe_count=len(baseline),
    )


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_retrieval_scenarios_never_log_or_diagnose_payload_canary(
    monkeypatch,
    scenario,
) -> None:
    from apps.documents.services import chunk_search

    recorder = _Recorder()
    snapshot = _snapshot(scenario)
    graph_scenario = scenario in {
        "direct",
        "extended",
        "combined",
        "timeout",
        "extractor",
        "memgraph",
    }

    def collect(*_args, **kwargs):
        return replace(snapshot, vector_error=kwargs["initial_vector_error"])

    def embedding(_query):
        if scenario == "embedding":
            raise RuntimeError(f"{CANARY}:embedding")
        return (0.1, 0.2)

    def graph_pool(current, *_args, **_kwargs):
        if scenario in {"timeout", "extractor", "memgraph"}:
            raise RuntimeError(f"{CANARY}:{scenario}")
        extras = {
            "direct": (chunk(3),),
            "extended": (chunk(4, _DOC_B),),
            "combined": (chunk(3), chunk(4, _DOC_B)),
        }[scenario]
        return (*current.baseline_candidates, *extras), {
            "graph_status": "hit",
            "graph_candidate_count": len(extras),
            "graph_seed_count": 1,
            "graph_ms": 0.0,
        }

    def rerank(_model, _query, rows, _top_k):
        if scenario == "reranker":
            raise RuntimeError(f"{CANARY}:reranker")
        return tuple(rows)

    def fallback(_model, rows, _top_k):
        return tuple(rows)

    monkeypatch.setattr(chunk_search, "logger", recorder)
    monkeypatch.setattr(chunk_search, "collect_hybrid_candidate_snapshot", collect)
    monkeypatch.setattr("aquillm.utils.get_embedding", embedding)
    monkeypatch.setattr(chunk_search, "hybrid_graph_candidate_pool", graph_pool)
    monkeypatch.setattr(chunk_search, "rerank_chunks", rerank)
    monkeypatch.setattr(chunk_search, "_fallback_rerank", fallback)
    model, _filters = _model([])
    dependencies = HybridGraphRetrievalDependencies(
        runtime=_Runtime(),
        settings=hybrid_settings(),
        materialize=lambda **_kwargs: (),
    )
    diagnostics: dict[str, object] = {}

    with override_settings(
        KG_OVERLAY_ENABLED=graph_scenario,
        RAG_CACHE_ENABLED=False,
    ):
        try:
            returned = text_chunk_search(
                model,
                f"{CANARY}:{scenario}",
                1,
                list(snapshot.documents),
                authorization_context=authorization(Policy()),
                hybrid_graph_dependencies=dependencies if graph_scenario else None,
            )
            diagnostics = returned[3]
        except RuntimeError:
            pass

    assert CANARY not in repr(recorder.events)
    assert CANARY not in repr(diagnostics)


def test_shared_logging_processor_drops_payload_and_exception_fields() -> None:
    from aquillm import settings_logging

    processor = getattr(settings_logging, "redact_retrieval_log_payloads", None)
    assert callable(processor)
    safe = {"event": "obs.rag.proxy", "reason": "internal_failure", "count": 0}
    payload = {
        **safe,
        "query": CANARY,
        "prompt": CANARY,
        "body": CANARY,
        "request_body": CANARY,
        "exact_terms": (CANARY,),
        "error": CANARY,
        "exception": CANARY,
        "exc_info": RuntimeError(CANARY),
    }

    redacted = processor(None, "warning", payload)

    assert redacted == safe
    assert CANARY not in repr(redacted)


def test_shared_sources_forbid_response_text_and_exception_rendering() -> None:
    for relative in TARGET_SOURCES:
        tree = ast.parse((REPO / relative).read_text(encoding="utf-8"))
        exception_names = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ExceptHandler) and node.name is not None
        }
        for node in ast.walk(tree):
            assert not (
                isinstance(node, ast.Attribute)
                and node.attr == "text"
                and isinstance(node.value, ast.Name)
                and node.value.id == "response"
            ), relative
            assert not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "str"
                and any(
                    isinstance(arg, ast.Name) and arg.id in exception_names
                    for arg in node.args
                )
            ), relative
            assert not (
                isinstance(node, ast.FormattedValue)
                and isinstance(node.value, ast.Name)
                and node.value.id in exception_names
            ), relative


def test_vllm_proxy_disables_request_body_logging() -> None:
    script = (REPO / "deploy/scripts/vllm_start.sh").read_text(encoding="utf-8")
    assert "cmd+=(--no-enable-log-requests)" in script
    assert "cmd+=(--disable-log-requests)" in script
