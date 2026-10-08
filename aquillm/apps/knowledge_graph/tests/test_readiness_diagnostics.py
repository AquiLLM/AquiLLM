"""Safe, specific observations for exact graph readiness rejections."""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter
from types import SimpleNamespace
from uuid import UUID

import pytest

from apps.documents.services.hybrid_graph_authorization import (
    HybridGraphRetrievalDependencies,
)
from apps.documents.services.hybrid_graph_orchestration import (
    hybrid_graph_candidate_pool,
)
from apps.documents.tests.hybrid_graph_test_support import (
    Policy,
    authorization,
    chunk,
    hybrid_settings,
    selected_snapshot,
)
from apps.knowledge_graph.projection.identifiers import (
    HmacSha256ProjectionIdentifierCodec,
)
from apps.knowledge_graph.retrieval import readiness_diagnostics
from apps.knowledge_graph.retrieval.production_runtime import (
    ProductionHybridBranchRuntime,
)
from apps.knowledge_graph.retrieval.ready_scope import (
    ReadyScopeError,
    ReadyScopeFailureReason,
    assemble_selected_ready_scope,
)
from apps.knowledge_graph.retrieval.ready_scope_repository import _load_authorities
from apps.knowledge_graph.tests.test_production_projection_read_aliases import (
    _ready_rows,
    _SourceManager,
)
from apps.knowledge_graph.tests.test_ready_scope import _DOC_A, _DOC_B, _authority
from lib.retrieval_redaction import RetrievalLogReason


@pytest.fixture
def readiness_events(monkeypatch):
    events = []

    class Sink:
        def info(self, event, **fields):
            events.append((event, fields))

    monkeypatch.setattr(readiness_diagnostics, "logger", Sink())
    return events


def _assert_event(events, reason):
    assert len(events) == 1
    event, fields = events[0]
    assert event == "obs.rag.graph_readiness_failed"
    assert set(fields) == {"reason", "count", "elapsed_ms"}
    assert fields["reason"] == reason
    assert fields["count"] == 1
    assert type(fields["elapsed_ms"]) is float
    assert 0.0 < fields["elapsed_ms"] <= 300_000.0
    assert "synthetic-source-sentinel" not in str(events)
    assert "synthetic-credential-sentinel" not in str(events)


def _scope_for(authorities):
    authorities = tuple(
        replace(row, embedding_model_signature="synthetic-source-sentinel")
        for row in authorities
    )
    return assemble_selected_ready_scope(
        authorization=authorization(Policy()),
        authorities=authorities,
        codec=HmacSha256ProjectionIdentifierCodec(
            b"synthetic-credential-sentinel", key_version="key-v1"
        ),
    )


@pytest.mark.parametrize(
    ("rows", "reason"),
    (
        ((_authority(7, _DOC_A, "1"),), "readiness_collection_coverage"),
        (
            (
                _authority(7, _DOC_A, "1"),
                _authority(9, _DOC_B, "2"),
                _authority(10, _DOC_B, "3"),
            ),
            "readiness_collection_coverage",
        ),
        (
            (_authority(7, _DOC_A, "1"), _authority(7, _DOC_B, "2")),
            "readiness_collection_coverage",
        ),
    ),
    ids=("missing-collection", "extra-collection", "duplicate-collection"),
)
def test_collection_coverage_keeps_public_readiness_category(
    rows, reason, readiness_events
):
    with pytest.raises(ReadyScopeError) as rejection:
        _scope_for(rows)
    assert rejection.value.reason is ReadyScopeFailureReason.READINESS_MISMATCH
    _assert_event(readiness_events, reason)


@pytest.mark.parametrize(
    ("rows", "reason"),
    (
        (
            (
                _authority(7, _DOC_A, "1"),
                replace(
                    _authority(9, _DOC_B, "2"),
                    documents=((UUID("33333333-3333-4333-8333-333333333333"), 209),),
                ),
            ),
            "readiness_document_coverage",
        ),
        (
            (_authority(7, _DOC_A, "1"), _authority(9, _DOC_A, "2")),
            "readiness_document_coverage",
        ),
    ),
    ids=("missing-document", "duplicate-document"),
)
def test_document_coverage_keeps_public_readiness_category(
    rows, reason, readiness_events
):
    with pytest.raises(ReadyScopeError) as rejection:
        _scope_for(rows)
    assert rejection.value.reason is ReadyScopeFailureReason.READINESS_MISMATCH
    _assert_event(readiness_events, reason)


def test_identifier_key_mismatch_keeps_public_readiness_category(readiness_events):
    rows = (
        _authority(7, _DOC_A, "1"),
        replace(_authority(9, _DOC_B, "2"), identifier_key_version="key-v2"),
    )
    with pytest.raises(ReadyScopeError) as rejection:
        _scope_for(rows)
    assert rejection.value.reason is ReadyScopeFailureReason.READINESS_MISMATCH
    _assert_event(readiness_events, "readiness_identifier_key")


@pytest.mark.parametrize(
    ("change", "reason"),
    (
        ("membership_missing", "readiness_membership_missing"),
        ("membership_and_artifact_missing", "readiness_membership_missing"),
        ("artifact_missing", "readiness_artifact_missing"),
        ("epoch_stale", "readiness_membership_stale"),
        ("checksum_stale", "readiness_membership_stale"),
        ("collection_snapshot_stale", "readiness_membership_stale"),
        ("artifact_snapshot_stale", "readiness_membership_stale"),
        ("artifact_binding", "readiness_artifact_binding"),
        ("stale_and_binding", "readiness_membership_stale"),
        ("artifact_collection", "readiness_artifact_collection"),
    ),
)
def test_repository_rejection_keeps_public_readiness_category(
    monkeypatch, change, reason, readiness_events
):
    from apps.knowledge_graph import models

    projections, states, artifacts, inputs = _ready_rows(
        (_authority(7, _DOC_A, "1"), _authority(9, _DOC_B, "2"))
    )
    if change in {"membership_missing", "membership_and_artifact_missing"}:
        states.pop(0)
    if change in {"artifact_missing", "membership_and_artifact_missing"}:
        artifacts.pop(0)
    elif change == "epoch_stale":
        states[0]["registry_epoch"] += 1
    elif change == "checksum_stale":
        states[0]["membership_checksum"] = "f" * 64
    elif change == "collection_snapshot_stale":
        projections[0]["collection_pk_snapshot"] += 1
    elif change == "artifact_snapshot_stale":
        projections[0]["artifact_pk_snapshot"] += 1
    elif change == "artifact_binding":
        states[0]["active_artifact_id"] += 1
    elif change == "stale_and_binding":
        states[0]["registry_epoch"] += 1
        states[0]["active_artifact_id"] += 1
    elif change == "artifact_collection":
        artifacts[0]["collection_scope_id"] += 1
    for model, rows in zip(
        (
            models.CollectionGraphProjection,
            models.CollectionGraphMembershipState,
            models.GraphArtifact,
            models.CollectionArtifactInput,
        ),
        (projections, states, artifacts, inputs),
        strict=True,
    ):
        monkeypatch.setattr(model, "objects", _SourceManager(rows, []))
    with pytest.raises(ReadyScopeError) as rejection:
        _load_authorities(
            authorization=authorization(Policy()),
            settings=SimpleNamespace(
                projection_schema_version="collection-graph-v1",
                projection_format_version="projection-v1",
                projection_identifier_key_version="key-v1",
            ),
            source_using="projection_source",
        )
    assert rejection.value.reason is ReadyScopeFailureReason.READINESS_MISMATCH
    _assert_event(readiness_events, reason)


def test_two_rejected_operations_emit_two_condition_events(readiness_events):
    with pytest.raises(ReadyScopeError, match="readiness_mismatch"):
        _scope_for((_authority(7, _DOC_A, "1"),))
    with pytest.raises(ReadyScopeError, match="readiness_mismatch"):
        _scope_for((_authority(7, _DOC_A, "1"), _authority(9, _DOC_A, "2")))
    assert [fields["reason"] for _, fields in readiness_events] == [
        "readiness_collection_coverage",
        "readiness_document_coverage",
    ]


def test_readiness_logger_rejects_free_text():
    with pytest.raises(TypeError):
        readiness_diagnostics.record_readiness_failure(
            reason="synthetic-source-sentinel", elapsed_ms=1.0
        )
    with pytest.raises(TypeError):
        readiness_diagnostics.record_readiness_failure(
            reason=RetrievalLogReason.COMPLETED, elapsed_ms=1.0
        )


def test_readiness_log_failure_preserves_original_rejection(monkeypatch):
    class BrokenSink:
        def info(self, *_args, **_kwargs):
            raise RuntimeError("synthetic-source-sentinel")

    monkeypatch.setattr(readiness_diagnostics, "logger", BrokenSink())
    with pytest.raises(ReadyScopeError) as rejection:
        _scope_for((_authority(7, _DOC_A, "1"),))
    assert str(rejection.value) == "readiness_mismatch"
    assert "synthetic-source-sentinel" not in str(rejection.value)
    assert "synthetic-credential-sentinel" not in str(rejection.value)


def test_readiness_log_failure_keeps_baseline_and_shared_failure(monkeypatch):
    class BrokenSink:
        def info(self, *_args, **_kwargs):
            raise RuntimeError("synthetic-source-sentinel")

    monkeypatch.setattr(readiness_diagnostics, "logger", BrokenSink())
    auth = authorization(Policy())
    settings = hybrid_settings()
    runtime = ProductionHybridBranchRuntime(
        authorization=auth,
        settings=settings,
        topology_loader=object(),
        codec=object(),
        scope_loader=lambda **_kwargs: _scope_for((_authority(7, _DOC_A, "1"),)),
        clock=perf_counter,
    )
    baseline = chunk(1)
    rows, diagnostics = hybrid_graph_candidate_pool(
        selected_snapshot(baseline=(baseline,)),
        "synthetic-query-sentinel",
        auth,
        HybridGraphRetrievalDependencies(runtime, settings, lambda **_kwargs: ()),
    )
    assert rows == (baseline,)
    assert diagnostics["graph_direct_reason"] == "readiness_mismatch"
    assert diagnostics["graph_extended_reason"] == "readiness_mismatch"
    assert "synthetic-source-sentinel" not in str(diagnostics)
    assert "synthetic-query-sentinel" not in str(diagnostics)
