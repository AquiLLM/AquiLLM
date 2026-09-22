from __future__ import annotations

from math import inf, nan

import pytest

from lib.retrieval_redaction import (
    MAX_RETRIEVAL_LOG_COUNT,
    MAX_RETRIEVAL_LOG_ELAPSED_MS,
    RetrievalLogReason,
    retrieval_log_fields,
)


def test_retrieval_log_fields_expose_only_reason_count_and_bounded_timing() -> None:
    fields = retrieval_log_fields(
        reason=RetrievalLogReason.UPSTREAM_UNAVAILABLE,
        count=7,
        elapsed_ms=12.5,
    )
    assert fields == {
        "reason": "upstream_unavailable",
        "count": 7,
        "elapsed_ms": 12.5,
    }
    assert tuple(fields) == ("reason", "count", "elapsed_ms")


def test_retrieval_log_reason_is_a_closed_fixed_enum() -> None:
    assert {reason.value for reason in RetrievalLogReason} == {
        "completed",
        "invalid_request",
        "authentication_failed",
        "payload_too_large",
        "upstream_unavailable",
        "provenance_mismatch",
        "mixed_ontology",
        "embedding_unavailable",
        "no_seeds",
        "ambiguous",
        "internal_failure",
    }
    with pytest.raises(ValueError):
        RetrievalLogReason("unique-user-span")


@pytest.mark.parametrize(
    ("overrides", "error"),
    (
        ({"reason": "completed"}, TypeError),
        ({"count": True}, TypeError),
        ({"count": -1}, ValueError),
        ({"count": MAX_RETRIEVAL_LOG_COUNT + 1}, ValueError),
        ({"elapsed_ms": True}, TypeError),
        ({"elapsed_ms": -0.1}, ValueError),
        ({"elapsed_ms": inf}, ValueError),
        ({"elapsed_ms": nan}, ValueError),
        ({"elapsed_ms": MAX_RETRIEVAL_LOG_ELAPSED_MS + 0.1}, ValueError),
    ),
)
def test_retrieval_log_fields_reject_untyped_or_unbounded_values(
    overrides: dict[str, object], error: type[Exception]
) -> None:
    values = {
        "reason": RetrievalLogReason.COMPLETED,
        "count": 0,
        "elapsed_ms": 0.0,
        **overrides,
    }
    with pytest.raises(error):
        retrieval_log_fields(**values)


def test_unknown_payload_shape_fails_closed_without_rendering_the_payload() -> None:
    class Canary:
        def __repr__(self) -> str:
            raise AssertionError("payload repr must not be evaluated")

    with pytest.raises(TypeError, match="unexpected keyword"):
        retrieval_log_fields(
            reason=RetrievalLogReason.INTERNAL_FAILURE,
            count=0,
            elapsed_ms=0.0,
            query=Canary(),
        )
