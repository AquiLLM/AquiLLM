"""Exact semantic binding between Task21 live traces and evaluator rows."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from .task21_hybrid_live_trace_schema import TASK21_HYBRID_ARMS


def _sequence(value, context):
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{context} must be a sequence")
    return tuple(value)


def _number(value, context):
    if type(value) not in (int, float) or not math.isfinite(float(value)) or value < 0:
        raise ValueError(f"{context} must be finite and nonnegative")
    return float(value)


def validate_live_trace_observations(payload, observations):
    from .task21_hybrid_live_trace import validate_live_trace

    validate_live_trace(payload)
    if (
        not isinstance(observations, Mapping)
        or tuple(observations) != TASK21_HYBRID_ARMS
    ):
        raise ValueError("live observations fields or order are not exact")
    for arm in TASK21_HYBRID_ARMS:
        traces = _sequence(payload["arms"][arm], f"{arm} traces")
        rows = _sequence(observations[arm], f"{arm} observations")
        if len(traces) != len(rows):
            raise ValueError("live trace and observation case counts differ")
        for trace, row in zip(traces, rows, strict=True):
            if not isinstance(row, Mapping) or trace["case_id"] != row.get("case_id"):
                raise ValueError("live trace and observation case ids differ")
            latency = _number(row.get("latency_ms"), "observation latency")
            if trace["timing_trace"]["total_ms"] != latency:
                raise ValueError(
                    "live trace total timing differs from evaluator latency"
                )
            candidates = _sequence(
                row.get("candidate_trace"), "observation candidate trace"
            )
            if tuple(trace["candidate_trace"]) != candidates:
                raise ValueError("live trace candidates differ from observations")
            ranked = _sequence(row.get("ranked_chunk_ids"), "ranked chunk ids")
            if tuple(candidate["chunk_id"] for candidate in candidates) != ranked:
                raise ValueError("observation candidate order differs from ranking")
            inaccessible = _sequence(
                row.get("inaccessible_result_chunk_ids"), "inaccessible results"
            )
            if trace["inaccessible_candidate_count"] != len(inaccessible):
                raise ValueError(
                    "live trace inaccessible count differs from observations"
                )
    return payload
