"""Closed, canonical evidence contract for Task21 live retrieval traces."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from .task21_hybrid_live_trace_observations import (
    validate_live_trace_observations as validate_live_trace_observations,
)
from .task21_hybrid_live_trace_schema import (
    CANDIDATE_FIELDS,
    CASE_FIELDS,
    FIXTURE_ID,
    FRESHNESS_FIELDS,
    HEX32,
    HEX40,
    HEX64,
    PARITY_FIELDS,
    SCHEMA,
    SOURCES,
    TASK21_HYBRID_ARMS,
    TIMING_FIELDS,
    TOKEN,
    TOP_FIELDS,
)


def _mapping(value, fields, context):
    if not isinstance(value, Mapping) or tuple(value) != fields:
        raise ValueError(f"{context} fields or order are not exact")
    return value


def _sequence(value, context):
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{context} must be a sequence")
    return tuple(value)


def _text(value, context):
    if type(value) is not str or TOKEN.fullmatch(value) is None:
        raise ValueError(f"{context} is invalid")
    return value


def _sha(value, context):
    if type(value) is not str or HEX64.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _integer(value, minimum, maximum, context):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{context} is outside its exact bounds")
    return value


def _number(value, context):
    if type(value) not in (int, float) or not math.isfinite(float(value)) or value < 0:
        raise ValueError(f"{context} must be finite and nonnegative")
    return float(value)


def _rank(value, context):
    if value is None:
        return None
    return _integer(value, 1, 2**31 - 1, context)


def _score(value, context, *, optional=False):
    if optional and value is None:
        return None
    if type(value) is not str:
        raise ValueError(f"{context} is not a canonical hexadecimal score")
    try:
        parsed = float.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{context} is not a hexadecimal score") from error
    if not math.isfinite(parsed) or parsed < 0 or parsed.hex() != value:
        raise ValueError(f"{context} is not canonical, finite, and nonnegative")
    return value


def _sha_sequence(value, context, *, sorted_required):
    result = tuple(_sha(item, context) for item in _sequence(value, context))
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{context} must be nonempty and unique")
    if sorted_required and result != tuple(sorted(result)):
        raise ValueError(f"{context} must be sorted")
    return result


def _validate_candidate(value, *, arm, ordinal, fixture_chunks):
    row = _mapping(value, CANDIDATE_FIELDS, "candidate trace")
    chunk_id = _text(row["chunk_id"], "candidate chunk id")
    if fixture_chunks and chunk_id not in fixture_chunks:
        raise ValueError("candidate chunk id is absent from the checked-in fixture")
    _integer(row["ordinal"], ordinal, ordinal, "candidate ordinal")
    sources = tuple(
        _text(item, "candidate source")
        for item in _sequence(row["sources"], "sources")
    )
    expected_sources = tuple(source for source in SOURCES if source in sources)
    if not sources or sources != expected_sources:
        raise ValueError("candidate sources must be a unique ordered subset")
    baseline_rank = _rank(row["baseline_rank"], "baseline rank")
    direct_rank = _rank(row["direct_rank"], "direct rank")
    extended_rank = _rank(row["extended_rank"], "extended rank")
    reranker_rank = _rank(row["reranker_rank"], "reranker rank")
    direct_score = _score(row["direct_score_hex"], "direct score", optional=True)
    extended_score = _score(row["extended_score_hex"], "extended score", optional=True)
    _score(row["fusion_score_hex"], "fusion score")
    memberships = {
        "baseline": baseline_rank is not None,
        "direct": direct_rank is not None and direct_score is not None,
        "extended": extended_rank is not None and extended_score is not None,
    }
    if any((source in sources) != present for source, present in memberships.items()):
        raise ValueError("candidate source ranks and scores are inconsistent")
    if (direct_rank is None) != (direct_score is None):
        raise ValueError("direct rank and score presence differs")
    if (extended_rank is None) != (extended_score is None):
        raise ValueError("extended rank and score presence differs")
    allowed_sources = {
        "vector_only": frozenset({"baseline"}),
        "direct": frozenset({"baseline", "direct"}),
        "extended": frozenset({"baseline", "extended"}),
        "combined": frozenset(SOURCES),
        "combined_reranked": frozenset(SOURCES),
    }
    if not set(sources).issubset(allowed_sources[arm]):
        raise ValueError("candidate sources violate the exact arm policy")
    if arm == "combined_reranked" and reranker_rank is None:
        raise ValueError("combined_reranked candidates require a reranker rank")
    if arm != "combined_reranked" and reranker_rank is not None:
        raise ValueError("reranker ranks belong only to combined_reranked")


def _validate_case(value, *, arm, fixture_chunks):
    row = _mapping(value, CASE_FIELDS, "case trace")
    case_id = _text(row["case_id"], "case id")
    candidates = validate_candidate_trace(
        row["candidate_trace"], arm=arm, fixture_chunk_ids=fixture_chunks
    )
    timings = _mapping(row["timing_trace"], TIMING_FIELDS, "timing trace")
    exact_timings = tuple(_number(timings[field], field) for field in TIMING_FIELDS)
    if exact_timings[-1] < max(exact_timings[:-1], default=0.0):
        raise ValueError("total timing cannot be smaller than a stage timing")
    status = row["authorization_status"]
    if status not in {"current", "absent_negative"}:
        raise ValueError("authorization status is invalid")
    if type(row["graph_scheduled"]) is not bool:
        raise ValueError("graph_scheduled must be an exact boolean")
    _integer(row["inaccessible_candidate_count"], 0, 0, "inaccessible count")
    if status == "absent_negative":
        if candidates or any(exact_timings) or row["graph_scheduled"]:
            raise ValueError("absent_negative traces must be empty and unscheduled")
    elif row["graph_scheduled"] != (arm != "vector_only"):
        raise ValueError("graph scheduling differs from the evaluated arm")
    return case_id


def validate_candidate_trace(value, *, arm, fixture_chunk_ids=()):
    candidates = _sequence(value, "candidate trace")
    fixture_chunks = frozenset(fixture_chunk_ids)
    for ordinal, candidate in enumerate(candidates, start=1):
        _validate_candidate(
            candidate, arm=arm, ordinal=ordinal, fixture_chunks=fixture_chunks
        )
    return candidates


def _validate_freshness(value):
    row = _mapping(value, FRESHNESS_FIELDS, "freshness attestation")
    for field in FRESHNESS_FIELDS[:4]:
        _sha_sequence(row[field], field, sorted_required=True)
    _text(row["ontology_version"], "ontology version")
    _sha(row["ontology_checksum"], "ontology checksum")


def _validate_parity(value):
    rows = _sequence(value, "backend parity inputs")
    if len(rows) != 2:
        raise ValueError("backend parity inputs require direct and extended")
    for expected_branch, raw in zip(("direct", "extended"), rows, strict=True):
        row = _mapping(raw, PARITY_FIELDS, "backend parity input")
        if row["branch"] != expected_branch:
            raise ValueError("backend parity branches are not exact")
        _sha(row["ready_bundle_checksum"], "ready bundle checksum")
        _sha(row["seed_checksum"], "seed checksum")
        _integer(row["seed_count"], 1, 64, "seed count")
        _integer(row["max_depth"], 1, 2, "max depth")
        _integer(row["max_nodes"], 1, 200, "max nodes")
        _integer(row["max_edges"], 1, 1000, "max edges")
        _integer(row["max_results"], 1, 20, "max results")
        for field in PARITY_FIELDS[8:]:
            _sha_sequence(row[field], field, sorted_required=False)


def validate_live_trace(payload, *, expected_case_ids=(), fixture_chunk_ids=()):
    top = _mapping(payload, TOP_FIELDS, "live trace")
    if top["schema"] != SCHEMA or top["fixture_id"] != FIXTURE_ID:
        raise ValueError("live trace schema or fixture identity is invalid")
    if type(top["run_id"]) is not str or HEX32.fullmatch(top["run_id"]) is None:
        raise ValueError("live trace run id is invalid")
    if (
        type(top["source_commit"]) is not str
        or HEX40.fullmatch(top["source_commit"]) is None
    ):
        raise ValueError("live trace source commit is invalid")
    _sha(top["fixture_checksum"], "fixture checksum")
    _sha(top["manifest_checksum"], "manifest checksum")
    fixture_chunks = frozenset(
        _text(item, "fixture chunk id") for item in fixture_chunk_ids
    )
    arms = _mapping(top["arms"], TASK21_HYBRID_ARMS, "live trace arms")
    observed_cases = None
    for arm in TASK21_HYBRID_ARMS:
        rows = _sequence(arms[arm], f"{arm} case traces")
        case_ids = tuple(
            _validate_case(row, arm=arm, fixture_chunks=fixture_chunks) for row in rows
        )
        if not case_ids or len(set(case_ids)) != len(case_ids):
            raise ValueError("live trace case ids must be nonempty and unique")
        if observed_cases is not None and case_ids != observed_cases:
            raise ValueError("live trace case order changed across arms")
        observed_cases = case_ids
    expected = tuple(expected_case_ids)
    if expected and observed_cases != expected:
        raise ValueError("live trace cases differ from the checked-in fixture")
    _validate_freshness(top["freshness_attestation"])
    _validate_parity(top["backend_parity_inputs"])
    return payload


def canonical_live_trace_bytes(payload, **validation) -> bytes:
    validate_live_trace(payload, **validation)
    body = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode()
    return body + b"\n"


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_live_trace(path, payload, **validation) -> Path:
    destination = Path(path)
    body = canonical_live_trace_bytes(payload, **validation)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if destination.exists():
        raise FileExistsError(destination)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.link(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
