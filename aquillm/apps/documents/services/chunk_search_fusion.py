# ruff: noqa: E501, E702
# fmt: off
"""Pure deterministic fusion for baseline and graph retrieval candidates."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import sha256
from math import fsum, isfinite
from typing import final
from uuid import UUID

RRF_K, MAX_BASELINE_CANDIDATES = 60, 606
MAX_BRANCH_CANDIDATES = MAX_GRAPH_ADDITIONS = 20
_DATABASE_ID_MAX, _CHUNK_NUMBER_MAX = 2**63 - 1, 2**31 - 1
_KEY = re.compile(r"[0-9a-f]{64}")
_INVALID_BASELINE, _INVALID_CAPS = "invalid baseline candidates", "invalid fusion caps"

class CandidateSource(StrEnum):
    BASELINE = "baseline"
    DIRECT = "direct"
    EXTENDED = "extended"
class FusionFailureReason(StrEnum):
    GRAPH_PROVENANCE_INVALID = "graph_provenance_invalid"

def _exact_int(value: object, minimum: int, maximum: int, name: str) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact int")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} is outside its hard cap")
def _key(value: object, name: str) -> None:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact str")
    if _KEY.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 64-hex opaque value")
def _coordinate(pk: object, document: object, chunk: object) -> None:
    _exact_int(pk, 1, _DATABASE_ID_MAX, "integer_chunk_pk")
    if type(document) is not UUID:
        raise TypeError("document_uuid must be an exact UUID")
    _exact_int(chunk, 0, _CHUNK_NUMBER_MAX, "chunk_number")

@final
@dataclass(frozen=True, slots=True)
class BaselineCandidate:
    integer_chunk_pk: int; document_uuid: UUID; chunk_number: int; candidate_object: object = field(repr=False)
    def __post_init__(self) -> None:
        _coordinate(self.integer_chunk_pk, self.document_uuid, self.chunk_number)
        if self.candidate_object is None:
            raise TypeError("candidate_object must be opaque and non-null")

@final
@dataclass(frozen=True, slots=True)
class GraphCandidate:
    integer_chunk_pk: int; document_uuid: UUID; chunk_number: int; candidate_object: object = field(repr=False)
    chunk_key: str; rank: int; score: float
    def __post_init__(self) -> None:
        _coordinate(self.integer_chunk_pk, self.document_uuid, self.chunk_number)
        if self.candidate_object is None:
            raise TypeError("candidate_object must be opaque and non-null")
        _key(self.chunk_key, "chunk_key")
        _exact_int(self.rank, 1, MAX_BRANCH_CANDIDATES, "rank")
        if type(self.score) is not float:
            raise TypeError("score must be an exact float")
        if not isfinite(self.score) or not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be finite and in [0, 1]")

def graph_candidate_order_checksum(candidates: tuple[GraphCandidate, ...]) -> str:
    if type(candidates) is not tuple or any(type(row) is not GraphCandidate for row in candidates):
        raise TypeError("candidates must contain exact GraphCandidate values")
    payload = [{"chunk_key": row.chunk_key, "rank": row.rank, "score": row.score.hex()} for row in candidates]
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

@final
@dataclass(frozen=True, slots=True)
class GraphBranchInput:
    source: CandidateSource; ready_bundle_checksum: str; candidate_count: int
    candidate_order_checksum: str; candidates: tuple[GraphCandidate, ...]
    def __post_init__(self) -> None:
        if type(self.source) is not CandidateSource or self.source is CandidateSource.BASELINE:
            raise TypeError("source must be an exact graph CandidateSource")
        _key(self.ready_bundle_checksum, "ready_bundle_checksum")
        _exact_int(self.candidate_count, 0, MAX_BRANCH_CANDIDATES, "candidate_count")
        _key(self.candidate_order_checksum, "candidate_order_checksum")
        if type(self.candidates) is not tuple or any(type(row) is not GraphCandidate for row in self.candidates):
            raise TypeError("candidates must contain exact GraphCandidate values")
        if len(self.candidates) > MAX_BRANCH_CANDIDATES:
            raise ValueError("candidates exceed their hard cap")

@final
@dataclass(frozen=True, slots=True)
class FusedCandidate:
    integer_chunk_pk: int; document_uuid: UUID; chunk_number: int; candidate_object: object = field(repr=False)
    sources: tuple[CandidateSource, ...]; baseline_rank: int | None
    direct_rank: int | None; extended_rank: int | None; rrf_score: float
    def __post_init__(self) -> None:
        _coordinate(self.integer_chunk_pk, self.document_uuid, self.chunk_number)
        if self.candidate_object is None:
            raise TypeError("candidate_object must be opaque and non-null")
        canonical = tuple(source for source in CandidateSource if source in self.sources)
        if type(self.sources) is not tuple or any(type(source) is not CandidateSource for source in self.sources) or not self.sources or self.sources != canonical:
            raise ValueError("sources must be exact, unique, and canonical")
        for source, rank, maximum in ((CandidateSource.BASELINE, self.baseline_rank, 606), (CandidateSource.DIRECT, self.direct_rank, 20), (CandidateSource.EXTENDED, self.extended_rank, 20)):
            if (source in self.sources) != (rank is not None):
                raise ValueError("source membership and ranks disagree")
            if rank is not None:
                _exact_int(rank, 1, maximum, f"{source.value}_rank")
        if type(self.rrf_score) is not float or not isfinite(self.rrf_score):
            raise TypeError("rrf_score must be an exact finite float")
        ranks = tuple(rank for rank in (self.direct_rank, self.extended_rank) if rank is not None)
        if self.rrf_score != fsum(1.0 / (RRF_K + rank) for rank in ranks):
            raise ValueError("rrf_score disagrees with branch ranks")

@final
@dataclass(frozen=True, slots=True)
class FusionDiagnostics:
    baseline_count: int; direct_input_count: int; extended_input_count: int
    baseline_duplicate_count: int; cross_branch_duplicate_count: int
    graph_only_considered: int; graph_only_selected: int; graph_only_dropped: int
    malformed_provenance: bool; failure_reason: FusionFailureReason | None
    def __post_init__(self) -> None:
        limits = (("baseline_count", 606), ("direct_input_count", 20), ("extended_input_count", 20), ("baseline_duplicate_count", 40), ("cross_branch_duplicate_count", 20), ("graph_only_considered", 40), ("graph_only_selected", 20), ("graph_only_dropped", 40))
        for name, maximum in limits:
            _exact_int(getattr(self, name), 0, maximum, name)
        if type(self.malformed_provenance) is not bool:
            raise TypeError("malformed_provenance must be an exact bool")
        if self.failure_reason is not None and type(self.failure_reason) is not FusionFailureReason:
            raise TypeError("failure_reason must be exact")
        if self.malformed_provenance != (self.failure_reason is FusionFailureReason.GRAPH_PROVENANCE_INVALID):
            raise ValueError("failure diagnostics disagree")
        if self.graph_only_considered != self.graph_only_selected + self.graph_only_dropped:
            raise ValueError("graph-only diagnostics disagree")
        if self.malformed_provenance:
            if any((self.baseline_duplicate_count, self.cross_branch_duplicate_count, self.graph_only_considered, self.graph_only_selected, self.graph_only_dropped)):
                raise ValueError("malformed diagnostics must not claim graph output")
        elif self.baseline_duplicate_count > self.direct_input_count + self.extended_input_count or self.baseline_duplicate_count > 2 * self.baseline_count or self.cross_branch_duplicate_count > min(self.direct_input_count, self.extended_input_count) or self.cross_branch_duplicate_count > self.graph_only_considered + self.baseline_duplicate_count // 2 or self.graph_only_considered > self.direct_input_count + self.extended_input_count - self.baseline_duplicate_count:
            raise ValueError("fusion diagnostic counts are incoherent")
def _graph_sort_key(row: FusedCandidate) -> tuple:
    best_rank = min(rank for rank in (row.direct_rank, row.extended_rank) if rank is not None)
    return -row.rrf_score, -len(row.sources), best_rank, row.document_uuid.int, row.chunk_number, row.integer_chunk_pk

@final
@dataclass(frozen=True, slots=True)
class FusionResult:
    candidates: tuple[FusedCandidate, ...]; rerank_candidates: tuple[object, ...] = field(repr=False)
    diagnostics: FusionDiagnostics
    def __post_init__(self) -> None:
        if type(self.candidates) is not tuple or any(type(row) is not FusedCandidate for row in self.candidates):
            raise TypeError("candidates must contain exact FusedCandidate values")
        if len(self.candidates) > 626 or len({row.integer_chunk_pk for row in self.candidates}) != len(self.candidates) or len({(row.document_uuid, row.chunk_number) for row in self.candidates}) != len(self.candidates):
            raise ValueError("fused candidates exceed bounds or contain duplicates")
        if type(self.rerank_candidates) is not tuple or len(self.rerank_candidates) != len(self.candidates):
            raise TypeError("rerank_candidates must be one exact tuple")
        if any(candidate is not fused.candidate_object for candidate, fused in zip(self.rerank_candidates, self.candidates, strict=True)):
            raise ValueError("rerank candidates disagree with fused candidates")
        if type(self.diagnostics) is not FusionDiagnostics:
            raise TypeError("diagnostics must be exact")
        if len(self.candidates) != self.diagnostics.baseline_count + self.diagnostics.graph_only_selected:
            raise ValueError("candidate count disagrees with diagnostics")
        baseline = self.candidates[:self.diagnostics.baseline_count]
        graph = self.candidates[self.diagnostics.baseline_count:]
        if tuple(row.baseline_rank for row in baseline) != tuple(range(1, len(baseline) + 1)) or any(CandidateSource.BASELINE not in row.sources for row in baseline):
            raise ValueError("baseline prefix is invalid")
        if any(CandidateSource.BASELINE in row.sources or row.baseline_rank is not None for row in graph) or tuple(sorted(graph, key=_graph_sort_key)) != graph:
            raise ValueError("graph suffix is invalid")
        direct_count = sum(CandidateSource.DIRECT in row.sources for row in self.candidates)
        extended_count = sum(CandidateSource.EXTENDED in row.sources for row in self.candidates)
        direct_ranks = tuple(row.direct_rank for row in self.candidates if row.direct_rank is not None)
        extended_ranks = tuple(row.extended_rank for row in self.candidates if row.extended_rank is not None)
        baseline_duplicates = sum(CandidateSource.DIRECT in row.sources for row in baseline) + sum(CandidateSource.EXTENDED in row.sources for row in baseline)
        if direct_count > self.diagnostics.direct_input_count or extended_count > self.diagnostics.extended_input_count or baseline_duplicates != self.diagnostics.baseline_duplicate_count or any(rank > self.diagnostics.direct_input_count for rank in direct_ranks) or any(rank > self.diagnostics.extended_input_count for rank in extended_ranks) or len(set(direct_ranks)) != len(direct_ranks) or len(set(extended_ranks)) != len(extended_ranks):
            raise ValueError("candidate memberships disagree with diagnostics")
        if self.diagnostics.malformed_provenance and (direct_count or extended_count or graph):
            raise ValueError("malformed result must contain plain baseline candidates only")

def _valid_branch(branch: object, expected: CandidateSource, ready: str) -> bool:
    if type(branch) is not GraphBranchInput or branch.source is not expected:
        return False
    try:
        rows = branch.candidates
        return branch.ready_bundle_checksum == ready and branch.candidate_count == len(rows) and branch.candidate_order_checksum == graph_candidate_order_checksum(rows) and tuple(row.rank for row in rows) == tuple(range(1, len(rows) + 1)) and len({row.integer_chunk_pk for row in rows}) == len(rows) and len({row.chunk_key for row in rows}) == len(rows) and rows == tuple(sorted(rows, key=lambda row: (-row.score, row.chunk_key)))
    except (AttributeError, TypeError, ValueError):
        return False
def _plain_baseline(rows: tuple[BaselineCandidate, ...]) -> tuple[FusedCandidate, ...]:
    return tuple(FusedCandidate(row.integer_chunk_pk, row.document_uuid, row.chunk_number, row.candidate_object, (CandidateSource.BASELINE,), rank, None, None, 0.0) for rank, row in enumerate(rows, start=1))
def _safe_count(branch: object) -> int:
    return min(len(branch.candidates), 20) if type(branch) is GraphBranchInput and type(branch.candidates) is tuple else 0
def _result(rows: tuple[FusedCandidate, ...], diagnostics: FusionDiagnostics) -> FusionResult:
    return FusionResult(rows, tuple(row.candidate_object for row in rows), diagnostics)
def _invalid_graph(baseline: tuple[FusedCandidate, ...], direct_count: int, extended_count: int) -> FusionResult:
    diagnostics = FusionDiagnostics(len(baseline), direct_count, extended_count, 0, 0, 0, 0, 0, True, FusionFailureReason.GRAPH_PROVENANCE_INVALID)
    return _result(baseline, diagnostics)
def _rrf(direct_rank: int | None, extended_rank: int | None) -> float:
    return fsum(1.0 / (RRF_K + rank) for rank in (direct_rank, extended_rank) if rank is not None)
def _complete_mapping_valid(baseline: tuple[BaselineCandidate, ...], direct: GraphBranchInput | None, extended: GraphBranchInput | None) -> bool:
    pk_coordinates = {row.integer_chunk_pk: (row.document_uuid, row.chunk_number) for row in baseline}
    coordinate_identities: dict[tuple[UUID, int], tuple[int, str | None]] = {(row.document_uuid, row.chunk_number): (row.integer_chunk_pk, None) for row in baseline}
    key_identities: dict[str, tuple[int, UUID, int]] = {}
    identity_keys: dict[tuple[int, UUID, int], str] = {}
    for branch in (direct, extended):
        for row in branch.candidates if branch is not None else ():
            coordinate = row.document_uuid, row.chunk_number
            identity = row.integer_chunk_pk, *coordinate
            if pk_coordinates.get(row.integer_chunk_pk, coordinate) != coordinate:
                return False
            coordinate_identity = coordinate_identities.get(coordinate)
            if coordinate_identity is not None and (coordinate_identity[0] != row.integer_chunk_pk or coordinate_identity[1] not in (None, row.chunk_key)):
                return False
            if key_identities.get(row.chunk_key, identity) != identity:
                return False
            if identity_keys.get(identity, row.chunk_key) != row.chunk_key:
                return False
            pk_coordinates[row.integer_chunk_pk] = coordinate
            coordinate_identities[coordinate] = row.integer_chunk_pk, row.chunk_key
            key_identities[row.chunk_key] = identity
            identity_keys[identity] = row.chunk_key
    return True

def fuse_candidates(
    *, baseline: tuple[BaselineCandidate, ...], direct: GraphBranchInput | None = None,
    extended: GraphBranchInput | None = None, expected_ready_bundle_checksum: str | None = None,
    baseline_cap: int = 606, direct_cap: int = 20, extended_cap: int = 20,
    graph_cap: int = 20, rrf_k: int = RRF_K,
) -> FusionResult:
    try:
        for value, maximum, name in ((baseline_cap, 606, "baseline_cap"), (direct_cap, 20, "direct_cap"), (extended_cap, 20, "extended_cap"), (graph_cap, 20, "graph_cap")):
            _exact_int(value, 0, maximum, name)
        if type(rrf_k) is not int or rrf_k != RRF_K:
            raise ValueError
    except (TypeError, ValueError) as error:
        raise ValueError(_INVALID_CAPS) from error
    if type(baseline) is not tuple or len(baseline) > 606 or any(type(row) is not BaselineCandidate for row in baseline) or len({row.integer_chunk_pk for row in baseline}) != len(baseline) or len({(row.document_uuid, row.chunk_number) for row in baseline}) != len(baseline):
        raise ValueError(_INVALID_BASELINE)
    full_baseline = baseline
    baseline = full_baseline[:baseline_cap]
    plain = _plain_baseline(baseline)
    baseline_by_pk = {row.integer_chunk_pk: row for row in baseline}
    direct_count, extended_count = _safe_count(direct), _safe_count(extended)
    graph_present = direct is not None or extended is not None
    ready_valid = type(expected_ready_bundle_checksum) is str and _KEY.fullmatch(expected_ready_bundle_checksum) is not None
    valid = (not graph_present or ready_valid) and (direct is None or _valid_branch(direct, CandidateSource.DIRECT, expected_ready_bundle_checksum)) and (extended is None or _valid_branch(extended, CandidateSource.EXTENDED, expected_ready_bundle_checksum))
    if not valid:
        return _invalid_graph(plain, direct_count, extended_count)
    if not _complete_mapping_valid(full_baseline, direct, extended):
        return _invalid_graph(plain, direct_count, extended_count)
    selected_direct = direct.candidates[:direct_cap] if direct is not None else ()
    selected_extended = extended.candidates[:extended_cap] if extended is not None else ()
    graph_rows: dict[int, GraphCandidate] = {}
    ranks: dict[int, dict[CandidateSource, int]] = {}
    baseline_duplicates = cross_duplicates = 0
    direct_pks = {row.integer_chunk_pk for row in selected_direct}
    for source, rows in ((CandidateSource.DIRECT, selected_direct), (CandidateSource.EXTENDED, selected_extended)):
        for row in rows:
            pk = row.integer_chunk_pk
            if source is CandidateSource.EXTENDED and pk in direct_pks:
                cross_duplicates += 1
            existing = baseline_by_pk.get(pk) or graph_rows.get(pk)
            if existing is not None and (existing.document_uuid != row.document_uuid or existing.chunk_number != row.chunk_number):
                return _invalid_graph(plain, direct_count, extended_count)
            if pk in baseline_by_pk:
                baseline_duplicates += 1
            else:
                graph_rows.setdefault(pk, row)
            ranks.setdefault(pk, {})[source] = row.rank
    fused_baseline: list[FusedCandidate] = []
    for baseline_rank, row in enumerate(baseline, start=1):
        memberships = ranks.get(row.integer_chunk_pk, {})
        direct_rank, extended_rank = memberships.get(CandidateSource.DIRECT), memberships.get(CandidateSource.EXTENDED)
        sources = (CandidateSource.BASELINE,) + tuple(source for source in (CandidateSource.DIRECT, CandidateSource.EXTENDED) if source in memberships)
        fused_baseline.append(FusedCandidate(row.integer_chunk_pk, row.document_uuid, row.chunk_number, row.candidate_object, sources, baseline_rank, direct_rank, extended_rank, _rrf(direct_rank, extended_rank)))
    graph_only: list[FusedCandidate] = []
    for pk, row in graph_rows.items():
        memberships = ranks[pk]
        direct_rank, extended_rank = memberships.get(CandidateSource.DIRECT), memberships.get(CandidateSource.EXTENDED)
        sources = tuple(source for source in (CandidateSource.DIRECT, CandidateSource.EXTENDED) if source in memberships)
        graph_only.append(FusedCandidate(pk, row.document_uuid, row.chunk_number, row.candidate_object, sources, None, direct_rank, extended_rank, _rrf(direct_rank, extended_rank)))
    graph_only.sort(key=_graph_sort_key)
    selected_graph = tuple(graph_only[:graph_cap])
    candidates = (*fused_baseline, *selected_graph)
    diagnostics = FusionDiagnostics(len(baseline), direct_count, extended_count, baseline_duplicates, cross_duplicates, len(graph_only), len(selected_graph), len(graph_only) - len(selected_graph), False, None)
    return _result(candidates, diagnostics)

__all__ = ["BaselineCandidate", "CandidateSource", "FusedCandidate", "FusionDiagnostics", "FusionFailureReason", "FusionResult", "GraphBranchInput", "GraphCandidate", "fuse_candidates", "graph_candidate_order_checksum"]
