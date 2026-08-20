# ruff: noqa: E501, E702
# fmt: off
import ast
import random
from dataclasses import FrozenInstanceError, fields, replace
from math import fsum
from pathlib import Path
from uuid import UUID

import pytest

from apps.documents.services.chunk_search_fusion import (
    BaselineCandidate,
    CandidateSource,
    FusedCandidate,
    FusionDiagnostics,
    FusionFailureReason,
    FusionResult,
    GraphBranchInput,
    GraphCandidate,
    fuse_candidates,
    graph_candidate_order_checksum,
)

READY = "a" * 64

def _baseline(pk: int, *, doc: int = 1, chunk: int = 0) -> BaselineCandidate:
    return BaselineCandidate(pk, UUID(int=doc), chunk, object())

def _graph(
    pk: int,
    rank: int,
    *,
    doc: int = 1,
    chunk: int = 0,
    score: float | None = None,
) -> GraphCandidate:
    return GraphCandidate(
        pk,
        UUID(int=doc),
        chunk,
        object(),
        f"{pk:064x}",
        rank,
        score if score is not None else 1.0 / rank,
    )

def _branch(
    source: CandidateSource,
    rows: tuple[GraphCandidate, ...],
    *,
    checksum: str | None = None,
    count: int | None = None,
    ready: str = READY,
) -> GraphBranchInput:
    return GraphBranchInput(
        source,
        ready,
        len(rows) if count is None else count,
        checksum or graph_candidate_order_checksum(rows),
        rows,
    )

def _fuse(
    baseline: tuple[BaselineCandidate, ...] = (),
    direct: GraphBranchInput | None = None,
    extended: GraphBranchInput | None = None,
    **caps: int,
) -> FusionResult:
    return fuse_candidates(
        baseline=baseline,
        direct=direct,
        extended=extended,
        expected_ready_bundle_checksum=READY,
        **caps,
    )

def test_contract_field_order_slots_immutability_and_pure_imports() -> None:
    assert tuple(CandidateSource) == ("baseline", "direct", "extended")
    assert tuple(field.name for field in fields(BaselineCandidate)) == (
        "integer_chunk_pk",
        "document_uuid",
        "chunk_number",
        "candidate_object",
    )
    assert tuple(field.name for field in fields(GraphCandidate)) == (
        "integer_chunk_pk",
        "document_uuid",
        "chunk_number",
        "candidate_object",
        "chunk_key",
        "rank",
        "score",
    )
    assert tuple(field.name for field in fields(GraphBranchInput)) == ("source", "ready_bundle_checksum", "candidate_count", "candidate_order_checksum", "candidates")
    assert tuple(field.name for field in fields(FusionDiagnostics)) == ("baseline_count", "direct_input_count", "extended_input_count", "baseline_duplicate_count", "cross_branch_duplicate_count", "graph_only_considered", "graph_only_selected", "graph_only_dropped", "malformed_provenance", "failure_reason")
    assert all(kind.__dataclass_params__.frozen and "__dict__" not in kind.__slots__ for kind in (BaselineCandidate, GraphCandidate, GraphBranchInput, FusedCandidate, FusionDiagnostics, FusionResult))
    assert tuple(field.name for field in fields(FusedCandidate)) == (
        "integer_chunk_pk",
        "document_uuid",
        "chunk_number",
        "candidate_object",
        "sources",
        "baseline_rank",
        "direct_rank",
        "extended_rank",
        "rrf_score",
    )
    assert tuple(field.name for field in fields(FusionResult)) == (
        "candidates",
        "rerank_candidates",
        "diagnostics",
    )
    row = _baseline(1)
    assert not hasattr(row, "__dict__")
    with pytest.raises(FrozenInstanceError):
        row.chunk_number = 3  # type: ignore[misc]
    module = Path(__file__).parents[1] / "services" / "chunk_search_fusion.py"
    imports = {
        node.module or ""
        for node in ast.walk(ast.parse(module.read_text(encoding="utf-8")))
        if isinstance(node, ast.ImportFrom)
    }
    assert not any(
        name.startswith(("django", "apps.knowledge_graph")) or "rerank" in name
        for name in imports
    )

def test_baseline_order_cap_and_fixed_fail_closed_validation() -> None:
    rows = (_baseline(3), _baseline(1), _baseline(2))
    result = _fuse(rows, baseline_cap=2)
    assert [row.integer_chunk_pk for row in result.candidates] == [3, 1]
    assert result.rerank_candidates == tuple(row.candidate_object for row in rows[:2])
    assert all(row.sources == (CandidateSource.BASELINE,) for row in result.candidates)
    assert all(row.rrf_score == 0.0 for row in result.candidates)
    for bad in ([rows[0]], (rows[0], rows[0]), (object(),)):
        with pytest.raises(ValueError) as captured:
            _fuse(bad)  # type: ignore[arg-type]
        assert str(captured.value) == "invalid baseline candidates"

def test_duplicate_baseline_keeps_position_object_and_adds_provenance() -> None:
    baseline = (_baseline(2, chunk=4), _baseline(1, chunk=3))
    duplicate = replace(_graph(1, 1, chunk=3), candidate_object=object())
    result = _fuse(
        baseline,
        _branch(CandidateSource.DIRECT, (duplicate,)),
        _branch(CandidateSource.EXTENDED, (_graph(1, 1, chunk=3),)),
    )
    assert [row.integer_chunk_pk for row in result.candidates] == [2, 1]
    fused = result.candidates[1]
    assert fused.candidate_object is baseline[1].candidate_object
    assert fused.sources == tuple(CandidateSource)
    assert (fused.baseline_rank, fused.direct_rank, fused.extended_rank) == (2, 1, 1)
    assert fused.rrf_score == fsum((1.0 / 61, 1.0 / 61))
    assert result.diagnostics.baseline_duplicate_count == 2

def test_graph_rrf_dedupe_order_caps_and_exact_float_vector() -> None:
    direct = _branch(
        CandidateSource.DIRECT,
        (_graph(20, 1, doc=2), _graph(30, 2, doc=3), _graph(40, 3, doc=4)),
    )
    extended = _branch(
        CandidateSource.EXTENDED,
        (_graph(30, 1, doc=3), _graph(50, 2, doc=5), _graph(60, 3, doc=6)),
    )
    result = _fuse(
        (_baseline(10),),
        direct,
        extended,
        direct_cap=3,
        extended_cap=2,
        graph_cap=3,
    )
    assert [row.integer_chunk_pk for row in result.candidates] == [10, 30, 20, 50]
    overlap = result.candidates[1]
    expected = fsum((1.0 / (60 + 2), 1.0 / (60 + 1)))
    assert overlap.rrf_score == expected
    assert overlap.rrf_score.hex() == expected.hex()
    assert overlap.sources == (CandidateSource.DIRECT, CandidateSource.EXTENDED)
    assert result.diagnostics == FusionDiagnostics(1, 3, 3, 0, 1, 4, 3, 1, False, None)

@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (_graph(2, 1, doc=2), _graph(1, 1, doc=1), [1, 2]),
        (_graph(2, 1, chunk=2), _graph(1, 1, chunk=1), [1, 2]),
        (_graph(2, 1), _graph(1, 1), [1, 2]),
    ],
)
def test_ties_fall_through_to_document_chunk_and_pk(left, right, expected) -> None:
    result = _fuse(
        direct=_branch(CandidateSource.DIRECT, (left,)),
        extended=_branch(CandidateSource.EXTENDED, (right,)),
    )
    assert [row.integer_chunk_pk for row in result.candidates] == expected

def test_independent_branch_caps_and_graph_cap() -> None:
    direct = _branch(CandidateSource.DIRECT, (_graph(1, 1), _graph(2, 2)))
    extended = _branch(CandidateSource.EXTENDED, (_graph(3, 1), _graph(4, 2)))
    result = _fuse(direct=direct, extended=extended, direct_cap=1, extended_cap=2)
    assert {row.integer_chunk_pk for row in result.candidates} == {1, 3, 4}
    capped = _fuse(direct=direct, extended=extended, graph_cap=1)
    assert len(capped.candidates) == 1
    assert capped.diagnostics.graph_only_dropped == 3


@pytest.mark.parametrize(
    "direct",
    [
        lambda rows: _branch(CandidateSource.EXTENDED, rows),
        lambda rows: _branch(CandidateSource.DIRECT, rows, checksum="b" * 64),
        lambda rows: _branch(CandidateSource.DIRECT, rows, count=1),
        lambda rows: _branch(
            CandidateSource.DIRECT, (rows[0], replace(rows[0], rank=2))
        ),
        lambda rows: _branch(
            CandidateSource.DIRECT, (rows[0], replace(rows[1], rank=3))
        ),
        lambda rows: _branch(CandidateSource.DIRECT, rows, ready="c" * 64),
    ],
)
def test_any_malformed_graph_provenance_drops_all_graph_rows(direct) -> None:
    rows = (_graph(2, 1), _graph(3, 2))
    result = _fuse(
        (_baseline(1),),
        direct(rows),
        _branch(CandidateSource.EXTENDED, (_graph(4, 1),)),
    )
    assert [row.integer_chunk_pk for row in result.candidates] == [1]
    assert result.diagnostics.malformed_provenance is True
    assert (
        result.diagnostics.failure_reason
        is FusionFailureReason.GRAPH_PROVENANCE_INVALID
    )


def test_cross_source_coordinate_mismatch_drops_graph() -> None:
    result = _fuse(
        direct=_branch(CandidateSource.DIRECT, (_graph(1, 1, doc=1),)),
        extended=_branch(CandidateSource.EXTENDED, (_graph(1, 1, doc=2),)),
    )
    assert result.candidates == ()
    assert result.diagnostics.malformed_provenance

def test_complete_uncapped_mapping_bijection_is_validated_before_branch_caps() -> None:
    baseline = (_baseline(1, doc=1), _baseline(9, doc=1))
    baseline_conflict = _branch(CandidateSource.DIRECT, (_graph(2, 1), _graph(9, 2, doc=2)))
    key = "f" * 64
    same_key_direct = _branch(CandidateSource.DIRECT, (_graph(2, 1), replace(_graph(3, 2), chunk_key=key)))
    same_key_extended = _branch(CandidateSource.EXTENDED, (_graph(4, 1), replace(_graph(5, 2), chunk_key=key)))
    same_pk_direct = _branch(CandidateSource.DIRECT, (_graph(2, 1), replace(_graph(6, 2), chunk_key="d" * 64)))
    same_pk_extended = _branch(CandidateSource.EXTENDED, (_graph(4, 1), replace(_graph(6, 2), chunk_key="e" * 64)))
    cases = ((baseline, baseline_conflict, None), ((), same_key_direct, same_key_extended), ((), same_pk_direct, same_pk_extended))
    for baseline_rows, direct, extended in cases:
        result = _fuse(baseline_rows, direct, extended, baseline_cap=1, direct_cap=1, extended_cap=1)
        assert result.rerank_candidates == tuple(row.candidate_object for row in baseline_rows[:1])
        assert result.diagnostics.malformed_provenance
        assert result.diagnostics.failure_reason is FusionFailureReason.GRAPH_PROVENANCE_INVALID
    valid_beyond_cap = _branch(CandidateSource.DIRECT, (_graph(9, 1, doc=1),))
    retained = _fuse(baseline, valid_beyond_cap, baseline_cap=1)
    assert [row.integer_chunk_pk for row in retained.candidates] == [1, 9]; assert retained.candidates[1].baseline_rank is None


def test_randomized_construction_order_has_canonical_output() -> None:
    signatures = set()
    for seed in range(20):
        order = ["direct", "extended"]
        random.Random(seed).shuffle(order)
        built = {name: _branch(CandidateSource(name), (_graph(5 if name == "direct" else 6, 1),)) for name in order}
        result = _fuse(direct=built["direct"], extended=built["extended"])
        signatures.add(tuple((row.integer_chunk_pk, row.sources, row.rrf_score.hex()) for row in result.candidates))
    assert len(signatures) == 1


class IntSubclass(int):
    pass


@pytest.mark.parametrize(
    "factory",
    [
        lambda: BaselineCandidate(True, UUID(int=1), 0, object()),
        lambda: BaselineCandidate(IntSubclass(1), UUID(int=1), 0, object()),
        lambda: BaselineCandidate(1, "not-a-uuid", 0, object()),
        lambda: BaselineCandidate(0, UUID(int=1), 0, object()),
        lambda: BaselineCandidate(1, UUID(int=1), -1, object()),
        lambda: GraphCandidate(1, UUID(int=1), 0, object(), "a" * 64, True, 0.5),
        lambda: GraphCandidate(1, UUID(int=1), 0, object(), "A" * 64, 1, 0.5),
        lambda: GraphCandidate(1, UUID(int=1), 0, object(), "a" * 64, 1, float("nan")),
    ],
)
def test_identity_and_order_inputs_reject_adversarial_types(factory) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory()


@pytest.mark.parametrize("rrf_k", [True, 59, 61, 60.0])
def test_rrf_k_is_fixed_to_exact_integer_sixty(rrf_k) -> None:
    with pytest.raises(ValueError, match="invalid fusion caps"):
        _fuse(rrf_k=rrf_k)
