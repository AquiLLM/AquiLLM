"""Pure contract tests for permission-scoped graph expansion values."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, fields
from uuid import UUID

import pytest

from apps.knowledge_graph.retrieval.ppr import PPRAlgorithmConfig
from apps.knowledge_graph.retrieval.types import (
    GraphExpansionDiagnostics,
    GraphExpansionRequest,
    GraphExpansionResult,
    GraphExpansionSeed,
)

_LIMITS = PPRAlgorithmConfig(canonical_resolver_version="contract-test-v1")
_DOC_A = UUID("11111111-1111-4111-8111-111111111111")
_DOC_B = UUID("22222222-2222-4222-8222-222222222222")
_HASH_A = "a" * 64
_HASH_B = "b" * 64


class _TupleSubclass(tuple):
    pass


def _seed(chunk_id: int = 1, rank: int = 1, weight: float = 1.0):
    return GraphExpansionSeed(
        chunk_id=chunk_id,
        rank=rank,
        restart_weight=weight,
    )


def _request(**overrides):
    values = {
        "seeds": (_seed(),),
        "allowed_doc_ids": (_DOC_A,),
        "allowed_collection_ids": (1,),
    }
    values.update(overrides)
    return GraphExpansionRequest(**values)


def _diagnostics(**overrides):
    values = {
        "status": "hit",
        "seed_count": 1,
        "candidate_count": 1,
        "elapsed_ms": 12.5,
        "algorithm_signature": _HASH_A,
        "graph_version_signature": _HASH_B,
    }
    values.update(overrides)
    return GraphExpansionDiagnostics(**values)


def test_seed_is_an_exact_frozen_value_with_normalized_positive_weight() -> None:
    seed = GraphExpansionSeed(chunk_id=7, rank=2, restart_weight=1)

    assert seed == GraphExpansionSeed(chunk_id=7, rank=2, restart_weight=1.0)
    assert seed.restart_weight == 1.0
    assert not hasattr(seed, "__dict__")
    with pytest.raises(FrozenInstanceError):
        seed.rank = 3  # type: ignore[misc]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"chunk_id": 0}, "chunk_id"),
        ({"chunk_id": True}, "chunk_id"),
        ({"chunk_id": "1"}, "chunk_id"),
        ({"rank": 0}, "rank"),
        ({"rank": True}, "rank"),
        ({"restart_weight": 0.0}, "restart_weight"),
        ({"restart_weight": -0.1}, "restart_weight"),
        ({"restart_weight": math.nan}, "restart_weight"),
        ({"restart_weight": math.inf}, "restart_weight"),
        ({"restart_weight": True}, "restart_weight"),
    ],
)
def test_seed_rejects_nonexact_or_nonpositive_values(
    overrides: dict[str, object], message: str
) -> None:
    values: dict[str, object] = {"chunk_id": 1, "rank": 1, "restart_weight": 1.0}
    values.update(overrides)

    with pytest.raises(ValueError, match=message):
        GraphExpansionSeed(**values)  # type: ignore[arg-type]


def test_request_preserves_exact_canonical_scope_tuples() -> None:
    request = GraphExpansionRequest(
        seeds=(_seed(1, 1, 0.75), _seed(2, 3, 0.25)),
        allowed_doc_ids=(_DOC_A, _DOC_B),
        allowed_collection_ids=(10, 20),
    )

    assert request.seeds == (_seed(1, 1, 0.75), _seed(2, 3, 0.25))
    assert request.allowed_doc_ids == (_DOC_A, _DOC_B)
    assert request.allowed_collection_ids == (10, 20)
    assert {field.name for field in fields(request)} == {
        "seeds",
        "allowed_doc_ids",
        "allowed_collection_ids",
    }


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"seeds": ()}, "seeds"),
        ({"seeds": (_seed(1, 1), _seed(1, 2))}, "chunk_id"),
        ({"seeds": (_seed(1, 1), _seed(2, 1))}, "rank"),
        ({"seeds": (_seed(1, 2), _seed(2, 1))}, "rank"),
        ({"seeds": (object(),)}, "seeds"),
        ({"allowed_doc_ids": ()}, "allowed_doc_ids"),
        ({"allowed_doc_ids": (_DOC_A, _DOC_A)}, "allowed_doc_ids"),
        ({"allowed_doc_ids": (_DOC_B, _DOC_A)}, "allowed_doc_ids"),
        ({"allowed_doc_ids": (str(_DOC_A),)}, "allowed_doc_ids"),
        ({"allowed_collection_ids": ()}, "allowed_collection_ids"),
        ({"allowed_collection_ids": (1, 1)}, "allowed_collection_ids"),
        ({"allowed_collection_ids": (2, 1)}, "allowed_collection_ids"),
        ({"allowed_collection_ids": (True,)}, "allowed_collection_ids"),
        ({"allowed_collection_ids": ("1",)}, "allowed_collection_ids"),
        ({"allowed_collection_ids": (object(),)}, "allowed_collection_ids"),
        (
            {"allowed_doc_ids": (_DOC_A,), "allowed_collection_ids": (1, 2)},
            "allowed_collection_ids",
        ),
    ],
)
def test_request_rejects_duplicate_unsorted_or_nonexact_scope_values(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _request(**overrides)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("seeds", [_seed()]),
        ("seeds", iter((_seed(),))),
        ("seeds", _TupleSubclass((_seed(),))),
        ("allowed_doc_ids", [_DOC_A]),
        ("allowed_doc_ids", iter((_DOC_A,))),
        ("allowed_doc_ids", _TupleSubclass((_DOC_A,))),
        ("allowed_collection_ids", [1]),
        ("allowed_collection_ids", iter((1,))),
        ("allowed_collection_ids", _TupleSubclass((1,))),
    ],
)
def test_request_rejects_nonexact_tuple_containers(
    field_name: str, value: object
) -> None:
    with pytest.raises(ValueError, match=field_name):
        _request(**{field_name: value})


def test_contracts_reject_scalar_and_value_object_subclasses() -> None:
    class IntSubclass(int):
        pass

    class FloatSubclass(float):
        pass

    class UUIDSubclass(UUID):
        pass

    class SeedSubclass(GraphExpansionSeed):
        pass

    class StringSubclass(str):
        pass

    with pytest.raises(ValueError, match="chunk_id"):
        GraphExpansionSeed(  # type: ignore[arg-type]
            chunk_id=IntSubclass(1), rank=1, restart_weight=1.0
        )
    with pytest.raises(ValueError, match="restart_weight"):
        GraphExpansionSeed(  # type: ignore[arg-type]
            chunk_id=1, rank=1, restart_weight=FloatSubclass(1.0)
        )
    with pytest.raises(ValueError, match="seeds"):
        _request(seeds=(SeedSubclass(1, 1, 1.0),))
    with pytest.raises(ValueError, match="allowed_doc_ids"):
        _request(allowed_doc_ids=(UUIDSubclass(str(_DOC_A)),))
    with pytest.raises(ValueError, match="allowed_collection_ids"):
        _request(allowed_collection_ids=(IntSubclass(1),))
    with pytest.raises(ValueError, match="status"):
        GraphExpansionDiagnostics(status=StringSubclass("miss"))
    with pytest.raises(ValueError, match="algorithm_signature"):
        GraphExpansionDiagnostics(
            status="miss", algorithm_signature=StringSubclass(_HASH_A)
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        (
            "seeds",
            tuple(
                _seed(index, index)
                for index in range(1, _LIMITS.max_seeds + 2)
            ),
        ),
        (
            "allowed_doc_ids",
            tuple(
                UUID(int=index)
                for index in range(1, _LIMITS.max_scope_documents + 2)
            ),
        ),
        (
            "allowed_collection_ids",
            tuple(range(1, _LIMITS.max_scope_collections + 2)),
        ),
    ],
)
def test_request_rejects_exact_tuples_beyond_hard_caps(
    field_name: str, value: object
) -> None:
    with pytest.raises(ValueError, match=field_name):
        _request(**{field_name: value})


def test_result_validates_novel_ordered_chunks_without_retaining_seed_ids() -> None:
    result = GraphExpansionResult(
        chunk_ids=(9, 3),
        diagnostics=_diagnostics(candidate_count=2),
        seed_chunk_ids=(1,),
    )

    assert result.chunk_ids == (9, 3)
    assert not hasattr(result, "seed_chunk_ids")
    assert not hasattr(result, "__dict__")


@pytest.mark.parametrize("status", ["miss", "timeout", "error"])
def test_nonhit_results_are_empty_and_retain_only_safe_diagnostics(
    status: str,
) -> None:
    diagnostics = GraphExpansionDiagnostics(
        status=status,
        seed_count=1,
        elapsed_ms=1.0,
        algorithm_signature=_HASH_A,
        graph_version_signature=_HASH_B,
    )
    result = GraphExpansionResult(
        chunk_ids=(), diagnostics=diagnostics, seed_chunk_ids=(1,)
    )

    assert result.chunk_ids == ()
    assert {field.name for field in fields(diagnostics)} == {
        "status",
        "seed_count",
        "candidate_count",
        "elapsed_ms",
        "algorithm_signature",
        "graph_version_signature",
    }


@pytest.mark.parametrize(
    ("chunk_ids", "seed_chunk_ids", "message"),
    [
        ((2, 2), (1,), "unique"),
        ((1, 2), (1,), "seed"),
        ((0,), (1,), "chunk_ids"),
        ((True,), (1,), "chunk_ids"),
        (("2",), (1,), "chunk_ids"),
        ((2,), (True,), "seed_chunk_ids"),
    ],
)
def test_result_rejects_duplicate_seed_or_nonexact_chunk_ids(
    chunk_ids: object, seed_chunk_ids: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        GraphExpansionResult(
            chunk_ids=chunk_ids,
            diagnostics=_diagnostics(),
            seed_chunk_ids=seed_chunk_ids,
        )


@pytest.mark.parametrize(
    ("chunk_ids", "seed_chunk_ids", "message"),
    [
        ([2], (1,), "chunk_ids"),
        (iter((2,)), (1,), "chunk_ids"),
        (_TupleSubclass((2,)), (1,), "chunk_ids"),
        ((2,), [1], "seed_chunk_ids"),
        ((2,), iter((1,)), "seed_chunk_ids"),
        ((2,), _TupleSubclass((1,)), "seed_chunk_ids"),
    ],
)
def test_result_rejects_nonexact_tuple_containers(
    chunk_ids: object, seed_chunk_ids: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        GraphExpansionResult(
            chunk_ids=chunk_ids,
            diagnostics=_diagnostics(),
            seed_chunk_ids=seed_chunk_ids,
        )


@pytest.mark.parametrize(
    ("chunk_ids", "seed_chunk_ids", "message"),
    [
        (
            tuple(range(2, _LIMITS.max_candidates + 3)),
            (1,),
            "chunk_ids",
        ),
        (
            (2,),
            tuple(range(1, _LIMITS.max_seeds + 2)),
            "seed_chunk_ids",
        ),
    ],
)
def test_result_rejects_exact_tuples_beyond_hard_caps(
    chunk_ids: object, seed_chunk_ids: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        GraphExpansionResult(
            chunk_ids=chunk_ids,
            diagnostics=_diagnostics(),
            seed_chunk_ids=seed_chunk_ids,
        )


@pytest.mark.parametrize("status", ["miss", "hit", "timeout", "error"])
def test_diagnostics_supports_only_operational_graph_statuses(status: str) -> None:
    diagnostics = GraphExpansionDiagnostics(status=status)

    assert diagnostics.status == status


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"status": "disabled"}, "status"),
        ({"status": []}, "status"),
        ({"seed_count": True}, "seed_count"),
        ({"seed_count": _LIMITS.max_seeds + 1}, "seed_count"),
        ({"candidate_count": -1}, "candidate_count"),
        ({"candidate_count": _LIMITS.max_candidates + 1}, "candidate_count"),
        ({"elapsed_ms": True}, "elapsed_ms"),
        ({"elapsed_ms": math.nan}, "elapsed_ms"),
        ({"elapsed_ms": math.inf}, "elapsed_ms"),
        ({"algorithm_signature": "A" * 64}, "algorithm_signature"),
        ({"algorithm_signature": "a" * 63}, "algorithm_signature"),
        ({"graph_version_signature": "graph-v1"}, "graph_version_signature"),
    ],
)
def test_diagnostics_rejects_unsafe_or_unbounded_metadata(
    overrides: dict[str, object], message: str
) -> None:
    values: dict[str, object] = {"status": "miss"}
    values.update(overrides)

    with pytest.raises(ValueError, match=message):
        GraphExpansionDiagnostics(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("chunk_ids", "seed_ids", "diagnostics", "message"),
    [
        ((2,), (1,), _diagnostics(candidate_count=0), "candidate_count"),
        ((2,), (1,), _diagnostics(seed_count=0), "seed_count"),
        ((), (1,), _diagnostics(status="hit", candidate_count=0), "status"),
        ((2,), (1,), _diagnostics(status="miss"), "status"),
        ((2,), (1,), "not diagnostics", "diagnostics"),
    ],
)
def test_result_rejects_inconsistent_or_unsafe_diagnostics(
    chunk_ids: object,
    seed_ids: object,
    diagnostics: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        GraphExpansionResult(
            chunk_ids=chunk_ids,
            diagnostics=diagnostics,
            seed_chunk_ids=seed_ids,
        )
