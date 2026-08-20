# fmt: off

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping

import pytest

from apps.knowledge_graph.retrieval import ppr as ppr_module
from apps.knowledge_graph.retrieval.ppr import (
    PPRAlgorithmConfig,
    RetrievalDirection,
    canonical_algorithm_json,
    edge_evidence_flow,
    graph_algorithm_signature,
    normalize_transition_rows,
    personalized_pagerank,
    raw_edge_weight,
    support_factor,
    transition_direction_factor,
    utility_factor,
)

_A = ("local", "a")
_B = ("local", "b")
_C = ("local", "c")
_CANONICAL = ("canonical", 7)


def test_ppr_v1_transition_factors_are_frozen() -> None:
    assert transition_direction_factor(RetrievalDirection.FORWARD) == 1.0
    assert transition_direction_factor(RetrievalDirection.UNDIRECTED) == 1.0
    assert transition_direction_factor(RetrievalDirection.REVERSE_DIRECTED) == 0.35
    assert support_factor(32) == 2.0
    assert support_factor(500) == 2.0
    assert utility_factor(0.0) == 0.5
    assert utility_factor(1.0) == 1.0


def test_raw_edge_weight_uses_direction_confidence_support_and_utility() -> None:
    forward = raw_edge_weight(
        direction=RetrievalDirection.FORWARD,
        confidence=0.8,
        support_count=32,
        destination_retrieval_utility=0.5,
    )
    reverse = raw_edge_weight(
        direction=RetrievalDirection.REVERSE_DIRECTED,
        confidence=0.8,
        support_count=32,
        destination_retrieval_utility=0.5,
    )
    assert forward == pytest.approx(1.2)
    assert reverse == pytest.approx(0.42)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"confidence": float("nan")}, "confidence"),
        ({"confidence": 10**400}, "confidence"),
        ({"confidence": -0.01}, "confidence"),
        ({"confidence": 1.01}, "confidence"),
        ({"support_count": 0}, "support_count"),
        ({"support_count": True}, "support_count"),
        ({"destination_retrieval_utility": float("inf")}, "utility"),
        ({"destination_retrieval_utility": -0.01}, "utility"),
        ({"destination_retrieval_utility": 1.01}, "utility"),
        ({"direction": "reverse"}, "direction"),
    ],
)
def test_raw_edge_weight_rejects_values_outside_the_persisted_contract(
    overrides: dict[str, object], message: str
) -> None:
    values: dict[str, object] = {
        "direction": RetrievalDirection.FORWARD,
        "confidence": 0.8,
        "support_count": 2,
        "destination_retrieval_utility": 0.5,
    }
    values.update(overrides)
    with pytest.raises(ValueError, match=message):
        raw_edge_weight(**values)  # type: ignore[arg-type]


def test_transition_rows_are_combined_normalized_sorted_and_complete() -> None:
    rows = normalize_transition_rows(
        {
            _B: ((_A, 0.0),),
            _A: ((_C, 3.0), (_B, 1.0), (_B, 1.0)),
        },
        nodes=(_CANONICAL,),
    )
    assert tuple(rows) == (_CANONICAL, _A, _B, _C)
    assert rows[_A] == ((_B, 0.4), (_C, 0.6))
    assert rows[_B] == ()
    assert rows[_C] == ()
    assert rows[_CANONICAL] == ()


def test_transition_normalization_is_independent_of_mapping_and_edge_order() -> None:
    first = normalize_transition_rows(
        {_A: ((_C, 3.0), (_B, 1.0), (_B, 1.0)), _B: ((_A, 2.0),)}
    )
    permuted = normalize_transition_rows(
        {_B: ((_A, 2.0),), _A: ((_B, 1.0), (_C, 3.0), (_B, 1.0))}
    )
    assert first == permuted


def test_personalized_pagerank_redistributes_dangling_mass_to_restart() -> None:
    scores = personalized_pagerank(
        {_A: 1.0},
        {_A: ((_B, 1.0),)},
        restart_probability=0.2,
        iterations=2,
    )
    assert scores[_A] == pytest.approx(0.84)
    assert scores[_B] == pytest.approx(0.16)
    assert sum(scores.values()) == pytest.approx(1.0)


def test_personalized_pagerank_runs_exact_fixed_cycle_iterations() -> None:
    one_iteration = personalized_pagerank(
        {_A: 0.75, _B: 0.25},
        {_A: ((_B, 1.0),), _B: ((_A, 1.0),)},
        restart_probability=0.25,
        iterations=1,
    )
    two_iterations = personalized_pagerank(
        {_A: 0.75, _B: 0.25},
        {_A: ((_B, 1.0),), _B: ((_A, 1.0),)},
        restart_probability=0.25,
        iterations=2,
    )
    assert one_iteration == pytest.approx({_A: 0.375, _B: 0.625})
    assert two_iterations == pytest.approx({_A: 0.65625, _B: 0.34375})


def test_personalized_pagerank_checks_one_private_deadline_per_iteration() -> None:
    observed_times: list[float] = []
    times = iter((0.0, 0.0, 1.1))
    def clock() -> float:
        value = next(times)
        observed_times.append(value)
        return value
    deadline = ppr_module._MonotonicDeadline(expires_at=1.0, clock=clock)
    with pytest.raises(TimeoutError):
        personalized_pagerank(
            {_A: 1.0},
            {_A: ((_B, 1.0),)},
            restart_probability=0.2,
            iterations=8,
            _deadline=deadline,
        )
    assert observed_times == [0.0, 0.0, 1.1]


def test_personalized_pagerank_is_independent_of_insertion_order() -> None:
    first = personalized_pagerank(
        {_A: 3.0, _CANONICAL: 1.0},
        {
            _A: ((_B, 1.0), (_C, 2.0)),
            _B: ((_CANONICAL, 1.0),),
            _C: ((_CANONICAL, 1.0),),
            _CANONICAL: ((_A, 1.0),),
        },
        restart_probability=0.2,
        iterations=8,
    )
    permuted = personalized_pagerank(
        {_CANONICAL: 1.0, _A: 3.0},
        {
            _CANONICAL: ((_A, 1.0),),
            _C: ((_CANONICAL, 1.0),),
            _B: ((_CANONICAL, 1.0),),
            _A: ((_C, 2.0), (_B, 1.0)),
        },
        restart_probability=0.2,
        iterations=8,
    )
    assert first == permuted


def test_legacy_ppr_literal_score_trace_ties_and_caps() -> None:
    scores = personalized_pagerank(
        {_A: 3.0, _CANONICAL: 1.0},
        {
            _A: ((_B, 1.0), (_C, 2.0)),
            _B: ((_CANONICAL, 1.0),),
            _C: ((_CANONICAL, 1.0),),
            _CANONICAL: ((_A, 1.0),),
        },
        restart_probability=0.2,
        iterations=8,
    )
    assert scores == {
        _CANONICAL: 0.35968832000000006,
        _A: 0.3370873600000001,
        _B: 0.10107477333333335,
        _C: 0.2021495466666667,
    }
    trace = json.dumps(
        [[list(key), scores[key].hex()] for key in scores], separators=(",", ":")
    ).encode()
    assert trace == (
        b'[[["canonical",7],"0x1.7052228c9cdc0p-2"],'
        b'[["local","a"],"0x1.592d6dcc61423p-2"],'
        b'[["local","b"],"0x1.9e0094dead2d7p-4"],'
        b'[["local","c"],"0x1.9e0094dead2d7p-3"]]'
    )
    tied = normalize_transition_rows({_A: ((_C, 1.0), (_B, 1.0))})
    assert tuple(tied) == (_A, _B, _C)
    assert tied[_A] == ((_B, 0.5), (_C, 0.5))
    cap = (("canonical", node_id) for node_id in range(1, 201))
    assert len(normalize_transition_rows({}, nodes=cap)) == 200


@pytest.mark.parametrize(
    ("restart", "rows", "restart_probability", "iterations", "message"),
    [
        ({_A: float("nan")}, {}, 0.2, 1, "restart"),
        ({_A: -1.0}, {}, 0.2, 1, "restart"),
        ({_A: 0.0}, {}, 0.2, 1, "positive"),
        ({_A: 1.0}, {_A: ((_B, -1.0),)}, 0.2, 1, "transition"),
        ({_A: 1.0}, {_A: ((_B, float("inf")),)}, 0.2, 1, "transition"),
        ({_A: 1.0}, {}, 0.0, 1, "restart_probability"),
        ({_A: 1.0}, {}, 1.0, 1, "restart_probability"),
        ({_A: 1.0}, {}, float("nan"), 1, "restart_probability"),
        ({_A: 1.0}, {}, 0.2, -1, "iterations"),
        ({_A: 1.0}, {}, 0.2, True, "iterations"),
    ],
)
def test_personalized_pagerank_rejects_malformed_or_nonfinite_math(
    restart: dict[tuple[str, str], float],
    rows: dict[tuple[str, str], tuple[tuple[tuple[str, str], float], ...]],
    restart_probability: float,
    iterations: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        personalized_pagerank(
            restart,
            rows,
            restart_probability=restart_probability,
            iterations=iterations,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("iterations", [0, 9])
def test_personalized_pagerank_enforces_the_v1_iteration_envelope(
    iterations: int,
) -> None:
    with pytest.raises(ValueError, match="iterations"):
        personalized_pagerank(
            {_A: 1.0},
            {},
            restart_probability=0.2,
            iterations=iterations,
        )


def test_transition_node_generator_stops_at_cap_plus_one() -> None:
    consumed: list[int] = []
    def node_stream():
        for node_id in range(1, 203):
            consumed.append(node_id)
            yield ("canonical", node_id)
    with pytest.raises(ValueError, match="node cap"):
        normalize_transition_rows({}, nodes=node_stream())
    assert consumed == list(range(1, 202))


def test_duplicate_transition_generator_stops_at_edge_cap_plus_one() -> None:
    consumed: list[int] = []
    def edge_stream():
        for edge_number in range(1, 1003):
            consumed.append(edge_number)
            yield (_B, 1.0)
    with pytest.raises(ValueError, match="edge cap"):
        normalize_transition_rows({_A: edge_stream()})
    assert consumed == list(range(1, 1002))


def test_malformed_mapping_cannot_stream_unbounded_duplicate_source_rows() -> None:
    consumed: list[int] = []
    class RepeatingRows(Mapping):
        def __getitem__(self, key):
            return ()
        def __iter__(self) -> Iterator[tuple[str, str]]:
            yield _A
        def __len__(self) -> int:
            return 1
        def items(self):
            for row_number in range(1, 203):
                consumed.append(row_number)
                yield (_A, ())
    with pytest.raises(ValueError, match="source row cap"):
        normalize_transition_rows(RepeatingRows())
    assert consumed == list(range(1, 202))


def test_transition_source_rows_require_exact_pair_tuples() -> None:
    class ListRows(Mapping):
        def __getitem__(self, key):
            return ()
        def __iter__(self) -> Iterator[tuple[str, str]]:
            yield _A
        def __len__(self) -> int:
            return 1
        def items(self):
            yield [_A, ()]

    with pytest.raises(ValueError, match="source row.*exact"):
        normalize_transition_rows(ListRows())


def test_transition_target_discovery_stops_at_node_cap_plus_one() -> None:
    consumed: list[int] = []

    def edge_stream():
        for node_id in range(1, 202):
            consumed.append(node_id)
            yield (("canonical", node_id), 1.0)

    with pytest.raises(ValueError, match="node cap"):
        normalize_transition_rows({_A: edge_stream()})

    # The source is node one, so the 200th distinct target is cap-plus-one.
    assert consumed == list(range(1, 201))


def test_runtime_accepts_the_exact_v1_node_and_raw_edge_ceilings() -> None:
    nodes = tuple(("canonical", node_id) for node_id in range(1, 201))
    assert len(normalize_transition_rows({}, nodes=nodes)) == 200

    edges = ((_B, 1.0) for _ in range(1_000))
    assert normalize_transition_rows({_A: edges})[_A] == ((_B, 1.0),)


def test_restart_mapping_is_rejected_before_materializing_node_cap_plus_one() -> None:
    restart = {("canonical", node_id): 1.0 for node_id in range(1, 202)}

    with pytest.raises(ValueError, match="node cap"):
        personalized_pagerank(
            restart,
            {},
            restart_probability=0.2,
            iterations=1,
        )


def test_ppr_numeric_inputs_require_exact_builtin_types() -> None:
    class FloatSubclass(float):
        pass

    with pytest.raises(ValueError, match="restart weight"):
        personalized_pagerank(
            {_A: FloatSubclass(1.0)},
            {},
            restart_probability=0.2,
            iterations=1,
        )
    with pytest.raises(ValueError, match="transition weight"):
        normalize_transition_rows({_A: ((_B, FloatSubclass(1.0)),)})
    with pytest.raises(ValueError, match="restart_probability"):
        personalized_pagerank(
            {_A: 1.0},
            {},
            restart_probability=FloatSubclass(0.2),
            iterations=1,
        )


@pytest.mark.parametrize(
    "rows",
    [
        {("canonical", "7"): ((_A, 1.0),)},
        {("local", 7): ((_A, 1.0),)},
        {("other", "a"): ((_A, 1.0),)},
        {_A: ((["local", "b"], 1.0),)},
    ],
)
def test_transition_rows_require_private_stable_identity_keys(rows: object) -> None:
    with pytest.raises(ValueError, match="node key"):
        normalize_transition_rows(rows)  # type: ignore[arg-type]


def test_edge_evidence_flow_is_the_damped_normalized_source_flow() -> None:
    assert edge_evidence_flow(
        restart_probability=0.2,
        source_score=0.5,
        normalized_share=0.25,
    ) == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("source_score", "normalized_share"),
    [
        (-0.1, 0.5),
        (float("nan"), 0.5),
        (0.5, -0.1),
        (0.5, 1.1),
        (0.5, float("inf")),
    ],
)
def test_edge_evidence_flow_rejects_invalid_scores_and_shares(
    source_score: float, normalized_share: float
) -> None:
    with pytest.raises(ValueError):
        edge_evidence_flow(
            restart_probability=0.2,
            source_score=source_score,
            normalized_share=normalized_share,
        )


def test_ppr_v1_algorithm_signature_pins_literal_canonical_json() -> None:
    config = PPRAlgorithmConfig(canonical_resolver_version="canonical-resolution-v1")
    expected = (
        b'{"algorithm":"ppr_v1","canonical_resolver_version":'
        b'"canonical-resolution-v1","evidence_version":"ppr_evidence_v1",'
        b'"max_candidates":20,"max_edges":1000,"max_evidence_per_edge":3,'
        b'"max_evidence_rows":3000,"max_fanout":10,"max_hops":2,'
        b'"max_mentions_per_entity":2,"max_nodes":200,"max_per_document":3,'
        b'"max_scope_collections":128,"max_scope_documents":10000,'
        b'"max_seeds":64,"mention_factor":"0.25","ppr_iterations":8,'
        b'"ppr_restart":"0.2","reverse_factor":"0.35","rrf_k":60,'
        b'"seed_version":"rrf_seed_v1","support_cap":32,"timeout_ms":150,'
        b'"transition_version":"ppr_transition_v1","utility_floor":"0.5"}'
    )

    assert canonical_algorithm_json(config) == expected
    assert graph_algorithm_signature(config) == (
        "73ec625d84514cd48ff2f5d7b6d1693d7317dfb0be50431ac086f2ef5f937dd2"
    )


def test_ppr_v1_signature_changes_when_an_effective_setting_changes() -> None:
    baseline = PPRAlgorithmConfig(canonical_resolver_version="canonical-resolution-v1")
    changed = PPRAlgorithmConfig(
        canonical_resolver_version="canonical-resolution-v1",
        max_candidates=19,
    )

    assert graph_algorithm_signature(baseline) != graph_algorithm_signature(changed)


@pytest.mark.parametrize(
    ("constant_name", "replacement", "payload_key", "canonical_value"),
    [
        ("MENTION_FACTOR", 0.3, "mention_factor", "0.3"),
        ("REVERSE_DIRECTION_FACTOR", 0.4, "reverse_factor", "0.4"),
        ("UTILITY_FLOOR", 0.6, "utility_floor", "0.6"),
        ("SUPPORT_CAP", 31, "support_cap", 31),
    ],
)
def test_signature_derives_every_frozen_computation_constant(
    monkeypatch: pytest.MonkeyPatch,
    constant_name: str,
    replacement: int | float,
    payload_key: str,
    canonical_value: int | str,
) -> None:
    config = PPRAlgorithmConfig(canonical_resolver_version="canonical-resolution-v1")
    baseline = graph_algorithm_signature(config)

    monkeypatch.setattr(ppr_module, constant_name, replacement)
    payload = json.loads(canonical_algorithm_json(config))

    assert payload[payload_key] == canonical_value
    assert graph_algorithm_signature(config) != baseline


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"canonical_resolver_version": ""}, "canonical_resolver_version"),
        ({"rrf_k": 1001}, "rrf_k"),
        ({"max_seeds": 65}, "max_seeds"),
        ({"max_hops": 3}, "max_hops"),
        ({"max_edges": 1001}, "max_edges"),
        ({"ppr_restart": float("nan")}, "ppr_restart"),
        ({"ppr_restart": 1.0}, "ppr_restart"),
        ({"ppr_iterations": 9}, "ppr_iterations"),
        ({"timeout_ms": 151}, "timeout_ms"),
        (
            {"max_scope_documents": 4, "max_scope_collections": 5},
            "max_scope_collections",
        ),
        ({"max_nodes": 50, "max_edges": 501}, "max_edges"),
        ({"max_evidence_rows": 2, "max_evidence_per_edge": 3}, "evidence"),
        ({"max_candidates": 2, "max_per_document": 3}, "max_per_document"),
    ],
)
def test_ppr_v1_config_rejects_values_outside_the_hard_envelope(
    overrides: dict[str, object], message: str
) -> None:
    values: dict[str, object] = {
        "canonical_resolver_version": "canonical-resolution-v1"
    }
    values.update(overrides)

    with pytest.raises(ValueError, match=message):
        PPRAlgorithmConfig(**values)  # type: ignore[arg-type]
