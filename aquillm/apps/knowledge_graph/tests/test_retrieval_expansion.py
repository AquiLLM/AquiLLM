"""Pure tests for deterministic ranking of authorized graph snapshots."""

from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError, fields, replace
from uuid import UUID

import pytest

from apps.knowledge_graph.retrieval import expansion
from apps.knowledge_graph.retrieval import ppr as ppr_module
from apps.knowledge_graph.retrieval.expansion import (
    AuthorizedChunkEvidence,
    AuthorizedGraphSnapshot,
    AuthorizedIdentityMention,
    AuthorizedRelationGroup,
    AuthorizedSeedIdentity,
    rank_authorized_graph_snapshot,
)
from apps.knowledge_graph.retrieval.ppr import (
    PPRAlgorithmConfig,
    RetrievalDirection,
)
from apps.knowledge_graph.retrieval.types import (
    GraphExpansionRequest,
    GraphExpansionSeed,
)

_DOC_A = UUID("11111111-1111-4111-8111-111111111111")
_DOC_B = UUID("22222222-2222-4222-8222-222222222222")
_SCOPE_SIGNATURE = "f" * 64
_A = ("canonical", 1)
_B = ("canonical", 2)
_C = ("canonical", 3)
_CONFIG = PPRAlgorithmConfig(
    canonical_resolver_version="expansion-test-v1",
    max_scope_documents=10,
    max_scope_collections=2,
    max_nodes=20,
    max_edges=100,
    max_evidence_rows=100,
    ppr_iterations=1,
    max_candidates=10,
)


class _TupleSubclass(tuple):
    pass


def _seed(
    chunk_id: int = 1,
    rank: int = 1,
    weight: float = 1.0,
) -> GraphExpansionSeed:
    return GraphExpansionSeed(chunk_id, rank, weight)


def _request(
    *,
    seeds: tuple[GraphExpansionSeed, ...] = (_seed(),),
    document_ids: tuple[UUID, ...] = (_DOC_A, _DOC_B),
    collection_ids: tuple[int, ...] = (1, 2),
) -> GraphExpansionRequest:
    return GraphExpansionRequest(seeds, document_ids, collection_ids)


def _evidence(
    chunk_id: int,
    *,
    document_id: UUID = _DOC_A,
    chunk_number: int | None = None,
    confidence: float = 1.0,
    provenance_key: str | None = None,
) -> AuthorizedChunkEvidence:
    return AuthorizedChunkEvidence(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_number=chunk_id if chunk_number is None else chunk_number,
        confidence=confidence,
        provenance_key=provenance_key or f"evidence-{chunk_id}",
    )


def _mapping(
    seed_chunk_id: int = 1,
    identity_key: tuple[str, object] = _A,
) -> AuthorizedSeedIdentity:
    return AuthorizedSeedIdentity(
        seed_chunk_id=seed_chunk_id,
        identity_key=identity_key,
    )


def _group(
    source: tuple[str, object],
    target: tuple[str, object],
    *,
    relation_type: str = "supports",
    raw_weight: float = 1.0,
    admission_hop: int = 1,
    evidence: tuple[AuthorizedChunkEvidence, ...] | None = None,
) -> AuthorizedRelationGroup:
    return AuthorizedRelationGroup(
        source_key=source,
        relation_type=relation_type,
        target_key=target,
        direction=RetrievalDirection.FORWARD,
        raw_weight=raw_weight,
        admission_hop=admission_hop,
        evidence=evidence or (_evidence(target[1]),),
    )


def _mention(
    identity_key: tuple[str, object],
    chunk_id: int,
    **evidence_overrides: object,
) -> AuthorizedIdentityMention:
    return AuthorizedIdentityMention(
        identity_key=identity_key,
        evidence=_evidence(chunk_id, **evidence_overrides),
    )


def _snapshot(
    *,
    config: PPRAlgorithmConfig = _CONFIG,
    load_max_hops: int = 2,
    document_ids: tuple[UUID, ...] = (_DOC_A, _DOC_B),
    collection_ids: tuple[int, ...] = (1, 2),
    identity_keys: tuple[tuple[str, object], ...] = (_A,),
    seed_identities: tuple[AuthorizedSeedIdentity, ...] = (_mapping(),),
    relation_groups: tuple[AuthorizedRelationGroup, ...] = (),
    mentions: tuple[AuthorizedIdentityMention, ...] = (),
    raw_audit_rows: tuple[tuple[int, str, tuple[object, ...]], ...] = (),
) -> AuthorizedGraphSnapshot:
    return AuthorizedGraphSnapshot(
        config=config,
        load_max_hops=load_max_hops,
        allowed_doc_ids=document_ids,
        allowed_collection_ids=collection_ids,
        scope_version_signature=_SCOPE_SIGNATURE,
        identity_keys=identity_keys,
        seed_identities=seed_identities,
        relation_groups=relation_groups,
        mentions=mentions,
        raw_audit_rows=raw_audit_rows,
    )


def _rank_with_trace(
    snapshot: AuthorizedGraphSnapshot,
    request: GraphExpansionRequest | None = None,
    *,
    effective_max_hops: int = 2,
):
    traces: list[bytes] = []
    result = rank_authorized_graph_snapshot(
        snapshot,
        request or _request(),
        effective_max_hops=effective_max_hops,
        _eval_trace=expansion._EvaluationTraceCapability(traces.append),
    )
    return result, traces


def _trace_candidates(trace: bytes) -> dict[int, float]:
    payload = json.loads(trace)
    return {
        row[0]: float.fromhex(row[1])
        for row in payload["candidate_contributions"]
    }


def test_snapshot_records_are_exact_frozen_provider_neutral_values() -> None:
    evidence = _evidence(2)
    group = _group(_A, _B, evidence=(evidence,))
    snapshot = _snapshot(
        identity_keys=(_A, _B),
        relation_groups=(group,),
        mentions=(_mention(_B, 3),),
    )

    assert snapshot.relation_groups == (group,)
    assert evidence.confidence == 1.0
    assert "query" not in {field.name for field in fields(snapshot)}
    assert not hasattr(snapshot, "__dict__")
    with pytest.raises(FrozenInstanceError):
        snapshot.load_max_hops = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("allowed_doc_ids", [_DOC_A, _DOC_B]),
        ("allowed_doc_ids", iter((_DOC_A, _DOC_B))),
        ("allowed_doc_ids", _TupleSubclass((_DOC_A, _DOC_B))),
        ("allowed_collection_ids", [1, 2]),
        ("allowed_collection_ids", iter((1, 2))),
        ("allowed_collection_ids", _TupleSubclass((1, 2))),
        ("identity_keys", [_A]),
        ("identity_keys", iter((_A,))),
        ("identity_keys", _TupleSubclass((_A,))),
        ("seed_identities", [_mapping()]),
        ("seed_identities", iter((_mapping(),))),
        ("seed_identities", _TupleSubclass((_mapping(),))),
        ("relation_groups", []),
        ("relation_groups", iter(())),
        ("relation_groups", _TupleSubclass(())),
        ("mentions", []),
        ("mentions", iter(())),
        ("mentions", _TupleSubclass(())),
    ],
)
def test_snapshot_rejects_nonexact_tuple_boundaries(
    field_name: str, value: object
) -> None:
    helper_field = {
        "allowed_doc_ids": "document_ids",
        "allowed_collection_ids": "collection_ids",
    }.get(field_name, field_name)
    with pytest.raises(ValueError, match=field_name):
        _snapshot(**{helper_field: value})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "evidence",
    [[_evidence(2)], iter((_evidence(2),)), _TupleSubclass((_evidence(2),))],
)
def test_relation_group_rejects_nonexact_evidence_tuples(
    evidence: object,
) -> None:
    with pytest.raises(ValueError, match="evidence"):
        AuthorizedRelationGroup(
            source_key=_A,
            relation_type="supports",
            target_key=_B,
            direction=RetrievalDirection.FORWARD,
            raw_weight=1.0,
            admission_hop=1,
            evidence=evidence,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: _evidence(2, confidence=math.nan), "confidence"),
        (lambda: _evidence(2, confidence=math.inf), "confidence"),
        (lambda: _evidence(True), "chunk_id"),
        (lambda: _group(_A, _B, raw_weight=math.nan), "raw_weight"),
        (lambda: _group(_A, _B, raw_weight=0.0), "raw_weight"),
        (lambda: _group(_A, _B, admission_hop=3), "admission_hop"),
        (lambda: _mapping(True), "seed_chunk_id"),
        (
            lambda: _mapping(identity_key=("local", "not-a-cluster-hash")),
            "identity_key",
        ),
    ],
)
def test_snapshot_records_reject_invalid_exact_or_nonfinite_values(
    factory, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        factory()


def test_identity_keys_reject_container_and_literal_subclasses() -> None:
    class StringSubclass(str):
        pass

    with pytest.raises(ValueError, match="identity_key"):
        _mapping(identity_key=_TupleSubclass(_A))
    with pytest.raises(ValueError, match="identity_key"):
        _mapping(identity_key=(StringSubclass("canonical"), 1))


def test_snapshot_rejects_subrecord_subclasses() -> None:
    class SeedIdentitySubclass(AuthorizedSeedIdentity):
        pass

    with pytest.raises(ValueError, match="seed_identities"):
        _snapshot(seed_identities=(SeedIdentitySubclass(1, _A),))


def test_snapshot_enforces_effective_storage_caps_and_memberships() -> None:
    config = replace(_CONFIG, max_nodes=1, max_edges=1, max_fanout=1)
    with pytest.raises(ValueError, match="identity_keys"):
        _snapshot(config=config, identity_keys=(_A, _B))
    with pytest.raises(ValueError, match="seed_identities"):
        _snapshot(
            config=config,
            seed_identities=(_mapping(), _mapping()),
        )
    edge_config = replace(
        _CONFIG,
        max_nodes=2,
        max_edges=1,
        max_fanout=1,
        max_evidence_rows=1,
        max_evidence_per_edge=1,
    )
    shared_evidence = (_evidence(2),)
    two_directions = _snapshot(
        config=edge_config,
        identity_keys=(_A, _B),
        relation_groups=(
            _group(
                _A,
                _B,
                relation_type="forward",
                evidence=shared_evidence,
            ),
            _group(
                _B,
                _A,
                relation_type="reverse",
                evidence=shared_evidence,
            ),
        ),
    )
    assert len(two_directions.relation_groups) == 2
    with pytest.raises(ValueError, match="relation evidence"):
        _snapshot(
            config=edge_config,
            identity_keys=(_A, _B),
            relation_groups=(
                _group(_A, _B, relation_type="forward"),
                _group(_B, _A, relation_type="reverse"),
            ),
        )
    with pytest.raises(ValueError, match="relation_groups"):
        _snapshot(
            config=edge_config,
            identity_keys=(_A, _B),
            relation_groups=(
                _group(_A, _B, relation_type="one"),
                _group(_B, _A, relation_type="two"),
                _group(_A, _B, relation_type="three"),
            ),
        )
    with pytest.raises(ValueError, match="identity_keys"):
        _snapshot(seed_identities=(_mapping(identity_key=_B),))


def test_snapshot_bounds_relation_evidence_and_fallback_mentions() -> None:
    evidence = tuple(_evidence(index) for index in range(2, 103))
    with pytest.raises(ValueError, match="max_evidence_rows"):
        _snapshot(
            identity_keys=(_A, _B),
            relation_groups=(_group(_A, _B, evidence=evidence),),
        )

    config = replace(_CONFIG, max_nodes=1, max_edges=10)
    with pytest.raises(ValueError, match="mentions"):
        _snapshot(
            config=config,
            mentions=(_mention(_A, 2), _mention(_A, 3), _mention(_A, 4)),
        )

    per_identity_config = replace(_CONFIG, max_mentions_per_entity=2)
    with pytest.raises(ValueError, match="mentions per identity"):
        _snapshot(
            config=per_identity_config,
            identity_keys=(_A, _B),
            mentions=(
                _mention(_A, 2),
                _mention(_A, 3),
                _mention(_A, 4),
            ),
        )


def test_seed_identity_dedupe_happens_before_equal_split() -> None:
    snapshot = _snapshot(
        identity_keys=(_A, _B),
        seed_identities=(_mapping(), _mapping(), _mapping(identity_key=_B)),
        mentions=(_mention(_A, 2), _mention(_B, 3, document_id=_DOC_B)),
    )

    result, traces = _rank_with_trace(snapshot)
    trace = json.loads(traces[0])

    assert result.chunk_ids == (2, 3)
    assert trace["restart_vector"] == [
        [["canonical", 1], "0x1.0000000000000p-1"],
        [["canonical", 2], "0x1.0000000000000p-1"],
    ]


def test_relation_evidence_suppresses_fallback_for_its_destination_identity() -> None:
    group = _group(_A, _B, evidence=(_evidence(2),))
    snapshot = _snapshot(
        identity_keys=(_A, _B),
        relation_groups=(group,),
        mentions=(_mention(_B, 2), _mention(_B, 3)),
    )

    result, traces = _rank_with_trace(snapshot)
    contributions = _trace_candidates(traces[0])

    assert result.chunk_ids == (2,)
    assert contributions[2] == pytest.approx(0.16)
    assert 3 not in contributions


def test_group_shares_remain_distinct_for_edge_evidence_projection() -> None:
    groups = (
        _group(
            _A,
            _B,
            relation_type="high",
            raw_weight=3.0,
            evidence=(_evidence(2),),
        ),
        _group(
            _A,
            _B,
            relation_type="low",
            raw_weight=1.0,
            evidence=(_evidence(3),),
        ),
    )
    snapshot = _snapshot(identity_keys=(_A, _B), relation_groups=groups)

    result, traces = _rank_with_trace(snapshot)
    contributions = _trace_candidates(traces[0])

    assert result.chunk_ids == (2, 3)
    assert contributions == pytest.approx({2: 0.12, 3: 0.04})


def test_contributions_use_max_within_identity_then_fsum_across_identities() -> None:
    snapshot = _snapshot(
        identity_keys=(_A, _B),
        seed_identities=(_mapping(), _mapping(identity_key=_B)),
        mentions=(
            _mention(_A, 2, provenance_key="a"),
            _mention(_A, 2, confidence=0.5, provenance_key="b"),
            _mention(_B, 2, provenance_key="c"),
        ),
    )

    result, traces = _rank_with_trace(snapshot)

    assert result.chunk_ids == (2,)
    assert _trace_candidates(traces[0])[2] == pytest.approx(0.25)


def test_fanout_uses_weight_then_stable_keys() -> None:
    fanout_config = replace(_CONFIG, max_fanout=1, max_edges=20)
    low = _group(_A, _B, raw_weight=1.0, evidence=(_evidence(2),))
    high = _group(_A, _C, raw_weight=2.0, evidence=(_evidence(3),))
    fanout_snapshot = _snapshot(
        config=fanout_config,
        identity_keys=(_A, _B, _C),
        relation_groups=(low, high),
    )

    fanout_result, _ = _rank_with_trace(fanout_snapshot)

    assert fanout_result.chunk_ids == (3,)


def test_global_edge_cap_replays_after_directional_group_expansion() -> None:
    config = replace(_CONFIG, max_fanout=2, max_edges=1)
    low = _group(_A, _B, raw_weight=1.0, evidence=(_evidence(2),))
    high = _group(_A, _C, raw_weight=2.0, evidence=(_evidence(3),))
    snapshot = _snapshot(
        config=config,
        identity_keys=(_A, _B, _C),
        relation_groups=(low, high),
    )

    result, trace = _rank_with_trace(snapshot)

    assert result.chunk_ids == (3,)
    assert len(json.loads(trace[0])["retained_groups"]) == 1


def test_per_edge_cap_selects_highest_confidence() -> None:
    edge_config = replace(_CONFIG, max_evidence_per_edge=1)
    edge_snapshot = _snapshot(
        config=edge_config,
        identity_keys=(_A, _B),
        relation_groups=(
            _group(
                _A,
                _B,
                evidence=(
                    _evidence(2, confidence=0.5),
                    _evidence(3, confidence=0.9),
                ),
            ),
        ),
    )
    edge_result, _ = _rank_with_trace(edge_snapshot)

    assert edge_result.chunk_ids == (3,)


def test_effective_hop_one_replay_equals_direct_hop_one_byte_for_byte() -> None:
    first = _group(_A, _B, evidence=(_evidence(2),))
    second = _group(
        _B,
        _C,
        relation_type="extends",
        admission_hop=2,
        evidence=(_evidence(3, document_id=_DOC_B),),
    )
    reciprocal = _group(
        _B,
        _A,
        relation_type="reciprocal",
        admission_hop=2,
        evidence=(_evidence(4, document_id=_DOC_B),),
    )
    two_hop = _snapshot(
        identity_keys=(_A, _B, _C),
        relation_groups=(second, reciprocal, first),
        raw_audit_rows=(
            (1, "physical_relation", (1, "first")),
            (2, "physical_relation", (2, "second")),
        ),
    )
    direct_config = replace(_CONFIG, max_hops=1)
    direct = _snapshot(
        config=direct_config,
        load_max_hops=1,
        identity_keys=(_A, _B),
        relation_groups=(first,),
        raw_audit_rows=((1, "physical_relation", (1, "first")),),
    )

    derived_result, derived_trace = _rank_with_trace(
        two_hop, effective_max_hops=1
    )
    direct_result, direct_trace = _rank_with_trace(
        direct, effective_max_hops=1
    )

    assert derived_result == direct_result
    assert derived_trace == direct_trace
    assert derived_result.chunk_ids == (2,)

    two_hop_result, _ = _rank_with_trace(two_hop, effective_max_hops=2)
    assert two_hop_result.chunk_ids == (4, 3, 2)


def test_raw_link_generation_changes_signature_even_when_semantics_match() -> None:
    first = _snapshot(
        mentions=(_mention(_A, 2),),
        raw_audit_rows=((0, "canonical_link", (10, 20, 30)),),
    )
    replacement = _snapshot(
        mentions=(_mention(_A, 2),),
        raw_audit_rows=((0, "canonical_link", (11, 20, 30)),),
    )

    first_result, _ = _rank_with_trace(first)
    replacement_result, _ = _rank_with_trace(replacement)

    assert first_result.chunk_ids == replacement_result.chunk_ids == (2,)
    assert (
        first_result.diagnostics.graph_version_signature
        != replacement_result.diagnostics.graph_version_signature
    )


def test_raw_audit_uuid_and_float_are_canonicalized_before_signature_json() -> None:
    snapshot = _snapshot(
        mentions=(_mention(_A, 2),),
        raw_audit_rows=((0, "relation_evidence", (_DOC_A, 0.5)),),
    )

    result, _ = _rank_with_trace(snapshot)

    assert snapshot.raw_audit_rows == (
        (0, "relation_evidence", (str(_DOC_A), 0.5.hex())),
    )
    assert result.chunk_ids == (2,)
    assert result.diagnostics.status == "hit"


def test_snapshot_and_evidence_insertion_order_do_not_change_bytes() -> None:
    first = _group(
        _A,
        _B,
        relation_type="alpha",
        evidence=(_evidence(2), _evidence(4, confidence=0.5)),
    )
    first_reversed = replace(first, evidence=tuple(reversed(first.evidence)))
    second = _group(
        _A,
        _C,
        relation_type="beta",
        evidence=(_evidence(3, document_id=_DOC_B),),
    )
    forward = _snapshot(
        identity_keys=(_A, _B, _C),
        relation_groups=(first, second),
    )
    reverse = _snapshot(
        identity_keys=(_C, _B, _A),
        relation_groups=(second, first_reversed),
    )

    forward_result, forward_trace = _rank_with_trace(forward)
    reverse_result, reverse_trace = _rank_with_trace(reverse)

    assert forward_result == reverse_result
    assert forward_trace == reverse_trace


def test_seed_chunks_are_removed_before_per_edge_evidence_cap() -> None:
    config = replace(_CONFIG, max_evidence_per_edge=1)
    group = _group(
        _A,
        _B,
        evidence=(
            _evidence(1, confidence=1.0),
            _evidence(2, confidence=0.9),
        ),
    )
    snapshot = _snapshot(
        config=config,
        identity_keys=(_A, _B),
        relation_groups=(group,),
    )

    result, _ = _rank_with_trace(snapshot)

    assert result.chunk_ids == (2,)


def test_candidate_order_applies_seed_rank_and_per_document_caps() -> None:
    config = replace(_CONFIG, max_per_document=1)
    request = _request(
        seeds=(_seed(1, 1, 1.0), _seed(10, 2, 1.0)),
    )
    snapshot = _snapshot(
        config=config,
        identity_keys=(_A, _B),
        seed_identities=(_mapping(1, _B), _mapping(10, _A)),
        mentions=(
            _mention(_A, 2, chunk_number=1),
            _mention(_A, 3, chunk_number=2),
            _mention(_B, 4, document_id=_DOC_B, chunk_number=1),
            _mention(_B, 5, document_id=_DOC_B, chunk_number=2),
        ),
    )

    result, _ = _rank_with_trace(snapshot, request)

    assert result.chunk_ids == (4, 2)


def test_candidate_uses_one_lexicographically_best_contributing_label() -> None:
    config = replace(_CONFIG, ppr_iterations=7)
    seeded_rank_ten = ("canonical", 4)
    rank_one_frontier = ("canonical", 5)
    request = _request(
        seeds=(_seed(1, 1, 1.0), _seed(10, 10, 1.0)),
    )
    snapshot = _snapshot(
        config=config,
        identity_keys=(_A, _B, _C, seeded_rank_ten, rank_one_frontier),
        seed_identities=(
            _mapping(1, _B),
            _mapping(10, seeded_rank_ten),
        ),
        relation_groups=(
            _group(
                seeded_rank_ten,
                _A,
                relation_type="rank_ten_hop_one",
                evidence=(_evidence(10),),
            ),
            _group(
                _B,
                rank_one_frontier,
                relation_type="rank_one_hop_one",
                evidence=(
                    _evidence(1, provenance_key="rank-one-hop-one"),
                ),
            ),
            _group(
                rank_one_frontier,
                _C,
                relation_type="rank_one_hop_two",
                admission_hop=2,
                evidence=(
                    _evidence(1, provenance_key="rank-one-hop-two"),
                ),
            ),
        ),
        mentions=(
            _mention(_A, 30, provenance_key="mention-hop-one-rank-ten"),
            _mention(_C, 30, provenance_key="mention-hop-two-rank-one"),
        ),
    )

    result, traces = _rank_with_trace(snapshot, request)
    candidate = next(
        row
        for row in json.loads(traces[0])["candidate_contributions"]
        if row[0] == 30
    )

    assert result.chunk_ids == (30,)
    assert candidate[2:4] == [1, 10]


def test_seed_restart_weights_change_the_candidate_ranking() -> None:
    snapshot = _snapshot(
        identity_keys=(_A, _B),
        seed_identities=(_mapping(1, _A), _mapping(10, _B)),
        mentions=(_mention(_A, 2), _mention(_B, 3, document_id=_DOC_B)),
    )
    heavy_a = _request(
        seeds=(_seed(1, 1, 0.9), _seed(10, 2, 0.1)),
    )
    heavy_b = _request(
        seeds=(_seed(1, 1, 0.1), _seed(10, 2, 0.9)),
    )

    heavy_a_result, _ = _rank_with_trace(snapshot, heavy_a)
    heavy_b_result, _ = _rank_with_trace(snapshot, heavy_b)

    assert heavy_a_result.chunk_ids == (2, 3)
    assert heavy_b_result.chunk_ids == (3, 2)


def test_total_candidate_cap_is_applied_after_normative_sorting() -> None:
    config = replace(_CONFIG, max_candidates=1, max_per_document=1)
    snapshot = _snapshot(
        config=config,
        identity_keys=(_A, _B),
        seed_identities=(_mapping(), _mapping(identity_key=_B)),
        mentions=(_mention(_A, 2), _mention(_B, 3, document_id=_DOC_B)),
    )

    result, _ = _rank_with_trace(snapshot)

    assert result.chunk_ids == (2,)


def test_scope_or_missing_seed_mapping_fails_open_to_bounded_miss() -> None:
    mismatched = _snapshot(
        document_ids=(_DOC_A,),
        collection_ids=(1,),
    )
    missing_mapping = _snapshot(seed_identities=())

    mismatch_result, _ = _rank_with_trace(mismatched)
    missing_result, _ = _rank_with_trace(missing_mapping)

    assert mismatch_result.chunk_ids == ()
    assert mismatch_result.diagnostics.status == "miss"
    assert missing_result.chunk_ids == ()
    assert missing_result.diagnostics.status == "miss"


def test_ungraphed_seed_is_ignored_when_another_seed_has_a_mapping() -> None:
    request = _request(
        seeds=(_seed(1, 1, 0.7), _seed(10, 2, 0.3)),
    )
    snapshot = _snapshot(
        seed_identities=(_mapping(1, _A),),
        mentions=(_mention(_A, 2),),
    )

    result, _ = _rank_with_trace(snapshot, request)

    assert result.chunk_ids == (2,)
    assert result.diagnostics.status == "hit"


def test_request_above_effective_seed_cap_fails_open_before_restart() -> None:
    config = replace(_CONFIG, max_seeds=1)
    request = _request(
        seeds=(_seed(1, 1, 1.0), _seed(10, 2, 1.0)),
    )
    snapshot = _snapshot(
        config=config,
        identity_keys=(_A, _B),
        seed_identities=(_mapping(1, _A), _mapping(10, _B)),
        mentions=(
            _mention(_A, 2),
            _mention(_B, 3, document_id=_DOC_B),
        ),
    )

    result, traces = _rank_with_trace(snapshot, request)

    assert result.chunk_ids == ()
    assert result.diagnostics.status == "miss"
    assert result.diagnostics.seed_count == 2
    assert traces == []


def test_timeout_and_malformed_snapshot_return_privacy_safe_empty_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def timeout(*args, **kwargs):
        raise TimeoutError

    monkeypatch.setattr(expansion, "personalized_pagerank", timeout)
    timeout_result = rank_authorized_graph_snapshot(
        _snapshot(), _request(), effective_max_hops=2
    )
    malformed_result = rank_authorized_graph_snapshot(
        object(),  # type: ignore[arg-type]
        _request(),
        effective_max_hops=2,
    )

    assert timeout_result.chunk_ids == ()
    assert timeout_result.diagnostics.status == "timeout"
    assert malformed_result.chunk_ids == ()
    assert malformed_result.diagnostics.status == "error"
    assert set(timeout_result.diagnostics.__dataclass_fields__) == {
        "status",
        "seed_count",
        "candidate_count",
        "elapsed_ms",
        "algorithm_signature",
        "graph_version_signature",
    }


def test_one_private_deadline_is_checked_between_ranking_phases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_times: list[float] = []
    times = iter((0.0, 1.1))

    def clock() -> float:
        value = next(times)
        observed_times.append(value)
        return value

    def forbidden_restart(*args, **kwargs):
        raise AssertionError("restart construction ran after the deadline")

    monkeypatch.setattr(expansion, "_build_restart_vector", forbidden_restart)
    deadline = ppr_module._MonotonicDeadline(expires_at=1.0, clock=clock)

    result = rank_authorized_graph_snapshot(
        _snapshot(),
        _request(),
        effective_max_hops=2,
        _deadline=deadline,
    )

    assert result.chunk_ids == ()
    assert result.diagnostics.status == "timeout"
    assert observed_times == [0.0, 1.1]


def test_raw_eval_trace_callable_is_rejected_without_receiving_private_data() -> None:
    traces: list[bytes] = []

    result = rank_authorized_graph_snapshot(
        _snapshot(),
        _request(),
        effective_max_hops=2,
        _eval_trace=traces.append,
    )

    assert result.chunk_ids == ()
    assert result.diagnostics.status == "error"
    assert traces == []


def test_retrieval_package_exports_only_composition_and_contract_types() -> None:
    from apps.knowledge_graph import retrieval

    assert retrieval.GraphExpansionSeed is GraphExpansionSeed
    assert retrieval.expand_chunk_candidates is expansion.expand_chunk_candidates
    assert not hasattr(retrieval, "AuthorizedGraphSnapshot")
    assert not hasattr(retrieval, "rank_authorized_graph_snapshot")
    assert not hasattr(retrieval, "_EvaluationTraceCapability")
    assert expansion.rank_authorized_graph_snapshot is rank_authorized_graph_snapshot
    assert "rank_authorized_graph_snapshot" not in expansion.__all__
