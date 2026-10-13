"""V1 admission caps may leave gaps in already authorized producer ranks."""

from dataclasses import replace
from types import SimpleNamespace

from django.test import override_settings

from apps.documents.services import chunk_rerank
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
    successful_branch,
)
from apps.knowledge_graph.retrieval.branch_contracts import GraphBranchCandidateV1
from apps.knowledge_graph.retrieval.materialization import MaterializedGraphChunkV1
from apps.knowledge_graph.retrieval.topology.contracts import HybridBranchKind


@override_settings(RAG_RRF_ENABLED=True, RAG_PROTECTED_EXACT_ENABLED=False)
def test_graph_cap_retains_rank_twenty_for_baseline_duplicate(monkeypatch):
    from math import fsum

    from apps.documents.services import chunk_search_rank_policy as policy

    records = []
    original = policy.fuse_ranked_channels_v2

    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        records.append(result)
        return result

    monkeypatch.setattr(policy, "fuse_ranked_channels_v2", record)
    x, y = chunk(1), chunk(2)
    fillers = tuple(chunk(i) for i in range(3, 19))
    direct_rows = tuple(chunk(i) for i in range(101, 120)) + (x,)
    extended_rows = tuple(chunk(i) for i in range(201, 221))
    materialized = {}

    def branch(kind, rows):
        candidates = []
        for rank, row in enumerate(rows, 1):
            key = f"{row.pk:064x}"
            materialized[key] = MaterializedGraphChunkV1(
                key, row.pk, row.doc_id, row.chunk_number, row
            )
            candidates.append(GraphBranchCandidateV1(key, rank, 1.0 / rank))
        return successful_branch(kind, candidates)

    direct = branch(HybridBranchKind.DIRECT, direct_rows)
    extended = branch(HybridBranchKind.EXTENDED, extended_rows)

    class Runtime:
        prepare_shared = prepare_extended = lambda *a, **k: object()

        def run_direct(self, **kwargs):
            return direct

        def run_extended(self, **kwargs):
            return extended

    snapshot = replace(
        selected_snapshot(baseline=(x, y, *fillers)),
        vector_results=(x, y),
        vector_chunk_ids=(x.pk, y.pk),
        trigram_results=(*fillers, y),
        trigram_chunk_ids=tuple(r.pk for r in (*fillers, y)),
    )
    pool, details = hybrid_graph_candidate_pool(
        snapshot,
        "query",
        authorization(Policy()),
        HybridGraphRetrievalDependencies(
            runtime=Runtime(),
            settings=hybrid_settings(),
            materialize=lambda **_: tuple(materialized.values()),
        ),
    )
    assert details["graph_status"] == "hit"
    assert details["graph_candidate_count"] == 20
    assert set(r.pk for r in pool) == {
        *(r.pk for r in snapshot.baseline_candidates),
        *(r.pk for r in direct_rows[:10]),
        *(r.pk for r in extended_rows[:10]),
    }
    assert tuple(r.pk for r in pool[:2]) == (y.pk, x.pk)
    x_record = next(c for c in records[0].candidates if c.chunk_id == x.pk)
    assert [(r.channel, r.rank) for r in x_record.channel_ranks] == [
        ("dense", 1),
        ("graph_direct", 20),
    ]
    assert x_record.rrf_score == fsum((1 / 61, 1 / 80))
    monkeypatch.setattr(chunk_rerank, "rerank_provider", lambda: "cohere")
    monkeypatch.setattr(
        chunk_rerank.apps,
        "get_app_config",
        lambda _: SimpleNamespace(cohere_client=None),
    )
    result = chunk_rerank.rerank_chunks_scored(object, "query", pool, 1)
    assert result.ranked_ids == (y.pk,)
    assert result.score_set.status == "unavailable"
