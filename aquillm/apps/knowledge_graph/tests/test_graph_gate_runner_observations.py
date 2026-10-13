"""Supplemental metrics survive actual Task21 scoring/payload boundaries."""

from copy import deepcopy

import pytest

from apps.knowledge_graph.evals.task21_hybrid_eval import (
    TASK21_HYBRID_ARMS,
    build_task21_hybrid_report,
)
from apps.knowledge_graph.evals.task21_hybrid_live_payloads import _split_observations
from apps.knowledge_graph.tests.test_retrieval_eval_metrics import task23_inputs
from apps.knowledge_graph.tests.test_task21_hybrid_live_payloads import _observation


def test_task21_no_relevance_does_not_certify_arbitrary_chunks_as_abstention():
    cases, rows, freshness, parity = task23_inputs()
    cases[0].update(
        expected_retrieval_chunk_ids=[],
        expected_no_answer=True,
        expected_coverage="partial",
    )
    for arm in rows.values():
        arm[0].update(
            coverage_status="partial", coverage_exhaustive=False, abstained=False
        )
    report = build_task21_hybrid_report(
        cases=cases, observations=rows, freshness=freshness, backend_parity=parity
    )
    assert report["graph_quality_v2"]["eligible"] is False
    result = report["arms"]["combined"]["cases"]["task23-case"]
    assert result["no_answer_correct"] is False
    assert result["precision_at_10"] == 0
    assert result["coverage_label_accuracy"] == 1


def test_live_payload_split_preserves_observed_coverage_without_labels():
    combined = {a: [_observation("case", a)] for a in TASK21_HYBRID_ARMS}
    for rows in combined.values():
        rows[0].update(
            coverage_status="partial", coverage_exhaustive=False, abstained=False
        )
    observations, _, _ = _split_observations(combined)
    assert observations["direct"][0]["coverage_status"] == "partial"
    mutated = deepcopy(combined)
    mutated["direct"][0]["expected_coverage"] = "partial"
    with pytest.raises(ValueError, match="not exact"):
        _split_observations(mutated)
