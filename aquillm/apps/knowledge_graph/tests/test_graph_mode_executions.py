"""Four current modes retain distinct executions of one statistical question."""

from copy import deepcopy

import pytest

from apps.knowledge_graph.evals.graph_activation_attachment_v2 import (
    validate_graph_activation_attachment_v2,
)
from apps.knowledge_graph.evals.graph_quality_gates_v2 import (
    GraphGatePolicyV2,
    evaluate_graph_gates_v2,
)
from apps.knowledge_graph.tests.graph_review_fixtures import four_mode_report
from apps.knowledge_graph.tests.test_graph_gate_applicability import attachment_pair


def test_four_modes_same_questions_count_once_with_distinct_invocations():
    report = four_mode_report()
    result = evaluate_graph_gates_v2(report=report, policy=GraphGatePolicyV2())
    assert result["eligible"], result["blocking_reasons"]
    assert result["retrieval"]["query_pairs"] == 150
    assert set(result["retrieval"]["modes"]) == {
        "baseline",
        "selection",
        "preservation",
        "combined",
    }


def test_actual_four_subject_attachment_has_four_distinct_event_ids(monkeypatch):
    report, quality = attachment_pair(monkeypatch)
    identities = {
        q["observations"][0]["lane_observations"]["graph"]["frontier_invocations"][0][
            "invocation_id"
        ]
        for q in quality.values()
    }
    assert len(identities) == 4
    result = validate_graph_activation_attachment_v2(report, quality_reports=quality)
    assert result["passed"], result
    assert result["attachment"]["result"]["retrieval"]["query_pairs"] == 150


@pytest.mark.parametrize(
    "mutation",
    [
        "copied_event",
        "missing_mode",
        "missing_request",
        "static_request",
        "event_request",
        "mode_regression",
    ],
)
def test_other_modes_and_signatures_cannot_hide(mutation):
    report = four_mode_report()
    query = report["queries"][0]
    pair = query["executions"]["selection"]
    candidate = pair["candidate"]
    if mutation == "copied_event":
        candidate["frontier_invocations"] = deepcopy(
            query["executions"]["baseline"]["candidate"]["frontier_invocations"]
        )
    elif mutation == "missing_mode":
        del query["executions"]["selection"]
    elif mutation == "missing_request":
        candidate["frontier_invocations"][0]["request_algorithm_signature"] = None
    elif mutation == "static_request":
        candidate["request_algorithm_signature"] = candidate["binding"][
            "algorithm_signature"
        ]
        candidate["frontier_invocations"][0]["request_algorithm_signature"] = candidate[
            "request_algorithm_signature"
        ]
    elif mutation == "event_request":
        candidate["frontier_invocations"][0]["request_algorithm_signature"] = "f" * 64
    else:
        for query in report["queries"]:
            if not query["labels"]["no_answer"]:
                observation = query["executions"]["selection"]["candidate"]
                observation.update(ranked=["bad"], repeated_ranked=["bad"])
    result = evaluate_graph_gates_v2(report=report, policy=GraphGatePolicyV2())
    assert not result["eligible"], result
    if mutation == "mode_regression":
        assert not result["retrieval"]["modes"]["baseline"]["blocking_reasons"]
        assert result["retrieval"]["modes"]["selection"]["blocking_reasons"]
