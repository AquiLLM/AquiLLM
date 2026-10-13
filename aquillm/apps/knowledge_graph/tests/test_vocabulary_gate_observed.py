"""Persisted vocabulary captures enter graph review; synthetic reports test binding only."""

from copy import deepcopy

import pytest

from apps.documents.services.source_loading import SourceRuntime, source_runtime_scope
from apps.knowledge_graph.evals.graph_effective_controls import (
    graph_effective_controls, validate_controls,
)
from apps.knowledge_graph.evals.graph_gate_observations import digest
from apps.knowledge_graph.evals.graph_quality_gates_v2 import (
    GraphGatePolicyV2, evaluate_graph_gates_v2,
)
from apps.knowledge_graph.resolution import vocabulary_capture as vocabulary
from apps.knowledge_graph.tests.graph_review_fixtures import rebind, rehash
from apps.knowledge_graph.tests.test_graph_gate_applicability import (
    attachment_pair, reports, validate,
)
from apps.knowledge_graph.tests.test_vocabulary_graph_capture import setup
from lib.retrieval.turn_budget import TurnBudget, TurnLimits


@pytest.fixture
def captured(monkeypatch, transactional_db):
    _, collection, authorization, _, _, pin = setup(monkeypatch)
    runtime = SourceRuntime(TurnBudget(TurnLimits()), authorization)
    with source_runtime_scope(runtime):
        capture = vocabulary.capture_retrieval_vocabulary(authorization)
        assert capture.get(collection.pk) is not None
        control = graph_effective_controls()
        assert control["scoped_vocabulary"] == {
            "state": "observed",
            "digest": digest(tuple(c.policy.checksum for c in capture.capsules)),
        }
        yield control, runtime, pin, authorization


def test_actual_persisted_capture_validates_observed_and_current(captured):
    control, _, _, _ = captured
    assert validate_controls(control, observed_only=True) == control
    assert validate_controls(control) == control


def test_observed_legacy_vocabulary_still_requires_report(captured):
    control, _, _, _ = captured
    assert control["identity_policy"] == control["remediation_mode"] == "legacy"
    result = validate(None, quality_reports=reports(control))
    assert result["passed"] is False
    assert result["applicable"] is True
    assert "graph_report_missing" in result["blocking_reasons"]


def test_genuine_capture_accepts_complete_synthetic_graph_attachment(monkeypatch, captured):
    report, quality = attachment_pair(monkeypatch)
    result = validate(report, quality_reports=quality)
    assert result["passed"] is True, result
    assert result["applicable"] is True


@pytest.mark.parametrize("observation", [
    None, {}, {"state": "unknown"}, {"state": "observed"},
    {"state": "observed", "digest": "not-sha"},
    {"state": "observed", "digest": "a" * 64, "extra": True},
])
def test_rehashed_malformed_vocabulary_is_not_a_valid_observation(captured, observation):
    control = deepcopy(captured[0])
    control["scoped_vocabulary"] = observation
    rehash(control)
    with pytest.raises(ValueError, match="graph_applicability_unknown"):
        validate_controls(control, observed_only=True)


@pytest.mark.parametrize("mutation", ["changed", "removed"])
def test_live_controls_reject_stale_or_removed_vocabulary(captured, mutation):
    control = deepcopy(captured[0])
    if mutation == "changed":
        control["scoped_vocabulary"]["digest"] = "f" * 64
    else:
        del control["scoped_vocabulary"]
    rehash(control)
    with pytest.raises(ValueError, match="graph_applicability_unknown"):
        validate_controls(control)


def test_detached_actual_pin_invalidates_capture_and_current_gate(captured):
    control, runtime, pin, authorization = captured
    pin.delete()
    with pytest.raises(vocabulary.VocabularyGraphUnavailable):
        vocabulary.capture_retrieval_vocabulary(authorization)
    assert runtime.observation["scoped_vocabulary"] == {"state": "unknown"}
    with pytest.raises(ValueError, match="graph_applicability_unknown"):
        validate_controls(control)


def test_rehashed_vocabulary_drift_cannot_be_declared_a_graph_treatment(monkeypatch, captured):
    report, _ = attachment_pair(monkeypatch)
    before = report["controls"]["baseline"]["scoped_vocabulary"]["digest"]
    report["controls"]["baseline"]["scoped_vocabulary"]["digest"] = "f" * 64
    rehash(report["controls"]["baseline"])
    rebind(report)
    report["treatment"]["scoped_vocabulary.digest"] = {
        "baseline": "f" * 64, "candidate": before,
    }
    result = evaluate_graph_gates_v2(report=report, policy=GraphGatePolicyV2())
    assert result["eligible"] is False
    assert "invalid_or_missing_observation:graph_non_treatment_control_drift" in result["blocking_reasons"]


@pytest.mark.parametrize("removed_arm", ["baseline", "candidate"])
def test_vocabulary_presence_cannot_be_added_or_removed_as_an_unlisted_treatment(monkeypatch, captured, removed_arm):
    report, _ = attachment_pair(monkeypatch)
    del report["controls"][removed_arm]["scoped_vocabulary"]
    rehash(report["controls"][removed_arm])
    rebind(report)
    result = evaluate_graph_gates_v2(report=report, policy=GraphGatePolicyV2())
    assert result["eligible"] is False
    assert "invalid_or_missing_observation:graph_non_treatment_control_drift" in result["blocking_reasons"]
