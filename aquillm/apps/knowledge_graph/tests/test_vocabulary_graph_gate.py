"""Optional vocabulary cannot inherit the unchanged-legacy graph exemption."""
import pytest

from apps.knowledge_graph.evals.graph_effective_controls import graph_effective_controls
from apps.knowledge_graph.tests.test_graph_gate_applicability import reports, validate


def test_default_off_retains_actual_legacy_exemption(monkeypatch):
    monkeypatch.delenv("KG_VOCABULARY_VETO_ENABLED", raising=False)
    controls = graph_effective_controls()
    assert "scoped_vocabulary" not in controls
    assert validate(None, quality_reports=reports(controls))["passed"]


@pytest.mark.parametrize("mutation", ["unknown", "changed", "removed"])
def test_enabled_unknown_changed_or_removed_observation_blocks_exemption(monkeypatch, mutation):
    legacy = graph_effective_controls()
    monkeypatch.setenv("KG_VOCABULARY_VETO_ENABLED", "1")
    controls = graph_effective_controls()
    assert controls["scoped_vocabulary"]["state"] == "unknown"
    if mutation == "changed":
        controls["scoped_vocabulary"] = {"state": "observed", "digest": "a" * 64}
    elif mutation == "removed":
        controls = legacy
    assert not validate(None, quality_reports=reports(controls))["passed"]
