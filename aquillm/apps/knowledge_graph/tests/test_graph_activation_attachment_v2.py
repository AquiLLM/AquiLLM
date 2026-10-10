"""Graph observations are additional evidence, never an activation authority."""

import importlib

import pytest

from apps.knowledge_graph.tests.graph_gate_test_fixtures import digest, fixture_report


def module():
    try:
        return importlib.import_module(
            "apps.knowledge_graph.evals.graph_activation_attachment_v2"
        )
    except ModuleNotFoundError:
        pytest.fail("G6 graph attachment validation missing")


def test_attachment_recomputes_result_instead_of_accepting_favorable_dictionary():
    report = fixture_report()
    report["queries"] = []
    report["eligible"] = True
    attached = module().graph_activation_attachment_v2(report)
    assert not attached["result"]["eligible"]
    assert attached["report_sha256"] == digest(report)


def test_missing_identity_cannot_opt_out():
    result = module().validate_graph_activation_attachment_v2(
        None, quality_reports={"baseline": {"observations": [{}]}}
    )
    assert not result["passed"]
    assert "graph_applicability_unknown" in result["blocking_reasons"]


def test_attachment_digest_binds_every_original_observation():
    report = fixture_report()
    attached = module().graph_activation_attachment_v2(report)
    report["queries"][0]["candidate"]["ranked"] = []
    changed = module().graph_activation_attachment_v2(report)
    assert changed["attachment_sha256"] != attached["attachment_sha256"]
    assert changed["report_sha256"] != attached["report_sha256"]


def test_synthetic_report_has_no_independent_activation_authority():
    attached = module().graph_activation_attachment_v2(fixture_report())
    assert "activation_eligible" not in attached
    assert attached["result"]["eligible"]
    assert not module().validate_graph_activation_attachment_v2(
        fixture_report(), quality_reports={}
    )["passed"]
