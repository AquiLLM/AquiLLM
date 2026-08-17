"""Offline contract tests for the knowledge-graph evaluation runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from apps.knowledge_graph.evals import run_kg_eval

EVALS_DIR = Path(__file__).resolve().parent.parent / "evals"


def test_loaders_return_immutable_fixture_cases_from_module_paths():
    extraction_cases = run_kg_eval.load_extraction_cases()
    retrieval_cases = run_kg_eval.load_retrieval_cases()

    assert {case["id"] for case in extraction_cases} >= {
        "acronym_definition_then_full_name",
        "ambiguous_acronym_two_meanings",
        "shared_model_name_across_documents",
        "nearby_model_versions_remain_distinct",
        "overlap_boundary_relation",
        "publisher_boilerplate_suppressed",
    }
    assert {case["id"] for case in retrieval_cases} >= {
        "conflicting_claims_across_collections",
        "inaccessible_collection_is_excluded",
        "canonical_identity_expands_a_to_b",
    }
    with pytest.raises(TypeError):
        extraction_cases[0]["id"] = "mutated"
    assert isinstance(extraction_cases[0]["documents"], tuple)


@pytest.mark.parametrize(
    ("filename", "contents", "message"),
    [
        ("bad-shape.yaml", "[]\n", "top level"),
        (
            "duplicate.yaml",
            (
                "schema_version: 1\ncase: &case\n  id: duplicate\n"
                "  description: duplicate test\n  privacy_intent: public\n"
                "  documents:\n    - doc_id: doc\n      collection_id: collection\n"
                "      chunks:\n        - chunk_id: chunk\n          text: evidence\n"
                "  expected:\n    entities: []\n    relations: []\n    auto_links: []\n"
                "    suppressed_evidence: []\ncases:\n  - *case\n  - *case\n"
            ),
            "duplicate",
        ),
        (
            "missing.yaml",
            "schema_version: 1\ncases:\n  - id: only-id\n",
            "missing required",
        ),
    ],
)
def test_loader_rejects_malformed_fixture_data(tmp_path, filename, contents, message):
    path = tmp_path / filename
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(run_kg_eval.FixtureValidationError, match=message):
        run_kg_eval.load_extraction_cases(path)


def test_extraction_metrics_are_set_based_with_empty_sets_scoring_one():
    case = {
        "expected": {
            "entities": [{"id": "e1"}, {"id": "e2"}],
            "relations": [{"source": "e1", "target": "e2", "type": "uses"}],
            "auto_links": [{"source": "e1", "target": "e2"}],
            "suppressed_evidence": [{"entity": "Publisher"}],
        }
    }
    prediction = {
        "entities": [{"id": "e1"}, {"id": "spurious"}],
        "relations": [],
        "auto_links": [{"source": "e1", "target": "e2"}],
        "suppressed_evidence": [],
    }

    report = run_kg_eval.score_extraction(case, prediction)

    assert report == {
        "entity_precision": 0.5,
        "entity_recall": 0.5,
        "relation_precision": 1.0,
        "relation_recall": 0.0,
        "auto_link_precision": 1.0,
        "suppression_precision": 1.0,
    }


def test_retrieval_recall_is_limited_to_first_ten_unique_output_ids():
    case = {"expected_retrieval_chunk_ids": ["chunk-02", "chunk-11"]}
    retrieved = (
        ["chunk-00", "chunk-02", "chunk-02"]
        + [f"chunk-{i:02}" for i in range(3, 12)]
        + ["chunk-11"]
    )

    report = run_kg_eval.score_retrieval(case, retrieved)

    assert report == {"retrieval_recall_at_10": 0.5}


def test_baseline_records_only_fixture_backed_results_and_marks_missing_as_skip():
    cases = (
        {"id": "available", "baseline_vector_result_ids": ["chunk-a", "chunk-b"]},
        {"id": "unavailable"},
    )

    records = run_kg_eval.build_baseline_records(cases)

    assert records == (
        {"id": "available", "result_ids": ["chunk-a", "chunk-b"], "status": "RECORDED"},
        {
            "id": "unavailable",
            "reason": "no fixture-backed or injected vector results",
            "status": "SKIP",
        },
    )


def test_cli_baseline_only_prints_deterministic_machine_readable_json(capsys):
    exit_code = run_kg_eval.main(["--baseline-only"])
    output = capsys.readouterr().out

    assert exit_code == 0
    payload = json.loads(output)
    assert payload["mode"] == "baseline-only"
    assert [record["id"] for record in payload["records"]] == sorted(
        record["id"] for record in payload["records"]
    )
    assert any(record["status"] == "SKIP" for record in payload["records"])
