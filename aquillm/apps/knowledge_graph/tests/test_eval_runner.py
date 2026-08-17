"""Offline contract tests for the knowledge-graph evaluation runner."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

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


def test_extraction_metrics_accept_immutable_loaded_fixture_records():
    case = run_kg_eval.load_extraction_cases()[0]

    report = run_kg_eval.score_extraction(case, case["expected"])

    assert report == {
        "entity_precision": 1.0,
        "entity_recall": 1.0,
        "relation_precision": 1.0,
        "relation_recall": 1.0,
        "auto_link_precision": 1.0,
        "suppression_precision": 1.0,
    }


def test_zero_gold_extraction_recall_is_one():
    report = run_kg_eval.score_extraction(
        {
            "expected": {
                "entities": [],
                "relations": [],
                "auto_links": [],
                "suppressed_evidence": [],
            }
        },
        {},
    )

    assert report["entity_recall"] == 1.0
    assert report["relation_recall"] == 1.0


def test_retrieval_recall_is_limited_to_first_ten_unique_output_ids():
    case = {"expected_retrieval_chunk_ids": ["chunk-02", "chunk-11"]}
    retrieved = (
        ["chunk-00", "chunk-02", "chunk-02"]
        + [f"chunk-{i:02}" for i in range(3, 12)]
        + ["chunk-11"]
    )

    report = run_kg_eval.score_retrieval(case, retrieved)

    assert report == {"retrieval_recall_at_10": 0.5}


def test_zero_gold_retrieval_recall_is_one():
    assert run_kg_eval.score_retrieval(
        {"expected_retrieval_chunk_ids": []}, ["unrelated-chunk"]
    ) == {"retrieval_recall_at_10": 1.0}


def _fixture_payload(filename):
    return yaml.safe_load((EVALS_DIR / filename).read_text(encoding="utf-8"))


def _write_payload(tmp_path, filename, payload):
    path = tmp_path / filename
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload["cases"][0]["documents"].append(
                deepcopy(payload["cases"][0]["documents"][0])
            ),
            "duplicate document id",
        ),
        (
            lambda payload: payload["cases"][0]["documents"][0]["chunks"].append(
                deepcopy(payload["cases"][0]["documents"][0]["chunks"][0])
            ),
            "duplicate chunk id",
        ),
        (
            lambda payload: payload["cases"][0]["expected"]["entities"][0].pop("text"),
            "missing required field 'text'",
        ),
        (
            lambda payload: payload["cases"][0]["expected"]["entities"][0].update(
                chunk_id="missing-chunk"
            ),
            "unknown chunk",
        ),
        (
            lambda payload: payload["cases"][0]["expected"]["relations"][0].update(
                target="unknown-entity"
            ),
            "unknown entity",
        ),
        (
            lambda payload: payload["cases"][0]["expected"]["auto_links"][0].update(
                target="unknown-entity"
            ),
            "unknown entity",
        ),
        (
            lambda payload: payload["cases"][-1]["expected"]["suppressed_evidence"][
                0
            ].update(chunk_id="missing-chunk"),
            "unknown chunk",
        ),
    ],
)
def test_extraction_loader_rejects_duplicate_and_dangling_records(
    tmp_path, mutate, message
):
    payload = _fixture_payload("extraction_cases.yaml")
    mutate(payload)

    with pytest.raises(run_kg_eval.FixtureValidationError, match=message):
        run_kg_eval.load_extraction_cases(
            _write_payload(tmp_path, "invalid-extraction.yaml", payload)
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload["cases"][0]["documents"].append(
                deepcopy(payload["cases"][0]["documents"][0])
            ),
            "duplicate document id",
        ),
        (
            lambda payload: payload["cases"][0]["documents"][0]["chunks"].append(
                deepcopy(payload["cases"][0]["documents"][0]["chunks"][0])
            ),
            "duplicate chunk id",
        ),
        (
            lambda payload: payload["cases"][0]["expected_retrieval_chunk_ids"].append(
                "unknown-chunk"
            ),
            "unknown chunk",
        ),
        (
            lambda payload: payload["cases"][0]["baseline_vector_result_ids"].append(
                "unknown-chunk"
            ),
            "unknown chunk",
        ),
        (
            lambda payload: payload["cases"][1].update(
                baseline_vector_result_ids=["private-incident-001"]
            ),
            "not in an accessible collection",
        ),
        (
            lambda payload: payload["cases"][-1]["canonical_identity_links"][0].update(
                target_chunk_id="unknown-chunk"
            ),
            "unknown chunk",
        ),
    ],
)
def test_retrieval_loader_rejects_duplicate_and_dangling_records(
    tmp_path, mutate, message
):
    payload = _fixture_payload("retrieval_cases.yaml")
    mutate(payload)

    with pytest.raises(run_kg_eval.FixtureValidationError, match=message):
        run_kg_eval.load_retrieval_cases(
            _write_payload(tmp_path, "invalid-retrieval.yaml", payload)
        )


def test_all_fixture_records_have_valid_text_anchors_and_references():
    extraction_cases = run_kg_eval.load_extraction_cases()
    for case in extraction_cases:
        chunks = {
            chunk["chunk_id"]: chunk["text"]
            for document in case["documents"]
            for chunk in document["chunks"]
        }
        entities = {entity["id"]: entity for entity in case["expected"]["entities"]}
        for entity in entities.values():
            assert entity["text"].lower() in chunks[entity["chunk_id"]].lower()
        for relation in case["expected"]["relations"]:
            assert relation["source"] in entities
            assert relation["target"] in entities
        for link in case["expected"]["auto_links"]:
            assert link["source"] in entities
            assert link["target"] in entities
        for evidence in case["expected"]["suppressed_evidence"]:
            assert evidence["entity"].lower() in chunks[evidence["chunk_id"]].lower()

    for case in run_kg_eval.load_retrieval_cases():
        chunks = {
            chunk["chunk_id"]
            for document in case["documents"]
            for chunk in document["chunks"]
        }
        assert set(case["expected_retrieval_chunk_ids"]) <= chunks
        for link in case.get("canonical_identity_links", ()):
            assert link["source_chunk_id"] in chunks
            assert link["target_chunk_id"] in chunks

    acronym_case = next(
        case
        for case in extraction_cases
        if case["id"] == "acronym_definition_then_full_name"
    )
    acronym_entities = {
        entity["id"]: entity for entity in acronym_case["expected"]["entities"]
    }
    acronym_chunks = {
        chunk["chunk_id"]: chunk["text"]
        for document in acronym_case["documents"]
        for chunk in document["chunks"]
    }
    assert acronym_entities["entity-rag-full-name"]["chunk_id"] == "rag-overview-002"
    assert (
        acronym_entities["entity-rag-full-name"]["text"]
        in acronym_chunks["rag-overview-002"]
    )

    overlap_case = next(
        case for case in extraction_cases if case["id"] == "overlap_boundary_relation"
    )
    overlap_chunks = {
        chunk["chunk_id"]: chunk["text"]
        for document in overlap_case["documents"]
        for chunk in document["chunks"]
    }
    assert "The retrieval service ranks passages" in overlap_chunks["latency-001"]
    assert "The retrieval service ranks passages" in overlap_chunks["latency-002"]


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


def test_cli_returns_two_for_invalid_fixture_input(tmp_path, capsys):
    invalid_fixture = tmp_path / "invalid.yaml"
    invalid_fixture.write_text("[]\n", encoding="utf-8")

    exit_code = run_kg_eval.main(
        ["--baseline-only", "--extraction-cases", str(invalid_fixture)]
    )

    assert exit_code == 2
    assert json.loads(capsys.readouterr().out)["status"] == "INVALID_FIXTURE"
