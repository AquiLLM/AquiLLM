"""Offline contract tests for the knowledge-graph evaluation runner."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from apps.knowledge_graph.evals import run_kg_eval

EVALS_DIR = Path(__file__).resolve().parent.parent / "evals"
AQUILLM_DIR = EVALS_DIR.parents[2]


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
        (
            "recursive.yaml",
            "schema_version: 1\ncases: &cases\n  - *cases\n",
            "recursive",
        ),
    ],
)
def test_loader_rejects_malformed_fixture_data(tmp_path, filename, contents, message):
    path = tmp_path / filename
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(run_kg_eval.FixtureValidationError, match=message):
        run_kg_eval.load_extraction_cases(path)


@pytest.mark.parametrize("schema_version", [True, 1.0])
def test_loader_requires_schema_version_to_be_exact_integer(tmp_path, schema_version):
    payload = _fixture_payload("extraction_cases.yaml")
    payload["schema_version"] = schema_version

    with pytest.raises(
        run_kg_eval.FixtureValidationError, match="schema_version must be integer 1"
    ):
        run_kg_eval.load_extraction_cases(
            _write_payload(tmp_path, "invalid-schema-version.yaml", payload)
        )


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
        "auto_link_precision": 0.0,
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


def test_semantic_matching_ignores_ids_but_keeps_span_and_direction():
    case = {
        "expected": {
            "entities": [
                {
                    "id": "gold-a",
                    "text": "Atlas",
                    "type": "model",
                    "chunk_id": "c",
                    "start": 1,
                    "end": 6,
                },
                {
                    "id": "gold-b",
                    "text": "Service",
                    "type": "service",
                    "chunk_id": "c",
                    "start": 7,
                    "end": 14,
                },
            ],
            "relations": [{"source": "gold-a", "target": "gold-b", "type": "uses"}],
            "auto_links": [
                {
                    "source": "gold-a",
                    "target": "gold-b",
                    "type": "canonical_identity",
                }
            ],
            "suppressed_evidence": [],
        }
    }
    prediction = {
        "entities": [
            {
                "id": "runtime-7",
                "text": "atlas",
                "type": "model",
                "chunk_id": "c",
                "start": 1,
                "end": 6,
                "confidence": 0.9,
            },
            {
                "id": "runtime-8",
                "text": "Service",
                "type": "service",
                "chunk_id": "c",
                "start": 7,
                "end": 14,
                "confidence": 0.2,
            },
        ],
        "relations": [
            {
                "source": "runtime-7",
                "target": "runtime-8",
                "type": "uses",
                "confidence": 0.1,
            }
        ],
        "auto_links": [
            {
                "source": "runtime-7",
                "target": "runtime-8",
                "type": "canonical_identity",
                "confidence": 0.8,
            }
        ],
    }
    assert run_kg_eval.score_extraction(case, prediction)["relation_recall"] == 1.0
    assert run_kg_eval.score_extraction(case, prediction)["auto_link_precision"] == 1.0
    prediction["entities"][0]["start"] = 2
    assert run_kg_eval.score_extraction(case, prediction)["entity_recall"] == 0.5
    prediction["entities"][0]["start"] = 1
    prediction["entities"][0]["chunk_id"] = "different-chunk"
    assert run_kg_eval.score_extraction(case, prediction)["entity_recall"] == 0.5
    assert run_kg_eval.score_extraction(case, prediction)["auto_link_precision"] == 0.0


def test_overlap_relation_scores_the_visible_same_chunk_mentions():
    case = next(
        case
        for case in run_kg_eval.load_extraction_cases()
        if case["id"] == "overlap_boundary_relation"
    )
    prediction = {
        "entities": [
            {
                **dict(case["expected"]["entities"][0]),
                "id": "runtime-atlas",
                "confidence": 0.9,
            },
            {
                **dict(case["expected"]["entities"][1]),
                "id": "runtime-service",
                "confidence": 0.8,
            },
        ],
        "relations": [
            {
                "source": "runtime-atlas",
                "target": "runtime-service",
                "type": "supplies_embeddings_to",
                "confidence": 0.7,
            }
        ],
    }

    report = run_kg_eval.score_extraction(case, prediction)

    assert report["relation_precision"] == 1.0
    assert report["relation_recall"] == 1.0


def test_baseline_accepts_deduplicated_positive_integer_ids():
    record = run_kg_eval.build_baseline_records(
        ({"id": "case"},), {"case": [2, 2, "alias"]}
    )[0]

    assert record["result_ids"] == (2, "alias")
    assert record["unresolved_result_ids"] == (2, "alias")
    assert record["unresolved_result_count"] == 2
    assert record["security_status"] == "UNKNOWN"


def test_retrieval_recall_is_limited_to_first_ten_unique_output_ids():
    case = {"expected_retrieval_chunk_ids": ["chunk-02", "chunk-11"]}
    retrieved = (
        ["chunk-00", "chunk-02", "chunk-02"]
        + [f"chunk-{i:02}" for i in range(3, 12)]
        + ["chunk-11"]
    )
    report = run_kg_eval.score_retrieval(case, retrieved)
    assert report == {"retrieval_recall_at_10": 0.5}


def test_baseline_reports_inaccessible_observed_results_without_counting_them():
    case = {
        "id": "mixed",
        "accessible_collection_ids": ["public"],
        "documents": (
            {"collection_id": "public", "chunks": ({"chunk_id": "public-1"},)},
            {"collection_id": "private", "chunks": ({"chunk_id": "private-1"},)},
        ),
        "baseline_vector_result_ids": ["public-1", "private-1"],
    }
    record = run_kg_eval.build_baseline_records((case,))[0]
    assert record["result_ids"] == ("public-1", "private-1")
    assert record["inaccessible_result_ids"] == ("private-1",)
    assert record["inaccessible_result_count"] == 1
    assert record["unresolved_result_ids"] == ()
    assert record["unresolved_result_count"] == 0
    assert record["security_status"] == "LEAKAGE"


@pytest.mark.parametrize(
    ("result_ids", "status", "inaccessible", "unresolved"),
    [
        ([101], "UNKNOWN", (), (101,)),
        (["unknown-result"], "UNKNOWN", (), ("unknown-result",)),
        (["private-1"], "LEAKAGE", ("private-1",), ()),
        (["private-1", 101], "LEAKAGE", ("private-1",), (101,)),
        (["public-1"], "OK", (), ()),
    ],
)
def test_baseline_security_status_requires_accessibility_evidence(
    result_ids, status, inaccessible, unresolved
):
    case = {
        "id": "case",
        "accessible_collection_ids": ("public",),
        "documents": (
            {"collection_id": "public", "chunks": ({"chunk_id": "public-1"},)},
            {
                "collection_id": "private",
                "chunks": ({"chunk_id": "private-1"},),
            },
        ),
    }

    record = run_kg_eval.build_baseline_records((case,), {"case": result_ids})[0]

    assert record["security_status"] == status
    assert record["inaccessible_result_ids"] == inaccessible
    assert record["unresolved_result_ids"] == unresolved


def test_structured_baseline_results_map_native_ids_to_collections():
    case = {"id": "case", "accessible_collection_ids": ("public",)}
    injected = {
        "case": {
            "result_ids": [101, 101],
            "id_collections": {"101": "public"},
        }
    }

    record = run_kg_eval.build_baseline_records((case,), injected)[0]

    assert record["result_ids"] == (101,)
    assert record["inaccessible_result_ids"] == ()
    assert record["unresolved_result_ids"] == ()
    assert record["security_status"] == "OK"


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
            lambda entity: entity.pop("start", None),
            "missing required field 'start'",
        ),
        (
            lambda entity: entity.pop("end", None),
            "missing required field 'end'",
        ),
        (lambda entity: entity.update(start="0"), "start must be an integer"),
        (lambda entity: entity.update(start=True), "start must be an integer"),
        (lambda entity: entity.update(end=False), "end must be an integer"),
        (lambda entity: entity.update(start=-1), "start must be non-negative"),
        (
            lambda entity: entity.update(start=10_000, end=10_001),
            "start is outside referenced chunk",
        ),
        (lambda entity: entity.update(end=0), "end must be greater than start"),
        (
            lambda entity: entity.update(end=10_000),
            "end is outside referenced chunk",
        ),
        (
            lambda entity: entity.update(start=1, end=31),
            "span does not exactly match text",
        ),
    ],
)
def test_extraction_loader_rejects_missing_or_invalid_entity_spans(
    tmp_path, mutate, message
):
    payload = _fixture_payload("extraction_cases.yaml")
    mutate(payload["cases"][0]["expected"]["entities"][0])

    with pytest.raises(run_kg_eval.FixtureValidationError, match=message):
        run_kg_eval.load_extraction_cases(
            _write_payload(tmp_path, "invalid-span.yaml", payload)
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


def test_retrieval_fixture_allows_observed_private_and_unresolved_baseline_ids(
    tmp_path,
):
    payload = _fixture_payload("retrieval_cases.yaml")
    payload["cases"][2]["baseline_vector_result_ids"] = [
        "public-token-001",
        "private-incident-001",
        "native-unresolved",
    ]

    cases = run_kg_eval.load_retrieval_cases(
        _write_payload(tmp_path, "observed-leakage.yaml", payload)
    )
    record = next(
        record
        for record in run_kg_eval.build_baseline_records(cases)
        if record["id"] == "inaccessible_collection_is_excluded"
    )

    assert record["security_status"] == "LEAKAGE"
    assert record["inaccessible_result_ids"] == ("private-incident-001",)
    assert record["unresolved_result_ids"] == ("native-unresolved",)


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
            assert type(entity["start"]) is int
            assert type(entity["end"]) is int
            chunk_text = chunks[entity["chunk_id"]]
            assert chunk_text[entity["start"] : entity["end"]] == entity["text"]
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
    acronym_mentions = {
        (
            entity["id"],
            entity["chunk_id"],
            entity["text"],
            entity["type"],
            entity["start"],
            entity["end"],
        )
        for entity in acronym_case["expected"]["entities"]
    }
    assert acronym_mentions == {
        (
            "entity-rag-full-name-001",
            "rag-overview-001",
            "Retrieval-augmented generation",
            "method",
            0,
            30,
        ),
        ("entity-rag-acronym-001", "rag-overview-001", "RAG", "method", 32, 35),
        (
            "entity-rag-full-name-002",
            "rag-overview-002",
            "Retrieval-augmented generation",
            "method",
            0,
            30,
        ),
        ("entity-rag-acronym-002", "rag-overview-002", "RAG", "method", 32, 35),
    }

    overlap_case = next(
        case for case in extraction_cases if case["id"] == "overlap_boundary_relation"
    )
    service_mentions = {
        (
            entity["id"],
            entity["chunk_id"],
            entity["text"],
            entity["type"],
            entity["start"],
            entity["end"],
        )
        for entity in overlap_case["expected"]["entities"]
        if entity["text"] == "retrieval service"
    }
    assert service_mentions == {
        (
            "entity-retrieval-service-001a",
            "latency-001",
            "retrieval service",
            "service",
            45,
            62,
        ),
        (
            "entity-retrieval-service-001b",
            "latency-001",
            "retrieval service",
            "service",
            68,
            85,
        ),
        (
            "entity-retrieval-service-002",
            "latency-002",
            "retrieval service",
            "service",
            4,
            21,
        ),
    }


def test_baseline_records_only_fixture_backed_results_and_marks_missing_as_skip():
    cases = (
        {
            "id": "available",
            "accessible_collection_ids": ("public",),
            "documents": (
                {
                    "collection_id": "public",
                    "chunks": ({"chunk_id": "chunk-a"}, {"chunk_id": "chunk-b"}),
                },
            ),
            "baseline_vector_result_ids": ("chunk-a", "chunk-b"),
        },
        {"id": "unavailable"},
    )

    records = run_kg_eval.build_baseline_records(cases)

    assert records == (
        {
            "id": "available",
            "result_ids": ("chunk-a", "chunk-b"),
            "inaccessible_result_ids": (),
            "inaccessible_result_count": 0,
            "unresolved_result_ids": (),
            "unresolved_result_count": 0,
            "security_status": "OK",
            "status": "RECORDED",
        },
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


def _run_cli(tmp_path, *args):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (str(AQUILLM_DIR), env.get("PYTHONPATH"))
        if value
    )
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "apps.knowledge_graph.evals.run_kg_eval",
            *map(str, args),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        timeout=10,
        check=False,
    )


def test_subprocess_cli_baseline_accepts_integer_ids_deterministically(tmp_path):
    results = tmp_path / "results.json"
    results.write_text(
        json.dumps(
            {
                "canonical_identity_expands_a_to_b": {
                    "result_ids": [101, 202],
                    "id_collections": {
                        "101": "collection-research-a",
                        "202": "collection-research-b",
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    first = _run_cli(tmp_path, "--baseline-only", "--retrieval-results", results)
    second = _run_cli(tmp_path, "--baseline-only", "--retrieval-results", results)

    assert first.returncode == second.returncode == 0
    assert first.stdout == second.stdout
    assert first.stderr == second.stderr == b""
    payload = json.loads(first.stdout)
    record = next(
        record
        for record in payload["records"]
        if record["id"] == "canonical_identity_expands_a_to_b"
    )
    assert record["result_ids"] == [101, 202]
    assert record["security_status"] == "OK"
    assert record["unresolved_result_ids"] == []


@pytest.mark.parametrize(
    ("kind", "contents", "extra_args"),
    [
        ("fixture", "[unterminated", ("--extraction-cases",)),
        (
            "recursive-fixture",
            "schema_version: 1\ncases: &cases\n  - *cases\n",
            ("--extraction-cases",),
        ),
        ("results", "[]", ("--retrieval-results",)),
        (
            "results",
            '{"canonical_identity_expands_a_to_b": {"result_ids": [1], '
            '"id_collections": []}}',
            ("--retrieval-results",),
        ),
        ("results", '{"unknown-case": [1]}', ("--retrieval-results",)),
        (
            "results",
            '{"canonical_identity_expands_a_to_b": [true]}',
            ("--retrieval-results",),
        ),
        (
            "results",
            '{"canonical_identity_expands_a_to_b": [0]}',
            ("--retrieval-results",),
        ),
        (
            "results",
            '{"canonical_identity_expands_a_to_b": [-1]}',
            ("--retrieval-results",),
        ),
    ],
)
def test_subprocess_cli_reports_invalid_inputs_without_tracebacks(
    tmp_path, kind, contents, extra_args
):
    invalid = tmp_path / f"invalid-{kind}.txt"
    invalid.write_text(contents, encoding="utf-8")

    completed = _run_cli(tmp_path, "--baseline-only", *extra_args, invalid)

    assert completed.returncode == 2
    lines = completed.stdout.splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["status"] == "INVALID_FIXTURE"
    assert completed.stderr == b""
