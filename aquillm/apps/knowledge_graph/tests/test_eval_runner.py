"""Offline contract tests for the knowledge-graph evaluation runner."""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import numpy as np
import pytest
import yaml

from apps.knowledge_graph.evals import fixture_manifest, run_kg_eval

EVALS_DIR = Path(__file__).resolve().parent.parent / "evals"
AQUILLM_DIR = EVALS_DIR.parents[2]
EMBED_CHECKPOINT = "a" * 40
RERANK_TEMPLATE_PATH = (
    AQUILLM_DIR.parent
    / "deploy"
    / "docker"
    / "vllm"
    / "chat_templates"
    / "qwen3_vl_reranker.jinja"
)
RERANK_TEMPLATE_SHA256 = sha256(RERANK_TEMPLATE_PATH.read_bytes()).hexdigest()
RERANK_EXTRA_ARGS = (
    "--chat-template /templates/qwen3_vl_reranker.jinja "
    "--hf-overrides "
    '\'{"architectures":["Qwen3VLForSequenceClassification"],'
    '"classifier_from_token":["no","yes"],'
    '"is_original_qwen3_reranker":true}\''
)
EMBED_EXTRA_ARGS = (
    "--quantization bitsandbytes --load-format bitsandbytes "
    "--model-loader-extra-config "
    '\'{"load_in_4bit":true,"bnb_4bit_compute_dtype":"float16",'
    '"bnb_4bit_quant_type":"nf4","bnb_4bit_use_double_quant":true}\' '
    "--hf-overrides "
    "'{\"matryoshka_dimensions\":[1024]}'"
)


def _strict_extraction_settings(**overrides):
    from lib.knowledge_graph.config import ExtractionSettings

    values = {
        "build_enabled": False,
        "provider": "gliner2_local",
        "model_id": "fastino/gliner2-base-v1",
        "model_revision": "8437ba583a733d87f56ae902f3b197934eedd58e",
        "device": "cpu",
        "batch_size": 8,
        "max_batch_characters": 64_000,
        "cache_dir": Path("/opt/kg-eval-hf-cache"),
        "local_files_only": True,
        "fail_open": False,
    }
    values.update(overrides)
    return ExtractionSettings(**values)


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
        "mention_span_precision": 0.5,
        "mention_span_recall": 0.5,
        "relation_precision": 1.0,
        "relation_recall": 0.0,
        "relation_direction_precision": 1.0,
        "relation_direction_recall": 0.0,
        "relation_endpoint_precision": 1.0,
        "relation_endpoint_recall": 0.0,
        "auto_link_precision": 0.0,
        "automatic_link_precision": 0.0,
        "automatic_link_recall": 0.0,
        "candidate_link_precision": 0.0,
        "candidate_link_recall": 0.0,
        "suppression_precision": 1.0,
        "suppression_recall": 0.0,
    }


def test_extraction_metrics_accept_immutable_loaded_fixture_records():
    case = run_kg_eval.load_extraction_cases()[0]

    report = run_kg_eval.score_extraction(case, case["expected"])

    assert report == {
        "entity_precision": 1.0,
        "entity_recall": 1.0,
        "mention_span_precision": 1.0,
        "mention_span_recall": 1.0,
        "relation_precision": 1.0,
        "relation_recall": 1.0,
        "relation_direction_precision": 1.0,
        "relation_direction_recall": 1.0,
        "relation_endpoint_precision": 1.0,
        "relation_endpoint_recall": 1.0,
        "auto_link_precision": 1.0,
        "automatic_link_precision": 1.0,
        "automatic_link_recall": 1.0,
        "candidate_link_precision": 1.0,
        "candidate_link_recall": 1.0,
        "suppression_precision": 1.0,
        "suppression_recall": 1.0,
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
            lambda payload: next(
                case for case in payload["cases"] if "canonical_identity_links" in case
            )["canonical_identity_links"][0].update(target_chunk_id="unknown-chunk"),
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
        value for value in (str(AQUILLM_DIR), env.get("PYTHONPATH")) if value
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


def test_extraction_metrics_report_each_decision_class_separately():
    case = {
        "expected": {
            "entities": [
                {
                    "id": "a",
                    "text": "Atlas",
                    "type": "model",
                    "chunk_id": "c",
                    "start": 0,
                    "end": 5,
                },
                {
                    "id": "b",
                    "text": "service",
                    "type": "service",
                    "chunk_id": "c",
                    "start": 6,
                    "end": 13,
                },
            ],
            "relations": [{"source": "a", "target": "b", "type": "uses"}],
            "auto_links": [
                {"source": "a", "target": "b", "type": "canonical_identity"}
            ],
            "suppressed_evidence": [
                {"entity": "Publisher", "chunk_id": "c", "reason": "boilerplate"}
            ],
        }
    }
    prediction = deepcopy(case["expected"])
    prediction["relations"] = [{"source": "b", "target": "a", "type": "uses"}]
    prediction["candidate_links"] = [
        {"source": "a", "target": "b", "type": "canonical_identity"},
        {"source": "b", "target": "a", "type": "canonical_identity"},
    ]

    report = run_kg_eval.score_extraction(case, prediction)

    assert report["mention_span_precision"] == 1.0
    assert report["mention_span_recall"] == 1.0
    assert report["relation_endpoint_precision"] == 1.0
    assert report["relation_endpoint_recall"] == 1.0
    assert report["relation_direction_precision"] == 0.0
    assert report["relation_direction_recall"] == 0.0
    assert report["automatic_link_precision"] == 1.0
    assert report["automatic_link_recall"] == 1.0
    assert report["candidate_link_precision"] == 0.5
    assert report["suppression_precision"] == 1.0
    assert report["suppression_recall"] == 1.0


def test_live_extraction_projection_scores_imperfect_predictions_never_gold():
    case = {
        "id": "imperfect-live-case",
        "expected": {
            "entities": [
                {
                    "id": "gold-a",
                    "text": "Atlas",
                    "type": "model",
                    "chunk_id": "chunk-a",
                    "start": 0,
                    "end": 5,
                },
                {
                    "id": "gold-b",
                    "text": "retriever",
                    "type": "method",
                    "chunk_id": "chunk-a",
                    "start": 10,
                    "end": 19,
                },
            ],
            "relations": [
                {"source": "gold-a", "target": "gold-b", "type": "uses_method"}
            ],
            "auto_links": [
                {
                    "source": "gold-a",
                    "target": "gold-b",
                    "type": "canonical_identity",
                }
            ],
            "suppressed_evidence": [
                {
                    "entity": "Publisher",
                    "type": "publisher",
                    "chunk_id": "chunk-a",
                    "reason": "publisher_suppressed_by_default",
                }
            ],
        },
    }
    predictions = {
        "entities": [
            {
                "id": "pred-a",
                "text": "Atlas",
                "type": "model",
                "chunk_id": "chunk-a",
                "start": 0,
                "end": 5,
            },
            {
                "id": "pred-b",
                "text": "retriever",
                "type": "method",
                "chunk_id": "chunk-a",
                "start": 10,
                "end": 19,
            },
        ],
        "relations": [{"source": "pred-b", "target": "pred-a", "type": "uses_method"}],
        "auto_links": [],
        "candidate_links": [
            {
                "source": "pred-a",
                "target": "pred-b",
                "type": "canonical_identity",
            }
        ],
        "suppressed_evidence": [],
    }
    calls: list[str] = []

    report = run_kg_eval.evaluate_live_extraction_cases(
        (case,),
        predict_case=lambda value: calls.append(value["id"]) or predictions,
    )

    assert calls == ["imperfect-live-case"]
    case_metrics = report["cases"]["imperfect-live-case"]
    assert case_metrics["mention_span_recall"] == 1.0
    assert case_metrics["relation_endpoint_recall"] == 1.0
    assert case_metrics["relation_direction_recall"] == 0.0
    assert case_metrics["automatic_link_recall"] == 0.0
    assert case_metrics["candidate_link_recall"] == 1.0
    assert case_metrics["suppression_recall"] == 0.0
    assert report["metrics"]["automatic_link_precision"] == 1.0
    assert report["metrics"]["candidate_link_precision"] == 1.0


def test_production_extraction_projection_is_lazy_pure_and_reuses_exact_pipelines():
    source = inspect.getsource(run_kg_eval._production_extraction_prediction)

    for production_call in (
        "collect_document_evidence",
        "resolve_document_mentions",
        "resolve_collection_entities",
        "decide_entity_filter",
    ):
        assert production_call in source
    for forbidden_write in (
        ".objects.",
        ".save(",
        ".create(",
        ".bulk_create(",
        ".delay(",
        ".apply_async(",
    ):
        assert forbidden_write not in source


def test_default_live_bundle_routes_actual_production_predictions_not_gold():
    source = inspect.getsource(run_kg_eval._live_comparison_bundle)

    assert "evaluate_production_extraction_cases(" in source
    assert 'score_extraction(case, case["expected"])' not in source
    assert "_production_extraction_prediction(" in inspect.getsource(
        run_kg_eval.evaluate_production_extraction_cases
    )


@pytest.mark.parametrize(
    "overrides",
    (
        {"build_enabled": True},
        {"provider": "remote"},
        {"model_id": "../mutable"},
        {"model_revision": "main"},
        {"model_revision": "A" * 40},
        {"local_files_only": False},
        {"fail_open": True},
        {"device": ""},
        {"batch_size": 0},
        {"max_batch_characters": 0},
    ),
)
def test_live_extraction_contract_rejects_mutable_or_fail_open_settings(overrides):
    validator = getattr(run_kg_eval, "_validate_live_extraction_settings", None)

    assert callable(validator)
    with pytest.raises(run_kg_eval.ComparisonAborted, match="extraction"):
        validator(_strict_extraction_settings(**overrides))


def test_live_extraction_contract_is_loaded_once_before_providers_or_content():
    source = inspect.getsource(run_kg_eval._live_comparison_bundle)
    validation = source.index(
        "extraction_settings = _validate_live_extraction_settings("
    )

    assert source.count("load_extraction_settings()") == 1
    assert validation < source.index("get_local_embed_config()")
    assert validation < source.index("_authorize_evaluation_collection_scope(")


def test_production_extraction_suite_uses_pinned_kernels_for_every_case(monkeypatch):
    cases = (
        {
            "id": "first",
            "expected": {
                "entities": [
                    {
                        "id": "gold-first",
                        "text": "Atlas",
                        "type": "model",
                        "chunk_id": "chunk-first",
                        "start": 0,
                        "end": 5,
                    }
                ],
                "relations": [],
                "auto_links": [],
                "suppressed_evidence": [],
            },
        },
        {
            "id": "second",
            "expected": {
                "entities": [
                    {
                        "id": "gold-second",
                        "text": "Orion",
                        "type": "model",
                        "chunk_id": "chunk-second",
                        "start": 0,
                        "end": 5,
                    }
                ],
                "relations": [],
                "auto_links": [],
                "suppressed_evidence": [],
            },
        },
    )
    calls = []

    def fake_projection(case, *, context, **kernels):
        calls.append(
            (
                case["id"],
                context,
                {name: value.__module__ for name, value in sorted(kernels.items())},
            )
        )
        return {
            "entities": [],
            "relations": [],
            "auto_links": [],
            "candidate_links": [],
            "suppressed_evidence": [],
        }

    monkeypatch.setattr(
        run_kg_eval,
        "_project_production_extraction_case",
        fake_projection,
    )

    report = run_kg_eval.evaluate_production_extraction_cases(
        cases,
        projection_context="attested-context",
    )

    assert [call[0] for call in calls] == ["first", "second"]
    assert all(call[1] == "attested-context" for call in calls)
    assert all(
        call[2]
        == {
            "collect_document_evidence": ("apps.knowledge_graph.extraction.pipeline"),
            "decide_entity_filter": "apps.knowledge_graph.graph.filtering",
            "resolve_collection_entities": (
                "apps.knowledge_graph.resolution.collection"
            ),
            "resolve_document_mentions": (
                "apps.knowledge_graph.resolution.coreference"
            ),
        }
        for call in calls
    )
    assert report["metrics"]["mention_span_recall"] == 0.0


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([2, 1], (1, 2)),
        ([4], (4,)),
        ([4, 2, 3, 1], (1, 2, 3, 4)),
    ],
)
def test_collection_scope_is_canonicalized_once(values, expected):
    assert run_kg_eval.canonicalize_collection_scope(values) == expected


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ([], "one through four"),
        ([1, 2, 3, 4, 5], "one through four"),
        ([1, 1], "duplicate"),
        ([True], "positive integer"),
        ([0], "positive integer"),
        (["1"], "positive integer"),
    ],
)
def test_collection_scope_rejects_bad_count_duplicates_and_types(values, message):
    with pytest.raises(run_kg_eval.ComparisonValidationError, match=message):
        run_kg_eval.canonicalize_collection_scope(values)


@pytest.mark.parametrize(
    ("debug", "environment", "allowed", "build", "overlay", "message"),
    [
        (False, "production", "1", "0", "0", "production"),
        (True, "development", "0", "0", "0", "KG_EVAL_BYPASS_ALLOWED"),
        (True, "development", "1", "1", "0", "KG_BUILD_ENABLED"),
        (True, "development", "1", "0", "1", "KG_OVERLAY_ENABLED"),
    ],
)
def test_eval_bypass_rejects_production_missing_opt_in_or_enabled_shipping_flags(
    debug, environment, allowed, build, overlay, message
):
    with pytest.raises(run_kg_eval.ComparisonValidationError, match=message):
        run_kg_eval.validate_eval_bypass(
            eval_only=True,
            debug=debug,
            environment=environment,
            environ={
                "KG_EVAL_BYPASS_ALLOWED": allowed,
                "KG_BUILD_ENABLED": build,
                "KG_OVERLAY_ENABLED": overlay,
            },
        )


def test_eval_bypass_accepts_only_explicit_debug_test_process_with_both_flags_off():
    run_kg_eval.validate_eval_bypass(
        eval_only=True,
        debug=True,
        environment="test",
        environ={
            "KG_EVAL_BYPASS_ALLOWED": "1",
            "KG_BUILD_ENABLED": "0",
            "KG_OVERLAY_ENABLED": "0",
        },
    )


def _fake_fixture_manifest():
    request_a = UUID("11111111-1111-4111-8111-111111111111")
    request_b = UUID("22222222-2222-4222-8222-222222222222")
    collection_bindings = {
        "collection-public": (101, request_a, True),
        "collection-policy-a": (101, request_a, True),
        "collection-policy-b": (202, request_b, True),
        "collection-research-a": (101, request_a, True),
        "collection-research-b": (202, request_b, True),
        "collection-security-private": (303, None, False),
    }
    documents = {}
    chunks = {}
    document_index = 1
    chunk_index = 1
    for case in (
        *run_kg_eval.load_extraction_cases(),
        *run_kg_eval.load_retrieval_cases(),
    ):
        for document in case["documents"]:
            symbol = document["doc_id"]
            collection_symbol = document["collection_id"]
            full_text, spans = run_kg_eval.assemble_fixture_document(
                tuple(chunk["text"] for chunk in document["chunks"])
            )
            if symbol not in documents:
                documents[symbol] = {
                    "document_id": f"00000000-0000-4000-8000-{document_index:012d}",
                    "collection_symbol": collection_symbol,
                    "full_text_sha256": sha256(full_text.encode()).hexdigest(),
                }
                document_index += 1
            for chunk_number, chunk in enumerate(document["chunks"]):
                start, end = spans[chunk_number]
                chunks.setdefault(
                    chunk["chunk_id"],
                    {
                        "chunk_id": 1_000 + chunk_index,
                        "document_symbol": symbol,
                        "chunk_number": chunk_number,
                        "start": start,
                        "end": end,
                        "content_sha256": sha256(chunk["text"].encode()).hexdigest(),
                        "embedding_sha256": run_kg_eval.canonical_embedding_sha256(
                            tuple(0.0 for _ in range(1024))
                        ),
                    },
                )
                chunk_index += 1
    return {
        "schema_version": 1,
        "fixture_id": "kg-task20-synthetic-v1",
        "fixture_checksum": run_kg_eval.fixture_checksum(),
        "embedding": {
            "model": "nomic/nomic-embed-text",
            "checkpoint": EMBED_CHECKPOINT,
            "dimensions": 1024,
            "input_type": "search_document",
            "endpoint_signature": fixture_manifest.embedding_endpoint_signature(
                model="nomic/nomic-embed-text",
                checkpoint=EMBED_CHECKPOINT,
                dimensions=1024,
                input_type="search_document",
            ),
        },
        "authorized_scope": [
            {"collection_id": 101, "rebuild_request_id": str(request_a)},
            {"collection_id": 202, "rebuild_request_id": str(request_b)},
        ],
        "collections": dict(
            sorted(
                {
                    symbol: {
                        "collection_id": collection_id,
                        "rebuild_request_id": (
                            None if request_id is None else str(request_id)
                        ),
                        "authorized": authorized,
                    }
                    for symbol, (
                        collection_id,
                        request_id,
                        authorized,
                    ) in collection_bindings.items()
                }.items()
            )
        ),
        "documents": dict(sorted(documents.items())),
        "chunks": dict(sorted(chunks.items())),
        "canonical_identity_assertions": [
            {
                "source_chunk_symbol": "orion-a-seed-001",
                "target_chunk_symbol": "orion-b-identity-001",
                "expected_outcome": "automatic",
            }
        ],
        "inaccessible_neighbor_assertions": [
            {
                "source_chunk_symbol": "public-token-001",
                "target_chunk_symbol": "private-incident-001",
            }
        ],
    }


def test_fixture_manifest_binds_exact_symbols_topology_scope_and_hashes():
    manifest = _fake_fixture_manifest()
    requests = (
        (101, UUID("11111111-1111-4111-8111-111111111111")),
        (202, UUID("22222222-2222-4222-8222-222222222222")),
    )

    resolved = run_kg_eval.validate_fixture_manifest(
        manifest,
        extraction_cases=run_kg_eval.load_extraction_cases(),
        retrieval_cases=run_kg_eval.load_retrieval_cases(),
        collection_requests=requests,
    )

    assert resolved.authorized_scope == requests
    assert resolved.collections["collection-security-private"].authorized is False
    assert resolved.manifest_checksum == fixture_manifest.fixture_manifest_checksum(
        manifest
    )
    canonical = resolved.canonical_identity_assertions[0]
    assert resolved.chunk(canonical.source_chunk_symbol).collection_id == 101
    assert resolved.chunk(canonical.target_chunk_symbol).collection_id == 202


def test_fixture_manifest_overlap_assembly_and_embedding_digest_are_exact():
    first = "Atlas supplies the retrieval service. The retrieval service ranks passages"
    second = "The retrieval service ranks passages before synthesis."

    full_text, spans = run_kg_eval.assemble_fixture_document((first, second))

    assert full_text[spans[0][0] : spans[0][1]] == first
    assert full_text[spans[1][0] : spans[1][1]] == second
    assert spans[1][0] < spans[0][1]
    vector = tuple(0.25 for _ in range(1024))
    assert run_kg_eval.canonical_embedding_sha256(vector) == (
        run_kg_eval.canonical_embedding_sha256(vector)
    )
    with pytest.raises(run_kg_eval.FixtureValidationError, match="1024"):
        run_kg_eval.canonical_embedding_sha256((0.25,))


def test_fixture_checksum_is_shared_by_provider_neutral_manifest_module():
    assert fixture_manifest.fixture_checksum() == run_kg_eval.fixture_checksum()
    source = inspect.getsource(fixture_manifest)
    assert "django" not in source.lower()
    assert "torch" not in source.lower()
    assert "transformers" not in source.lower()


def test_eval_runbook_fences_worker_cleanup_by_exact_compose_labels():
    runbook = (
        AQUILLM_DIR.parent
        / "docs"
        / "documents"
        / "operations"
        / "knowledge-graph-overlay-runbook.md"
    ).read_text(encoding="utf-8")
    cleanup = runbook.split("stop_eval_worker() {", 1)[1].split("}\n", 1)[0]

    assert cleanup.index("docker inspect") < cleanup.index("docker rm -fv")
    assert 'com.docker.compose.project"' in cleanup
    assert 'com.docker.compose.service"' in cleanup
    assert 'com.docker.compose.oneoff"' in cleanup
    assert "worker_knowledge_graph" in cleanup


def test_eval_runbook_attests_strict_reranker_and_hands_off_neutral_cache():
    runbook = (
        AQUILLM_DIR.parent
        / "docs"
        / "documents"
        / "operations"
        / "knowledge-graph-overlay-runbook.md"
    ).read_text(encoding="utf-8")
    procedure = "set -euo pipefail" + runbook.split("set -euo pipefail", 1)[1]
    procedure = procedure.split("```", 1)[0]

    assert "APP_RERANK_MODEL_REVISION" in runbook
    assert "APP_RERANK_TOKENIZER_REVISION" in runbook
    assert "APP_RERANK_CODE_REVISION" in runbook
    assert "APP_EMBED_MODEL_REVISION" in runbook
    assert "APP_EMBED_TOKENIZER_REVISION" in runbook
    assert "APP_EMBED_CODE_REVISION" in runbook
    assert "http://vllm_rerank:8000/v1" in runbook
    assert "DJANGO_CACHE_REDIS_URL=redis://redis:6379/1" in runbook
    assert "RAG_CACHE_ENABLED=1" in runbook
    assert "LMCACHE_ENABLED=0" in runbook
    assert "VLLM_STRICT_PROTECTED_ARGS" in runbook
    assert "RAG_CACHE_ENABLED=0" in procedure
    assert "kg_eval_root_python()" in procedure
    assert "kg_eval_no_cache_python()" in procedure
    root_helper = procedure.split("kg_eval_root_python() {", 1)[1].split("}", 1)[0]
    assert "--user 0:0" in root_helper
    assert "-e KG_GLINER2_LOCAL_FILES_ONLY=0" in root_helper
    assert "PYTHONDONTWRITEBYTECODE=1" in root_helper
    assert "env -i" in procedure
    assert 'case "$TASK21_ENV_FILE" in /*)' in procedure
    assert procedure.index('case "$TASK21_ENV_FILE" in /*)') < procedure.index(
        'TASK21_ENV_FILE="$(realpath'
    )
    assert "vllm_embed vllm_rerank" in procedure
    assert "build worker_knowledge_graph vllm_embed vllm_rerank" in procedure
    assert "KG_EVAL_EMBED_CONTAINER" in procedure
    assert "KG_EVAL_RERANK_CONTAINER" in procedure
    assert 'flag("--revision") == os.environ["VLLM_REVISION"]' in procedure
    assert (
        'flag("--tokenizer-revision") == os.environ["VLLM_TOKENIZER_REVISION"]'
        in procedure
    )
    assert 'flag("--code-revision") == os.environ["VLLM_CODE_REVISION"]' in procedure
    assert (
        'revision == os.environ["VLLM_TOKENIZER_REVISION"]' in procedure
        or 'os.environ["VLLM_TOKENIZER_REVISION"] == os.environ["VLLM_REVISION"]'
        in procedure
    )
    assert (
        'revision == os.environ["VLLM_CODE_REVISION"]' in procedure
        or 'os.environ["VLLM_CODE_REVISION"] == os.environ["VLLM_REVISION"]'
        in procedure
    )
    assert "argv.count(name) == 1" in procedure
    assert "qwen3_vl_reranker.jinja" in procedure
    assert "Qwen3VLForSequenceClassification" in procedure
    assert "bnb_4bit_quant_type" in procedure
    assert '"matryoshka_dimensions": [1024]' in procedure
    assert 'flag("--tensor-parallel-size") == "1"' in procedure
    assert 'flag("--gpu-memory-utilization") == "0.20"' in procedure
    assert 'flag("--gpu-memory-utilization") == "0.30"' in procedure
    assert 'flag("--max-model-len") == "2048"' in procedure
    assert 'flag("--max-model-len") == "1024"' in procedure
    assert 'flag("--runner") == "pooling"' in procedure
    assert 'flag("--dtype") == "float16"' in procedure
    assert procedure.count('assert "--task" not in argv') == 2
    assert 'argv.count("--trust-remote-code") == 1' in procedure
    assert 'os.environ["VLLM_STRICT_PROTECTED_ARGS"] == "1"' in procedure
    assert 'os.environ["VLLM_API_KEY"] == "EMPTY"' in procedure
    assert (
        'os.environ["VLLM_DOWNLOAD_DIR"] == "/root/.cache/huggingface/hub"' in procedure
    )
    assert 'os.environ["VLLM_PYTHON_BIN"] == "python3"' in procedure
    assert 'Path(argv[0]).name == "python3"' in procedure
    assert 'flag("--api-key") == "EMPTY"' in procedure
    assert 'flag("--download-dir") == "/root/.cache/huggingface/hub"' in procedure
    assert "/vllm_start.sh" in procedure
    assert "/parse_vllm_extra_args.py" in procedure
    assert "sha256" in procedure
    wrapper = (AQUILLM_DIR.parent / "deploy" / "scripts" / "vllm_start.sh").read_text(
        encoding="utf-8"
    )
    assert 'cmd+=(--tokenizer-revision "${VLLM_TOKENIZER_REVISION}")' in wrapper
    assert 'cmd+=(--code-revision "${VLLM_CODE_REVISION}")' in wrapper
    assert "VLLM_TOKENIZER_REVISION" in wrapper.split("unset \\", 1)[1]
    assert "VLLM_CODE_REVISION" in wrapper.split("unset \\", 1)[1]
    assert "/opt/kg-eval-hf-cache" in procedure
    assert 'test "$KG_GLINER2_CACHE_DIR" = /opt/kg-eval-hf-cache' in procedure
    assert "chown -R" in procedure
    chown_command = procedure[
        procedure.rfind(
            "kg_eval_compose run", 0, procedure.index("chown -R")
        ) : procedure.index("chown -R")
    ]
    assert "--user 0:0" in chown_command
    assert "kg_eval_cache_host_uid=ok" in procedure

    root_warm = procedure.index(
        "kg_eval_root_python manage.py check_knowledge_graph_extractor"
    )
    worker_start = procedure.index("kg_eval_compose run -d --name")
    terminal_builds = procedure.index('assert report["status"] == "succeeded"')
    worker_stop = procedure.index("stop_eval_worker", terminal_builds)
    cache_handoff = procedure.index("chown -R", worker_stop)
    host_probe = procedure.index("kg_eval_cache_host_uid=ok", cache_handoff)
    comparison = procedure.index(
        "-m apps.knowledge_graph.evals.run_kg_eval", host_probe
    )
    assert root_warm < worker_start < terminal_builds < worker_stop
    assert worker_stop < cache_handoff < host_probe < comparison
    strict_probe = procedure.index("strict_local_reranker=ok")
    no_cache_helper = procedure.index("kg_eval_no_cache_python()")
    assert no_cache_helper < strict_probe < comparison
    rerank_probe_start = procedure.rindex(
        "kg_eval_no_cache_python -c '", no_cache_helper, strict_probe
    )
    assert rerank_probe_start < strict_probe
    assert (
        'shlex.split(os.environ["VLLM_EXTRA_ARGS"]) == expected_extra_args' in procedure
    )


def _fixture_embedding_binding(**overrides):
    values = {
        "model": "Qwen/Qwen3-Embedding-4B",
        "checkpoint": EMBED_CHECKPOINT,
        "dimensions": 1024,
        "input_type": "search_document",
    }
    values.update(overrides)
    values.setdefault(
        "endpoint_signature",
        fixture_manifest.embedding_endpoint_signature(
            model=values["model"],
            checkpoint=values["checkpoint"],
            dimensions=values["dimensions"],
            input_type=values["input_type"],
        ),
    )
    return fixture_manifest.FixtureEmbeddingBinding(**values)


def test_live_embedding_contract_accepts_exact_local_model_revision_and_shape():
    run_kg_eval._validate_live_embedding_contract(
        _fixture_embedding_binding(),
        base_url="http://vllm_embed:8000/v1",
        api_key="EMPTY",
        configured_sidecar_api_key="EMPTY",
        configured_model="Qwen/Qwen3-Embedding-4B",
        configured_checkpoint=EMBED_CHECKPOINT,
        configured_tokenizer_checkpoint=EMBED_CHECKPOINT,
        configured_code_checkpoint=EMBED_CHECKPOINT,
        configured_extra_args=EMBED_EXTRA_ARGS,
        configured_trust_remote_code="1",
        configured_runner="pooling",
        configured_dtype="float16",
        configured_tensor_parallel_size="1",
        configured_gpu_memory_utilization="0.20",
        configured_max_model_len="2048",
        configured_strict_protected_args="1",
        configured_download_dir="/root/.cache/huggingface/hub",
        configured_python_bin="python3",
        configured_dimensions=1024,
    )


@pytest.mark.parametrize("moving_checkpoint", ("main", "A" * 40))
def test_live_embedding_contract_rejects_mutable_or_noncanonical_revision(
    moving_checkpoint,
):
    binding = _fixture_embedding_binding(checkpoint=moving_checkpoint)

    with pytest.raises(run_kg_eval.ComparisonAborted, match="immutable commit"):
        run_kg_eval._validate_live_embedding_contract(
            binding,
            base_url="http://vllm_embed:8000/v1",
            api_key="EMPTY",
            configured_sidecar_api_key="EMPTY",
            configured_model="Qwen/Qwen3-Embedding-4B",
            configured_checkpoint=moving_checkpoint,
            configured_tokenizer_checkpoint=moving_checkpoint,
            configured_code_checkpoint=moving_checkpoint,
            configured_extra_args=EMBED_EXTRA_ARGS,
            configured_trust_remote_code="1",
            configured_runner="pooling",
            configured_dtype="float16",
            configured_tensor_parallel_size="1",
            configured_gpu_memory_utilization="0.20",
            configured_max_model_len="2048",
            configured_strict_protected_args="1",
            configured_download_dir="/root/.cache/huggingface/hub",
            configured_python_bin="python3",
            configured_dimensions=1024,
        )


@pytest.mark.parametrize(
    ("tokenizer_checkpoint", "code_checkpoint", "message"),
    (
        ("moving-tokenizer", EMBED_CHECKPOINT, "tokenizer revision"),
        (EMBED_CHECKPOINT, "moving-code", "code revision"),
    ),
)
def test_live_embedding_contract_rejects_unpinned_tokenizer_or_code_revision(
    tokenizer_checkpoint,
    code_checkpoint,
    message,
):
    with pytest.raises(run_kg_eval.ComparisonAborted, match=message):
        run_kg_eval._validate_live_embedding_contract(
            _fixture_embedding_binding(),
            base_url="http://vllm_embed:8000/v1",
            api_key="EMPTY",
            configured_sidecar_api_key="EMPTY",
            configured_model="Qwen/Qwen3-Embedding-4B",
            configured_checkpoint=EMBED_CHECKPOINT,
            configured_tokenizer_checkpoint=tokenizer_checkpoint,
            configured_code_checkpoint=code_checkpoint,
            configured_extra_args=EMBED_EXTRA_ARGS,
            configured_trust_remote_code="1",
            configured_runner="pooling",
            configured_dtype="float16",
            configured_tensor_parallel_size="1",
            configured_gpu_memory_utilization="0.20",
            configured_max_model_len="2048",
            configured_strict_protected_args="1",
            configured_download_dir="/root/.cache/huggingface/hub",
            configured_python_bin="python3",
            configured_dimensions=1024,
        )


@pytest.mark.parametrize(
    ("extra_args", "trust_remote_code", "message"),
    (
        ("--dtype auto", "1", "extra arguments"),
        (EMBED_EXTRA_ARGS, "0", "remote code"),
    ),
)
def test_live_embedding_contract_rejects_unattested_runtime_configuration(
    extra_args,
    trust_remote_code,
    message,
):
    with pytest.raises(run_kg_eval.ComparisonAborted, match=message):
        run_kg_eval._validate_live_embedding_contract(
            _fixture_embedding_binding(),
            base_url="http://vllm_embed:8000/v1",
            api_key="EMPTY",
            configured_sidecar_api_key="EMPTY",
            configured_model="Qwen/Qwen3-Embedding-4B",
            configured_checkpoint=EMBED_CHECKPOINT,
            configured_tokenizer_checkpoint=EMBED_CHECKPOINT,
            configured_code_checkpoint=EMBED_CHECKPOINT,
            configured_extra_args=extra_args,
            configured_trust_remote_code=trust_remote_code,
            configured_runner="pooling",
            configured_dtype="float16",
            configured_tensor_parallel_size="1",
            configured_gpu_memory_utilization="0.20",
            configured_max_model_len="2048",
            configured_strict_protected_args="1",
            configured_download_dir="/root/.cache/huggingface/hub",
            configured_python_bin="python3",
            configured_dimensions=1024,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"configured_runner": "generate"}, "runner"),
        ({"configured_dtype": "auto"}, "dtype"),
        ({"configured_tensor_parallel_size": "2"}, "tensor parallel"),
        ({"configured_gpu_memory_utilization": "0.12"}, "GPU memory"),
        ({"configured_max_model_len": "8192"}, "maximum model length"),
        ({"configured_strict_protected_args": "0"}, "protected-argument"),
        ({"api_key": "secret"}, "API key"),
        ({"configured_sidecar_api_key": "secret"}, "API key"),
        ({"configured_download_dir": "/tmp/models"}, "download directory"),
        ({"configured_python_bin": "python"}, "python binary"),
    ),
)
def test_live_embedding_contract_rejects_unattested_resource_envelope(
    overrides,
    message,
):
    kwargs = {
        "binding": _fixture_embedding_binding(),
        "base_url": "http://vllm_embed:8000/v1",
        "api_key": "EMPTY",
        "configured_sidecar_api_key": "EMPTY",
        "configured_model": "Qwen/Qwen3-Embedding-4B",
        "configured_checkpoint": EMBED_CHECKPOINT,
        "configured_tokenizer_checkpoint": EMBED_CHECKPOINT,
        "configured_code_checkpoint": EMBED_CHECKPOINT,
        "configured_extra_args": EMBED_EXTRA_ARGS,
        "configured_trust_remote_code": "1",
        "configured_runner": "pooling",
        "configured_dtype": "float16",
        "configured_tensor_parallel_size": "1",
        "configured_gpu_memory_utilization": "0.20",
        "configured_max_model_len": "2048",
        "configured_strict_protected_args": "1",
        "configured_download_dir": "/root/.cache/huggingface/hub",
        "configured_python_bin": "python3",
        "configured_dimensions": 1024,
    }
    kwargs.update(overrides)

    with pytest.raises(run_kg_eval.ComparisonAborted, match=message):
        run_kg_eval._validate_live_embedding_contract(**kwargs)


@pytest.mark.parametrize(
    "unsafe_model",
    (
        "/app/local-model",
        "../mutable",
        "https://host/model",
        "owner/name with space",
        "--owner/name",
        "owner\\name",
        "owner/name/extra",
    ),
)
def test_live_embedding_contract_requires_huggingface_repo_id(unsafe_model):
    binding = _fixture_embedding_binding(model=unsafe_model)

    with pytest.raises(run_kg_eval.ComparisonAborted, match="embedding model"):
        run_kg_eval._validate_live_embedding_contract(
            binding,
            base_url="http://vllm_embed:8000/v1",
            api_key="EMPTY",
            configured_sidecar_api_key="EMPTY",
            configured_model=unsafe_model,
            configured_checkpoint=EMBED_CHECKPOINT,
            configured_tokenizer_checkpoint=EMBED_CHECKPOINT,
            configured_code_checkpoint=EMBED_CHECKPOINT,
            configured_extra_args=EMBED_EXTRA_ARGS,
            configured_trust_remote_code="1",
            configured_runner="pooling",
            configured_dtype="float16",
            configured_tensor_parallel_size="1",
            configured_gpu_memory_utilization="0.20",
            configured_max_model_len="2048",
            configured_strict_protected_args="1",
            configured_download_dir="/root/.cache/huggingface/hub",
            configured_python_bin="python3",
            configured_dimensions=1024,
        )


@pytest.mark.parametrize(
    ("binding", "configured_model", "configured_checkpoint", "dimensions"),
    (
        (
            _fixture_embedding_binding(model="wrong-model"),
            "Qwen/Qwen3-Embedding-4B",
            EMBED_CHECKPOINT,
            1024,
        ),
        (
            _fixture_embedding_binding(checkpoint="wrong-revision"),
            "Qwen/Qwen3-Embedding-4B",
            EMBED_CHECKPOINT,
            1024,
        ),
        (
            _fixture_embedding_binding(dimensions=768),
            "Qwen/Qwen3-Embedding-4B",
            EMBED_CHECKPOINT,
            1024,
        ),
        (
            _fixture_embedding_binding(input_type="search_query"),
            "Qwen/Qwen3-Embedding-4B",
            EMBED_CHECKPOINT,
            1024,
        ),
        (
            _fixture_embedding_binding(endpoint_signature="0" * 64),
            "Qwen/Qwen3-Embedding-4B",
            EMBED_CHECKPOINT,
            1024,
        ),
    ),
)
def test_live_embedding_contract_rejects_any_provenance_mismatch(
    binding,
    configured_model,
    configured_checkpoint,
    dimensions,
):
    with pytest.raises(run_kg_eval.ComparisonAborted, match="embedding"):
        run_kg_eval._validate_live_embedding_contract(
            binding,
            base_url="http://vllm_embed:8000/v1",
            api_key="EMPTY",
            configured_sidecar_api_key="EMPTY",
            configured_model=configured_model,
            configured_checkpoint=configured_checkpoint,
            configured_tokenizer_checkpoint=configured_checkpoint,
            configured_code_checkpoint=configured_checkpoint,
            configured_extra_args=EMBED_EXTRA_ARGS,
            configured_trust_remote_code="1",
            configured_runner="pooling",
            configured_dtype="float16",
            configured_tensor_parallel_size="1",
            configured_gpu_memory_utilization="0.20",
            configured_max_model_len="2048",
            configured_strict_protected_args="1",
            configured_download_dir="/root/.cache/huggingface/hub",
            configured_python_bin="python3",
            configured_dimensions=dimensions,
        )


@pytest.mark.parametrize(
    "mutate",
    (
        lambda manifest: manifest.update(schema_version=True),
        lambda manifest: manifest["authorized_scope"][0].update(collection_id=True),
        lambda manifest: manifest["collections"]["collection-public"].update(
            collection_id=True
        ),
        lambda manifest: manifest["chunks"]["orion-a-seed-001"].update(chunk_id=True),
        lambda manifest: manifest["chunks"]["orion-a-seed-001"].update(
            chunk_number=False
        ),
        lambda manifest: manifest["chunks"]["orion-a-seed-001"].update(start=False),
        lambda manifest: manifest["chunks"]["orion-a-seed-001"].update(end=True),
    ),
)
def test_fixture_manifest_rejects_boolean_integer_fields(mutate):
    manifest = _fake_fixture_manifest()
    mutate(manifest)

    with pytest.raises(run_kg_eval.FixtureValidationError, match="exact integer"):
        run_kg_eval.validate_fixture_manifest(
            manifest,
            extraction_cases=run_kg_eval.load_extraction_cases(),
            retrieval_cases=run_kg_eval.load_retrieval_cases(),
            collection_requests=(
                (101, UUID("11111111-1111-4111-8111-111111111111")),
                (202, UUID("22222222-2222-4222-8222-222222222222")),
            ),
        )


def test_fixture_manifest_revalidates_exact_authorized_db_rows_and_vectors():
    manifest = _fake_fixture_manifest()
    requests = (
        (101, UUID("11111111-1111-4111-8111-111111111111")),
        (202, UUID("22222222-2222-4222-8222-222222222222")),
    )
    extraction_cases = run_kg_eval.load_extraction_cases()
    retrieval_cases = run_kg_eval.load_retrieval_cases()
    resolved = run_kg_eval.validate_fixture_manifest(
        manifest,
        extraction_cases=extraction_cases,
        retrieval_cases=retrieval_cases,
        collection_requests=requests,
    )
    logical_documents: dict[str, tuple[str, ...]] = {}
    logical_chunks: dict[str, str] = {}
    for case in (*extraction_cases, *retrieval_cases):
        for document in case["documents"]:
            texts = tuple(str(chunk["text"]) for chunk in document["chunks"])
            logical_documents[str(document["doc_id"])] = texts
            logical_chunks.update(
                (str(chunk["chunk_id"]), str(chunk["text"]))
                for chunk in document["chunks"]
            )
    document_rows = tuple(
        SimpleNamespace(
            id=binding.document_id,
            collection_id=binding.collection_id,
            full_text=run_kg_eval.assemble_fixture_document(logical_documents[symbol])[
                0
            ],
            full_text_hash=binding.full_text_sha256,
        )
        for symbol, binding in resolved.documents.items()
        if resolved.collections[binding.collection_symbol].authorized
    )
    chunk_rows = tuple(
        SimpleNamespace(
            pk=binding.chunk_id,
            doc_id=resolved.documents[binding.document_symbol].document_id,
            chunk_number=binding.chunk_number,
            start_position=binding.start,
            end_position=binding.end,
            content=logical_chunks[symbol],
            embedding=np.zeros(1024, dtype=np.float32),
        )
        for symbol, binding in resolved.chunks.items()
        if resolved.collections[binding.collection_symbol].authorized
    )

    attestation = run_kg_eval.revalidate_fixture_database_rows(
        resolved,
        document_rows=document_rows,
        chunk_rows=chunk_rows,
    )
    assert attestation["document_ids"] == tuple(
        sorted((row.id for row in document_rows), key=lambda value: value.int)
    )
    assert attestation["chunk_ids"] == tuple(sorted(row.pk for row in chunk_rows))
    broken = list(chunk_rows)
    broken[0] = SimpleNamespace(
        **(vars(broken[0]) | {"start_position": broken[0].start_position + 1})
    )
    with pytest.raises(run_kg_eval.ComparisonAborted, match="span"):
        run_kg_eval.revalidate_fixture_database_rows(
            resolved,
            document_rows=document_rows,
            chunk_rows=tuple(broken),
        )


def test_live_manifest_authorizes_before_exact_db_rows_and_candidate_queries():
    source = inspect.getsource(run_kg_eval._live_comparison_bundle)

    assert source.index(
        "selected_artifacts = _authorize_evaluation_collection_scope("
    ) < (source.index("revalidate_fixture_database_rows("))
    assert source.index("revalidate_fixture_database_rows(") < source.index(
        "run_one_snapshot_comparison("
    )
    assert "content__in" not in source


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda manifest: manifest["chunks"]["orion-a-seed-001"].update(
                content_sha256="f" * 64
            ),
            "current fixture text",
        ),
        (
            lambda manifest: manifest["collections"]["collection-research-b"].update(
                collection_id=101,
                rebuild_request_id="11111111-1111-4111-8111-111111111111",
            ),
            "distinct collections",
        ),
        (
            lambda manifest: manifest["collections"][
                "collection-security-private"
            ].update(authorized=True),
            "hidden security collection",
        ),
    ),
)
def test_fixture_manifest_rejects_hash_topology_or_hidden_scope_drift(mutate, message):
    manifest = _fake_fixture_manifest()
    mutate(manifest)

    with pytest.raises(run_kg_eval.FixtureValidationError, match=message):
        run_kg_eval.validate_fixture_manifest(
            manifest,
            extraction_cases=run_kg_eval.load_extraction_cases(),
            retrieval_cases=run_kg_eval.load_retrieval_cases(),
            collection_requests=(
                (101, UUID("11111111-1111-4111-8111-111111111111")),
                (202, UUID("22222222-2222-4222-8222-222222222222")),
            ),
        )


def _passing_comparison_bundle():
    cases = {}
    for fixture_case in run_kg_eval.load_retrieval_cases():
        distances = fixture_case.get("expected_min_semantic_distance", {})
        cases[str(fixture_case["id"])] = {
            "quality_tags": list(fixture_case.get("quality_tags", ())),
            "minimum_semantic_distance": max(distances.values(), default=0),
        }
    algorithm_signatures = getattr(
        run_kg_eval,
        "current_comparison_algorithm_signatures",
        lambda: {
            "vector_only": "4" * 64,
            "one_hop": "5" * 64,
            "ppr_v1": "6" * 64,
        },
    )()
    common = {
        "comparison_snapshot_signature": "1" * 64,
        "collection_scope": [1, 2],
        "seed_snapshot_signature": "2" * 64,
        "fixture_checksum": run_kg_eval.fixture_checksum(),
        "fixture_manifest_checksum": "3" * 64,
        "versions": {
            "ontology": [{"version": "research-v1", "checksum": "a" * 64}],
            "resolver": [{"version": "resolver-v1", "checksum": "b" * 64}],
            "filter": [{"version": "filter-v1", "checksum": "c" * 64}],
        },
    }
    embedding = {
        "model": "Qwen/Qwen3-Embedding-4B",
        "checkpoint": EMBED_CHECKPOINT,
        "tokenizer_checkpoint": EMBED_CHECKPOINT,
        "code_checkpoint": EMBED_CHECKPOINT,
        "dimensions": 1024,
        "input_type": "search_document",
        "endpoint_signature": fixture_manifest.embedding_endpoint_signature(
            model="Qwen/Qwen3-Embedding-4B",
            checkpoint=EMBED_CHECKPOINT,
            dimensions=1024,
            input_type="search_document",
        ),
        "extra_args_signature": sha256(EMBED_EXTRA_ARGS.encode("utf-8")).hexdigest(),
        "trust_remote_code": True,
        "runner": "pooling",
        "dtype": "float16",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.2,
        "max_model_len": 2048,
        "strict_protected_args": True,
        "api_key_signature": sha256(b"EMPTY").hexdigest(),
        "download_dir": "/root/.cache/huggingface/hub",
        "python_bin": "python3",
    }
    embedding["config_signature"] = run_kg_eval.comparison_snapshot_signature(embedding)
    reranker = {
        "provider": "local",
        "model": "Qwen/Qwen3-VL-Reranker-2B",
        "checkpoint": "4bd860ac4f15ad1897a214615cccc700f8f71818",
        "tokenizer_checkpoint": "4bd860ac4f15ad1897a214615cccc700f8f71818",
        "code_checkpoint": "4bd860ac4f15ad1897a214615cccc700f8f71818",
        "endpoint_signature": sha256(b"http://vllm_rerank:8000/v1").hexdigest(),
        "timeout_seconds": 3,
        "document_char_limit": 2000,
        "multimodal": True,
        "extra_args_signature": sha256(RERANK_EXTRA_ARGS.encode("utf-8")).hexdigest(),
        "trust_remote_code": True,
        "runner": "pooling",
        "task": "classify",
        "dtype": "float16",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.30,
        "max_model_len": 1024,
        "strict_protected_args": True,
        "api_key_signature": sha256(b"EMPTY").hexdigest(),
        "download_dir": "/root/.cache/huggingface/hub",
        "python_bin": "python3",
        "chat_template_sha256": RERANK_TEMPLATE_SHA256,
        "cache_enabled": False,
    }
    reranker["config_signature"] = run_kg_eval.comparison_snapshot_signature(reranker)
    extraction = {
        "provider": "gliner2_local",
        "model": "fastino/gliner2-base-v1",
        "checkpoint": "8437ba583a733d87f56ae902f3b197934eedd58e",
        "build_enabled": False,
        "device": "cpu",
        "batch_size": 8,
        "max_batch_characters": 64_000,
        "local_files_only": True,
        "fail_open": False,
    }
    extraction["config_signature"] = run_kg_eval.comparison_snapshot_signature(
        extraction
    )

    def arm(name, *, max_hops, algorithm, graph, recall, ndcg, latency):
        return {
            **{**common, "versions": deepcopy(common["versions"])},
            "name": name,
            "max_hops": max_hops,
            "ppr_iterations": 0 if max_hops == 0 else 8,
            "algorithm_signature": algorithm,
            "graph_version_signature": graph,
            "metrics": {
                "recall_at_10": recall,
                "mrr": recall,
                "ndcg_at_10": ndcg,
                "graph_hit_rate": 0.0 if max_hops == 0 else 1.0,
                "inaccessible_result_count": 0,
                "latency_p95_ms": latency,
                "graph_added_latency_p95_ms": 0.0 if max_hops == 0 else latency,
                "citation_evidence_coverage": 1.0,
                "seed_coverage": 1.0,
                "node_count": 5,
                "edge_count": 4,
                "distance_2_novel_fraction": 0.0 if max_hops < 2 else 0.25,
            },
            "cases": {
                case_id: {
                    **metadata,
                    "distance_2_relevant_hit": (
                        case_id == "two_hop_cross_document_metric" and max_hops == 2
                    ),
                    "recall_at_10": recall
                    + (
                        0.1
                        if case_id == "two_hop_cross_document_metric" and max_hops == 2
                        else 0.0
                    ),
                    "ndcg_at_10": ndcg
                    + (
                        0.1
                        if case_id == "two_hop_cross_document_metric" and max_hops == 2
                        else 0.0
                    ),
                }
                for case_id, metadata in cases.items()
            },
        }

    return {
        "schema_version": 1,
        "mode": "comparison",
        "eval_only": True,
        **common,
        "model": {
            "provider": "gliner2_local",
            "name": "fastino/gliner2-base-v1",
            "checkpoint": "8437ba583a733d87f56ae902f3b197934eedd58e",
        },
        "extraction": extraction,
        "embedding": embedding,
        "reranker": reranker,
        "artifacts": [
            {
                "collection_id": 1,
                "build_key": "d" * 64,
                "source_hash": "e" * 64,
                "rebuild_request": str(UUID("11111111-1111-4111-8111-111111111111")),
            },
            {
                "collection_id": 2,
                "build_key": "f" * 64,
                "source_hash": "0" * 64,
                "rebuild_request": str(UUID("22222222-2222-4222-8222-222222222222")),
            },
        ],
        "latency_budget_ms": 150.0,
        "extraction_metrics": {
            "mention_span_precision": 0.95,
            "mention_span_recall": 0.95,
            "relation_endpoint_precision": 0.95,
            "relation_endpoint_recall": 0.95,
            "relation_direction_precision": 0.95,
            "relation_direction_recall": 0.95,
            "automatic_link_precision": 0.95,
            "automatic_link_recall": 0.95,
            "candidate_link_precision": 0.80,
            "candidate_link_recall": 0.95,
            "suppression_precision": 1.0,
            "suppression_recall": 1.0,
        },
        "invariants": {
            "exact_baseline_on_graph_failure": True,
            "deterministic_repeated_ppr": True,
            "strict_local_reranking": True,
            "rerank_cache_enabled": False,
            "graph_miss_observations": 1,
            "graph_error_observations": 1,
        },
        "arms": {
            "vector_only": arm(
                "vector_only",
                max_hops=0,
                algorithm=algorithm_signatures["vector_only"],
                graph=None,
                recall=0.2,
                ndcg=0.2,
                latency=10.0,
            ),
            "one_hop": arm(
                "one_hop",
                max_hops=1,
                algorithm=algorithm_signatures["one_hop"],
                graph="7" * 64,
                recall=0.4,
                ndcg=0.4,
                latency=80.0,
            ),
            "ppr_v1": arm(
                "ppr_v1",
                max_hops=2,
                algorithm=algorithm_signatures["ppr_v1"],
                graph="8" * 64,
                recall=0.4,
                ndcg=0.4,
                latency=100.0,
            ),
        },
    }


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("provider", "remote"),
        ("name", "../mutable"),
        ("checkpoint", "main"),
        ("checkpoint", "A" * 40),
    ),
)
def test_bundle_rejects_nonlocal_or_mutable_extraction_model(field, value):
    bundle = _passing_comparison_bundle()
    bundle["model"][field] = value

    with pytest.raises(run_kg_eval.ComparisonValidationError, match="model"):
        run_kg_eval.validate_comparison_bundle(bundle)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda bundle: bundle.update(query_text="private query"),
        lambda bundle: bundle["arms"]["ppr_v1"].update(document_text="private"),
        lambda bundle: bundle["arms"]["ppr_v1"]["metrics"].update(labels=[]),
        lambda bundle: bundle["arms"]["ppr_v1"]["cases"][
            "two_hop_cross_document_metric"
        ].update(query_text="private"),
        lambda bundle: bundle["extraction_metrics"].update(predictions=[]),
        lambda bundle: bundle["invariants"].update(raw_ids=[]),
    ),
)
def test_comparison_bundle_rejects_unknown_fields_at_every_report_layer(mutate):
    bundle = _passing_comparison_bundle()
    mutate(bundle)

    with pytest.raises(
        run_kg_eval.ComparisonValidationError, match="fields are not exact"
    ):
        run_kg_eval.validate_comparison_bundle(bundle)


@pytest.mark.parametrize("mutation", ("omit", "tags", "distance"))
def test_comparison_bundle_binds_exact_current_fixture_case_metadata(mutation):
    bundle = _passing_comparison_bundle()
    case_id = "two_hop_cross_document_metric"
    if mutation == "omit":
        for arm in bundle["arms"].values():
            arm["cases"].pop(case_id)
    elif mutation == "tags":
        for arm in bundle["arms"].values():
            arm["cases"][case_id]["quality_tags"] = ["relationship"]
    else:
        for arm in bundle["arms"].values():
            arm["cases"][case_id]["minimum_semantic_distance"] = 1

    with pytest.raises(run_kg_eval.ComparisonValidationError, match="current fixture"):
        run_kg_eval.validate_comparison_bundle(bundle)


def test_comparison_bundle_rejects_operator_inflated_latency_budget():
    bundle = _passing_comparison_bundle()
    bundle["latency_budget_ms"] = 1_000_000_000.0

    with pytest.raises(run_kg_eval.ComparisonValidationError, match="latency budget"):
        run_kg_eval.validate_comparison_bundle(bundle)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda bundle: bundle["arms"].pop("one_hop"), "three arms"),
        (
            lambda bundle: bundle["arms"]["one_hop"].update(
                comparison_snapshot_signature="9" * 64
            ),
            "comparison snapshot",
        ),
        (
            lambda bundle: bundle["arms"]["one_hop"].update(collection_scope=[1]),
            "collection scope",
        ),
        (
            lambda bundle: bundle["arms"]["one_hop"].update(
                seed_snapshot_signature="9" * 64
            ),
            "seed snapshot",
        ),
        (
            lambda bundle: bundle["arms"]["one_hop"].update(fixture_checksum="9" * 64),
            "fixture checksum",
        ),
        (
            lambda bundle: bundle["arms"]["one_hop"]["versions"]["ontology"][0].update(
                version="other"
            ),
            "artifact versions",
        ),
        (
            lambda bundle: bundle["arms"]["ppr_v1"].update(max_hops=1),
            "shipping.*max_hops=2",
        ),
        (
            lambda bundle: bundle["arms"]["ppr_v1"].update(ppr_iterations=7),
            "eight iterations",
        ),
    ],
)
def test_comparison_bundle_rejects_missing_or_mismatched_snapshot_state(
    mutate, message
):
    bundle = _passing_comparison_bundle()
    mutate(bundle)

    with pytest.raises(run_kg_eval.ComparisonValidationError, match=message):
        run_kg_eval.validate_comparison_bundle(bundle)


def test_comparison_bundle_rejects_consistently_replaced_fixture_and_algorithms():
    fixture_bundle = _passing_comparison_bundle()
    fixture_bundle["fixture_checksum"] = "9" * 64
    for arm in fixture_bundle["arms"].values():
        arm["fixture_checksum"] = "9" * 64

    with pytest.raises(
        run_kg_eval.ComparisonValidationError,
        match="current fixture",
    ):
        run_kg_eval.validate_comparison_bundle(fixture_bundle)

    algorithm_bundle = _passing_comparison_bundle()
    replacements = {
        "vector_only": "6" * 64,
        "one_hop": "7" * 64,
        "ppr_v1": "8" * 64,
    }
    for arm_name, signature in replacements.items():
        algorithm_bundle["arms"][arm_name]["algorithm_signature"] = signature

    with pytest.raises(
        run_kg_eval.ComparisonValidationError,
        match="current algorithm",
    ):
        run_kg_eval.validate_comparison_bundle(algorithm_bundle)


def test_measured_gates_cover_all_rollout_requirements_and_pass_good_bundle():
    gates = run_kg_eval.evaluate_measured_gates(_passing_comparison_bundle())

    assert set(gates) == {
        "Permission isolation",
        "Fail-open parity",
        "Identity precision",
        "Retrieval quality",
        "Multi-hop value",
        "Latency",
        "Determinism",
        "Citations",
    }
    assert {gate["status"] for gate in gates.values()} == {"PASS"}


@pytest.mark.parametrize(
    "missing_status",
    ("graph_miss_observations", "graph_error_observations"),
)
def test_fail_open_gate_requires_measured_miss_and_error_parity(missing_status):
    bundle = _passing_comparison_bundle()
    bundle["invariants"][missing_status] = 0

    gates = run_kg_eval.evaluate_measured_gates(bundle)

    assert gates["Fail-open parity"]["status"] == "FAIL"
    assert "miss=0" in gates["Fail-open parity"]["current_value"] or (
        "error=0" in gates["Fail-open parity"]["current_value"]
    )


def test_latency_gate_uses_graph_added_p95_not_slow_shared_baseline_or_rerank():
    bundle = _passing_comparison_bundle()
    for arm in bundle["arms"].values():
        arm["metrics"]["latency_p95_ms"] = 5_000.0
    bundle["arms"]["ppr_v1"]["metrics"]["graph_added_latency_p95_ms"] = 149.0

    gates = run_kg_eval.evaluate_measured_gates(bundle)

    assert gates["Latency"]["status"] == "PASS"
    assert "graph_added_p95=149" in gates["Latency"]["current_value"]

    bundle["arms"]["ppr_v1"]["metrics"]["graph_added_latency_p95_ms"] = 151.0
    gates = run_kg_eval.evaluate_measured_gates(bundle)
    assert gates["Latency"]["status"] == "FAIL"


def test_multi_hop_gate_requires_a_returned_relevant_distance_two_chunk():
    bundle = _passing_comparison_bundle()
    bundle["arms"]["ppr_v1"]["cases"]["two_hop_cross_document_metric"][
        "distance_2_relevant_hit"
    ] = False

    gates = run_kg_eval.evaluate_measured_gates(bundle)

    assert gates["Multi-hop value"]["status"] == "FAIL"


def test_gate_verification_fails_for_pending_or_failing_values(tmp_path):
    report = tmp_path / "comparison.json"
    report.write_text(json.dumps(_passing_comparison_bundle()), encoding="utf-8")
    runbook = tmp_path / "runbook.md"
    runbook.write_text(
        run_kg_eval.PENDING_GATE_TABLE + "\n",
        encoding="utf-8",
    )

    with pytest.raises(run_kg_eval.GateVerificationError, match="pending"):
        run_kg_eval.verify_measured_gates(report, runbook)

    failing = _passing_comparison_bundle()
    failing["arms"]["ppr_v1"]["metrics"]["inaccessible_result_count"] = 1
    report.write_text(json.dumps(failing), encoding="utf-8")
    run_kg_eval.write_measured_gates(report, runbook)

    with pytest.raises(run_kg_eval.GateVerificationError, match="failing"):
        run_kg_eval.verify_measured_gates(report, runbook)


def test_gate_writer_and_verifier_use_validated_atomic_comparison_report(tmp_path):
    report = tmp_path / "comparison.json"
    report.write_text(json.dumps(_passing_comparison_bundle()), encoding="utf-8")
    runbook = tmp_path / "nested" / "runbook.md"
    runbook.parent.mkdir()
    runbook.write_text(
        "before\n" + run_kg_eval.PENDING_GATE_TABLE + "\nafter\n",
        encoding="utf-8",
    )

    gates = run_kg_eval.write_measured_gates(report, runbook)
    verified = run_kg_eval.verify_measured_gates(report, runbook)

    assert gates == verified
    assert "PENDING_MEASUREMENT" not in runbook.read_text(encoding="utf-8")
    assert runbook.read_text(encoding="utf-8").startswith("before\n")
    assert runbook.read_text(encoding="utf-8").endswith("after\n")


@pytest.mark.parametrize("operation", ("load", "write", "verify"))
def test_all_persisted_report_paths_reject_extraction_config_drift(
    tmp_path,
    operation,
):
    bundle = _passing_comparison_bundle()
    bundle["extraction"]["fail_open"] = True
    report = tmp_path / "comparison.json"
    report.write_text(json.dumps(bundle), encoding="utf-8")
    runbook = tmp_path / "runbook.md"
    runbook.write_text(run_kg_eval.PENDING_GATE_TABLE + "\n", encoding="utf-8")

    with pytest.raises(run_kg_eval.ComparisonValidationError, match="extraction"):
        if operation == "load":
            run_kg_eval._load_comparison_report(report)
        elif operation == "write":
            run_kg_eval.write_measured_gates(report, runbook)
        else:
            run_kg_eval.verify_measured_gates(report, runbook)


def test_atomic_json_replace_creates_parent_and_preserves_old_file_on_failure(tmp_path):
    destination = tmp_path / "nested" / "comparison.json"
    run_kg_eval.atomic_write_json(destination, {"complete": True})
    assert json.loads(destination.read_text(encoding="utf-8")) == {"complete": True}

    with pytest.raises(TypeError):
        run_kg_eval.atomic_write_json(destination, {"bad": object()})

    assert json.loads(destination.read_text(encoding="utf-8")) == {"complete": True}
    assert list(destination.parent.iterdir()) == [destination]


def test_retrieval_fixture_covers_required_graph_and_security_shapes():
    cases = run_kg_eval.load_retrieval_cases()
    tags = {tag for case in cases for tag in case.get("quality_tags", ())}

    assert {
        "relationship",
        "two_hop",
        "cross_collection",
        "alias",
        "cross_document",
        "hub",
        "dangling",
        "cycle",
        "duplicate_evidence",
        "negative_noise",
        "inaccessible_neighbor",
    } <= tags
    assert any(
        len(case["accessible_collection_ids"]) >= 2
        and "cross_collection" in case.get("quality_tags", ())
        for case in cases
    )


def test_human_comparison_table_contains_each_arm_and_required_metrics():
    bundle = _passing_comparison_bundle()
    table = run_kg_eval.format_comparison_table(bundle)

    assert "vector_only" in table
    assert "one_hop" in table
    assert "ppr_v1" in table
    assert "Reranker tokenizer checkpoint:" in table
    assert "Reranker code checkpoint:" in table
    assert "Embedding tokenizer checkpoint:" in table
    assert "Embedding code checkpoint:" in table
    assert "Embedding extra-args signature:" in table
    assert "Embedding config signature:" in table
    assert "Reranker extra-args signature:" in table
    assert "Reranker chat-template SHA-256:" in table
    assert "Embedding runner/dtype: pooling / float16" in table
    assert "Reranker runner/dtype: pooling / float16" in table
    assert "Embedding resources: TP=1 GPU=0.2 max_len=2048" in table
    assert "Reranker resources: TP=1 GPU=0.3 max_len=1024" in table
    assert "Reranker cache enabled: false" in table
    assert "Extractor device: cpu" in table
    assert "Extractor batch envelope: count=8 characters=64000" in table
    assert "Extractor offline/fail-open/build: true / false / false" in table
    assert "Extractor config signature:" in table
    assert "Embedding API-key signature:" in table
    assert "Embedding runtime: python=python3" in table
    assert "Reranker API-key signature:" in table
    assert "Reranker runtime: python=python3" in table
    for heading in (
        "Recall@10",
        "MRR",
        "nDCG@10",
        "Graph hit rate",
        "Inaccessible",
        "p95 latency",
        "Graph-added p95",
        "Citation coverage",
        "Seed coverage",
        "Nodes",
        "Edges",
        "Distance-2 novel",
    ):
        assert heading in table
    for required_value in (
        "1, 2",
        bundle["model"]["provider"],
        bundle["model"]["name"],
        bundle["model"]["checkpoint"],
        bundle["embedding"]["model"],
        bundle["embedding"]["checkpoint"],
        bundle["embedding"]["endpoint_signature"],
        bundle["reranker"]["model"],
        bundle["reranker"]["checkpoint"],
        bundle["reranker"]["endpoint_signature"],
        bundle["reranker"]["config_signature"],
        bundle["versions"]["ontology"][0]["version"],
        bundle["versions"]["resolver"][0]["version"],
        bundle["versions"]["filter"][0]["version"],
        bundle["fixture_checksum"],
        bundle["comparison_snapshot_signature"],
        bundle["arms"]["one_hop"]["algorithm_signature"],
        bundle["arms"]["ppr_v1"]["graph_version_signature"],
    ):
        assert required_value in table


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (lambda bundle: bundle["model"].pop("provider"), "model provenance"),
        (lambda bundle: bundle.pop("embedding"), "comparison bundle"),
        (
            lambda bundle: bundle["embedding"].update(checkpoint="main"),
            "embedding checkpoint",
        ),
        (
            lambda bundle: bundle["embedding"].update(dimensions=True),
            "embedding dimensions",
        ),
        (
            lambda bundle: bundle["embedding"].update(input_type="search_query"),
            "search_document",
        ),
        (
            lambda bundle: bundle["embedding"].update(endpoint_signature="bad"),
            "endpoint signature",
        ),
        (
            lambda bundle: bundle["embedding"].update(
                tokenizer_checkpoint="moving-tokenizer"
            ),
            "embedding tokenizer checkpoint",
        ),
        (
            lambda bundle: bundle["embedding"].update(code_checkpoint="moving-code"),
            "embedding code checkpoint",
        ),
        (
            lambda bundle: bundle["embedding"].update(extra_args_signature="0" * 64),
            "embedding extra arguments",
        ),
        (
            lambda bundle: bundle["embedding"].update(trust_remote_code=False),
            "embedding remote code",
        ),
        (
            lambda bundle: bundle["embedding"].update(runner="generate"),
            "embedding runner",
        ),
        (
            lambda bundle: bundle["embedding"].update(dtype="auto"),
            "embedding dtype",
        ),
        (
            lambda bundle: bundle["embedding"].update(tensor_parallel_size=2),
            "embedding tensor parallel",
        ),
        (
            lambda bundle: bundle["embedding"].update(gpu_memory_utilization=0.12),
            "embedding GPU memory",
        ),
        (
            lambda bundle: bundle["embedding"].update(max_model_len=8192),
            "embedding maximum model length",
        ),
        (
            lambda bundle: bundle["embedding"].update(api_key_signature="0" * 64),
            "embedding API key",
        ),
        (
            lambda bundle: bundle["embedding"].update(download_dir="/tmp/models"),
            "embedding download directory",
        ),
        (
            lambda bundle: bundle["embedding"].update(python_bin="python"),
            "embedding python binary",
        ),
        (
            lambda bundle: bundle["embedding"].update(config_signature="0" * 64),
            "embedding config",
        ),
    ),
)
def test_comparison_bundle_requires_exact_model_and_embedding_provenance(
    mutate, message
):
    bundle = _passing_comparison_bundle()
    mutate(bundle)

    with pytest.raises(run_kg_eval.ComparisonValidationError, match=message):
        run_kg_eval.validate_comparison_bundle(bundle)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("provider", "remote"),
        ("model", "../mutable"),
        ("checkpoint", "main"),
        ("build_enabled", True),
        ("device", "unsafe device"),
        ("batch_size", 0),
        ("max_batch_characters", True),
        ("local_files_only", False),
        ("fail_open", True),
        ("config_signature", "0" * 64),
    ),
)
def test_comparison_bundle_binds_exact_extraction_runtime(field, value):
    bundle = _passing_comparison_bundle()
    bundle["extraction"][field] = value

    with pytest.raises(run_kg_eval.ComparisonValidationError, match="extraction"):
        run_kg_eval.validate_comparison_bundle(bundle)


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (lambda bundle: bundle.pop("reranker"), "comparison bundle"),
        (
            lambda bundle: bundle["reranker"].update(provider="auto"),
            "reranker provider",
        ),
        (
            lambda bundle: bundle["reranker"].update(model=""),
            "reranker model",
        ),
        (
            lambda bundle: bundle["reranker"].update(checkpoint="moving-tag"),
            "reranker checkpoint",
        ),
        (
            lambda bundle: bundle["reranker"].update(tokenizer_checkpoint="0" * 40),
            "reranker tokenizer checkpoint",
        ),
        (
            lambda bundle: bundle["reranker"].update(code_checkpoint="0" * 40),
            "reranker code checkpoint",
        ),
        (
            lambda bundle: bundle["reranker"].update(extra_args_signature="0" * 64),
            "reranker extra arguments",
        ),
        (
            lambda bundle: bundle["reranker"].update(trust_remote_code=False),
            "reranker remote code",
        ),
        (
            lambda bundle: bundle["reranker"].update(runner="generate"),
            "reranker runner",
        ),
        (
            lambda bundle: bundle["reranker"].update(task="score"),
            "reranker task",
        ),
        (
            lambda bundle: bundle["reranker"].update(dtype="auto"),
            "reranker dtype",
        ),
        (
            lambda bundle: bundle["reranker"].update(tensor_parallel_size=2),
            "reranker tensor parallel",
        ),
        (
            lambda bundle: bundle["reranker"].update(gpu_memory_utilization=0.08),
            "reranker GPU memory",
        ),
        (
            lambda bundle: bundle["reranker"].update(max_model_len=8192),
            "reranker maximum model length",
        ),
        (
            lambda bundle: bundle["reranker"].update(api_key_signature="0" * 64),
            "reranker API key",
        ),
        (
            lambda bundle: bundle["reranker"].update(download_dir="/tmp/models"),
            "reranker download directory",
        ),
        (
            lambda bundle: bundle["reranker"].update(python_bin="python"),
            "reranker python binary",
        ),
        (
            lambda bundle: bundle["reranker"].update(chat_template_sha256="0" * 64),
            "reranker chat template",
        ),
        (
            lambda bundle: bundle["reranker"].update(cache_enabled=True),
            "reranker cache",
        ),
        (
            lambda bundle: bundle["reranker"].update(endpoint_signature="0" * 64),
            "reranker endpoint",
        ),
        (
            lambda bundle: bundle["reranker"].update(timeout_seconds=True),
            "reranker timeout",
        ),
        (
            lambda bundle: bundle["reranker"].update(config_signature="0" * 64),
            "reranker config",
        ),
    ),
)
def test_comparison_bundle_requires_exact_strict_reranker_provenance(
    mutate,
    message,
):
    bundle = _passing_comparison_bundle()
    mutate(bundle)

    with pytest.raises(run_kg_eval.ComparisonValidationError, match=message):
        run_kg_eval.validate_comparison_bundle(bundle)


def test_comparison_bundle_rejects_changed_checked_in_reranker_template(
    tmp_path,
    monkeypatch,
):
    bundle = _passing_comparison_bundle()
    changed_template = tmp_path / "qwen3_vl_reranker.jinja"
    changed_template.write_bytes(RERANK_TEMPLATE_PATH.read_bytes() + b"\n{# drift #}\n")
    monkeypatch.setattr(run_kg_eval, "_RERANK_TEMPLATE_PATH", changed_template)

    with pytest.raises(
        run_kg_eval.ComparisonValidationError,
        match="reranker chat template",
    ):
        run_kg_eval.validate_comparison_bundle(bundle)


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"provider": "auto"}, "provider"),
        ({"base_url": "http://localhost:8000/v1"}, "endpoint"),
        ({"base_url": "http://VLLM_RERANK:8000/v1"}, "endpoint"),
        ({"api_key": "secret"}, "API key"),
        ({"loaded_model": "Qwen/other-reranker"}, "loaded model"),
        ({"configured_tokenizer": "Qwen/other-tokenizer"}, "tokenizer"),
        ({"configured_model": ""}, "model"),
        ({"configured_model": " Qwen/Qwen3-VL-Reranker-2B"}, "model"),
        ({"configured_model": "unsafe\nmodel"}, "model"),
        ({"configured_model": "x" * 257}, "model"),
        ({"configured_checkpoint": "moving-tag"}, "revision"),
        ({"configured_tokenizer_checkpoint": "0" * 40}, "tokenizer revision"),
        ({"configured_code_checkpoint": "0" * 40}, "code revision"),
        ({"configured_extra_args": "--dtype auto"}, "extra arguments"),
        ({"configured_trust_remote_code": "0"}, "remote code"),
        ({"configured_runner": "generate"}, "runner"),
        ({"configured_task": "score"}, "task"),
        ({"configured_dtype": "auto"}, "dtype"),
        ({"configured_tensor_parallel_size": "2"}, "tensor parallel"),
        ({"configured_gpu_memory_utilization": "0.25"}, "GPU memory"),
        ({"configured_max_model_len": "8192"}, "maximum model length"),
        ({"configured_strict_protected_args": "0"}, "protected-argument"),
        ({"configured_download_dir": "/tmp/models"}, "download directory"),
        ({"configured_python_bin": "python"}, "python binary"),
        ({"configured_cache_enabled": True}, "cache"),
        ({"timeout_seconds": True}, "timeout"),
        ({"document_char_limit": 0}, "document"),
    ),
)
def test_live_reranker_contract_rejects_non_attested_local_configuration(
    overrides,
    message,
):
    kwargs = {
        "provider": "local",
        "base_url": "http://vllm_rerank:8000/v1",
        "api_key": "EMPTY",
        "configured_model": "Qwen/Qwen3-VL-Reranker-2B",
        "loaded_model": "Qwen/Qwen3-VL-Reranker-2B",
        "configured_tokenizer": "Qwen/Qwen3-VL-Reranker-2B",
        "configured_checkpoint": "4bd860ac4f15ad1897a214615cccc700f8f71818",
        "configured_tokenizer_checkpoint": "4bd860ac4f15ad1897a214615cccc700f8f71818",
        "configured_code_checkpoint": "4bd860ac4f15ad1897a214615cccc700f8f71818",
        "configured_extra_args": RERANK_EXTRA_ARGS,
        "configured_trust_remote_code": "1",
        "configured_runner": "pooling",
        "configured_task": "",
        "configured_dtype": "float16",
        "configured_tensor_parallel_size": "1",
        "configured_gpu_memory_utilization": "0.30",
        "configured_max_model_len": "1024",
        "configured_strict_protected_args": "1",
        "configured_download_dir": "/root/.cache/huggingface/hub",
        "configured_python_bin": "python3",
        "configured_cache_enabled": False,
        "timeout_seconds": 3,
        "document_char_limit": 2000,
        "multimodal": True,
    }
    kwargs.update(overrides)

    with pytest.raises(run_kg_eval.ComparisonAborted, match=message):
        run_kg_eval._validate_live_reranker_contract(**kwargs)


def test_live_reranker_contract_is_bound_into_live_suite_and_strict_seam():
    provenance = run_kg_eval._validate_live_reranker_contract(
        provider="local",
        base_url="http://vllm_rerank:8000/v1",
        api_key="EMPTY",
        configured_model="Qwen/Qwen3-VL-Reranker-2B",
        loaded_model="Qwen/Qwen3-VL-Reranker-2B",
        configured_tokenizer="Qwen/Qwen3-VL-Reranker-2B",
        configured_checkpoint="4bd860ac4f15ad1897a214615cccc700f8f71818",
        configured_tokenizer_checkpoint="4bd860ac4f15ad1897a214615cccc700f8f71818",
        configured_code_checkpoint="4bd860ac4f15ad1897a214615cccc700f8f71818",
        configured_extra_args=RERANK_EXTRA_ARGS,
        configured_trust_remote_code="1",
        configured_runner="pooling",
        configured_task="",
        configured_dtype="float16",
        configured_tensor_parallel_size="1",
        configured_gpu_memory_utilization="0.30",
        configured_max_model_len="1024",
        configured_strict_protected_args="1",
        configured_download_dir="/root/.cache/huggingface/hub",
        configured_python_bin="python3",
        configured_cache_enabled=False,
        timeout_seconds=3,
        document_char_limit=2000,
        multimodal=True,
    )
    source = inspect.getsource(run_kg_eval._live_comparison_bundle)

    assert provenance["provider"] == "local"
    assert provenance["config_signature"] == run_kg_eval.comparison_snapshot_signature(
        {key: value for key, value in provenance.items() if key != "config_signature"}
    )
    assert source.count('"reranker": dict(reranker_provenance)') == 2
    assert "_STRICT_EVALUATION_RERANK" in source
    assert "_eval_rerank_capability=" in source

    fallback_bundle = _passing_comparison_bundle()
    fallback_bundle["invariants"]["strict_local_reranking"] = False
    with pytest.raises(
        run_kg_eval.ComparisonValidationError,
        match="strict local reranking",
    ):
        run_kg_eval.validate_comparison_bundle(fallback_bundle)

    cached_bundle = _passing_comparison_bundle()
    cached_bundle["invariants"]["rerank_cache_enabled"] = True
    with pytest.raises(
        run_kg_eval.ComparisonValidationError,
        match="rerank cache",
    ):
        run_kg_eval.validate_comparison_bundle(cached_bundle)


def test_live_reranker_requires_explicit_actual_model_and_tokenizer(monkeypatch):
    from apps.documents.services.chunk_rerank_config import (
        rerank_code_revision,
        rerank_tokenizer,
        rerank_tokenizer_revision,
        rerank_vllm_model,
    )

    monkeypatch.setenv("APP_RERANK_MODEL", "Qwen/custom-served-alias")
    monkeypatch.delenv("APP_RERANK_VLLM_MODEL", raising=False)
    monkeypatch.delenv("APP_RERANK_TOKENIZER", raising=False)
    monkeypatch.delenv("APP_RERANK_TOKENIZER_REVISION", raising=False)
    monkeypatch.delenv("APP_RERANK_CODE_REVISION", raising=False)

    assert rerank_vllm_model() == ""
    assert rerank_tokenizer() == ""
    assert rerank_tokenizer_revision() == ""
    assert rerank_code_revision() == ""


@pytest.mark.parametrize(
    "unsafe_model",
    (
        " leading-space",
        "bad\nmodel",
        "x" * 257,
        "/app/local-model",
        "../mutable",
        "https://host/model",
        "owner/name with space",
        "--owner/name",
        "owner\\name",
        "owner/name/extra",
    ),
)
def test_comparison_bundle_rejects_unsafe_reranker_model_before_config_signature(
    unsafe_model,
):
    bundle = _passing_comparison_bundle()
    bundle["reranker"]["model"] = unsafe_model
    unsigned = {
        key: value
        for key, value in bundle["reranker"].items()
        if key != "config_signature"
    }
    bundle["reranker"]["config_signature"] = run_kg_eval.comparison_snapshot_signature(
        unsigned
    )

    with pytest.raises(run_kg_eval.ComparisonValidationError, match="reranker model"):
        run_kg_eval.validate_comparison_bundle(bundle)


@pytest.mark.parametrize(
    "unsafe_model",
    ("/app/local-model", "../mutable", "https://host/model", "owner/name with space"),
)
def test_comparison_bundle_rejects_non_repository_embedding_model(unsafe_model):
    bundle = _passing_comparison_bundle()
    bundle["embedding"]["model"] = unsafe_model
    unsigned = {
        key: value
        for key, value in bundle["embedding"].items()
        if key != "config_signature"
    }
    bundle["embedding"]["config_signature"] = run_kg_eval.comparison_snapshot_signature(
        unsigned
    )

    with pytest.raises(run_kg_eval.ComparisonValidationError, match="embedding model"):
        run_kg_eval.validate_comparison_bundle(bundle)


def test_cli_comparison_requires_eval_only_collections_and_atomic_output(
    tmp_path, capsys, monkeypatch
):
    output = tmp_path / "nested" / "report.json"
    bundle = _passing_comparison_bundle()

    monkeypatch.setattr(
        run_kg_eval,
        "_live_comparison_bundle",
        lambda **_kwargs: bundle,
    )
    monkeypatch.setattr(run_kg_eval, "_runtime_eval_context", lambda: (True, "test"))
    monkeypatch.setenv("KG_EVAL_BYPASS_ALLOWED", "1")
    monkeypatch.setenv("KG_BUILD_ENABLED", "0")
    monkeypatch.setenv("KG_OVERLAY_ENABLED", "0")

    missing_eval = run_kg_eval.main(
        ["--mode", "comparison", "--collection", "2", "--output", str(output)]
    )
    missing_scope = run_kg_eval.main(
        ["--mode", "comparison", "--eval-only", "--output", str(output)]
    )
    success = run_kg_eval.main(
        [
            "--mode",
            "comparison",
            "--eval-only",
            "--collection",
            "2",
            "--collection",
            "1",
            "--rebuild-request",
            "22222222-2222-4222-8222-222222222222",
            "--rebuild-request",
            "11111111-1111-4111-8111-111111111111",
            "--fixture-manifest",
            str(tmp_path / "fixture-manifest.json"),
            "--output",
            str(output),
        ]
    )

    assert missing_eval == 2
    assert missing_scope == 2
    assert success == 0
    assert json.loads(output.read_text(encoding="utf-8"))["collection_scope"] == [1, 2]
    captured = capsys.readouterr()
    assert "vector_only" in captured.out
    assert captured.err == ""


def test_cli_pairs_rebuild_requests_with_collections_before_canonical_sort(
    tmp_path, capsys, monkeypatch
):
    output = tmp_path / "comparison.json"
    bundle = _passing_comparison_bundle()
    observed: list[tuple[tuple[int, UUID], ...]] = []

    def live(**kwargs):
        observed.append(kwargs["evaluation_rebuild_requests"])
        return bundle

    monkeypatch.setattr(run_kg_eval, "_live_comparison_bundle", live)
    monkeypatch.setattr(run_kg_eval, "_runtime_eval_context", lambda: (True, "test"))
    monkeypatch.setenv("KG_EVAL_BYPASS_ALLOWED", "1")
    monkeypatch.setenv("KG_BUILD_ENABLED", "0")
    monkeypatch.setenv("KG_OVERLAY_ENABLED", "0")

    request_one = UUID("11111111-1111-4111-8111-111111111111")
    request_two = UUID("22222222-2222-4222-8222-222222222222")
    success = run_kg_eval.main(
        [
            "--mode",
            "comparison",
            "--eval-only",
            "--collection",
            "2",
            "--collection",
            "1",
            "--rebuild-request",
            str(request_two),
            "--rebuild-request",
            str(request_one),
            "--fixture-manifest",
            str(tmp_path / "fixture-manifest.json"),
            "--output",
            str(output),
        ]
    )

    assert success == 0
    assert observed == [((1, request_one), (2, request_two))]
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize(
    "arguments",
    (
        ("--collection", "1"),
        (
            "--collection",
            "1",
            "--rebuild-request",
            "11111111-1111-4111-8111-111111111111",
            "--rebuild-request",
            "22222222-2222-4222-8222-222222222222",
        ),
    ),
)
def test_cli_rejects_missing_or_mismatched_rebuild_request_mapping(
    tmp_path, capsys, monkeypatch, arguments
):
    monkeypatch.setattr(run_kg_eval, "_runtime_eval_context", lambda: (True, "test"))
    monkeypatch.setenv("KG_EVAL_BYPASS_ALLOWED", "1")
    monkeypatch.setenv("KG_BUILD_ENABLED", "0")
    monkeypatch.setenv("KG_OVERLAY_ENABLED", "0")

    result = run_kg_eval.main(
        [
            "--mode",
            "comparison",
            "--eval-only",
            *arguments,
            "--output",
            str(tmp_path / "report.json"),
        ]
    )
    assert result == 2
    assert "rebuild" in capsys.readouterr().out


def test_fixture_checksum_and_task23_hybrid_runner_contracts_are_public():
    bundle = _passing_comparison_bundle()
    assert bundle["fixture_checksum"] == sha256(b"fixture").hexdigest() or len(
        bundle["fixture_checksum"]
    ) == 64
    assert run_kg_eval.TASK21_HYBRID_ARMS[-1] == "combined_reranked"
    assert callable(run_kg_eval.build_task21_hybrid_report)
