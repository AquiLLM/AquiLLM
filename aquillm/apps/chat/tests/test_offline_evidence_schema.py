"""Contracts for frozen, independently reviewable offline evidence fixtures."""

from __future__ import annotations

import ast
import hashlib
from collections import Counter
from pathlib import Path

import pytest
import yaml

from apps.chat.evals.offline.schema import (
    canonical_json_bytes,
    load_dataset,
    sha256_file,
    validate_dataset,
    validate_test_manifest,
)

FIXTURE_DIR = Path(__file__).parents[1] / "evals" / "offline" / "fixtures"
MANIFEST_PATH = Path(__file__).parents[1] / "evals" / "offline" / "test_manifest.yaml"
KINDS = ("routing", "evidence", "memory")
STRATA = {"favorable", "unfavorable", "ambiguous", "adversarial_boundary"}


def _valid_case(kind: str) -> dict:
    base = {
        "id": f"{kind}-case-001",
        "stratum": "favorable",
        "rationale": "A synthetic contract example.",
    }
    if kind == "routing":
        return {
            **base,
            "input": {
                "text": "Search the documents for calibration notes.",
                "selected_collection_ids": ["collection-public-a"],
                "prior_tools": [],
            },
            "gold": {
                "classifier": {
                    "requires_rag": True,
                    "wants_figures": False,
                    "wants_whole_document": False,
                    "is_retry": False,
                    "requires_local_tools": False,
                },
                "reason": "explicit_search",
                "production_action": "retrieve",
                "expected_query": "Search the documents for calibration notes.",
            },
        }
    if kind == "evidence":
        return {
            **base,
            "question": "What calibration context is reported?",
            "answer_target": "The synthetic calibration context.",
            "token_budget": 64,
            "candidates": [
                {
                    "evidence_id": "evidence-001",
                    "doc_id": "doc-public-a",
                    "chunk_id": 1,
                    "rank": 1,
                    "text": "Synthetic calibration context.",
                    "estimated_tokens": 7,
                    "citation": "[doc:doc-public-a chunk:1]",
                    "relevant": True,
                }
            ],
            "gold": {
                "relevant_evidence_ids": ["evidence-001"],
                "relevant_document_ids": ["doc-public-a"],
            },
        }
    return {
        **base,
        "input": {
            "user_content": "Remember that the project uses public test data.",
            "assistant_content": "Acknowledged.",
        },
        "gold": {"normalized_facts": ["The project uses public test data."]},
    }


def _valid_dataset(kind: str) -> dict:
    return {
        "schema_version": "1.1",
        "dataset_id": f"{kind}-v2",
        "frozen_at": "2026-08-06T17:00:00Z",
        "provenance": "synthetic_public",
        "sensitivity": "synthetic_public",
        "rubric_version": "1.1",
        "annotation": {
            "author_role": "fixture_author",
            "annotated_at": "2026-08-06",
        },
        "review": {
            "status": "approved",
            "record": "review.yaml",
        },
        "cases": [_valid_case(kind)],
    }


def _write_approved_fixture_set(
    tmp_path,
    *,
    review_mutator=None,
    dataset_mutator=None,
) -> Path:
    datasets = {kind: _valid_dataset(kind) for kind in KINDS}
    for dataset in datasets.values():
        dataset["review"]["status"] = "approved"
    if dataset_mutator:
        dataset_mutator(datasets)
    for kind, dataset in datasets.items():
        (tmp_path / f"{kind}.yaml").write_text(
            yaml.safe_dump(dataset, sort_keys=False), encoding="utf-8"
        )

    review = {
        "schema_version": "1.1",
        "review_id": "synthetic-review-v2",
        "rubric_version": "1.1",
        "status": "approved",
        "reviewer_identity": "codex-agent:independent-reviewer",
        "reviewer_role": "independent_reviewer",
        "review_date": "2026-08-06",
        "production_functions_executed": False,
        "fixture_hashes": {
            f"{kind}.yaml": sha256_file(tmp_path / f"{kind}.yaml") for kind in KINDS
        },
        "label_changes": [
            {
                "case_id": "routing-case-001",
                "field": "gold.reason",
                "before": "old",
                "after": "explicit_search",
                "reason": "Synthetic review correction.",
            }
        ],
        "adjudications": [
            {
                "case_id": "routing-case-001",
                "decision": "The explicit search reading controls.",
            }
        ],
        "evidence_relevance_adjudications": [
            {
                "case_id": "evidence-case-001",
                "decision": "The only candidate states the answer target.",
                "relevant_evidence_ids": ["evidence-001"],
            }
        ],
        "retained_ambiguities": [],
        "approval": True,
    }
    if review_mutator:
        review_mutator(review)
    (tmp_path / "review.yaml").write_text(
        yaml.safe_dump(review, sort_keys=False), encoding="utf-8"
    )
    return tmp_path / "routing.yaml"


@pytest.mark.parametrize(
    "field", ["schema_version", "frozen_at", "provenance", "rubric_version"]
)
def test_dataset_envelope_requires_version_freeze_provenance_and_rubric(field):
    dataset = _valid_dataset("routing")
    del dataset[field]

    with pytest.raises(ValueError, match=field):
        validate_dataset(dataset, "routing")


def test_dataset_contract_requires_version_1_1_and_v2_dataset_identity():
    dataset = _valid_dataset("routing")
    dataset["schema_version"] = "1.0"
    with pytest.raises(ValueError, match="schema_version"):
        validate_dataset(dataset, "routing")

    dataset = _valid_dataset("routing")
    dataset["dataset_id"] = "routing-v1"
    with pytest.raises(ValueError, match="dataset_id"):
        validate_dataset(dataset, "routing")


def test_dataset_requires_synthetic_public_sensitivity_and_fixture_author_role():
    dataset = _valid_dataset("routing")
    dataset["sensitivity"] = "private"
    with pytest.raises(ValueError, match="synthetic_public"):
        validate_dataset(dataset, "routing")

    dataset = _valid_dataset("routing")
    dataset["annotation"]["author_role"] = "unknown"
    with pytest.raises(ValueError, match="fixture_author"):
        validate_dataset(dataset, "routing")


def test_canonical_validation_requires_approved_independent_review():
    dataset = _valid_dataset("routing")
    dataset["review"]["status"] = "pending_independent_review"

    with pytest.raises(ValueError, match="approved independent review"):
        validate_dataset(dataset, "routing")

    validate_dataset(dataset, "routing", allow_pending=True)


def test_invalid_review_status_is_rejected_even_during_authoring():
    dataset = _valid_dataset("routing")
    dataset["review"]["status"] = "author_approved"

    with pytest.raises(ValueError, match="review status"):
        validate_dataset(dataset, "routing", allow_pending=True)


def test_duplicate_case_ids_are_rejected():
    dataset = _valid_dataset("memory")
    dataset["cases"].append(dict(dataset["cases"][0]))

    with pytest.raises(ValueError, match="duplicate case id"):
        validate_dataset(dataset, "memory")


@pytest.mark.parametrize("kind", KINDS)
def test_missing_gold_is_rejected(kind):
    dataset = _valid_dataset(kind)
    del dataset["cases"][0]["gold"]

    with pytest.raises(ValueError, match="gold"):
        validate_dataset(dataset, kind)


def test_malformed_or_inconsistent_evidence_citations_are_rejected():
    malformed = _valid_dataset("evidence")
    malformed["cases"][0]["candidates"][0]["citation"] = "doc-public-a:1"
    with pytest.raises(ValueError, match="citation"):
        validate_dataset(malformed, "evidence")

    inconsistent = _valid_dataset("evidence")
    inconsistent["cases"][0]["candidates"][0]["citation"] = "[doc:other chunk:1]"
    with pytest.raises(ValueError, match="citation.*identity"):
        validate_dataset(inconsistent, "evidence")


def test_evidence_identity_gold_and_relevance_flags_are_consistent():
    dataset = _valid_dataset("evidence")
    dataset["cases"][0]["candidates"][0]["relevant"] = False

    with pytest.raises(ValueError, match="relevant_evidence_ids"):
        validate_dataset(dataset, "evidence")


@pytest.mark.parametrize("field", ["question", "answer_target"])
def test_evidence_cases_require_explicit_question_and_answer_target(field):
    dataset = _valid_dataset("evidence")
    del dataset["cases"][0][field]

    with pytest.raises(ValueError, match=field):
        validate_dataset(dataset, "evidence")


def test_evidence_candidate_token_estimate_matches_documented_approximation():
    dataset = _valid_dataset("evidence")
    dataset["cases"][0]["candidates"][0]["estimated_tokens"] = 999

    with pytest.raises(ValueError, match="estimated_tokens"):
        validate_dataset(dataset, "evidence")


def test_duplicate_citations_and_observed_conflicts_preserve_candidate_identity():
    dataset = _valid_dataset("evidence")
    first = dataset["cases"][0]["candidates"][0]
    duplicate = {
        **first,
        "evidence_id": "evidence-002",
        "relevant": False,
        "observed_citation": "[doc:other-public-doc chunk:2]",
    }
    dataset["cases"][0]["candidates"].append(duplicate)

    validate_dataset(dataset, "evidence")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda case: case["gold"]["classifier"].update(requires_local_tools=True),
            "reason priority",
        ),
        (
            lambda case: case["gold"].update(
                production_action="prompt_select_collection"
            ),
            "collection/action consistency",
        ),
    ],
)
def test_routing_gold_enforces_reason_priority_and_action_consistency(mutate, message):
    dataset = _valid_dataset("routing")
    mutate(dataset["cases"][0])

    with pytest.raises(ValueError, match=message):
        validate_dataset(dataset, "routing")


def test_memory_validation_is_independent_of_production_fact_cleaner(monkeypatch):
    import lib.memory.extraction.stable_facts as stable_facts

    monkeypatch.setattr(
        stable_facts,
        "clean_stable_facts",
        lambda _facts: (_ for _ in ()).throw(
            AssertionError("production helper called")
        ),
    )

    validate_dataset(_valid_dataset("memory"), "memory")


def test_memory_gold_requires_sentence_cased_candidates():
    dataset = _valid_dataset("memory")
    dataset["cases"][0]["gold"]["normalized_facts"] = ["lowercase durable fact."]

    with pytest.raises(ValueError, match="sentence-cased"):
        validate_dataset(dataset, "memory")


def test_manifest_rejects_file_only_selector():
    manifest = {
        "schema_version": "1.0",
        "entries": [
            {
                "node_id": "apps/chat/tests/test_rag_intent.py",
                "status": "included",
                "prerequisite": "none",
                "reason": "Pure deterministic unit test.",
            }
        ],
    }

    with pytest.raises(ValueError, match="exact pytest node id"):
        validate_test_manifest(manifest)


def test_manifest_requires_known_status_prerequisite_reason_and_unique_nodes():
    entry = {
        "node_id": "apps/chat/tests/test_rag_intent.py::test_brand_new_chat_no_rag",
        "status": "included",
        "prerequisite": "none",
        "reason": "Pure deterministic unit test.",
    }
    validate_test_manifest({"schema_version": "1.0", "entries": [entry]})

    for missing in ("status", "prerequisite", "reason"):
        invalid = dict(entry)
        invalid.pop(missing)
        with pytest.raises(ValueError, match=missing):
            validate_test_manifest({"schema_version": "1.0", "entries": [invalid]})

    duplicate = {"schema_version": "1.0", "entries": [entry, dict(entry)]}
    with pytest.raises(ValueError, match="duplicate.*node"):
        validate_test_manifest(duplicate)


def test_manifest_static_resolution_rejects_nonexistent_node(tmp_path):
    test_file = tmp_path / "test_example.py"
    test_file.write_text("def test_present():\n    pass\n", encoding="utf-8")
    manifest = {
        "schema_version": "1.0",
        "entries": [
            {
                "node_id": "test_example.py::test_missing",
                "status": "included",
                "prerequisite": "none",
                "reason": "Synthetic static-resolution contract.",
            }
        ],
    }

    with pytest.raises(ValueError, match="does not resolve"):
        validate_test_manifest(manifest, project_root=tmp_path)


@pytest.mark.parametrize(
    ("review_mutator", "message"),
    [
        (lambda review: review.pop("reviewer_identity"), "reviewer_identity"),
        (
            lambda review: review["label_changes"][0].pop("reason"),
            "label_changes.*reason",
        ),
        (
            lambda review: review["adjudications"][0].update(case_id="unknown-case"),
            "unknown case id",
        ),
        (
            lambda review: review.update(evidence_relevance_adjudications=[]),
            "complete evidence adjudications",
        ),
    ],
)
def test_approved_review_requires_structured_complete_valid_audit(
    tmp_path, review_mutator, message
):
    routing_path = _write_approved_fixture_set(tmp_path, review_mutator=review_mutator)

    with pytest.raises(ValueError, match=message):
        load_dataset(routing_path, "routing")


def test_approved_review_requires_every_retained_ambiguous_case(tmp_path):
    routing_path = _write_approved_fixture_set(
        tmp_path,
        dataset_mutator=lambda datasets: datasets["routing"]["cases"][0].update(
            stratum="ambiguous"
        ),
    )

    with pytest.raises(ValueError, match="complete retained ambiguity"):
        load_dataset(routing_path, "routing")


def test_canonical_json_and_sha256_are_deterministic(tmp_path):
    first = {"z": [3, 2, 1], "a": {"b": True}}
    second = {"a": {"b": True}, "z": [3, 2, 1]}

    expected = b'{"a":{"b":true},"z":[3,2,1]}\n'
    assert canonical_json_bytes(first) == expected
    assert canonical_json_bytes(second) == expected

    path = tmp_path / "canonical.json"
    path.write_bytes(expected)
    assert sha256_file(path) == hashlib.sha256(expected).hexdigest()


def test_approved_fixtures_are_frozen_balanced_synthetic_and_review_hashes_match():
    minimums = {"routing": 60, "evidence": 24, "memory": 40}
    datasets = {}
    for kind in KINDS:
        data = load_dataset(FIXTURE_DIR / f"{kind}.yaml", kind)
        datasets[kind] = data
        assert len(data["cases"]) >= minimums[kind]
        assert data["provenance"] == "synthetic_public"
        assert data["sensitivity"] == "synthetic_public"
        assert data["review"] == {
            "status": "approved",
            "record": "review.yaml",
        }
        assert data["schema_version"] == "1.1"
        assert data["dataset_id"] == f"{kind}-v2"
        assert data["rubric_version"] == "1.1"
        assert data["frozen_at"] == "2026-08-06T17:00:00Z"
        counts = Counter(case["stratum"] for case in data["cases"])
        assert set(counts) == STRATA
        assert max(counts.values()) - min(counts.values()) <= 1

    routing_actions = Counter(
        case["gold"]["production_action"] for case in datasets["routing"]["cases"]
    )
    assert set(routing_actions) == {
        "retrieve",
        "prompt_select_collection",
        "skip_normal_tool_loop",
        "local_tool_handling",
    }
    assert max(routing_actions.values()) - min(routing_actions.values()) <= 1

    review = yaml.safe_load((FIXTURE_DIR / "review.yaml").read_text(encoding="utf-8"))
    assert review["review_id"] == "offline-evidence-fixtures-v2"
    assert review["status"] == "approved"
    assert review["reviewer_identity"] == (
        "codex-agent:/root/offline_fixtures_spec_review"
    )
    assert review["reviewer_role"] == "independent_reviewer"
    assert review["review_date"] == "2026-08-06"
    assert review["production_functions_executed"] is False
    assert review["approval"] is True
    assert review["label_changes"]
    assert len(review["evidence_relevance_adjudications"]) == 24
    ambiguous_count = sum(
        case["stratum"] == "ambiguous"
        for dataset in datasets.values()
        for case in dataset["cases"]
    )
    assert len(review["retained_ambiguities"]) == ambiguous_count
    assert review["fixture_hashes"] == {
        f"{kind}.yaml": sha256_file(FIXTURE_DIR / f"{kind}.yaml") for kind in KINDS
    }


def test_pending_fixtures_fail_canonical_approval_validation():
    dataset = _valid_dataset("routing")
    dataset["review"]["status"] = "pending_independent_review"
    with pytest.raises(ValueError, match="approved independent review"):
        validate_dataset(dataset, "routing")
    validate_dataset(dataset, "routing", allow_pending=True)


def test_exact_test_manifest_is_deterministically_ordered_and_covers_required_nodes():
    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    validate_test_manifest(manifest, project_root=Path(__file__).parents[3])
    nodes = [entry["node_id"] for entry in manifest["entries"]]
    assert nodes == sorted(nodes)

    for filename in (
        "test_rag_intent.py",
        "test_rag_query.py",
        "test_rag_evidence.py",
        "test_rag_eval_runner.py",
    ):
        tree = ast.parse(
            (Path(__file__).with_name(filename)).read_text(encoding="utf-8")
        )
        expected_functions = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("test_")
        }
        manifested = {
            node.rsplit("::", 1)[-1]
            for node in nodes
            if node.startswith(f"apps/chat/tests/{filename}::")
        }
        assert manifested == expected_functions

    required = {
        "lib/tools/search/tests/test_vector_search_pack.py::VectorSearchPackTests::test_pack_includes_image_url_when_storage_has_image",
        "lib/tools/search/tests/test_vector_search_pack.py::VectorSearchPackTests::test_pack_omits_image_url_when_storage_missing_file",
        "lib/tools/search/tests/test_vector_search_pack.py::VectorSearchPackTests::test_pack_empty_results_explains_no_relevant_passages",
        "lib/memory/tests/test_stable_facts_quality.py::test_explicit_remember_directive_normalizes_to_durable_fact",
        "lib/memory/tests/test_stable_facts_quality.py::test_durable_project_tooling_statement_is_retained",
        "lib/memory/tests/test_stable_facts_quality.py::test_transient_tactical_turn_is_not_promoted_to_memory",
        "lib/memory/tests/test_stable_facts_quality.py::test_vague_self_referential_remember_text_is_filtered",
        "lib/memory/tests/test_mem0_search_isolation.py::test_parse_mem0_search_items_requires_matching_user_and_excludes_current_session",
        "tests/integration/test_architecture_import_boundaries.py::test_no_direct_aquillm_models_imports_in_runtime_modules",
        "tests/integration/test_settings_security_flags.py::test_celery_accept_content_excludes_pickle",
        "tests/integration/test_settings_security_flags.py::test_celery_tasks_do_not_force_pickle_serializer",
        "apps/chat/tests/test_tool_payload_compaction.py::PackChunkSearchTests::test_pack_chunk_search_results_compact_fields_preserved",
        "apps/chat/tests/test_tool_payload_compaction.py::PackChunkSearchTests::test_pack_chunk_search_results_verbose_list_items",
        "apps/documents/tests/test_chunk_rerank_parse.py::test_parse_rerank_results_accepts_results_and_data_shapes",
        "apps/documents/tests/test_chunk_rerank_parse.py::test_parse_rerank_results_deduplicates_and_skips_invalid_indexes",
        "apps/documents/tests/test_chunk_rerank_parse.py::test_score_parsers_accept_supported_response_shapes",
    }
    assert required <= set(nodes)

    citation_node = (
        "apps/documents/tests/test_citation_api.py::"
        "test_citation_sources_groups_and_enforces_access"
    )
    citation = next(
        entry for entry in manifest["entries"] if entry["node_id"] == citation_node
    )
    assert citation["status"] == "prerequisite_blocked"
    assert citation["prerequisite"] == "postgresql_test_database"
