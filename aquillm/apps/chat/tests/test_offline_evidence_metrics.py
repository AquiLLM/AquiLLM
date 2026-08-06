import pytest

from apps.chat.evals.offline import policies
from apps.chat.evals.offline.metrics import (
    aggregate_evidence,
    binary_metrics,
    categorical_conformance,
    citation_diagnostics,
    compare_policies,
    exact_set_metrics,
    memory_stratum_errors,
    query_conformance,
    score_evidence_case,
)
from apps.chat.evals.offline.policies import sequential_select
from apps.chat.services import rag_evidence


def test_binary_metrics_reports_counts_denominators_and_f1():
    result = binary_metrics(
        [True, True, False, False],
        [True, False, True, False],
    )

    assert result["confusion"] == {"tp": 1, "fn": 1, "fp": 1, "tn": 1}
    assert result["accuracy"] == {
        "status": "ok",
        "value": 0.5,
        "numerator": 2,
        "denominator": 4,
    }
    assert result["precision"] == {
        "status": "ok",
        "value": 0.5,
        "numerator": 1,
        "denominator": 2,
    }
    assert result["recall"] == {
        "status": "ok",
        "value": 0.5,
        "numerator": 1,
        "denominator": 2,
    }
    assert result["f1"] == {
        "status": "ok",
        "value": 0.5,
        "numerator": 2,
        "denominator": 4,
    }
    assert result["support"] == 4


def test_binary_metrics_marks_zero_denominators_not_applicable():
    result = binary_metrics([], [])

    for name in ("accuracy", "precision", "recall", "f1"):
        assert result[name] == {
            "status": "not_applicable",
            "value": None,
            "numerator": 0,
            "denominator": 0,
        }


def test_binary_metrics_rejects_mismatched_case_counts():
    with pytest.raises(ValueError, match="same length"):
        binary_metrics([True], [])


def test_categorical_conformance_reports_support_and_full_confusion_matrix():
    result = categorical_conformance(
        ["explicit_search", "retry", "retry", "none"],
        ["explicit_search", "explicit_search", "retry", "none"],
        labels=["explicit_search", "retry", "figure", "none"],
    )

    assert result["support"] == 4
    assert result["conformance"] == {
        "status": "ok",
        "value": 0.75,
        "numerator": 3,
        "denominator": 4,
    }
    assert result["by_label"]["retry"] == {
        "support": 2,
        "conformance": {
            "status": "ok",
            "value": 0.5,
            "numerator": 1,
            "denominator": 2,
        },
    }
    assert result["by_label"]["figure"]["support"] == 0
    assert result["by_label"]["figure"]["conformance"]["status"] == "not_applicable"
    assert result["confusion_matrix"]["retry"]["explicit_search"] == 1
    assert result["confusion_matrix"]["figure"] == {
        "explicit_search": 0,
        "retry": 0,
        "figure": 0,
        "none": 0,
    }


@pytest.mark.parametrize("unknown_side", ["expected", "actual"])
def test_categorical_conformance_rejects_unknown_labels(unknown_side):
    expected = ["known"] if unknown_side == "actual" else ["unknown"]
    actual = ["unknown"] if unknown_side == "actual" else ["known"]

    with pytest.raises(ValueError, match="unknown label"):
        categorical_conformance(expected, actual, labels=["known"])


def test_query_conformance_strips_only_leading_and_trailing_whitespace():
    result = query_conformance(
        ["  alpha  ", "two  spaces", "Case"],
        ["alpha", "two spaces", "case"],
    )

    assert result["normalized_expected"] == ["alpha", "two  spaces", "Case"]
    assert result["normalized_actual"] == ["alpha", "two spaces", "case"]
    assert result["conformance"] == {
        "status": "ok",
        "value": pytest.approx(1 / 3),
        "numerator": 1,
        "denominator": 3,
    }
    assert result["support"] == 3


def test_query_conformance_reports_empty_support_and_rejects_mismatched_counts():
    assert query_conformance([], [])["conformance"]["status"] == "not_applicable"
    with pytest.raises(ValueError, match="same length"):
        query_conformance(["query"], [])


def test_exact_set_metrics_use_exact_identity_and_remove_duplicates_in_order():
    result = exact_set_metrics(
        [["alpha", "beta"], [], ["Case"]],
        [["alpha", "alpha", "gamma"], [], ["case"]],
    )

    assert result["normalized_actual"] == [["alpha", "gamma"], [], ["case"]]
    assert result["precision"] == {
        "status": "ok",
        "value": pytest.approx(1 / 3),
        "numerator": 1,
        "denominator": 3,
    }
    assert result["recall"]["value"] == pytest.approx(1 / 3)
    assert result["f1"]["value"] == pytest.approx(1 / 3)
    assert result["exact_set_conformance"] == {
        "status": "ok",
        "value": pytest.approx(1 / 3),
        "numerator": 1,
        "denominator": 3,
    }
    assert result["duplicate_count"] == 1
    assert result["duplicate_rate"] == {
        "status": "ok",
        "value": 0.25,
        "numerator": 1,
        "denominator": 4,
    }
    assert result["support"] == 3


def test_exact_set_metrics_make_empty_item_metrics_explicit():
    result = exact_set_metrics([[]], [[]])

    for name in ("precision", "recall", "f1", "duplicate_rate"):
        assert result[name]["status"] == "not_applicable"
        assert result[name]["value"] is None
    assert result["exact_set_conformance"]["value"] == 1.0


def test_exact_set_metrics_reject_mismatched_case_counts():
    with pytest.raises(ValueError, match="same length"):
        exact_set_metrics([[]], [])


def test_citation_diagnostics_measure_raw_syntax_consistency_and_prefix_behavior():
    selected = [
        {
            "doc_id": "doc-a",
            "chunk_id": 1,
            "citation": "[doc:doc-a chunk:1]",
            "image_url": "/aquillm/images/a.png",
        },
        {
            "doc_id": "doc-b",
            "chunk_id": 2,
            "citation": "[doc:doc-a chunk:1]",
            "image_url": "https://example.invalid/a.png",
        },
        {
            "doc_id": "doc-c",
            "chunk_id": 3,
            "citation": "not-a-citation",
            "image_url": 42,
        },
    ]

    result = citation_diagnostics(selected)

    assert result["syntax_validity"] == {
        "status": "ok",
        "value": pytest.approx(2 / 3),
        "numerator": 2,
        "denominator": 3,
    }
    assert result["chunk_consistency"] == {
        "status": "ok",
        "value": pytest.approx(1 / 3),
        "numerator": 1,
        "denominator": 3,
    }
    assert result["duplicate_count"] == 1
    assert result["conflict_count"] == 1
    assert result["image_path_prefix_behavior"] == {
        "label": "prefix_behavior",
        "prefix": "/aquillm/",
        "status": "ok",
        "value": 0.5,
        "numerator": 1,
        "denominator": 2,
    }
    assert "author" not in repr(result).lower()


def test_citation_diagnostics_reject_nonnumeric_chunk_syntax():
    result = citation_diagnostics(
        [
            {
                "doc_id": "doc-a",
                "chunk_id": "not-a-number",
                "citation": "[doc:doc-a chunk:not-a-number]",
            }
        ]
    )

    assert result["syntax_validity"]["numerator"] == 0
    assert result["chunk_consistency"]["numerator"] == 0


def test_citation_conflicts_include_reused_malformed_tokens_before_deduplication():
    result = citation_diagnostics(
        [
            {"doc_id": "doc-a", "chunk_id": 1, "citation": "malformed"},
            {"doc_id": "doc-b", "chunk_id": 2, "citation": "malformed"},
        ]
    )

    assert result["syntax_validity"]["numerator"] == 0
    assert result["duplicate_count"] == 1
    assert result["conflict_count"] == 1


def _evidence_case(case_id="case-a", *, gold=None, budget=1):
    return {
        "id": case_id,
        "stratum": "distributed",
        "token_budget": budget,
        "gold": {"relevant_evidence_ids": gold if gold is not None else ["e1", "e2"]},
        "candidates": [
            {"evidence_id": "e1", "doc_id": "doc-a", "chunk_id": 1, "text": "abcd"},
            {"evidence_id": "e2", "doc_id": "doc-b", "chunk_id": 2, "text": "efgh"},
            {"evidence_id": "e3", "doc_id": "doc-b", "chunk_id": 3, "text": "ijkl"},
        ],
    }


def test_score_evidence_case_uses_evidence_identity_and_reports_budget_diagnostics():
    result = score_evidence_case(
        _evidence_case(),
        [
            {"evidence_id": "e1", "doc_id": "doc-a", "chunk_id": 1, "text": "abcd"},
            {"evidence_id": "e3", "doc_id": "doc-b", "chunk_id": 3, "text": "ijkl"},
        ],
    )

    assert result["case_id"] == "case-a"
    assert result["relevant_evidence_recall"] == {
        "status": "ok",
        "value": 0.5,
        "numerator": 1,
        "denominator": 2,
    }
    assert result["relevant_document_coverage"] == {
        "status": "ok",
        "value": 1.0,
        "numerator": 2,
        "denominator": 2,
    }
    assert result["distinct_selected_documents"] == 2
    assert result["estimated_token_use"] == 2
    assert result["overrun_tokens"] == 1


def test_score_evidence_case_marks_zero_gold_recall_not_applicable():
    result = score_evidence_case(_evidence_case(gold=[]), [])

    assert result["relevant_evidence_recall"]["status"] == "not_applicable"
    assert result["relevant_document_coverage"]["status"] == "not_applicable"
    assert result["distinct_selected_documents"] == 0


def test_relevant_document_coverage_uses_selected_doc_identity_not_evidence_id():
    case = {
        "id": "same-document",
        "token_budget": 10,
        "gold": {
            "relevant_evidence_ids": ["gold-chunk"],
            "relevant_document_ids": ["doc-a"],
        },
        "candidates": [
            {"evidence_id": "gold-chunk", "doc_id": "doc-a", "text": "gold"},
            {"evidence_id": "other-chunk", "doc_id": "doc-a", "text": "context"},
        ],
    }

    result = score_evidence_case(case, [case["candidates"][1]])

    assert result["relevant_evidence_recall"]["value"] == 0.0
    assert result["relevant_document_coverage"] == {
        "status": "ok",
        "value": 1.0,
        "numerator": 1,
        "denominator": 1,
    }


def test_aggregate_evidence_reports_macro_micro_recall_and_support():
    records = [
        score_evidence_case(
            _evidence_case("half"), [_evidence_case()["candidates"][0]]
        ),
        score_evidence_case(_evidence_case("all"), _evidence_case()["candidates"][:2]),
        score_evidence_case(_evidence_case("empty", gold=[]), []),
    ]

    result = aggregate_evidence(records)

    assert result["support"] == 3
    assert result["applicable_support"] == 2
    assert result["macro_relevant_evidence_recall"] == {
        "status": "ok",
        "value": 0.75,
        "numerator": 1.5,
        "denominator": 2,
    }
    assert result["micro_relevant_evidence_recall"] == {
        "status": "ok",
        "value": 0.75,
        "numerator": 3,
        "denominator": 4,
    }


def test_aggregate_evidence_marks_all_zero_gold_not_applicable():
    result = aggregate_evidence([score_evidence_case(_evidence_case(gold=[]), [])])

    assert result["macro_relevant_evidence_recall"]["status"] == "not_applicable"
    assert result["micro_relevant_evidence_recall"]["status"] == "not_applicable"
    assert result["applicable_support"] == 0


def test_compare_policies_counts_metric_specific_favorable_and_unfavorable_cases():
    records = [
        {
            "sequential": {"relevant_evidence_recall": 0.5},
            "aquillm": {"relevant_evidence_recall": 1.0},
        },
        {
            "sequential": {"relevant_evidence_recall": 1.0},
            "aquillm": {"relevant_evidence_recall": 0.5},
        },
        {
            "sequential": {"relevant_evidence_recall": 0.5},
            "aquillm": {"relevant_evidence_recall": 0.5},
        },
        {
            "sequential": {"relevant_evidence_recall": None},
            "aquillm": {"relevant_evidence_recall": None},
        },
    ]

    assert compare_policies(records, "relevant_evidence_recall") == {
        "metric": "relevant_evidence_recall",
        "higher_is_better": True,
        "wins": 1,
        "ties": 1,
        "losses": 1,
        "support": 3,
        "not_applicable": 1,
    }
    assert compare_policies(
        [{"sequential": {"overrun_tokens": 2}, "aquillm": {"overrun_tokens": 0}}],
        "overrun_tokens",
    )["wins"] == 1


@pytest.mark.parametrize(
    "metric",
    ["estimated_token_use", "duplicate_count", "conflict_count", "overrun_tokens"],
)
def test_compare_policies_declares_cost_and_error_metrics_lower_is_better(metric):
    result = compare_policies(
        [{"sequential": {metric: 2}, "aquillm": {metric: 1}}], metric
    )

    assert result["higher_is_better"] is False
    assert result["wins"] == 1


def test_compare_policies_rejects_metrics_without_an_explicit_direction():
    with pytest.raises(ValueError, match="unknown policy-comparison metric"):
        compare_policies([], "overrunish")


def test_memory_stratum_errors_report_exact_fp_fn_conformance_and_duplicates():
    records = [
        {
            "stratum": "stable_preference",
            "expected": ["likes tea", "uses vim"],
            "actual": ["likes tea", "likes tea", "likes coffee"],
        },
        {
            "stratum": "stable_preference",
            "expected": [],
            "actual": [],
        },
        {"stratum": "transient", "expected": [], "actual": ["do this now"]},
    ]

    result = memory_stratum_errors(records)

    stable = result["strata"]["stable_preference"]
    assert stable["support"] == 2
    assert stable["false_positive_count"] == 1
    assert stable["false_negative_count"] == 1
    assert stable["exact_set_conformance"]["numerator"] == 1
    assert stable["duplicate_count"] == 1
    assert stable["duplicate_rate"]["value"] == pytest.approx(1 / 3)
    assert result["strata"]["transient"]["false_positive_count"] == 1
    assert result["support"] == 3


def test_sequential_policy_imports_production_text_and_token_helpers():
    assert policies._estimate_tokens is rag_evidence._estimate_tokens
    assert policies._chunk_text is rag_evidence._chunk_text


def test_sequential_policy_preserves_stable_order_metadata_and_packet_fields():
    chunks = [
        {
            "evidence_id": "e2",
            "doc_id": "doc-b",
            "chunk_id": 2,
            "rank": 1,
            "text": "abcdefgh",
            "citation": "[doc:doc-b chunk:2]",
            "image_url": "/aquillm/images/b.png",
            "custom": "preserved",
        },
        {
            "evidence_id": "e1",
            "d": "doc-a",
            "c": 1,
            "rank": 1,
            "x": "abcd",
            "ref": "[doc:doc-a chunk:1]",
            "u": "https://example.invalid/a.png",
        },
    ]

    result = sequential_select(chunks, token_budget=3)

    assert result == {
        "chunks": chunks,
        "image_urls": ["/aquillm/images/b.png"],
        "citation_tokens": ["[doc:doc-b chunk:2]", "[doc:doc-a chunk:1]"],
        "total_tokens": 3,
        "overrun_tokens": 0,
    }
    assert result["chunks"][0] is chunks[0]


def test_sequential_policy_admits_first_oversized_chunk_and_reports_overrun():
    first = {"evidence_id": "large", "text": "x" * 20, "metadata": {"keep": True}}
    later = {"evidence_id": "later", "text": "abcd"}

    result = sequential_select([first, later], token_budget=2)

    assert result["chunks"] == [first]
    assert result["chunks"][0] is first
    assert result["total_tokens"] == rag_evidence._estimate_tokens(first["text"])
    assert result["overrun_tokens"] == 3


def test_sequential_policy_stops_at_first_later_chunk_that_exceeds_budget():
    chunks = [
        {"evidence_id": "first", "text": "abcd"},
        {"evidence_id": "blocked", "text": "x" * 12},
        {"evidence_id": "would-fit", "text": "abcd"},
    ]

    result = sequential_select(chunks, token_budget=2)

    assert [chunk["evidence_id"] for chunk in result["chunks"]] == ["first"]
    assert result["total_tokens"] == 1
    assert result["overrun_tokens"] == 0


def test_sequential_policy_empty_input_has_explicit_zero_accounting():
    assert sequential_select([], token_budget=5) == {
        "chunks": [],
        "image_urls": [],
        "citation_tokens": [],
        "total_tokens": 0,
        "overrun_tokens": 0,
    }
