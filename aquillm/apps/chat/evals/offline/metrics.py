"""JSON-serializable metrics for deterministic offline evaluation."""

from __future__ import annotations

import re

from apps.chat.services.rag_evidence import _chunk_text, _estimate_tokens

_CITATION_RE = re.compile(r"^\[doc:([^\[\]\s]+) chunk:(\d+)\]$")
_POLICY_METRIC_DIRECTIONS = {
    "relevant_evidence_recall": True,
    "relevant_document_coverage": True,
    "distinct_selected_documents": True,
    "syntax_validity": True,
    "chunk_consistency": True,
    "estimated_token_use": False,
    "duplicate_count": False,
    "conflict_count": False,
    "image_path_prefix_behavior": True,
    "overrun_tokens": False,
}


def _ratio(numerator: int | float, denominator: int | float) -> dict:
    if denominator == 0:
        return {
            "status": "not_applicable",
            "value": None,
            "numerator": numerator,
            "denominator": denominator,
        }
    return {
        "status": "ok",
        "value": numerator / denominator,
        "numerator": numerator,
        "denominator": denominator,
    }


def binary_metrics(expected: list[bool], actual: list[bool]) -> dict:
    """Return confusion counts and fixed-set binary conformance metrics."""
    if len(expected) != len(actual):
        raise ValueError("expected and actual must have the same length")

    tp = sum(want and got for want, got in zip(expected, actual))
    fn = sum(want and not got for want, got in zip(expected, actual))
    fp = sum(not want and got for want, got in zip(expected, actual))
    tn = sum(not want and not got for want, got in zip(expected, actual))
    return {
        "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
        "accuracy": _ratio(tp + tn, len(expected)),
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "support": len(expected),
    }


def categorical_conformance(
    expected: list[str], actual: list[str], *, labels: list[str] | None = None
) -> dict:
    """Score single-label reason or action conformance with explicit support."""
    if len(expected) != len(actual):
        raise ValueError("expected and actual must have the same length")

    known_labels = _unique(labels if labels is not None else [*expected, *actual])
    known_label_set = set(known_labels)
    unknown = [label for label in [*expected, *actual] if label not in known_label_set]
    if unknown:
        raise ValueError(f"unknown label: {unknown[0]}")

    confusion = {want: {got: 0 for got in known_labels} for want in known_labels}
    for want, got in zip(expected, actual):
        confusion[want][got] += 1

    by_label = {}
    for label in known_labels:
        support = expected.count(label)
        by_label[label] = {
            "support": support,
            "conformance": _ratio(confusion[label][label], support),
        }
    conforming = sum(want == got for want, got in zip(expected, actual))
    return {
        "labels": known_labels,
        "support": len(expected),
        "conformance": _ratio(conforming, len(expected)),
        "by_label": by_label,
        "confusion_matrix": confusion,
    }


def query_conformance(expected: list[str], actual: list[str]) -> dict:
    """Score exact query strings after trimming edge whitespace only."""
    if len(expected) != len(actual):
        raise ValueError("expected and actual must have the same length")
    normalized_expected = [query.strip() for query in expected]
    normalized_actual = [query.strip() for query in actual]
    conforming = sum(
        want == got for want, got in zip(normalized_expected, normalized_actual)
    )
    return {
        "normalized_expected": normalized_expected,
        "normalized_actual": normalized_actual,
        "support": len(expected),
        "conformance": _ratio(conforming, len(expected)),
    }


def exact_set_metrics(expected: list[list[str]], actual: list[list[str]]) -> dict:
    """Micro-score exact string sets after stable duplicate removal per case."""
    if len(expected) != len(actual):
        raise ValueError("expected and actual must have the same length")

    normalized_expected = [_unique(items) for items in expected]
    normalized_actual = [_unique(items) for items in actual]
    true_positive_count = sum(
        len(set(want) & set(got))
        for want, got in zip(normalized_expected, normalized_actual)
    )
    expected_count = sum(map(len, normalized_expected))
    actual_count = sum(map(len, normalized_actual))
    raw_actual_count = sum(map(len, actual))
    duplicate_count = raw_actual_count - actual_count
    conforming_cases = sum(
        set(want) == set(got)
        for want, got in zip(normalized_expected, normalized_actual)
    )

    return {
        "normalized_expected": normalized_expected,
        "normalized_actual": normalized_actual,
        "precision": _ratio(true_positive_count, actual_count),
        "recall": _ratio(true_positive_count, expected_count),
        "f1": _ratio(2 * true_positive_count, actual_count + expected_count),
        "exact_set_conformance": _ratio(conforming_cases, len(expected)),
        "duplicate_count": duplicate_count,
        "duplicate_rate": _ratio(duplicate_count, raw_actual_count),
        "support": len(expected),
    }


def score_evidence_case(case: dict, selected: list[dict]) -> dict:
    """Score one controlled evidence-selection case by stable evidence ID."""
    gold = case.get("gold") or {}
    gold_ids = _unique(
        gold.get("relevant_evidence_ids", case.get("relevant_evidence_ids", []))
    )
    gold_id_set = set(gold_ids)
    selected_ids = _unique(
        chunk["evidence_id"]
        for chunk in selected
        if chunk.get("evidence_id") is not None
    )
    selected_relevant_ids = [item for item in selected_ids if item in gold_id_set]

    candidates_by_id = {
        chunk.get("evidence_id"): chunk
        for chunk in case.get("candidates", [])
        if chunk.get("evidence_id") is not None
    }
    gold_doc_ids = _unique(
        gold.get("relevant_document_ids", [])
        or (
            _doc_id(candidates_by_id[item])
            for item in gold_ids
            if item in candidates_by_id
        )
    )
    gold_doc_set = set(gold_doc_ids)
    selected_doc_ids = _unique(_doc_id(chunk) for chunk in selected)
    selected_relevant_doc_ids = [
        doc_id for doc_id in selected_doc_ids if doc_id in gold_doc_set
    ]
    total_tokens = sum(_estimate_tokens(_chunk_text(chunk)) for chunk in selected)
    budget = case.get("token_budget", 0)

    return {
        "case_id": case.get("id"),
        "stratum": case.get("stratum"),
        "gold_relevant_evidence_ids": gold_ids,
        "selected_relevant_evidence_ids": selected_relevant_ids,
        "relevant_evidence_recall": _ratio(len(selected_relevant_ids), len(gold_ids)),
        "gold_relevant_document_ids": gold_doc_ids,
        "selected_relevant_document_ids": selected_relevant_doc_ids,
        "relevant_document_coverage": _ratio(
            len(selected_relevant_doc_ids), len(gold_doc_ids)
        ),
        "distinct_selected_documents": len(selected_doc_ids),
        "estimated_token_use": total_tokens,
        "token_budget": budget,
        "overrun_tokens": max(0, total_tokens - budget),
        "citation_diagnostics": citation_diagnostics(selected),
    }


def aggregate_evidence(records: list[dict]) -> dict:
    """Aggregate case evidence recall without treating zero-gold cases as zeroes."""
    applicable = [
        record["relevant_evidence_recall"]
        for record in records
        if record["relevant_evidence_recall"]["status"] == "ok"
    ]
    macro_numerator = sum(metric["value"] for metric in applicable)
    micro_numerator = sum(metric["numerator"] for metric in applicable)
    micro_denominator = sum(metric["denominator"] for metric in applicable)
    return {
        "support": len(records),
        "applicable_support": len(applicable),
        "macro_relevant_evidence_recall": _ratio(macro_numerator, len(applicable)),
        "micro_relevant_evidence_recall": _ratio(micro_numerator, micro_denominator),
    }


def compare_policies(records: list[dict], metric: str) -> dict:
    """Count paired AquiLLM wins, ties, and losses for one named metric."""
    try:
        higher_is_better = _POLICY_METRIC_DIRECTIONS[metric]
    except KeyError as exc:
        raise ValueError(f"unknown policy-comparison metric: {metric}") from exc
    wins = ties = losses = not_applicable = 0
    for record in records:
        aquillm_value = _metric_value(record.get("aquillm", {}).get(metric))
        sequential_value = _metric_value(record.get("sequential", {}).get(metric))
        if aquillm_value is None or sequential_value is None:
            not_applicable += 1
            continue
        if aquillm_value == sequential_value:
            ties += 1
        elif (aquillm_value > sequential_value) == higher_is_better:
            wins += 1
        else:
            losses += 1
    return {
        "metric": metric,
        "higher_is_better": higher_is_better,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "support": wins + ties + losses,
        "not_applicable": not_applicable,
    }


def citation_diagnostics(selected: list[dict]) -> dict:
    """Report citation and image-prefix behavior before packet-level deduplication."""
    citations: list[str | None] = []
    valid_count = consistent_count = 0
    mappings: dict[str, set[tuple[str, str]]] = {}
    image_paths: list[str] = []
    for chunk in selected:
        citation = chunk.get("citation") or chunk.get("ref")
        citations.append(citation if isinstance(citation, str) else None)
        match = _CITATION_RE.fullmatch(citation) if isinstance(citation, str) else None
        if isinstance(citation, str):
            actual_pair = (str(_doc_id(chunk)), str(_chunk_id(chunk)))
            mappings.setdefault(citation, set()).add(actual_pair)
        if match:
            valid_count += 1
            citation_pair = match.groups()
            if citation_pair == actual_pair:
                consistent_count += 1

        image_path = chunk.get("image_url") or chunk.get("u")
        if isinstance(image_path, str):
            image_paths.append(image_path)

    citation_strings = [item for item in citations if item is not None]
    duplicate_count = len(citation_strings) - len(set(citation_strings))
    conflict_count = sum(len(pairs) - 1 for pairs in mappings.values())
    prefix_count = sum(path.startswith("/aquillm/") for path in image_paths)
    image_metric = _ratio(prefix_count, len(image_paths))
    return {
        "syntax_validity": _ratio(valid_count, len(selected)),
        "chunk_consistency": _ratio(consistent_count, len(selected)),
        "duplicate_count": duplicate_count,
        "conflict_count": conflict_count,
        "image_path_prefix_behavior": {
            "label": "prefix_behavior",
            "prefix": "/aquillm/",
            **image_metric,
        },
    }


def memory_stratum_errors(records: list[dict]) -> dict:
    """Report exact fact-set errors and duplicate rates by fixture stratum."""
    strata: dict[str, list[dict]] = {}
    for record in records:
        strata.setdefault(record["stratum"], []).append(record)

    output: dict[str, dict] = {}
    for stratum, group in strata.items():
        expected = [record.get("expected", []) for record in group]
        actual = [record.get("actual", []) for record in group]
        metrics = exact_set_metrics(expected, actual)
        false_positive_count = false_negative_count = 0
        for want, got in zip(
            metrics["normalized_expected"], metrics["normalized_actual"]
        ):
            false_positive_count += len(set(got) - set(want))
            false_negative_count += len(set(want) - set(got))
        output[stratum] = {
            **metrics,
            "false_positive_count": false_positive_count,
            "false_negative_count": false_negative_count,
        }

    return {
        "support": len(records),
        "strata": output,
        "overall": exact_set_metrics(
            [record.get("expected", []) for record in records],
            [record.get("actual", []) for record in records],
        ),
    }


def _unique(items) -> list:
    return list(dict.fromkeys(items))


def _doc_id(chunk: dict):
    if chunk.get("doc_id") is not None:
        return chunk["doc_id"]
    return chunk.get("d", "")


def _chunk_id(chunk: dict):
    return (
        chunk.get("chunk_id")
        if chunk.get("chunk_id") is not None
        else chunk.get("c", "")
    )


def _metric_value(value):
    if isinstance(value, dict):
        return value.get("value") if value.get("status", "ok") == "ok" else None
    return value
