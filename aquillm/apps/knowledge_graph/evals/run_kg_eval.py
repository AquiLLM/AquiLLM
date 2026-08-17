"""Pure-Python, deterministic evaluation support for the knowledge graph.

The loaders resolve their default fixture paths beside this module, so they are
safe to use from any working directory.  They make no Django, ORM, LLM,
GLiNER2, database, or network calls.

Metrics use structural set equality: each expected/predicted record is
canonicalized as sorted JSON before set intersection.  Precision and recall use
``1.0`` when their denominator is zero (no predicted records for precision, or
no gold records for recall); this makes empty-vs-empty a perfect match.

Usage from ``aquillm/``::

    python -m apps.knowledge_graph.evals.run_kg_eval --baseline-only

``--baseline-only`` prints compact, key-sorted JSON.  It records vector result
IDs only when injected by ``--retrieval-results`` or supplied by a fixture; a
case with neither emits ``SKIP`` and never fabricates IDs or scores.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

_HERE = Path(__file__).resolve().parent
_DEFAULT_EXTRACTION_CASES_PATH = _HERE / "extraction_cases.yaml"
_DEFAULT_RETRIEVAL_CASES_PATH = _HERE / "retrieval_cases.yaml"


class FixtureValidationError(ValueError):
    """Raised when an offline gold fixture cannot be evaluated deterministically."""


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(value[key]) for key in sorted(value)}
        )
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FixtureValidationError(f"{context} must be a mapping")
    return value


def _require_nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FixtureValidationError(f"{context} must be a non-empty string")
    return value


def _require_sequence(
    value: Any, context: str, *, nonempty: bool = False
) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise FixtureValidationError(f"{context} must be a list")
    if nonempty and not value:
        raise FixtureValidationError(f"{context} must not be empty")
    return value


def _validate_documents(value: Any, context: str) -> None:
    for document_index, document in enumerate(
        _require_sequence(value, f"{context}.documents", nonempty=True)
    ):
        document = _require_mapping(document, f"{context}.documents[{document_index}]")
        document_context = f"{context}.documents[{document_index}]"
        for field in ("doc_id", "collection_id", "chunks"):
            if field not in document:
                raise FixtureValidationError(
                    f"{document_context} missing required field {field!r}"
                )
        _require_nonempty_string(
            document["doc_id"], f"{context}.documents[{document_index}].doc_id"
        )
        _require_nonempty_string(
            document["collection_id"],
            f"{context}.documents[{document_index}].collection_id",
        )
        for chunk_index, chunk in enumerate(
            _require_sequence(
                document["chunks"],
                f"{context}.documents[{document_index}].chunks",
                nonempty=True,
            )
        ):
            chunk = _require_mapping(
                chunk, f"{context}.documents[{document_index}].chunks[{chunk_index}]"
            )
            chunk_context = f"{document_context}.chunks[{chunk_index}]"
            for field in ("chunk_id", "text"):
                if field not in chunk:
                    raise FixtureValidationError(
                        f"{chunk_context} missing required field {field!r}"
                    )
                _require_nonempty_string(
                    chunk[field],
                    f"{context}.documents[{document_index}].chunks[{chunk_index}].{field}",
                )


def _validate_expected(value: Any, context: str) -> None:
    expected = _require_mapping(value, f"{context}.expected")
    for field in ("entities", "relations", "auto_links", "suppressed_evidence"):
        if field not in expected:
            raise FixtureValidationError(
                f"{context}.expected missing required field {field!r}"
            )
        for index, record in enumerate(
            _require_sequence(expected[field], f"{context}.expected.{field}")
        ):
            _require_mapping(record, f"{context}.expected.{field}[{index}]")


def _validate_extraction_case(case: Mapping[str, Any], index: int) -> None:
    context = f"cases[{index}]"
    for field in ("id", "description", "privacy_intent", "documents", "expected"):
        if field not in case:
            raise FixtureValidationError(f"{context} missing required field {field!r}")
    _require_nonempty_string(case["id"], f"{context}.id")
    _require_nonempty_string(case["description"], f"{context}.description")
    _require_nonempty_string(case["privacy_intent"], f"{context}.privacy_intent")
    _validate_documents(case["documents"], context)
    _validate_expected(case["expected"], context)


def _validate_retrieval_case(case: Mapping[str, Any], index: int) -> None:
    context = f"cases[{index}]"
    for field in (
        "id",
        "description",
        "privacy_intent",
        "query",
        "accessible_collection_ids",
        "documents",
        "expected_retrieval_chunk_ids",
    ):
        if field not in case:
            raise FixtureValidationError(f"{context} missing required field {field!r}")
    for field in ("id", "description", "privacy_intent", "query"):
        _require_nonempty_string(case[field], f"{context}.{field}")
    for field in ("accessible_collection_ids", "expected_retrieval_chunk_ids"):
        for item_index, item in enumerate(
            _require_sequence(case[field], f"{context}.{field}", nonempty=True)
        ):
            _require_nonempty_string(item, f"{context}.{field}[{item_index}]")
    if len(set(case["expected_retrieval_chunk_ids"])) != len(
        case["expected_retrieval_chunk_ids"]
    ):
        raise FixtureValidationError(
            f"{context}.expected_retrieval_chunk_ids must be unique"
        )
    _validate_documents(case["documents"], context)
    if "baseline_vector_result_ids" in case:
        for item_index, item in enumerate(
            _require_sequence(
                case["baseline_vector_result_ids"],
                f"{context}.baseline_vector_result_ids",
            )
        ):
            _require_nonempty_string(
                item, f"{context}.baseline_vector_result_ids[{item_index}]"
            )


def _load_cases(path: Path, validator: Any) -> tuple[Mapping[str, Any], ...]:
    try:
        with path.open(encoding="utf-8") as fixture_file:
            payload = yaml.safe_load(fixture_file)
    except (OSError, yaml.YAMLError) as exc:
        raise FixtureValidationError(f"could not read fixture {path}: {exc}") from exc
    payload = _require_mapping(payload, "top level")
    if payload.get("schema_version") != 1:
        raise FixtureValidationError("top level schema_version must be 1")
    if "cases" not in payload:
        raise FixtureValidationError("top level missing required field 'cases'")
    cases = _require_sequence(payload["cases"], "top level cases", nonempty=True)
    normalized: list[Mapping[str, Any]] = []
    case_ids: set[str] = set()
    for index, raw_case in enumerate(cases):
        case = _require_mapping(raw_case, f"cases[{index}]")
        case_id = case.get("id")
        _require_nonempty_string(case_id, f"cases[{index}].id")
        if case_id in case_ids:
            raise FixtureValidationError(f"duplicate case id {case_id!r}")
        case_ids.add(case_id)
        validator(case, index)
        normalized.append(_freeze(case))
    return tuple(normalized)


def load_extraction_cases(path: Path | None = None) -> tuple[Mapping[str, Any], ...]:
    """Load validated, deep-immutable extraction fixtures without external services."""
    return _load_cases(
        path or _DEFAULT_EXTRACTION_CASES_PATH, _validate_extraction_case
    )


def load_retrieval_cases(path: Path | None = None) -> tuple[Mapping[str, Any], ...]:
    """Load validated, deep-immutable retrieval fixtures without external services."""
    return _load_cases(path or _DEFAULT_RETRIEVAL_CASES_PATH, _validate_retrieval_case)


def _structural_set(records: Any) -> set[str]:
    if records is None:
        return set()
    return {
        json.dumps(record, sort_keys=True, separators=(",", ":")) for record in records
    }


def _precision(gold: set[str], predicted: set[str]) -> float:
    return 1.0 if not predicted else len(gold & predicted) / len(predicted)


def _recall(gold: set[str], predicted: set[str]) -> float:
    return 1.0 if not gold else len(gold & predicted) / len(gold)


def score_extraction(
    case: Mapping[str, Any], predictions: Mapping[str, Any]
) -> Mapping[str, float]:
    """Return set-based offline extraction metrics for injected prediction records."""
    expected = _require_mapping(case.get("expected"), "case.expected")
    report: dict[str, float] = {}
    for name, gold_key in (("entity", "entities"), ("relation", "relations")):
        gold = _structural_set(expected.get(gold_key))
        predicted = _structural_set(predictions.get(gold_key))
        report[f"{name}_precision"] = _precision(gold, predicted)
        report[f"{name}_recall"] = _recall(gold, predicted)
    for metric_name, key in (
        ("auto_link_precision", "auto_links"),
        ("suppression_precision", "suppressed_evidence"),
    ):
        report[metric_name] = _precision(
            _structural_set(expected.get(key)), _structural_set(predictions.get(key))
        )
    return MappingProxyType(report)


def score_retrieval(
    case: Mapping[str, Any], retrieved_chunk_ids: Sequence[str]
) -> Mapping[str, float]:
    """Return recall@10 after stable de-duplication of injected retrieval output IDs."""
    gold = set(case.get("expected_retrieval_chunk_ids", ()))
    seen: set[str] = set()
    top_ten: list[str] = []
    for chunk_id in retrieved_chunk_ids:
        if chunk_id not in seen:
            seen.add(chunk_id)
            top_ten.append(chunk_id)
        if len(top_ten) == 10:
            break
    return MappingProxyType({"retrieval_recall_at_10": _recall(gold, set(top_ten))})


def build_baseline_records(
    cases: Sequence[Mapping[str, Any]],
    injected_results: Mapping[str, Sequence[str]] | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """Record actual vector IDs, returning explicit ``SKIP`` if none are available."""
    injected_results = injected_results or {}
    records: list[Mapping[str, Any]] = []
    for case in sorted(cases, key=lambda item: str(item["id"])):
        case_id = str(case["id"])
        result_ids = injected_results.get(
            case_id, case.get("baseline_vector_result_ids")
        )
        if result_ids is None:
            record: dict[str, Any] = {
                "id": case_id,
                "reason": "no fixture-backed or injected vector results",
                "status": "SKIP",
            }
        else:
            if (
                isinstance(result_ids, (str, bytes))
                or not isinstance(result_ids, Sequence)
                or not all(isinstance(item, str) and item for item in result_ids)
            ):
                raise FixtureValidationError(
                    f"baseline IDs for {case_id!r} must be a list of non-empty strings"
                )
            record = {
                "id": case_id,
                "result_ids": list(result_ids),
                "status": "RECORDED",
            }
        records.append(MappingProxyType(record))
    return tuple(records)


def _load_injected_results(path: Path | None) -> Mapping[str, Sequence[str]]:
    if path is None:
        return MappingProxyType({})
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FixtureValidationError(
            f"could not read injected retrieval results {path}: {exc}"
        ) from exc
    payload = _require_mapping(payload, "injected retrieval results")
    return _freeze(payload)


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Offline knowledge-graph evaluation runner"
    )
    parser.add_argument(
        "--extraction-cases", type=Path, default=_DEFAULT_EXTRACTION_CASES_PATH
    )
    parser.add_argument(
        "--retrieval-cases", type=Path, default=_DEFAULT_RETRIEVAL_CASES_PATH
    )
    parser.add_argument(
        "--retrieval-results",
        type=Path,
        help="JSON mapping of case ID to existing vector result IDs",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Record existing vector-only result IDs without scoring",
    )
    args = parser.parse_args(argv)
    try:
        # Validate both fixture classes for a meaningful invalid-fixture exit code.
        load_extraction_cases(args.extraction_cases)
        retrieval_cases = load_retrieval_cases(args.retrieval_cases)
        if args.baseline_only:
            payload = {
                "mode": "baseline-only",
                "records": build_baseline_records(
                    retrieval_cases, _load_injected_results(args.retrieval_results)
                ),
            }
        else:
            payload = {
                "mode": "fixtures-validated",
                "retrieval_case_ids": [case["id"] for case in retrieval_cases],
            }
    except FixtureValidationError as exc:
        print(
            json.dumps(
                {"error": str(exc), "status": "INVALID_FIXTURE"},
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 2
    print(json.dumps(_thaw(payload), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
