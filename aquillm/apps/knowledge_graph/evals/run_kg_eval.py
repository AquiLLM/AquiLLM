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

Retrieval-result JSON maps each case ID either to a simple result-ID list or to
``{"result_ids": [...], "id_collections": {"<id>": "<collection_id>"}}``.
Native integer IDs need this optional collection evidence to receive a resolved
security status; JSON object keys use the string form of each result ID.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Hashable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

_HERE = Path(__file__).resolve().parent
_DEFAULT_EXTRACTION_CASES_PATH = _HERE / "extraction_cases.yaml"
_DEFAULT_RETRIEVAL_CASES_PATH = _HERE / "retrieval_cases.yaml"


class FixtureValidationError(ValueError):
    """Raised when an offline gold fixture cannot be evaluated deterministically."""


def _freeze(value: Any, active_ids: set[int] | None = None) -> Any:
    """Deep-freeze JSON-like fixture data without coercing YAML key types."""
    active_ids = active_ids if active_ids is not None else set()
    if isinstance(value, (Mapping, list, tuple)):
        identity = id(value)
        if identity in active_ids:
            raise FixtureValidationError("recursive fixture aliases are not supported")
        active_ids.add(identity)
        try:
            if isinstance(value, Mapping):
                if not all(isinstance(key, str) for key in value):
                    raise FixtureValidationError("mapping keys must be strings")
                return MappingProxyType(
                    {
                        key: _freeze(value[key], active_ids)
                        for key in sorted(value)
                    }
                )
            return tuple(_freeze(item, active_ids) for item in value)
        finally:
            active_ids.remove(identity)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise FixtureValidationError(
        f"unsupported fixture value type {type(value).__name__}"
    )


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


def _require_fields(
    record: Mapping[str, Any], fields: Sequence[str], context: str
) -> None:
    for field in fields:
        if field not in record:
            raise FixtureValidationError(f"{context} missing required field {field!r}")


def _validate_documents(value: Any, context: str) -> dict[str, tuple[str, str]]:
    """Return validated ``chunk_id -> (text, collection_id)`` evidence."""
    document_ids: set[str] = set()
    chunks: dict[str, tuple[str, str]] = {}
    for document_index, document in enumerate(
        _require_sequence(value, f"{context}.documents", nonempty=True)
    ):
        document_context = f"{context}.documents[{document_index}]"
        document = _require_mapping(document, document_context)
        _require_fields(
            document, ("doc_id", "collection_id", "chunks"), document_context
        )
        document_id = _require_nonempty_string(
            document["doc_id"], f"{document_context}.doc_id"
        )
        if document_id in document_ids:
            raise FixtureValidationError(
                f"{document_context} has duplicate document id {document_id!r}"
            )
        document_ids.add(document_id)
        collection_id = _require_nonempty_string(
            document["collection_id"], f"{document_context}.collection_id"
        )
        for chunk_index, chunk in enumerate(
            _require_sequence(
                document["chunks"], f"{document_context}.chunks", nonempty=True
            )
        ):
            chunk_context = f"{document_context}.chunks[{chunk_index}]"
            chunk = _require_mapping(chunk, chunk_context)
            _require_fields(chunk, ("chunk_id", "text"), chunk_context)
            chunk_id = _require_nonempty_string(
                chunk["chunk_id"], f"{chunk_context}.chunk_id"
            )
            text = _require_nonempty_string(chunk["text"], f"{chunk_context}.text")
            if chunk_id in chunks:
                raise FixtureValidationError(
                    f"{chunk_context} has duplicate chunk id {chunk_id!r}"
                )
            chunks[chunk_id] = (text, collection_id)
    return chunks


def _validate_expected(
    value: Any, context: str, chunks: Mapping[str, tuple[str, str]]
) -> None:
    expected = _require_mapping(value, f"{context}.expected")
    for field in ("entities", "relations", "auto_links", "suppressed_evidence"):
        if field not in expected:
            raise FixtureValidationError(
                f"{context}.expected missing required field {field!r}"
            )
        _require_sequence(expected[field], f"{context}.expected.{field}")

    entity_ids: set[str] = set()
    for index, raw_entity in enumerate(expected["entities"]):
        entity_context = f"{context}.expected.entities[{index}]"
        entity = _require_mapping(raw_entity, entity_context)
        _require_fields(
            entity,
            ("id", "text", "type", "chunk_id", "start", "end"),
            entity_context,
        )
        entity_id = _require_nonempty_string(entity["id"], f"{entity_context}.id")
        text = _require_nonempty_string(entity["text"], f"{entity_context}.text")
        _require_nonempty_string(entity["type"], f"{entity_context}.type")
        chunk_id = _require_nonempty_string(
            entity["chunk_id"], f"{entity_context}.chunk_id"
        )
        start = entity["start"]
        end = entity["end"]
        if type(start) is not int:
            raise FixtureValidationError(f"{entity_context}.start must be an integer")
        if type(end) is not int:
            raise FixtureValidationError(f"{entity_context}.end must be an integer")
        if start < 0:
            raise FixtureValidationError(
                f"{entity_context}.start must be non-negative"
            )
        if end <= start:
            raise FixtureValidationError(
                f"{entity_context}.end must be greater than start"
            )
        if entity_id in entity_ids:
            raise FixtureValidationError(
                f"{entity_context} has duplicate entity id {entity_id!r}"
            )
        if chunk_id not in chunks:
            raise FixtureValidationError(
                f"{entity_context} references unknown chunk {chunk_id!r}"
            )
        chunk_text = chunks[chunk_id][0]
        if start >= len(chunk_text):
            raise FixtureValidationError(
                f"{entity_context}.start is outside referenced chunk {chunk_id!r}"
            )
        if end > len(chunk_text):
            raise FixtureValidationError(
                f"{entity_context}.end is outside referenced chunk {chunk_id!r}"
            )
        if chunk_text[start:end] != text:
            raise FixtureValidationError(
                f"{entity_context} span does not exactly match text"
            )
        entity_ids.add(entity_id)

    for field in ("relations", "auto_links"):
        seen: set[tuple[str, str, str]] = set()
        for index, raw_record in enumerate(expected[field]):
            record_context = f"{context}.expected.{field}[{index}]"
            record = _require_mapping(raw_record, record_context)
            _require_fields(record, ("source", "target", "type"), record_context)
            source = _require_nonempty_string(
                record["source"], f"{record_context}.source"
            )
            target = _require_nonempty_string(
                record["target"], f"{record_context}.target"
            )
            relation_type = _require_nonempty_string(
                record["type"], f"{record_context}.type"
            )
            if source not in entity_ids or target not in entity_ids:
                raise FixtureValidationError(
                    f"{record_context} references unknown entity"
                )
            identity = (source, target, relation_type)
            if identity in seen:
                raise FixtureValidationError(
                    f"{record_context} has duplicate {field[:-1]} record"
                )
            seen.add(identity)

    seen_suppression: set[tuple[str, str, str, str]] = set()
    for index, raw_evidence in enumerate(expected["suppressed_evidence"]):
        evidence_context = f"{context}.expected.suppressed_evidence[{index}]"
        evidence = _require_mapping(raw_evidence, evidence_context)
        _require_fields(
            evidence, ("entity", "type", "chunk_id", "reason"), evidence_context
        )
        entity = _require_nonempty_string(
            evidence["entity"], f"{evidence_context}.entity"
        )
        evidence_type = _require_nonempty_string(
            evidence["type"], f"{evidence_context}.type"
        )
        chunk_id = _require_nonempty_string(
            evidence["chunk_id"], f"{evidence_context}.chunk_id"
        )
        reason = _require_nonempty_string(
            evidence["reason"], f"{evidence_context}.reason"
        )
        if chunk_id not in chunks:
            raise FixtureValidationError(
                f"{evidence_context} references unknown chunk {chunk_id!r}"
            )
        if entity.lower() not in chunks[chunk_id][0].lower():
            raise FixtureValidationError(
                f"{evidence_context}.entity is not anchored in chunk {chunk_id!r}"
            )
        identity = (entity, evidence_type, chunk_id, reason)
        if identity in seen_suppression:
            raise FixtureValidationError(
                f"{evidence_context} has duplicate suppression record"
            )
        seen_suppression.add(identity)


def _validate_extraction_case(case: Mapping[str, Any], index: int) -> None:
    context = f"cases[{index}]"
    for field in ("id", "description", "privacy_intent", "documents", "expected"):
        if field not in case:
            raise FixtureValidationError(f"{context} missing required field {field!r}")
    _require_nonempty_string(case["id"], f"{context}.id")
    _require_nonempty_string(case["description"], f"{context}.description")
    _require_nonempty_string(case["privacy_intent"], f"{context}.privacy_intent")
    chunks = _validate_documents(case["documents"], context)
    _validate_expected(case["expected"], context, chunks)


def _validate_retrieval_case(case: Mapping[str, Any], index: int) -> None:
    context = f"cases[{index}]"
    _require_fields(
        case,
        (
            "id",
            "description",
            "privacy_intent",
            "query",
            "accessible_collection_ids",
            "documents",
            "expected_retrieval_chunk_ids",
        ),
        context,
    )
    for field in ("id", "description", "privacy_intent", "query"):
        _require_nonempty_string(case[field], f"{context}.{field}")
    accessible_collections: set[str] = set()
    for item_index, item in enumerate(
        _require_sequence(
            case["accessible_collection_ids"],
            f"{context}.accessible_collection_ids",
            nonempty=True,
        )
    ):
        collection_id = _require_nonempty_string(
            item, f"{context}.accessible_collection_ids[{item_index}]"
        )
        if collection_id in accessible_collections:
            raise FixtureValidationError(
                f"{context} has duplicate accessible collection id {collection_id!r}"
            )
        accessible_collections.add(collection_id)
    chunks = _validate_documents(case["documents"], context)
    expected_ids: set[str] = set()
    for item_index, item in enumerate(
        _require_sequence(
            case["expected_retrieval_chunk_ids"],
            f"{context}.expected_retrieval_chunk_ids",
            nonempty=False,
        )
    ):
        chunk_id = _require_nonempty_string(
            item, f"{context}.expected_retrieval_chunk_ids[{item_index}]"
        )
        if chunk_id not in chunks:
            raise FixtureValidationError(
                f"{context} references unknown chunk {chunk_id!r}"
            )
        if chunks[chunk_id][1] not in accessible_collections:
            raise FixtureValidationError(
                f"{context} retrieval evidence is not in an accessible collection"
            )
        if chunk_id in expected_ids:
            raise FixtureValidationError(
                f"{context} has duplicate retrieval chunk id {chunk_id!r}"
            )
        expected_ids.add(chunk_id)
    if "baseline_vector_result_ids" in case:
        baseline_ids: set[str | int] = set()
        for item_index, item in enumerate(
            _require_sequence(
                case["baseline_vector_result_ids"],
                f"{context}.baseline_vector_result_ids",
            )
        ):
            item_context = f"{context}.baseline_vector_result_ids[{item_index}]"
            if isinstance(item, str):
                result_id: str | int = _require_nonempty_string(item, item_context)
            elif type(item) is int and item > 0:
                result_id = item
            else:
                raise FixtureValidationError(
                    f"{item_context} must be a positive integer or non-empty string"
                )
            if result_id in baseline_ids:
                raise FixtureValidationError(
                    f"{context} has duplicate baseline result id {result_id!r}"
                )
            baseline_ids.add(result_id)
    if "canonical_identity_links" in case:
        seen_links: set[tuple[str, str, str]] = set()
        for link_index, raw_link in enumerate(
            _require_sequence(
                case["canonical_identity_links"],
                f"{context}.canonical_identity_links",
            )
        ):
            link_context = f"{context}.canonical_identity_links[{link_index}]"
            link = _require_mapping(raw_link, link_context)
            _require_fields(
                link, ("source_chunk_id", "target_chunk_id", "type"), link_context
            )
            source = _require_nonempty_string(
                link["source_chunk_id"], f"{link_context}.source_chunk_id"
            )
            target = _require_nonempty_string(
                link["target_chunk_id"], f"{link_context}.target_chunk_id"
            )
            link_type = _require_nonempty_string(link["type"], f"{link_context}.type")
            if source not in chunks or target not in chunks:
                raise FixtureValidationError(f"{link_context} references unknown chunk")
            identity = (source, target, link_type)
            if identity in seen_links:
                raise FixtureValidationError(
                    f"{link_context} has duplicate canonical identity link"
                )
            seen_links.add(identity)


def _load_cases(path: Path, validator: Any) -> tuple[Mapping[str, Any], ...]:
    try:
        with path.open(encoding="utf-8") as fixture_file:
            payload = yaml.safe_load(fixture_file)
    except (OSError, yaml.YAMLError) as exc:
        raise FixtureValidationError(f"could not read fixture {path}: {exc}") from exc
    payload = _require_mapping(_freeze(payload), "top level")
    schema_version = payload.get("schema_version")
    if type(schema_version) is not int or schema_version != 1:
        raise FixtureValidationError("top level schema_version must be integer 1")
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
        normalized.append(case)
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
        json.dumps(_thaw(record), sort_keys=True, separators=(",", ":"))
        for record in records
    }


def _entity_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    """Prediction entity contract: type, text, chunk_id, start, and end.

    IDs and confidence are intentionally ignored; ``id`` is only a legacy fallback
    for old hand-written unit predictions that omit all semantic fields.
    """
    if all(key in record for key in ("type", "text", "chunk_id")):
        return (
            record["type"],
            str(record["text"]).casefold(),
            record["chunk_id"],
            record.get("start"),
            record.get("end"),
        )
    return ("legacy-id", record.get("id"))


def _entity_set(records: Any) -> set[tuple[Any, ...]]:
    return {
        _entity_key(_require_mapping(record, "prediction entity"))
        for record in records or ()
    }


def _relation_set(records: Any, entities: Any) -> set[tuple[Any, ...]]:
    by_id = {
        record.get("id"): _entity_key(_require_mapping(record, "prediction entity"))
        for record in entities or ()
    }
    result = set()
    for record in records or ():
        record = _require_mapping(record, "prediction relation")
        result.add(
            (
                record.get("type"),
                by_id.get(record.get("source"), record.get("source")),
                by_id.get(record.get("target"), record.get("target")),
            )
        )
    return result


def _precision(gold: AbstractSet[Hashable], predicted: AbstractSet[Hashable]) -> float:
    return 1.0 if not predicted else len(gold & predicted) / len(predicted)


def _recall(gold: AbstractSet[Hashable], predicted: AbstractSet[Hashable]) -> float:
    return 1.0 if not gold else len(gold & predicted) / len(gold)


def score_extraction(
    case: Mapping[str, Any], predictions: Mapping[str, Any]
) -> Mapping[str, float]:
    """Return set-based offline extraction metrics for injected prediction records."""
    expected = _require_mapping(case.get("expected"), "case.expected")
    report: dict[str, float] = {}
    gold_entities = _entity_set(expected.get("entities"))
    predicted_entities = _entity_set(predictions.get("entities"))
    report["entity_precision"] = _precision(gold_entities, predicted_entities)
    report["entity_recall"] = _recall(gold_entities, predicted_entities)
    gold_relations = _relation_set(expected.get("relations"), expected.get("entities"))
    predicted_relations = _relation_set(
        predictions.get("relations"), predictions.get("entities")
    )
    report["relation_precision"] = _precision(gold_relations, predicted_relations)
    report["relation_recall"] = _recall(gold_relations, predicted_relations)
    gold_links = _relation_set(expected.get("auto_links"), expected.get("entities"))
    predicted_links = _relation_set(
        predictions.get("auto_links"), predictions.get("entities")
    )
    report["auto_link_precision"] = _precision(gold_links, predicted_links)
    report["suppression_precision"] = _precision(
        _structural_set(expected.get("suppressed_evidence")),
        _structural_set(predictions.get("suppressed_evidence")),
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
    injected_results: Mapping[str, Any] | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """Record vector IDs and classify security only from collection evidence."""
    injected_results = injected_results or {}
    records: list[Mapping[str, Any]] = []
    for case in sorted(cases, key=lambda item: str(item["id"])):
        case_id = str(case["id"])
        raw_results = injected_results.get(
            case_id, case.get("baseline_vector_result_ids")
        )
        if raw_results is None:
            record: dict[str, Any] = {
                "id": case_id,
                "reason": "no fixture-backed or injected vector results",
                "status": "SKIP",
            }
        else:
            id_collections: Mapping[str, Any] = MappingProxyType({})
            if isinstance(raw_results, Mapping):
                _require_fields(
                    raw_results, ("result_ids",), f"results for {case_id!r}"
                )
                result_ids = raw_results["result_ids"]
                if "id_collections" in raw_results:
                    id_collections = _require_mapping(
                        raw_results["id_collections"],
                        f"id_collections for {case_id!r}",
                    )
                    for result_key, collection_id in id_collections.items():
                        _require_nonempty_string(
                            result_key,
                            f"id_collections key for {case_id!r}",
                        )
                        _require_nonempty_string(
                            collection_id,
                            f"collection for result {result_key!r}",
                        )
            else:
                result_ids = raw_results
            if isinstance(result_ids, (str, bytes)) or not isinstance(
                result_ids, Sequence
            ):
                raise FixtureValidationError(
                    f"baseline IDs for {case_id!r} must be a list of valid IDs"
                )
            deduped: list[str | int] = []
            for item in result_ids:
                if not (
                    (isinstance(item, str) and item)
                    or (
                        isinstance(item, int)
                        and not isinstance(item, bool)
                        and item > 0
                    )
                ):
                    raise FixtureValidationError(
                        f"baseline IDs for {case_id!r} must be valid IDs"
                    )
                if item not in deduped:
                    deduped.append(item)
            chunk_collections = {
                chunk["chunk_id"]: document["collection_id"]
                for document in case.get("documents", ())
                for chunk in document.get("chunks", ())
            }
            accessible = set(case.get("accessible_collection_ids", ()))
            inaccessible: list[str | int] = []
            unresolved: list[str | int] = []
            for item in deduped:
                collection_id = (
                    chunk_collections.get(item)
                    if isinstance(item, str) and item in chunk_collections
                    else id_collections.get(str(item))
                )
                if collection_id is None:
                    unresolved.append(item)
                elif collection_id not in accessible:
                    inaccessible.append(item)
            if inaccessible:
                security_status = "LEAKAGE"
            elif unresolved:
                security_status = "UNKNOWN"
            else:
                security_status = "OK"
            record = {
                "id": case_id,
                "result_ids": tuple(deduped),
                "inaccessible_result_ids": tuple(inaccessible),
                "inaccessible_result_count": len(inaccessible),
                "unresolved_result_ids": tuple(unresolved),
                "unresolved_result_count": len(unresolved),
                "security_status": security_status,
                "status": "RECORDED",
            }
        records.append(MappingProxyType(record))
    return tuple(records)


def _load_injected_results(path: Path | None) -> Mapping[str, Any]:
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
        help=(
            "JSON mapping of case ID to result IDs, optionally with ID-to-collection "
            "evidence"
        ),
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
        injected_results = _load_injected_results(args.retrieval_results)
        unknown_case_ids = set(injected_results) - {
            case["id"] for case in retrieval_cases
        }
        if unknown_case_ids:
            raise FixtureValidationError(
                f"unknown injected case IDs: {sorted(unknown_case_ids)!r}"
            )
        if args.baseline_only:
            payload = {
                "mode": "baseline-only",
                "records": build_baseline_records(retrieval_cases, injected_results),
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
