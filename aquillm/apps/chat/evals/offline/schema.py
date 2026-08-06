"""Schema and deterministic serialization helpers for offline evaluation data."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

_KINDS = {"routing", "evidence", "memory"}
_STRATA = {"favorable", "unfavorable", "ambiguous", "adversarial_boundary"}
_REVIEW_STATUSES = {"approved", "pending_independent_review"}
_ROUTING_REASONS = {
    "retry_request",
    "local_tool_request",
    "figure_request",
    "explicit_search",
    "collection_backed_question",
    "no_retrieval_needed",
}
_ROUTING_ACTIONS = {
    "retrieve",
    "prompt_select_collection",
    "skip_normal_tool_loop",
    "local_tool_handling",
}
_CLASSIFIER_FIELDS = {
    "requires_rag",
    "wants_figures",
    "wants_whole_document",
    "is_retry",
    "requires_local_tools",
}
_CITATION_RE = re.compile(r"\[doc:([^\]\s]+) chunk:(\d+)\]")


def canonical_json_bytes(value: object) -> bytes:
    """Serialize JSON deterministically as compact, newline-terminated UTF-8."""
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 hex digest of a file's exact bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_dataset(path: Path, kind: str, *, allow_pending: bool = False) -> dict:
    """Load YAML, validate its dataset contract, and verify its review record."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    validate_dataset(data, kind, allow_pending=allow_pending)

    review_path = path.parent / data["review"]["record"]
    if not review_path.is_file():
        raise ValueError(f"review record does not exist: {review_path.name}")
    with review_path.open("r", encoding="utf-8") as handle:
        review_record = yaml.safe_load(handle)
    _validate_review_record(review_record, data, path)
    return data


def validate_dataset(data: dict, kind: str, *, allow_pending: bool = False) -> None:
    """Validate one routing, evidence, or memory fixture dataset."""
    if kind not in _KINDS:
        raise ValueError(f"unknown dataset kind: {kind}")
    if not isinstance(data, dict):
        raise ValueError("dataset must be a mapping")

    _require_equal(data, "schema_version", "1.0")
    _require_nonempty_string(data, "dataset_id")
    frozen_at = _require_nonempty_string(data, "frozen_at")
    try:
        parsed = datetime.fromisoformat(frozen_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("frozen_at must be an ISO-8601 timestamp") from exc
    if not frozen_at.endswith("Z") or parsed.utcoffset() is None:
        raise ValueError("frozen_at must be a UTC timestamp ending in Z")
    _require_equal(data, "provenance", "synthetic_public")
    _require_equal(data, "sensitivity", "synthetic_public")
    _require_equal(data, "rubric_version", "1.0")

    annotation = _require_mapping(data, "annotation")
    _require_equal(annotation, "author_role", "fixture_author")
    _require_nonempty_string(annotation, "annotated_at")

    review = _require_mapping(data, "review")
    status = review.get("status")
    if status not in _REVIEW_STATUSES:
        raise ValueError(f"invalid review status: {status!r}")
    _require_nonempty_string(review, "record")
    if Path(review["record"]).name != review["record"] or not review["record"].endswith(
        ".yaml"
    ):
        raise ValueError("review record must be a local YAML filename")
    if status != "approved" and not allow_pending:
        raise ValueError("dataset requires approved independent review")

    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a non-empty list")
    seen: set[str] = set()
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"case {index} must be a mapping")
        case_id = _require_nonempty_string(case, "id")
        if case_id in seen:
            raise ValueError(f"duplicate case id: {case_id}")
        seen.add(case_id)
        if case.get("stratum") not in _STRATA:
            raise ValueError(f"case {case_id} has invalid stratum")
        _require_nonempty_string(case, "rationale")
        _require_mapping(case, "gold")
        if kind == "routing":
            _validate_routing_case(case)
        elif kind == "evidence":
            _validate_evidence_case(case)
        else:
            _validate_memory_case(case)


def validate_test_manifest(data: dict) -> None:
    """Validate an ordered manifest of exact pytest node IDs."""
    if not isinstance(data, dict):
        raise ValueError("test manifest must be a mapping")
    _require_equal(data, "schema_version", "1.0")
    entries = data.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("entries must be a non-empty list")
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"manifest entry {index} must be a mapping")
        node_id = _require_nonempty_string(entry, "node_id")
        if "::" not in node_id or not node_id.split("::", 1)[1]:
            raise ValueError(
                f"manifest entry must use an exact pytest node id: {node_id}"
            )
        if node_id in seen:
            raise ValueError(f"duplicate manifest node: {node_id}")
        seen.add(node_id)
        status = entry.get("status")
        if status not in {"included", "prerequisite_blocked"}:
            raise ValueError(f"manifest entry {node_id} has invalid or missing status")
        prerequisite = _require_nonempty_string(entry, "prerequisite")
        _require_nonempty_string(entry, "reason")
        if status == "included" and prerequisite != "none":
            raise ValueError(
                f"included manifest entry {node_id} prerequisite must be none"
            )
        if status == "prerequisite_blocked" and prerequisite == "none":
            raise ValueError(
                f"blocked manifest entry {node_id} requires a prerequisite"
            )


def _validate_routing_case(case: dict) -> None:
    case_id = case["id"]
    inputs = _require_mapping(case, "input")
    _require_nonempty_string(inputs, "text")
    if not isinstance(inputs.get("selected_collection_ids"), list):
        raise ValueError(
            f"routing case {case_id} selected_collection_ids must be a list"
        )
    if not isinstance(inputs.get("prior_tools"), list):
        raise ValueError(f"routing case {case_id} prior_tools must be a list")

    gold = case["gold"]
    classifier = _require_mapping(gold, "classifier")
    if set(classifier) != _CLASSIFIER_FIELDS or not all(
        isinstance(classifier[field], bool) for field in _CLASSIFIER_FIELDS
    ):
        raise ValueError(
            f"routing case {case_id} classifier fields must be exact booleans"
        )
    if gold.get("reason") not in _ROUTING_REASONS:
        raise ValueError(f"routing case {case_id} has invalid gold reason")
    if gold.get("production_action") not in _ROUTING_ACTIONS:
        raise ValueError(f"routing case {case_id} has invalid gold production action")
    if "expected_query" in gold and not isinstance(gold["expected_query"], str):
        raise ValueError(f"routing case {case_id} expected_query must be a string")


def _validate_evidence_case(case: dict) -> None:
    case_id = case["id"]
    budget = case.get("token_budget")
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise ValueError(
            f"evidence case {case_id} token_budget must be a positive integer"
        )
    candidates = case.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError(f"evidence case {case_id} candidates must be a list")

    evidence_ids: set[str] = set()
    relevant_ids: list[str] = []
    relevant_docs: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError(f"evidence case {case_id} candidate must be a mapping")
        evidence_id = _require_nonempty_string(candidate, "evidence_id")
        if evidence_id in evidence_ids:
            raise ValueError(
                f"evidence case {case_id} duplicate evidence_id: {evidence_id}"
            )
        evidence_ids.add(evidence_id)
        doc_id = _require_nonempty_string(candidate, "doc_id")
        chunk_id = candidate.get("chunk_id")
        if isinstance(chunk_id, bool) or not isinstance(chunk_id, (int, float)):
            raise ValueError(f"evidence case {case_id} chunk_id must be numeric")
        rank = candidate.get("rank")
        if isinstance(rank, bool) or not isinstance(rank, (int, float)) or rank < 1:
            raise ValueError(f"evidence case {case_id} rank must be positive numeric")
        if not isinstance(candidate.get("text"), str):
            raise ValueError(f"evidence case {case_id} text must be a string")
        citation = candidate.get("citation")
        match = _CITATION_RE.fullmatch(citation) if isinstance(citation, str) else None
        if not match:
            raise ValueError(f"evidence case {case_id} has malformed citation")
        if match.group(1) != doc_id or int(match.group(2)) != chunk_id:
            raise ValueError(
                f"evidence case {case_id} citation identity is inconsistent"
            )
        if not isinstance(candidate.get("relevant"), bool):
            raise ValueError(f"evidence case {case_id} relevant must be boolean")
        image_url = candidate.get("image_url")
        if image_url is not None and (
            not isinstance(image_url, str) or not image_url.startswith("/aquillm/")
        ):
            raise ValueError(
                f"evidence case {case_id} image_url must be a synthetic /aquillm/ path"
            )
        if candidate["relevant"]:
            relevant_ids.append(evidence_id)
            if doc_id not in relevant_docs:
                relevant_docs.append(doc_id)

    gold = case["gold"]
    gold_ids = gold.get("relevant_evidence_ids")
    gold_docs = gold.get("relevant_document_ids")
    if gold_ids != relevant_ids:
        raise ValueError(
            f"evidence case {case_id} relevant_evidence_ids do not match flags"
        )
    if gold_docs != relevant_docs:
        raise ValueError(
            f"evidence case {case_id} relevant_document_ids do not match evidence"
        )


def _validate_memory_case(case: dict) -> None:
    case_id = case["id"]
    inputs = _require_mapping(case, "input")
    for field in ("user_content", "assistant_content"):
        if not isinstance(inputs.get(field), str):
            raise ValueError(f"memory case {case_id} {field} must be a string")
    facts = case["gold"].get("normalized_facts")
    if not isinstance(facts, list) or not all(
        isinstance(fact, str) and fact for fact in facts
    ):
        raise ValueError(
            f"memory case {case_id} normalized_facts must be non-empty strings"
        )
    if len(facts) != len(set(facts)):
        raise ValueError(
            f"memory case {case_id} normalized_facts must be duplicate-free"
        )
    from lib.memory.extraction.stable_facts import clean_stable_facts

    if clean_stable_facts(facts) != facts:
        raise ValueError(
            f"memory case {case_id} gold facts are not canonically normalized"
        )


def _validate_review_record(record: Any, dataset: dict, dataset_path: Path) -> None:
    if not isinstance(record, dict):
        raise ValueError("review record must be a mapping")
    _require_equal(record, "schema_version", "1.0")
    _require_equal(record, "rubric_version", dataset["rubric_version"])
    if record.get("status") != dataset["review"]["status"]:
        raise ValueError("review record status does not match dataset")
    reviewer = _require_mapping(record, "reviewer")
    _require_equal(reviewer, "role", "independent_reviewer")
    fixture_hashes = _require_mapping(record, "fixture_hashes")
    expected_hash = fixture_hashes.get(dataset_path.name)
    if expected_hash != sha256_file(dataset_path):
        raise ValueError(f"review record fixture hash mismatch for {dataset_path.name}")
    if record["status"] == "approved":
        _require_nonempty_string(reviewer, "identity")
        _require_nonempty_string(reviewer, "reviewed_at")
        if record.get("approval") is not True:
            raise ValueError("approved review record must set approval true")
    elif record.get("approval") is not None:
        raise ValueError("pending review record approval must be null")
    for field in ("label_changes", "retained_ambiguities", "adjudications"):
        if not isinstance(record.get(field), list):
            raise ValueError(f"review record {field} must be a list")


def _require_mapping(data: dict, field: str) -> dict:
    value = data.get(field)
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a mapping")
    return value


def _require_nonempty_string(data: dict, field: str) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _require_equal(data: dict, field: str, expected: object) -> None:
    if data.get(field) != expected:
        raise ValueError(f"{field} must be {expected!r}")


__all__ = [
    "canonical_json_bytes",
    "load_dataset",
    "sha256_file",
    "validate_dataset",
    "validate_test_manifest",
]
