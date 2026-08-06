"""Schema and deterministic serialization helpers for offline evaluation data."""

from __future__ import annotations

import ast
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
_DATASET_SCHEMA_VERSION = "1.1"
_RUBRIC_VERSION = "1.1"
SOURCE_TEXT_HASH_ALGORITHM = "sha256-utf8-lf-v1"


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


def sha256_canonical_text(path: Path) -> str:
    """Hash UTF-8 source text after canonicalizing every newline form to LF."""

    text = Path(path).read_bytes().decode("utf-8")
    canonical = text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


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

    _require_equal(data, "schema_version", _DATASET_SCHEMA_VERSION)
    _require_equal(data, "dataset_id", f"{kind}-v2")
    frozen_at = _require_nonempty_string(data, "frozen_at")
    try:
        parsed = datetime.fromisoformat(frozen_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("frozen_at must be an ISO-8601 timestamp") from exc
    if not frozen_at.endswith("Z") or parsed.utcoffset() is None:
        raise ValueError("frozen_at must be a UTC timestamp ending in Z")
    _require_equal(data, "provenance", "synthetic_public")
    _require_equal(data, "sensitivity", "synthetic_public")
    _require_equal(data, "rubric_version", _RUBRIC_VERSION)

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


def validate_test_manifest(data: dict, *, project_root: Path | None = None) -> None:
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
        if "allow_skip" in entry:
            raise ValueError("test manifest entries may not define allow_skip")
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
        if project_root is not None:
            _validate_static_node(node_id, Path(project_root))


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
    direct_action = gold.get("direct_pipeline_action")
    if direct_action is not None and direct_action not in _ROUTING_ACTIONS:
        raise ValueError(f"routing case {case_id} has invalid direct_pipeline_action")

    is_retry = classifier["is_retry"]
    local = classifier["requires_local_tools"]
    figures = classifier["wants_figures"]
    requires_rag = classifier["requires_rag"]
    reason = gold["reason"]
    if is_retry:
        expected_reason = "retry_request"
    elif local:
        expected_reason = "local_tool_request"
    elif figures:
        expected_reason = "figure_request"
    elif requires_rag:
        expected_reason = reason
        if reason not in {"explicit_search", "collection_backed_question"}:
            raise ValueError(f"routing case {case_id} violates reason priority")
    else:
        expected_reason = "no_retrieval_needed"
    if reason != expected_reason:
        raise ValueError(f"routing case {case_id} violates reason priority")
    if figures and not requires_rag:
        raise ValueError(f"routing case {case_id} figure request must require RAG")
    if local and requires_rag:
        raise ValueError(
            f"routing case {case_id} local tools and RAG are mutually exclusive"
        )

    selected = inputs["selected_collection_ids"]
    expected_action = (
        "local_tool_handling"
        if local
        else "retrieve"
        if requires_rag and selected
        else "prompt_select_collection"
        if requires_rag
        else "skip_normal_tool_loop"
    )
    if gold["production_action"] != expected_action:
        raise ValueError(
            f"routing case {case_id} violates collection/action consistency"
        )


def _validate_evidence_case(case: dict) -> None:
    case_id = case["id"]
    _require_nonempty_string(case, "question")
    _require_nonempty_string(case, "answer_target")
    budget = case.get("token_budget")
    _require_positive_integer(budget, f"evidence case {case_id} token_budget")
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
        _require_positive_integer(chunk_id, f"evidence case {case_id} chunk_id")
        rank = candidate.get("rank")
        _require_positive_integer(rank, f"evidence case {case_id} rank")
        if not isinstance(candidate.get("text"), str):
            raise ValueError(f"evidence case {case_id} text must be a string")
        estimated_tokens = candidate.get("estimated_tokens")
        _require_positive_integer(
            estimated_tokens, f"evidence case {case_id} estimated_tokens"
        )
        expected_tokens = max(1, len(candidate["text"]) // 4)
        if estimated_tokens != expected_tokens:
            raise ValueError(
                f"evidence case {case_id} estimated_tokens must be {expected_tokens}"
            )
        citation = candidate.get("citation")
        match = _CITATION_RE.fullmatch(citation) if isinstance(citation, str) else None
        if not match:
            raise ValueError(f"evidence case {case_id} has malformed citation")
        if match.group(1) != doc_id or int(match.group(2)) != chunk_id:
            raise ValueError(
                f"evidence case {case_id} citation identity is inconsistent"
            )
        observed_citation = candidate.get("observed_citation")
        if observed_citation is not None and not (
            isinstance(observed_citation, str)
            and _CITATION_RE.fullmatch(observed_citation)
        ):
            raise ValueError(f"evidence case {case_id} observed_citation is malformed")
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
    for fact in facts:
        if fact != " ".join(fact.split()) or fact != fact.strip(" \t\r\n\"'"):
            raise ValueError(
                f"memory case {case_id} gold facts violate "
                "the static normalization contract"
            )
        first_alpha = next(
            (character for character in fact if character.isalpha()), None
        )
        if first_alpha is not None and not first_alpha.isupper():
            raise ValueError(f"memory case {case_id} gold facts must be sentence-cased")


def _validate_review_record(record: Any, dataset: dict, dataset_path: Path) -> None:
    if not isinstance(record, dict):
        raise ValueError("review record must be a mapping")
    _require_equal(record, "schema_version", _DATASET_SCHEMA_VERSION)
    _require_equal(record, "rubric_version", dataset["rubric_version"])
    if record.get("status") != dataset["review"]["status"]:
        raise ValueError("review record status does not match dataset")
    _require_equal(record, "reviewer_role", "independent_reviewer")
    _require_nonempty_string(record, "reviewer_identity")
    _require_nonempty_string(record, "review_date")
    if record.get("production_functions_executed") is not False:
        raise ValueError("review record must state production_functions_executed false")
    _require_equal(
        record, "source_hash_algorithm", SOURCE_TEXT_HASH_ALGORITHM
    )
    fixture_hashes = _require_mapping(record, "fixture_hashes")
    expected_hash = fixture_hashes.get(dataset_path.name)
    if expected_hash != sha256_canonical_text(dataset_path):
        raise ValueError(f"review record fixture hash mismatch for {dataset_path.name}")
    if record["status"] == "approved":
        if record.get("approval") is not True:
            raise ValueError("approved review record must set approval true")
        _validate_approved_audit(record, dataset, dataset_path)
    elif record.get("approval") is not None:
        raise ValueError("pending review record approval must be null")
    for field in ("label_changes", "retained_ambiguities", "adjudications"):
        if not isinstance(record.get(field), list):
            raise ValueError(f"review record {field} must be a list")


def _validate_approved_audit(record: dict, dataset: dict, dataset_path: Path) -> None:
    inventory = _fixture_case_inventory(dataset, dataset_path)
    case_ids = set(inventory)

    label_changes = record.get("label_changes")
    if not isinstance(label_changes, list) or not label_changes:
        raise ValueError("approved review record label_changes must be non-empty")
    for entry in label_changes:
        _validate_audit_entry(
            entry,
            "label_changes",
            case_ids,
            required=("case_id", "field", "before", "after", "reason"),
        )

    adjudications = record.get("adjudications")
    if not isinstance(adjudications, list) or not adjudications:
        raise ValueError("approved review record adjudications must be non-empty")
    for entry in adjudications:
        _validate_audit_entry(
            entry,
            "adjudications",
            case_ids,
            required=("case_id", "decision"),
        )

    evidence_entries = record.get("evidence_relevance_adjudications")
    if not isinstance(evidence_entries, list):
        raise ValueError("approved review requires complete evidence adjudications")
    evidence_cases = {
        case_id: case
        for case_id, case in inventory.items()
        if case_id.startswith("evidence-")
    }
    evidence_ids = [
        entry.get("case_id") for entry in evidence_entries if isinstance(entry, dict)
    ]
    if len(evidence_entries) != len(evidence_cases) or set(evidence_ids) != set(
        evidence_cases
    ):
        raise ValueError("approved review requires complete evidence adjudications")
    for entry in evidence_entries:
        _validate_audit_entry(
            entry,
            "evidence_relevance_adjudications",
            case_ids,
            required=("case_id", "decision", "relevant_evidence_ids"),
        )
        expected_ids = evidence_cases[entry["case_id"]]["gold"]["relevant_evidence_ids"]
        if entry["relevant_evidence_ids"] != expected_ids:
            raise ValueError(
                "evidence adjudication relevant_evidence_ids must match fixture gold"
            )

    ambiguity_entries = record.get("retained_ambiguities")
    if not isinstance(ambiguity_entries, list):
        raise ValueError("approved review requires complete retained ambiguity entries")
    expected_ambiguous = {
        case_id
        for case_id, case in inventory.items()
        if case.get("stratum") == "ambiguous"
    }
    ambiguity_ids = [
        entry.get("case_id") for entry in ambiguity_entries if isinstance(entry, dict)
    ]
    if (
        len(ambiguity_entries) != len(expected_ambiguous)
        or set(ambiguity_ids) != expected_ambiguous
    ):
        raise ValueError("approved review requires complete retained ambiguity entries")
    for entry in ambiguity_entries:
        _validate_audit_entry(
            entry,
            "retained_ambiguities",
            case_ids,
            required=("case_id", "decision"),
        )


def _fixture_case_inventory(dataset: dict, dataset_path: Path) -> dict[str, dict]:
    inventory: dict[str, dict] = {}
    for kind in sorted(_KINDS):
        sibling = dataset_path.parent / f"{kind}.yaml"
        if sibling.resolve() == dataset_path.resolve():
            sibling_data = dataset
        elif sibling.is_file():
            with sibling.open("r", encoding="utf-8") as handle:
                sibling_data = yaml.safe_load(handle)
        else:
            continue
        for case in sibling_data.get("cases", []):
            case_id = case.get("id")
            if not isinstance(case_id, str) or case_id in inventory:
                raise ValueError(
                    "fixture inventory contains invalid or duplicate case id"
                )
            inventory[case_id] = case
    return inventory


def _validate_audit_entry(
    entry: Any,
    field: str,
    case_ids: set[str],
    *,
    required: tuple[str, ...],
) -> None:
    if not isinstance(entry, dict):
        raise ValueError(f"approved review {field} entry must be a mapping")
    for key in required:
        if key not in entry:
            raise ValueError(f"approved review {field} entry missing {key}")
        if key not in {"before", "after", "relevant_evidence_ids"} and (
            not isinstance(entry[key], str) or not entry[key].strip()
        ):
            raise ValueError(f"approved review {field} entry has invalid {key}")
    if entry["case_id"] not in case_ids:
        raise ValueError(f"approved review {field} references unknown case id")
    if "relevant_evidence_ids" in required and not isinstance(
        entry["relevant_evidence_ids"], list
    ):
        raise ValueError("approved review evidence relevance entry IDs must be a list")


def _validate_static_node(node_id: str, project_root: Path) -> None:
    file_selector, *selectors = node_id.split("::")
    root = project_root.resolve()
    test_path = (root / file_selector).resolve()
    try:
        test_path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"pytest node escapes project root: {node_id}") from exc
    if not test_path.is_file() or test_path.suffix != ".py":
        raise ValueError(f"pytest node does not resolve to a test file: {node_id}")
    tree = ast.parse(test_path.read_text(encoding="utf-8"), filename=str(test_path))
    if len(selectors) == 1:
        found = any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == selectors[0]
            for node in tree.body
        )
    elif len(selectors) == 2:
        found = any(
            isinstance(node, ast.ClassDef)
            and node.name == selectors[0]
            and any(
                isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                and child.name == selectors[1]
                for child in node.body
            )
            for node in tree.body
        )
    else:
        found = False
    if not found:
        raise ValueError(f"pytest node does not resolve statically: {node_id}")


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


def _require_positive_integer(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


__all__ = [
    "SOURCE_TEXT_HASH_ALGORITHM",
    "canonical_json_bytes",
    "load_dataset",
    "sha256_canonical_text",
    "sha256_file",
    "validate_dataset",
    "validate_test_manifest",
]
