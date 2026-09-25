"""Validate versioned, offline evidence snapshots; no database or model calls."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import yaml

from apps.chat.services.rag_selection_types import SelectionCandidate, SelectionLimits


@dataclass(frozen=True)
class ReplayCase:
    identifier: str
    split: str
    family: str
    question: str
    profile: str | None
    limits: SelectionLimits
    candidates: tuple[SelectionCandidate, ...]
    labels: dict
    score_status: str
    scoring_observations: dict


def _ids(values: object) -> tuple[int, ...]:
    if not isinstance(values, list) or any(
        type(i) is not int or i <= 0 for i in values
    ):
        raise ValueError("label IDs must be positive integers")
    if len(set(values)) != len(values):
        raise ValueError("duplicate label IDs")
    return tuple(values)


def parse_case(raw: dict) -> ReplayCase:
    if not isinstance(raw, dict):
        raise ValueError("case must be a mapping")
    for key in ("id", "question", "split"):
        if not isinstance(raw.get(key), str) or not raw[key].strip():
            raise ValueError(f"missing {key}")
    if raw["split"] not in ("regression", "development", "held_out"):
        raise ValueError("unknown split")
    if raw.get("profile") not in (None, "focused", "balanced", "breadth"):
        raise ValueError("unknown profile")
    try:
        limits = SelectionLimits(**raw["limits"])
    except (KeyError, TypeError) as exc:
        raise ValueError("invalid limits") from exc
    if (
        type(limits.max_passages) is not int
        or not 0 <= limits.max_passages <= 45
        or type(limits.max_per_document) is not int
        or limits.max_per_document < 1
        or type(limits.token_budget) is not int
        or limits.token_budget < 0
    ):
        raise ValueError("invalid limits")
    source_rows = raw.get("candidates")
    if not isinstance(source_rows, list) or len(source_rows) > 45:
        raise ValueError("candidate cap exceeded or invalid pool")
    candidates = []
    for index, row in enumerate(source_rows):
        if not isinstance(row, dict):
            raise ValueError("candidate must be a mapping")
        pk, relevance = row.get("chunk_id"), row.get("relevance")
        if type(pk) is not int or pk <= 0:
            raise ValueError("invalid chunk ID")
        if (
            type(relevance) not in (float, int)
            or not math.isfinite(relevance)
            or not 0 <= relevance <= 1
        ):
            raise ValueError("invalid normalized relevance")
        text, document = row.get("text"), row.get("doc_id")
        if not isinstance(text, str) or not isinstance(document, str) or not document:
            raise ValueError("invalid candidate text/document")
        rank = row.get("fused_rank", index + 1)
        number = row.get("chunk_number", index)
        if type(rank) is not int or rank < 1 or type(number) is not int or number < 0:
            raise ValueError("invalid candidate coordinates/rank")
        doc_id = str(uuid5(NAMESPACE_URL, f"evidence-replay:{document}"))
        public = {
            "chunk_id": pk,
            "doc_id": doc_id,
            "chunk": number,
            "text": text,
            "rank": rank,
            "citation": f"[doc:{doc_id} chunk:{pk}]",
        }
        candidates.append(
            SelectionCandidate(
                pk,
                doc_id,
                number,
                text,
                float(relevance),
                rank,
                hashlib.sha256(text.encode()).hexdigest(),
                public,
            )
        )
    identifiers = {c.chunk_id for c in candidates}
    if len(identifiers) != len(candidates):
        raise ValueError("duplicate candidate IDs")
    grades = raw.get("grades", {})
    if not isinstance(grades, dict):
        raise ValueError("grades must be a mapping")
    for key, grade in grades.items():
        if (
            not str(key).isdigit()
            or int(key) <= 0
            or type(grade) is not int
            or not 0 <= grade <= 3
        ):
            raise ValueError("invalid relevance grade")
    labels = {"grades": {int(k): v for k, v in grades.items()}}
    for key in ("required_chunks", "forbidden_chunks", "expected_adaptive"):
        labels[key] = _ids(raw.get(key, []))
    if not set(labels["expected_adaptive"]) <= identifiers:
        raise ValueError("expected IDs outside candidate pool")
    for key in ("required_pairs", "redundancy_groups"):
        groups = raw.get(key, [])
        if not isinstance(groups, list):
            raise ValueError("invalid evidence groups")
        labels[key] = tuple(_ids(group) for group in groups)
        if any(len(group) < 2 for group in labels[key]):
            raise ValueError("evidence groups need at least two IDs")
    aspects = raw.get("aspects", {})
    if not isinstance(aspects, dict):
        raise ValueError("aspects must be a mapping")
    labels["aspects"] = {str(k): _ids(v) for k, v in aspects.items()}
    status = raw.get("score_status", "model")
    if status not in ("model", "rank_fallback"):
        raise ValueError("invalid score status")
    observations = raw.get("scoring_observations", {})
    if not isinstance(observations, dict) or set(observations) - {
        "reused_pairs",
        "new_pairs",
        "stage_ms",
    }:
        raise ValueError("invalid scoring observations")
    if any(
        type(v) not in (int, float) or not math.isfinite(v) or v < 0
        for v in observations.values()
    ):
        raise ValueError("invalid scoring observations")
    return ReplayCase(
        raw["id"],
        raw["split"],
        str(raw.get("family", "unspecified")),
        raw["question"],
        raw.get("profile"),
        limits,
        tuple(candidates),
        labels,
        status,
        observations,
    )


def load_cases(path: Path | str) -> tuple[ReplayCase, ...]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError("unsupported replay schema")
    if not isinstance(payload.get("cases"), list):
        raise ValueError("missing cases")
    cases = tuple(parse_case(raw) for raw in payload["cases"])
    if len({case.identifier for case in cases}) != len(cases):
        raise ValueError("duplicate case IDs")
    return cases
