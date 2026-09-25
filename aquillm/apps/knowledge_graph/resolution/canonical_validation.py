"""Exact scalar and normalized-key validation for canonical resolution."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from hashlib import sha256
from typing import TYPE_CHECKING

from .normalization import normalize_entity_label

if TYPE_CHECKING:
    from .canonical import CanonicalComponent, CanonicalDecision

_ACRONYM_PATTERN = re.compile(r"[A-Z][A-Z0-9-]{1,11}")
def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or not 1 <= value <= 2**63 - 1:
        raise ValueError(f"{label} must be a positive database integer")
    return value


def _bounded_text(
    value: object,
    label: str,
    *,
    maximum: int,
    allow_empty: bool = False,
) -> str:
    if type(value) is not str or value != value.strip() or "\x00" in value:
        raise ValueError(f"{label} must be an exact trimmed string")
    if (not value and not allow_empty) or len(value) > maximum:
        emptiness = "possibly empty" if allow_empty else "nonempty"
        raise ValueError(f"{label} must be a bounded {emptiness} string")
    return value


def _hash_payload(payload: object) -> str:
    return sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _hash_resolution_records(
    *, resolver_version: str, components: Iterable[object], decisions: Iterable[object]
) -> str:
    """Hash the v1 sorted JSON envelope without copying corpus-sized record lists."""

    digest = sha256()
    encoder = json.JSONEncoder(
        sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )
    digest.update(b'{"components":[')
    for index, record in enumerate(components):
        if index:
            digest.update(b",")
        digest.update(encoder.encode(record).encode("utf-8"))
    digest.update(b'],"decisions":[')
    for index, record in enumerate(decisions):
        if index:
            digest.update(b",")
        digest.update(encoder.encode(record).encode("utf-8"))
    digest.update(b'],"resolver_version":')
    digest.update(encoder.encode(resolver_version).encode("utf-8"))
    digest.update(b"}")
    return digest.hexdigest()


def _hash_canonical_resolution(
    *, resolver_version: str,
    components: Iterable[CanonicalComponent],
    decisions: Iterable[CanonicalDecision],
) -> str:
    """Adapt canonical result records to the stable streamed checksum envelope."""

    return _hash_resolution_records(
        resolver_version=resolver_version,
        components=(
            {
                "identity_key": item.identity_key,
                "entity_ids": list(item.entity_ids),
                "collection_ids": list(item.collection_ids),
                "label": item.label,
                "normalized_label": item.normalized_label,
                "entity_type": item.entity_type,
                "version_signature": item.version_signature,
                "method": item.method,
            }
            for item in components
        ),
        decisions=(
            {
                "left": item.left_entity_id,
                "right": item.right_entity_id,
                "score": item.score,
                "method": item.method,
                "outcome": item.outcome.value,
                "reason": item.reason,
                "evidence_key": item.evidence_key,
                "metadata": list(item.metadata),
            }
            for item in decisions
        ),
    )


def _normalized_key(value: str) -> str:
    return normalize_entity_label(value).key


def _is_acronym(value: str) -> bool:
    return bool(_ACRONYM_PATTERN.fullmatch(value))
