"""Immutable canonical association decision checksums."""

import json
from hashlib import sha256

from django.core.exceptions import ValidationError


def canonical_link_decision_checksum(link: object) -> str:
    """Hash one immutable canonical decision without its lifecycle status."""

    payload = {
        "canonical_entity_id": getattr(link, "canonical_entity_id", None),
        "collection_entity_id": getattr(link, "collection_entity_id", None),
        "metadata": getattr(link, "metadata", None),
        "method": getattr(link, "method", None),
        "outcome": getattr(link, "outcome", None),
        "reason": getattr(link, "reason", None),
        "resolver_version": getattr(link, "resolver_version", None),
        "score": getattr(link, "score", None),
    }
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            {"decision_checksum": "Canonical decision is not serializable."}
        ) from exc
    return sha256(encoded).hexdigest()
