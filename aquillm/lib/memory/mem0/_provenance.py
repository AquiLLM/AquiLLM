"""Canonical source attribution must include a valid UUID and SHA256 digest."""
import re
from uuid import UUID


def validated_source_provenance(message_uuid, digest):
    normalized_uuid = None
    if isinstance(message_uuid, str) and 0 < len(message_uuid) <= 36:
        try:
            normalized_uuid = str(UUID(message_uuid))
        except (ValueError, TypeError, AttributeError):
            pass
    normalized_digest = (digest.lower() if isinstance(digest, str)
        and re.fullmatch(r'[0-9a-fA-F]{64}', digest) else None)
    return normalized_uuid, normalized_digest
