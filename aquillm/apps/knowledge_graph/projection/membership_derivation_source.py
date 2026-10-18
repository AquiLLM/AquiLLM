"""One import-time source identity for the fixed trusted membership derivation."""

from hashlib import sha256
from pathlib import Path

from . import (
    identifiers,
    membership_checksum,
    membership_identifiers,
    membership_source_fingerprint,
    membership_source_query,
    memberships,
    records,
    serialization,
)


def source_revision(modules):
    """Missing source disables certificates; nonsemantic edits invalidate too."""
    digest = sha256()
    try:
        for module in modules:
            path = Path(module.__file__)
            if path.suffix != ".py":
                return None
            content = path.read_bytes().replace(b"\r\n", b"\n")
            name = module.__name__.encode("utf-8")
            for value in (name, content):
                digest.update(len(value).to_bytes(8, "big"))
                digest.update(value)
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    return digest.hexdigest()


NATIVE_SOURCE_REVISION = source_revision(
    (
        identifiers,
        memberships,
        membership_checksum,
        membership_identifiers,
        serialization,
        records,
        membership_source_query,
        membership_source_fingerprint,
    )
)
