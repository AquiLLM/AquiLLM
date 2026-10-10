"""Frozen source identities and representations, independent of application I/O.

Artifact digests identify bytes. Occurrences identify an ordered archive location;
neither grants access to a document or to its output.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Literal


class SourceContractError(ValueError):
    """Source identity or archive location cannot be proven."""


@dataclass(frozen=True)
class SourceArtifactRefV1:
    sha256: str
    size_bytes: int
    storage_key: str
    member_path: str | None
    container_sha256: str | None


@dataclass(frozen=True)
class ArchiveMemberStepV1:
    container_sha256: str
    member_path: str
    archive_ordinal: int


@dataclass(frozen=True)
class SourceArtifactOccurrenceRefV1:
    occurrence_id: int
    occurrence_key: str
    artifact: SourceArtifactRefV1
    archive_lineage: tuple[ArchiveMemberStepV1, ...]


@dataclass(frozen=True)
class CellV1:
    row: int
    column: int
    address: str
    text: str | None
    formula: str | None
    value_status: Literal["literal", "cached", "uncached", "blank"]
    row_span: int = 1
    column_span: int = 1
    header_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class SourceElementV1:
    element_key: str
    kind: Literal["heading", "paragraph", "table", "formula", "figure", "separator"]
    start: int
    end: int
    page: int | None
    sheet: str | None
    slide: int | None
    parent_key: str | None
    cells: tuple[CellV1, ...] = ()
    bbox: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class ExtractionV1:
    parser_key: str
    artifact: SourceArtifactRefV1
    canonical_text: str
    elements: tuple[SourceElementV1, ...]
    quality: Literal["complete", "partial", "unsupported"]
    issue_codes: tuple[str, ...]


@dataclass(frozen=True)
class SourceChunkV1:
    start: int
    end: int
    source_text: str
    embedding_text: str
    element_keys: tuple[str, ...]
    source_sha256: str
    input_sha256: str
    modality: Literal["text", "image", "caption_only"] = "text"
    media_artifact: SourceArtifactRefV1 | None = None


def validate_member_path(path: str) -> None:
    """Reject unsafe or noncanonical archive names without altering identity."""
    if (
        not isinstance(path, str)
        or not path
        or "\\" in path
        or ":" in path
        or "\x00" in path
        or any(part in {"", ".", ".."} for part in path.split("/"))
    ):
        raise SourceContractError("unsafe_archive_member_path")
    try:
        path.encode("utf-8")
    except UnicodeError as exc:
        raise SourceContractError("unsafe_archive_member_path") from exc


def _validate_digest(digest: str) -> None:
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise SourceContractError("invalid_source_digest")


def artifact_storage_key(digest: str) -> str:
    _validate_digest(digest)
    return f"source_artifacts/v1/sha256/{digest[:2]}/{digest}"


def source_artifact_ref(
    data: bytes, *, member_path: str | None, container_sha256: str | None
) -> SourceArtifactRefV1:
    if not isinstance(data, bytes):
        raise SourceContractError("source_bytes_required")
    digest = sha256(data).hexdigest()
    ref = SourceArtifactRefV1(
        digest, len(data), artifact_storage_key(digest), member_path, container_sha256
    )
    validate_artifact_ref(ref)
    return ref


def validate_artifact_ref(
    ref: SourceArtifactRefV1, *, data: bytes | None = None
) -> None:
    if not isinstance(ref, SourceArtifactRefV1):
        raise SourceContractError("source_artifact_required")
    _validate_digest(ref.sha256)
    if (
        type(ref.size_bytes) is not int
        or ref.size_bytes < 0
        or ref.storage_key != artifact_storage_key(ref.sha256)
    ):
        raise SourceContractError("invalid_source_artifact")
    if (ref.member_path is None) != (ref.container_sha256 is None):
        raise SourceContractError("source_container_member_mismatch")
    if ref.member_path is not None:
        validate_member_path(ref.member_path)
        _validate_digest(ref.container_sha256)
    if data is not None and (
        not isinstance(data, bytes)
        or len(data) != ref.size_bytes
        or sha256(data).hexdigest() != ref.sha256
    ):
        raise SourceContractError("source_bytes_digest_mismatch")


def validate_archive_lineage(
    artifact: SourceArtifactRefV1, archive_lineage: tuple[ArchiveMemberStepV1, ...]
) -> None:
    validate_artifact_ref(artifact)
    if not isinstance(archive_lineage, tuple):
        raise SourceContractError("immutable_archive_lineage_required")
    for step in archive_lineage:
        if not isinstance(step, ArchiveMemberStepV1):
            raise SourceContractError("invalid_archive_lineage")
        _validate_digest(step.container_sha256)
        validate_member_path(step.member_path)
        if type(step.archive_ordinal) is not int or step.archive_ordinal < 0:
            raise SourceContractError("invalid_archive_ordinal")
    if archive_lineage:
        leaf = archive_lineage[-1]
        if (
            leaf.container_sha256 != artifact.container_sha256
            or leaf.member_path != artifact.member_path
        ):
            raise SourceContractError("source_lineage_mismatch")
    elif artifact.member_path is not None or artifact.container_sha256 is not None:
        raise SourceContractError("source_lineage_mismatch")


def occurrence_key(
    artifact: SourceArtifactRefV1,
    *,
    original_filename: str,
    content_type: str,
    archive_lineage: tuple[ArchiveMemberStepV1, ...],
) -> str:
    """SHA256 of canonical JSON v1; duplicate ordinals remain distinct."""
    validate_archive_lineage(artifact, archive_lineage)
    for value in (original_filename, content_type):
        if not isinstance(value, str) or "\x00" in value:
            raise SourceContractError("invalid_source_name_or_content_type")
        try:
            value.encode("utf-8")
        except UnicodeError as exc:
            raise SourceContractError("invalid_source_name_or_content_type") from exc
    if not original_filename:
        raise SourceContractError("original_filename_required")
    manifest = {
        "version": "source-occurrence-v1",
        "artifact_sha256": artifact.sha256,
        "archive_lineage": [asdict(step) for step in archive_lineage],
        "original_filename": original_filename,
        "content_type": content_type,
    }
    try:
        wire = json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except UnicodeError as exc:
        raise SourceContractError("invalid_source_name_or_content_type") from exc
    return sha256(wire).hexdigest()
