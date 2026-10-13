from dataclasses import dataclass, field
from typing import Any

from lib.parsers.source_contracts import (
    ArchiveMemberStepV1, ExtractionV1, SourceArtifactOccurrenceRefV1, SourceArtifactRefV1,
)


@dataclass
class ExtractedTextPayload:
    title: str
    normalized_type: str
    full_text: str
    modality: str = "text"
    media_bytes: bytes | None = None
    media_filename: str | None = None
    media_content_type: str | None = None
    provider: str | None = None
    model: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    source_artifact: SourceArtifactRefV1 | None = None
    extraction: ExtractionV1 | None = None
    source_occurrence: SourceArtifactOccurrenceRefV1 | None = None
    extraction_revision_id: int | None = None
    media_artifact: SourceArtifactRefV1 | None = None
    # Transient parser -> worker bytes. Never serialize these into output metadata.
    source_bytes: bytes | None = field(default=None, repr=False)
    source_filename: str | None = None
    source_content_type: str | None = None
    archive_lineage: tuple[ArchiveMemberStepV1, ...] = ()
    source_containers: tuple[tuple[str, bytes, str], ...] = field(default=(), repr=False)


class UnsupportedFileTypeError(ValueError):
    pass


class ExtractionError(RuntimeError):
    pass
