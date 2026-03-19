from dataclasses import dataclass, field
from typing import Any


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


class UnsupportedFileTypeError(ValueError):
    pass


class ExtractionError(RuntimeError):
    pass
