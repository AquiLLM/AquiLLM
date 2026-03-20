"""
Parser configuration and file type detection.
"""

import mimetypes
import os

DOCUMENT_EXTENSIONS = {
    ".pdf",
    ".doc",
    ".docx",
    ".odt",
    ".rtf",
    ".txt",
    ".md",
    ".html",
    ".htm",
    ".epub",
}
TABULAR_EXTENSIONS = {".csv", ".tsv", ".xls", ".xlsx", ".ods"}
PRESENTATION_EXTENSIONS = {".ppt", ".pptx", ".odp"}
STRUCTURED_EXTENSIONS = {".json", ".jsonl", ".xml", ".yaml", ".yml"}
TRANSCRIPT_EXTENSIONS = {".vtt", ".srt"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".heic", ".heif"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi", ".mpeg", ".mpg"}

SUPPORTED_EXTENSIONS = (
    DOCUMENT_EXTENSIONS
    | TABULAR_EXTENSIONS
    | PRESENTATION_EXTENSIONS
    | STRUCTURED_EXTENSIONS
    | TRANSCRIPT_EXTENSIONS
    | IMAGE_EXTENSIONS
    | AUDIO_EXTENSIONS
    | VIDEO_EXTENSIONS
    | {".zip"}
)


def clean_filename(filename: str) -> str:
    """Clean and normalize filename."""
    base = os.path.basename(filename or "").strip()
    if not base:
        return "upload"
    return base


def get_stem(filename: str) -> str:
    """Get filename stem (without extension)."""
    clean = clean_filename(filename)
    stem, _ = os.path.splitext(clean)
    return stem or "upload"


def guess_content_type(filename: str, fallback: str = "application/octet-stream") -> str:
    """Guess content type from filename."""
    guessed = mimetypes.guess_type(filename)[0]
    return guessed or fallback


def detect_ingest_type(filename: str, content_type: str | None = None) -> str:
    """Detect ingestion type from filename and content type."""
    extension = os.path.splitext(clean_filename(filename))[1].lower()
    if extension in DOCUMENT_EXTENSIONS:
        return "document"
    if extension in TABULAR_EXTENSIONS:
        return "tabular"
    if extension in PRESENTATION_EXTENSIONS:
        return "presentation"
    if extension in STRUCTURED_EXTENSIONS:
        return "structured"
    if extension in TRANSCRIPT_EXTENSIONS:
        return "transcript"
    if extension in IMAGE_EXTENSIONS:
        return "image"
    if extension in AUDIO_EXTENSIONS:
        return "audio"
    if extension in VIDEO_EXTENSIONS:
        return "video"
    if extension == ".zip":
        return "archive"

    normalized_ct = (content_type or "").lower()
    if normalized_ct.startswith("image/"):
        return "image"
    if normalized_ct.startswith("audio/"):
        return "audio"
    if normalized_ct.startswith("video/"):
        return "video"
    if normalized_ct in ("text/csv", "text/tab-separated-values"):
        return "tabular"
    if normalized_ct in ("application/json", "application/xml", "text/xml"):
        return "structured"
    if normalized_ct in ("text/plain", "text/markdown", "text/html"):
        return "document"

    raise ValueError(f"Unsupported file type for {filename!r} ({content_type or 'unknown content-type'})")


__all__ = [
    'DOCUMENT_EXTENSIONS',
    'TABULAR_EXTENSIONS',
    'PRESENTATION_EXTENSIONS',
    'STRUCTURED_EXTENSIONS',
    'TRANSCRIPT_EXTENSIONS',
    'IMAGE_EXTENSIONS',
    'AUDIO_EXTENSIONS',
    'VIDEO_EXTENSIONS',
    'SUPPORTED_EXTENSIONS',
    'clean_filename',
    'get_stem',
    'guess_content_type',
    'detect_ingest_type',
]
