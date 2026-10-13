"""
File parsing and extraction for ingestion pipeline.

This module integrates lib/parsers with the full ingestion pipeline,
including figure extraction and media transcription.
"""

import io
import structlog
import os
from dataclasses import replace

from aquillm.ocr_utils import extract_text_from_image
from aquillm.vtt import coalesce_captions, parse as parse_vtt, to_text as vtt_to_text

from .media import transcribe_media_bytes
from .parser_archive import ArchiveBudget, extract_archive_payloads
from .parser_figures import extract_figure_payloads_for_format
from .types import ExtractedTextPayload, ExtractionError, UnsupportedFileTypeError
from lib.parsers.source_contracts import ArchiveMemberStepV1, ExtractionV1, SourceElementV1, source_artifact_ref
from lib.parsers.structural import extract_structure

# Import from lib/parsers
from lib.parsers import (
    DOCUMENT_EXTENSIONS,
    TABULAR_EXTENSIONS,
    PRESENTATION_EXTENSIONS,
    STRUCTURED_EXTENSIONS,
    TRANSCRIPT_EXTENSIONS,
    IMAGE_EXTENSIONS,
    AUDIO_EXTENSIONS,
    VIDEO_EXTENSIONS,
    SUPPORTED_EXTENSIONS,
    clean_filename as _clean_name,
    get_stem as _stem,
    guess_content_type as _guess_content_type,
    detect_ingest_type as _detect_ingest_type_lib,
    read_text_bytes as _read_text_bytes,
    extract_html_text,
    extract_epub_text,
    extract_odp_text,
    extract_json_text,
    extract_jsonl_text,
    extract_xml_text,
    extract_yaml_text,
    iter_srt_text,
)

logger = structlog.stdlib.get_logger(__name__)


def detect_ingest_type(filename: str, content_type: str | None = None) -> str:
    try:
        return _detect_ingest_type_lib(filename, content_type)
    except ValueError as e:
        raise UnsupportedFileTypeError(str(e)) from e


def _native_text_payload(filename, data, normalized_type):
    artifact = source_artifact_ref(data, member_path=None, container_sha256=None)
    extraction = extract_structure(filename, data, artifact=artifact)
    return ExtractedTextPayload(title=_stem(filename), normalized_type=normalized_type,
                                full_text=extraction.canonical_text, extraction=extraction)


def _extract_pdf(filename: str, data: bytes) -> list[ExtractedTextPayload]:
    """Extract text and figures from a PDF."""
    payloads: list[ExtractedTextPayload] = []
    payloads.append(_native_text_payload(filename, data, 'pdf'))
    
    extract_figure_payloads_for_format(filename, data, "pdf", payloads)
    
    return payloads


def _extract_html(filename: str, data: bytes) -> ExtractedTextPayload:
    return ExtractedTextPayload(title=_stem(filename), normalized_type="html", full_text=extract_html_text(data))


def _extract_json(filename: str, data: bytes) -> ExtractedTextPayload:
    return ExtractedTextPayload(title=_stem(filename), normalized_type="json", full_text=extract_json_text(data))


def _extract_jsonl(filename: str, data: bytes) -> ExtractedTextPayload:
    return ExtractedTextPayload(title=_stem(filename), normalized_type="jsonl", full_text=extract_jsonl_text(data))


def _extract_xml(filename: str, data: bytes) -> ExtractedTextPayload:
    return ExtractedTextPayload(title=_stem(filename), normalized_type="xml", full_text=extract_xml_text(data))


def _extract_yaml(filename: str, data: bytes) -> ExtractedTextPayload:
    return ExtractedTextPayload(title=_stem(filename), normalized_type="yaml", full_text=extract_yaml_text(data))


def _extract_csv_like(filename: str, data: bytes, delimiter: str) -> ExtractedTextPayload:
    return _native_text_payload(filename, data, 'tabular')


def _extract_docx(filename: str, data: bytes) -> list[ExtractedTextPayload]:
    payloads = [_native_text_payload(filename, data, 'docx')]
    extract_figure_payloads_for_format(filename, data, "docx", payloads)
    return payloads


def _extract_xlsx(filename: str, data: bytes) -> list[ExtractedTextPayload]:
    payloads = [_native_text_payload(filename, data, 'xlsx')]
    extract_figure_payloads_for_format(filename, data, "xlsx", payloads)
    return payloads


def _extract_xls(filename: str, data: bytes) -> ExtractedTextPayload:
    return _native_text_payload(filename, data, 'xls')


def _extract_ods(filename: str, data: bytes) -> list[ExtractedTextPayload]:
    payloads = [_native_text_payload(filename, data, 'ods')]
    extract_figure_payloads_for_format(filename, data, "ods", payloads)
    return payloads


def _extract_pptx(filename: str, data: bytes) -> list[ExtractedTextPayload]:
    payloads = [_native_text_payload(filename, data, 'pptx')]
    extract_figure_payloads_for_format(filename, data, "pptx", payloads)
    return payloads


def _extract_odp(filename: str, data: bytes) -> ExtractedTextPayload:
    return ExtractedTextPayload(title=_stem(filename), normalized_type="odp", full_text=extract_odp_text(data))


def _extract_epub(filename: str, data: bytes) -> list[ExtractedTextPayload]:
    text = extract_epub_text(data)
    payloads = [ExtractedTextPayload(title=_stem(filename), normalized_type="epub", full_text=text)]
    extract_figure_payloads_for_format(filename, data, "epub", payloads)
    return payloads


def _extract_vtt(filename: str, data: bytes) -> ExtractedTextPayload:
    captions = parse_vtt(io.BytesIO(data))
    text = vtt_to_text(coalesce_captions(captions))
    return ExtractedTextPayload(title=_stem(filename), normalized_type="vtt", full_text=text)


def _extract_srt(filename: str, data: bytes) -> ExtractedTextPayload:
    text = _read_text_bytes(data)
    lines = list(iter_srt_text(text.splitlines()))
    return ExtractedTextPayload(title=_stem(filename), normalized_type="srt", full_text="\n".join(lines))


def _extract_image_ocr(filename: str, data: bytes) -> ExtractedTextPayload:
    result = extract_text_from_image(data, convert_to_latex=False)
    text = (result.get("extracted_text") or "").strip()
    if not text:
        raise ExtractionError("OCR returned empty text.")
    clean_name = _clean_name(filename)
    return ExtractedTextPayload(
        title=_stem(filename),
        normalized_type="image_ocr",
        full_text=text,
        modality="image",
        media_bytes=data,
        media_filename=clean_name,
        media_content_type=_guess_content_type(clean_name, fallback="image/jpeg"),
        provider=(result.get("provider") or "") or None,
        model=(result.get("model") or "") or None,
    )


def _extract_media_transcript(filename: str, data: bytes, normalized_type: str) -> ExtractedTextPayload:
    transcript = transcribe_media_bytes(data, filename=filename)
    clean_name = _clean_name(filename)
    modality = "video" if normalized_type == "video_transcript" else "audio"
    return ExtractedTextPayload(
        title=_stem(filename),
        normalized_type=normalized_type,
        full_text=(transcript or "").strip(),
        modality=modality,
        media_bytes=data,
        media_filename=clean_name,
        media_content_type=_guess_content_type(clean_name),
        provider=((os.getenv("INGEST_TRANSCRIBE_PROVIDER") or "").strip().lower() or None),
        model=((os.getenv("INGEST_TRANSCRIBE_MODEL") or "").strip() or None),
    )


def extract_text_payloads(
    filename: str, data: bytes, content_type: str | None = None, depth: int = 0, *,
    _archive_lineage: tuple[ArchiveMemberStepV1, ...] = (),
    _source_containers: tuple[tuple[str, bytes, str], ...] = (),
    _archive_budget: ArchiveBudget | None = None,
) -> list[ExtractedTextPayload]:
    if depth > 2:
        raise ExtractionError("Nested archive depth exceeded.")
    if detect_ingest_type(filename, content_type) == "archive":
        return extract_archive_payloads(
            filename, data, depth, archive_lineage=_archive_lineage,
            source_containers=_source_containers, budget=_archive_budget,
            content_type=content_type,
        )
    payloads = _extract_unbound_text_payloads(filename, data, content_type, depth)
    leaf = _archive_lineage[-1] if _archive_lineage else None
    artifact = source_artifact_ref(
        data, member_path=leaf.member_path if leaf else None,
        container_sha256=leaf.container_sha256 if leaf else None,
    )
    body = next((payload for payload in payloads if payload.modality == 'text'), payloads[0] if payloads else None)
    extraction = None
    if body is not None:
        if body.extraction is not None:
            extraction = replace(body.extraction, artifact=artifact)
        else:
            text = body.full_text
            elements = (SourceElementV1('element-1', 'paragraph', 0, len(text), None, None, None, None),) if text else ()
            extraction = ExtractionV1(f'existing-{body.normalized_type}-text-v1', artifact, text, elements, 'partial', ('structure_unavailable',))
    for payload in payloads:
        payload.source_artifact = artifact
        payload.source_bytes = data
        payload.source_filename = filename
        payload.source_content_type = content_type or _guess_content_type(filename)
        payload.archive_lineage = _archive_lineage
        payload.source_containers = _source_containers
        payload.extraction = extraction
        if payload.media_bytes is not None:
            payload.media_artifact = source_artifact_ref(payload.media_bytes, member_path=None, container_sha256=None)
    return payloads


def _extract_unbound_text_payloads(filename: str, data: bytes, content_type: str | None = None, depth: int = 0) -> list[ExtractedTextPayload]:
    if depth > 2:
        raise ExtractionError("Nested archive depth exceeded.")

    extension = os.path.splitext(_clean_name(filename))[1].lower()
    
    try:
        ingest_type = detect_ingest_type(filename, content_type)
    except ValueError as e:
        raise UnsupportedFileTypeError(str(e)) from e

    if ingest_type == "archive":
        return extract_archive_payloads(filename, data, depth)
    if ingest_type == "image":
        return [_extract_image_ocr(filename, data)]
    if ingest_type == "audio":
        return [_extract_media_transcript(filename, data, normalized_type="audio_transcript")]
    if ingest_type == "video":
        return [_extract_media_transcript(filename, data, normalized_type="video_transcript")]

    if extension == ".pdf":
        return _extract_pdf(filename, data)
    if extension in (".txt", ".md"):
        return [_native_text_payload(filename, data, extension.lstrip('.'))]
    if extension in ('.doc', '.ppt', '.rtf', '.odt'):
        # Unsupported legacy paths still carry a typed extraction snapshot;
        # binary decoding is never advertised as successful native extraction.
        return [_native_text_payload(filename, data, extension.lstrip('.'))]
    if extension in (".html", ".htm"):
        return [_extract_html(filename, data)]
    if extension == ".docx":
        return _extract_docx(filename, data)
    if extension == ".epub":
        return _extract_epub(filename, data)
    if extension == ".csv":
        return [_extract_csv_like(filename, data, delimiter=",")]
    if extension == ".tsv":
        return [_extract_csv_like(filename, data, delimiter="\t")]
    if extension == ".xlsx":
        return _extract_xlsx(filename, data)
    if extension == ".xls":
        return [_extract_xls(filename, data)]
    if extension == ".ods":
        return _extract_ods(filename, data)
    if extension == ".pptx":
        return _extract_pptx(filename, data)
    if extension == ".odp":
        return [_extract_odp(filename, data)]
    if extension == ".json":
        return [_extract_json(filename, data)]
    if extension == ".jsonl":
        return [_extract_jsonl(filename, data)]
    if extension in (".xml",):
        return [_extract_xml(filename, data)]
    if extension in (".yaml", ".yml"):
        return [_extract_yaml(filename, data)]
    if extension == ".vtt":
        return [_extract_vtt(filename, data)]
    if extension == ".srt":
        return [_extract_srt(filename, data)]

    import mimetypes
    guessed_mime = content_type or mimetypes.guess_type(filename)[0] or ""
    if guessed_mime.startswith("text/"):
        return [ExtractedTextPayload(title=_stem(filename), normalized_type="text", full_text=_read_text_bytes(data))]

    raise UnsupportedFileTypeError(f"No extractor available for {filename!r} ({content_type or 'unknown'})")
