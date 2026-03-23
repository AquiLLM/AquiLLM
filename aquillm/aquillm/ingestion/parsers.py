"""
File parsing and extraction for ingestion pipeline.

This module integrates lib/parsers with the full ingestion pipeline,
including figure extraction and media transcription.
"""

import io
import logging
import os

from aquillm.ocr_utils import extract_text_from_image
from aquillm.vtt import coalesce_captions, parse as parse_vtt, to_text as vtt_to_text

from .media import transcribe_media_bytes
from .parser_archive import extract_archive_payloads
from .parser_figures import extract_figure_payloads_for_format
from .types import ExtractedTextPayload, ExtractionError, UnsupportedFileTypeError

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
    extract_pdf_text,
    extract_html_text,
    extract_docx_text,
    extract_epub_text,
    extract_xlsx_text,
    extract_xls_text,
    extract_ods_text,
    extract_csv_text,
    extract_pptx_text,
    extract_odp_text,
    extract_json_text,
    extract_jsonl_text,
    extract_xml_text,
    extract_yaml_text,
    iter_srt_text,
)

logger = logging.getLogger(__name__)


def detect_ingest_type(filename: str, content_type: str | None = None) -> str:
    try:
        return _detect_ingest_type_lib(filename, content_type)
    except ValueError as e:
        raise UnsupportedFileTypeError(str(e)) from e


def _extract_pdf(filename: str, data: bytes) -> list[ExtractedTextPayload]:
    """Extract text and figures from a PDF."""
    payloads: list[ExtractedTextPayload] = []
    full_text = extract_pdf_text(data)
    
    payloads.append(ExtractedTextPayload(
        title=_stem(filename),
        normalized_type="pdf",
        full_text=full_text,
    ))
    
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
    return ExtractedTextPayload(title=_stem(filename), normalized_type="tabular", full_text=extract_csv_text(data, delimiter))


def _extract_docx(filename: str, data: bytes) -> list[ExtractedTextPayload]:
    text = extract_docx_text(data)
    payloads = [ExtractedTextPayload(title=_stem(filename), normalized_type="docx", full_text=text)]
    extract_figure_payloads_for_format(filename, data, "docx", payloads)
    return payloads


def _extract_xlsx(filename: str, data: bytes) -> list[ExtractedTextPayload]:
    text = extract_xlsx_text(data)
    payloads = [ExtractedTextPayload(title=_stem(filename), normalized_type="xlsx", full_text=text)]
    extract_figure_payloads_for_format(filename, data, "xlsx", payloads)
    return payloads


def _extract_xls(filename: str, data: bytes) -> ExtractedTextPayload:
    return ExtractedTextPayload(title=_stem(filename), normalized_type="xls", full_text=extract_xls_text(data))


def _extract_ods(filename: str, data: bytes) -> list[ExtractedTextPayload]:
    text = extract_ods_text(data)
    payloads = [ExtractedTextPayload(title=_stem(filename), normalized_type="ods", full_text=text)]
    extract_figure_payloads_for_format(filename, data, "ods", payloads)
    return payloads


def _extract_pptx(filename: str, data: bytes) -> list[ExtractedTextPayload]:
    text = extract_pptx_text(data)
    payloads = [ExtractedTextPayload(title=_stem(filename), normalized_type="pptx", full_text=text)]
    _extract_figures_for_format(filename, data, "pptx", payloads)
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


def extract_text_payloads(filename: str, data: bytes, content_type: str | None = None, depth: int = 0) -> list[ExtractedTextPayload]:
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
    if extension in (".txt", ".md", ".doc", ".rtf"):
        return [ExtractedTextPayload(title=_stem(filename), normalized_type=extension.lstrip("."), full_text=_read_text_bytes(data))]
    if extension in (".html", ".htm"):
        return [_extract_html(filename, data)]
    if extension == ".docx":
        return _extract_docx(filename, data)
    if extension == ".odt":
        return [ExtractedTextPayload(title=_stem(filename), normalized_type="odt", full_text=_read_text_bytes(data))]
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
    if extension == ".ppt":
        return [ExtractedTextPayload(title=_stem(filename), normalized_type="ppt", full_text=_read_text_bytes(data))]
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
