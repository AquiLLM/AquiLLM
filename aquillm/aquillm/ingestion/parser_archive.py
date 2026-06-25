"""ZIP archive expansion for nested ingestion (lazy-imports extract_text_payloads)."""
from __future__ import annotations

import io
import os
import zipfile

from lib.parsers import SUPPORTED_EXTENSIONS, clean_filename as _clean_name, guess_content_type as _guess_content_type

from .types import ExtractionError, ExtractedTextPayload


def extract_archive_payloads(filename: str, data: bytes, depth: int) -> list[ExtractedTextPayload]:
    max_files = int((os.getenv("INGEST_ARCHIVE_MAX_FILES") or "100").strip())
    max_total_bytes = int((os.getenv("INGEST_ARCHIVE_MAX_TOTAL_BYTES") or "52428800").strip())
    payloads: list[ExtractedTextPayload] = []
    total_bytes = 0

    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            if len(payloads) >= max_files:
                break
            if info.file_size <= 0:
                continue
            total_bytes += info.file_size
            if total_bytes > max_total_bytes:
                raise ExtractionError("Archive expanded size exceeds INGEST_ARCHIVE_MAX_TOTAL_BYTES.")
            inner_name = info.filename
            inner_ext = os.path.splitext(inner_name)[1].lower()
            if inner_ext not in SUPPORTED_EXTENSIONS or inner_ext == ".zip":
                continue
            with archive.open(info, "r") as file_obj:
                inner_data = file_obj.read()
            from aquillm.ingestion.parsers import extract_text_payloads

            payloads.extend(
                extract_text_payloads(
                    inner_name,
                    inner_data,
                    content_type=_guess_content_type(inner_name, fallback="application/octet-stream"),
                    depth=depth + 1,
                )
            )

    return payloads


__all__ = ["extract_archive_payloads"]
