"""Bounded archive expansion preserving exact bytes and root-to-leaf locations."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import io
import os
import zipfile

from lib.parsers import SUPPORTED_EXTENSIONS, guess_content_type
from lib.parsers.source_contracts import ArchiveMemberStepV1, SourceContractError, validate_member_path

from .types import ExtractionError, ExtractedTextPayload


@dataclass
class ArchiveBudget:
    max_files: int
    max_total_bytes: int
    members: int = 0
    expanded_bytes: int = 0

    @classmethod
    def from_environment(cls):
        return cls(
            max_files=int((os.getenv("INGEST_ARCHIVE_MAX_FILES") or "100").strip()),
            max_total_bytes=int((os.getenv("INGEST_ARCHIVE_MAX_TOTAL_BYTES") or "52428800").strip()),
        )


def extract_archive_payloads(
    filename: str, data: bytes, depth: int, *,
    archive_lineage: tuple[ArchiveMemberStepV1, ...] = (),
    source_containers: tuple[tuple[str, bytes, str], ...] = (),
    budget: ArchiveBudget | None = None,
    content_type: str | None = None,
) -> list[ExtractedTextPayload]:
    budget = budget if budget is not None else ArchiveBudget.from_environment()
    payloads: list[ExtractedTextPayload] = []
    container_digest = sha256(data).hexdigest()
    containers = source_containers + ((filename, data, content_type or guess_content_type(filename)),)
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for ordinal, info in enumerate(archive.infolist()):
                # Validate even skipped/unsupported names; no normalization changes identity.
                validate_member_path(info.filename[:-1] if info.is_dir() else info.filename)
                if info.is_dir():
                    continue
                if budget.members >= budget.max_files:
                    break
                budget.members += 1
                if info.file_size <= 0:
                    continue
                remaining = budget.max_total_bytes - budget.expanded_bytes
                if info.file_size > remaining:
                    raise ExtractionError("Archive expanded size exceeds INGEST_ARCHIVE_MAX_TOTAL_BYTES.")
                # Charge all actual members, including unsupported formats, to one tree budget.
                budget.expanded_bytes += info.file_size
                if os.path.splitext(info.filename)[1].lower() not in SUPPORTED_EXTENSIONS:
                    continue
                with archive.open(info, "r") as file_obj:
                    inner_data = file_obj.read(remaining + 1)
                if len(inner_data) != info.file_size or len(inner_data) > remaining:
                    raise ExtractionError("Archive member size could not be verified.")
                from aquillm.ingestion.parsers import extract_text_payloads

                payloads.extend(extract_text_payloads(
                    info.filename, inner_data,
                    content_type=guess_content_type(info.filename), depth=depth + 1,
                    _archive_lineage=archive_lineage + (ArchiveMemberStepV1(container_digest, info.filename, ordinal),),
                    _source_containers=containers, _archive_budget=budget,
                ))
    except SourceContractError as exc:
        raise ExtractionError("Unsafe archive member path.") from exc
    except (zipfile.BadZipFile, RuntimeError, EOFError) as exc:
        if isinstance(exc, ExtractionError):
            raise
        raise ExtractionError("Archive member integrity could not be verified.") from exc
    return payloads


__all__ = ["extract_archive_payloads"]
