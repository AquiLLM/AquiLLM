"""Bounded archive expansion preserving exact bytes and root-to-leaf locations."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import io
import os
import zipfile

from lib.parsers import SUPPORTED_EXTENSIONS, guess_content_type
from lib.parsers.source_contracts import ArchiveMemberStepV1, SourceContractError, canonical_archive_entry_name

from .types import ExtractionError, ExtractedTextPayload
from .parser_archive_failures import archive_failure


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
                member_name = canonical_archive_entry_name(
                    info.orig_filename, info.filename, is_directory=info.is_dir()
                )
                if info.is_dir():
                    continue
                if budget.members >= budget.max_files:
                    payloads.append(archive_failure(filename, data, 'archive_member_limit',
                        lineage=archive_lineage, containers=source_containers,
                        content_type=content_type or guess_content_type(filename)))
                    break
                budget.members += 1
                remaining = budget.max_total_bytes - budget.expanded_bytes
                if info.file_size > remaining:
                    payloads.append(archive_failure(filename, data, 'archive_byte_limit',
                        lineage=archive_lineage, containers=source_containers,
                        content_type=content_type or guess_content_type(filename)))
                    break
                # Charge all actual members, including unsupported formats, to one tree budget.
                budget.expanded_bytes += info.file_size
                with archive.open(info, "r") as file_obj:
                    inner_data = file_obj.read(remaining + 1)
                if len(inner_data) != info.file_size or len(inner_data) > remaining:
                    raise ExtractionError("Archive member size could not be verified.")
                from aquillm.ingestion.parsers import extract_text_payloads

                lineage = archive_lineage + (ArchiveMemberStepV1(container_digest, member_name, ordinal),)
                member_type = guess_content_type(member_name)
                failure = 'empty_extraction' if not inner_data else (
                    'unsupported_member' if os.path.splitext(member_name)[1].lower() not in SUPPORTED_EXTENSIONS else '')
                if not failure:
                    try:
                        payloads.extend(extract_text_payloads(member_name, inner_data,
                            content_type=member_type, depth=depth + 1, _archive_lineage=lineage,
                            _source_containers=containers, _archive_budget=budget))
                    except SourceContractError:
                        raise
                    except Exception as exc:
                        from lib.embeddings.errors import require_transient
                        try:
                            require_transient(exc)
                        except Exception:
                            pass
                        else:
                            raise
                        failure = 'member_extraction_failed'
                if failure:
                    payloads.append(archive_failure(member_name, inner_data, failure,
                        lineage=lineage, containers=containers, content_type=member_type))
    except SourceContractError as exc:
        raise ExtractionError("Unsafe archive member path.") from exc
    except (zipfile.BadZipFile, RuntimeError, EOFError) as exc:
        if isinstance(exc, ExtractionError):
            raise
        from lib.embeddings.errors import require_transient
        try:
            require_transient(exc)
        except Exception:
            pass
        else:
            raise
        raise ExtractionError("Archive member integrity could not be verified.") from exc
    return payloads


__all__ = ["extract_archive_payloads"]
