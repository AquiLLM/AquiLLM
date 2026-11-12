"""Native archive-member byte and ordinal proofs for source occurrences."""
from hashlib import sha256
import zipfile

from apps.ingestion.models import SourceArtifact
from lib.parsers.source_contracts import (
    ArchiveMemberStepV1, SourceArtifactRefV1, SourceContractError,
    canonical_archive_entry_name,
)
from .source_artifacts import SourceArtifactIntegrityError


def verify_archive_member(
    container: SourceArtifact, child: SourceArtifact, step: ArchiveMemberStepV1,
    *, storage, verify_blob,
) -> None:
    """Attest each registered child's exact ordinal/name/bytes in its container."""
    container_ref = SourceArtifactRefV1(
        container.sha256,
        container.size_bytes,
        container.immutable_storage_key,
        None,
        None,
    )
    verify_blob(container_ref)
    try:
        with storage.open(container.immutable_storage_key, "rb") as file_obj:
            with zipfile.ZipFile(file_obj) as archive:
                entries = archive.infolist()
                if step.archive_ordinal >= len(entries):
                    raise SourceArtifactIntegrityError("source_archive_ordinal_missing")
                entry = entries[step.archive_ordinal]
                entry_name = canonical_archive_entry_name(
                    entry.orig_filename, entry.filename, is_directory=entry.is_dir()
                )
                if (
                    entry.is_dir()
                    or entry_name != step.member_path
                    or entry.file_size != child.size_bytes
                ):
                    raise SourceArtifactIntegrityError("source_archive_member_mismatch")
                digest = sha256()
                size = 0
                with archive.open(entry) as member:
                    while chunk := member.read(65536):
                        size += len(chunk)
                        if size > child.size_bytes:
                            raise SourceArtifactIntegrityError(
                                "source_archive_bytes_mismatch"
                            )
                        digest.update(chunk)
                if size != child.size_bytes or digest.hexdigest() != child.sha256:
                    raise SourceArtifactIntegrityError("source_archive_bytes_mismatch")
    except (SourceContractError, zipfile.BadZipFile, RuntimeError, EOFError) as exc:
        raise SourceArtifactIntegrityError("source_archive_integrity_failed") from exc
