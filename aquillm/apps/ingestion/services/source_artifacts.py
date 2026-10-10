"""Verified put-if-absent blob persistence and occurrence registration.

No source URL or digest authorization is provided here. A caller must bind the
resolved occurrence to its own output/document and enforce that document's scope.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

from django.core.files.storage import FileSystemStorage, default_storage

from apps.ingestion.models import SourceArtifact, SourceArtifactOccurrence
from lib.parsers.source_contracts import (
    ArchiveMemberStepV1,
    SourceArtifactOccurrenceRefV1,
    SourceArtifactRefV1,
    SourceContractError,
    canonical_archive_entry_name,
    occurrence_key,
    source_artifact_ref,
    validate_archive_lineage,
    validate_artifact_ref,
)


class SourceArtifactIntegrityError(SourceContractError):
    """Exact immutable source bytes or their occurrence cannot be proven."""


class UnsupportedArtifactStorage(RuntimeError):
    """Storage cannot guarantee a conditional, non-overwriting object creation."""


def _put_if_absent(ref: SourceArtifactRefV1, data: bytes) -> None:
    if isinstance(default_storage, FileSystemStorage):
        # Django save() can rename on collision. Create the exact key exclusively.
        target = Path(default_storage.path(ref.storage_key))
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with target.open("xb") as file_obj:
                file_obj.write(data)
        except FileExistsError:
            pass
        return

    from storages.backends.s3 import S3Storage

    if isinstance(default_storage, S3Storage):
        from botocore.exceptions import ClientError

        client = default_storage.connection.meta.client
        operation = client.meta.service_model.operation_model("PutObject")
        if "IfNoneMatch" not in operation.input_shape.members:
            raise UnsupportedArtifactStorage(
                "Source storage lacks conditional PutObject support."
            )
        parameters = default_storage.get_object_parameters(ref.storage_key)
        parameters.update(
            Bucket=default_storage.bucket_name,
            Key=default_storage._normalize_name(ref.storage_key),
            Body=data,
            IfNoneMatch="*",
        )
        try:
            client.put_object(**parameters)
        except ClientError as exc:
            if exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode") != 412:
                raise
            # Existing object is accepted only after digest/size readback below.
        return
    raise UnsupportedArtifactStorage(
        "Source storage does not support conditional immutable creation."
    )


def _verify_blob(ref: SourceArtifactRefV1) -> None:
    digest = sha256()
    size = 0
    try:
        with default_storage.open(ref.storage_key, "rb") as file_obj:
            while chunk := file_obj.read(65536):
                size += len(chunk)
                if size > ref.size_bytes:
                    raise SourceArtifactIntegrityError(
                        "source_artifact_integrity_failed"
                    )
                digest.update(chunk)
    except FileNotFoundError as exc:
        raise SourceArtifactIntegrityError("source_artifact_integrity_failed") from exc
    if size != ref.size_bytes or digest.hexdigest() != ref.sha256:
        raise SourceArtifactIntegrityError("source_artifact_integrity_failed")


def _registered_artifact(ref: SourceArtifactRefV1, *, using: str) -> SourceArtifact:
    row = SourceArtifact.objects.using(using).filter(sha256=ref.sha256).first()
    if (
        row is None
        or row.size_bytes != ref.size_bytes
        or row.immutable_storage_key != ref.storage_key
    ):
        raise SourceArtifactIntegrityError("source_artifact_registration_mismatch")
    return row


def _verify_archive_member(
    container: SourceArtifact, child: SourceArtifact, step: ArchiveMemberStepV1
) -> None:
    """Attest each registered child's exact ordinal/name/bytes in its container."""
    container_ref = SourceArtifactRefV1(
        container.sha256,
        container.size_bytes,
        container.immutable_storage_key,
        None,
        None,
    )
    _verify_blob(container_ref)
    try:
        with default_storage.open(container.immutable_storage_key, "rb") as file_obj:
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


def persist_artifact(
    data: bytes,
    *,
    member_path: str | None,
    container_sha256: str | None,
    using: str,
) -> SourceArtifactRefV1:
    ref = source_artifact_ref(
        data, member_path=member_path, container_sha256=container_sha256
    )
    _put_if_absent(ref, data)
    _verify_blob(ref)
    # A crash here can leave an unreferenced verified blob; never the reverse.
    row, _ = SourceArtifact.objects.using(using).get_or_create(
        sha256=ref.sha256,
        defaults={
            "size_bytes": ref.size_bytes,
            "immutable_storage_key": ref.storage_key,
        },
    )
    if row.size_bytes != ref.size_bytes or row.immutable_storage_key != ref.storage_key:
        raise SourceArtifactIntegrityError("source_artifact_registration_mismatch")
    return ref


def persist_occurrence(
    artifact: SourceArtifactRefV1,
    *,
    original_filename: str,
    content_type: str,
    archive_lineage: tuple[ArchiveMemberStepV1, ...],
    using: str,
) -> SourceArtifactOccurrenceRefV1:
    try:
        key = occurrence_key(
            artifact,
            original_filename=original_filename,
            content_type=content_type,
            archive_lineage=archive_lineage,
        )
    except SourceContractError as exc:
        raise SourceArtifactIntegrityError("source_occurrence_invalid") from exc
    row = _registered_artifact(artifact, using=using)
    _verify_blob(artifact)
    ancestors = {
        ancestor.sha256: ancestor
        for ancestor in SourceArtifact.objects.using(using).filter(
            sha256__in=[step.container_sha256 for step in archive_lineage],
        )
    }
    if any(step.container_sha256 not in ancestors for step in archive_lineage):
        raise SourceArtifactIntegrityError("source_container_unregistered")
    for index, step in enumerate(archive_lineage):
        child = (
            ancestors[archive_lineage[index + 1].container_sha256]
            if index + 1 < len(archive_lineage)
            else row
        )
        _verify_archive_member(ancestors[step.container_sha256], child, step)
    leaf = archive_lineage[-1] if archive_lineage else None
    lineage_json = [asdict(step) for step in archive_lineage]
    fields = {
        "artifact_id": row.pk,
        "container_id": ancestors[leaf.container_sha256].pk if leaf else None,
        "member_path": leaf.member_path if leaf else None,
        "archive_ordinal": leaf.archive_ordinal if leaf else None,
        "original_filename": original_filename,
        "content_type": content_type,
        "archive_lineage": lineage_json,
    }
    occurrence, _ = SourceArtifactOccurrence.objects.using(using).get_or_create(
        occurrence_key=key,
        defaults=fields,
    )
    if any(getattr(occurrence, field) != value for field, value in fields.items()):
        raise SourceArtifactIntegrityError("source_occurrence_registration_mismatch")
    return SourceArtifactOccurrenceRefV1(occurrence.pk, key, artifact, archive_lineage)


def _validate_transient_source(payload) -> None:
    from .source_artifact_transient import validate_transient_source

    validate_transient_source(payload)


def bind_payload_source(payload, *, using: str) -> SourceArtifactOccurrenceRefV1:
    """Verify transient ancestry and bytes, then resolve the durable occurrence.

    I2 supplies validated extraction/revision separately; I3 requires those plus
    this occurrence before registering its strict durable output manifest.
    """
    _validate_transient_source(payload)
    for index, (filename, container_bytes, content_type) in enumerate(
        payload.source_containers
    ):
        parent = payload.archive_lineage[index - 1] if index else None
        container_ref = persist_artifact(
            container_bytes,
            member_path=parent.member_path if parent else None,
            container_sha256=parent.container_sha256 if parent else None,
            using=using,
        )
        persist_occurrence(
            container_ref,
            original_filename=filename,
            content_type=content_type,
            archive_lineage=payload.archive_lineage[:index],
            using=using,
        )
    artifact = persist_artifact(
        payload.source_bytes,
        member_path=payload.source_artifact.member_path,
        container_sha256=payload.source_artifact.container_sha256,
        using=using,
    )
    occurrence = persist_occurrence(
        artifact,
        original_filename=payload.source_filename,
        content_type=payload.source_content_type or "application/octet-stream",
        archive_lineage=payload.archive_lineage,
        using=using,
    )
    if (
        payload.source_occurrence is not None
        and payload.source_occurrence != occurrence
    ):
        raise SourceArtifactIntegrityError("source_occurrence_binding_mismatch")
    payload.source_artifact = artifact
    payload.source_occurrence = occurrence
    if payload.media_bytes is not None:
        payload.media_artifact = persist_artifact(
            payload.media_bytes, member_path=None, container_sha256=None, using=using
        )
    return occurrence
