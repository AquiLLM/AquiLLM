"""Content-addressed persistence must never overwrite or register unchecked bytes."""

import io
import zipfile
from hashlib import sha256
from unittest.mock import patch

import pytest
from django.core.files.storage import FileSystemStorage


@pytest.fixture
def artifact_storage(settings, tmp_path):
    settings.STORAGES = {
        "default": {
            "BACKEND": "django.core.files.storage.FileSystemStorage",
            "OPTIONS": {"location": str(tmp_path)},
        },
        "staticfiles": {
            "BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"
        },
    }
    return FileSystemStorage(location=str(tmp_path))


@pytest.mark.django_db
def test_artifact_deduplication_and_occurrence_ordinal_identity(artifact_storage):
    from apps.ingestion.models import SourceArtifact, SourceArtifactOccurrence
    from apps.ingestion.services.source_artifacts import (
        persist_artifact,
        persist_occurrence,
    )
    from lib.parsers.source_contracts import ArchiveMemberStepV1

    data = b"same member"
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("a.txt", data)
        with pytest.warns(UserWarning):
            archive.writestr("a.txt", data)
    container = persist_artifact(
        stream.getvalue(), member_path=None, container_sha256=None, using="default"
    )
    ref = persist_artifact(
        data, member_path="a.txt", container_sha256=container.sha256, using="default"
    )
    again = persist_artifact(
        data, member_path="b.txt", container_sha256=container.sha256, using="default"
    )
    assert ref.storage_key == again.storage_key
    assert artifact_storage.open(ref.storage_key).read() == data
    occurrences = [
        persist_occurrence(
            ref,
            original_filename="a.txt",
            content_type="text/plain",
            archive_lineage=(ArchiveMemberStepV1(container.sha256, "a.txt", ordinal),),
            using="default",
        )
        for ordinal in (0, 1)
    ]
    assert occurrences[0].occurrence_key != occurrences[1].occurrence_key
    assert occurrences[0].occurrence_id != occurrences[1].occurrence_id
    assert SourceArtifact.objects.count() == 2
    assert SourceArtifactOccurrence.objects.count() == 2
    assert (
        persist_occurrence(
            ref,
            original_filename="a.txt",
            content_type="text/plain",
            archive_lineage=(ArchiveMemberStepV1(container.sha256, "a.txt", 0),),
            using="default",
        )
        == occurrences[0]
    )


@pytest.mark.django_db
def test_corrupt_existing_blob_is_never_overwritten_or_registered(artifact_storage):
    from django.core.files.base import ContentFile

    from apps.ingestion.models import SourceArtifact
    from apps.ingestion.services.source_artifacts import (
        SourceArtifactIntegrityError,
        persist_artifact,
    )

    digest = sha256(b"expected").hexdigest()
    key = f"source_artifacts/v1/sha256/{digest[:2]}/{digest}"
    artifact_storage.save(key, ContentFile(b"corrupt!"))
    with pytest.raises(SourceArtifactIntegrityError):
        persist_artifact(
            b"expected", member_path=None, container_sha256=None, using="default"
        )
    assert artifact_storage.open(key).read() == b"corrupt!"
    assert not SourceArtifact.objects.exists()


@pytest.mark.django_db
def test_unsupported_conditional_storage_fails_before_registration(artifact_storage):
    from apps.ingestion.models import SourceArtifact
    from apps.ingestion.services.source_artifacts import (
        UnsupportedArtifactStorage,
        persist_artifact,
    )

    with patch("apps.ingestion.services.source_artifacts.default_storage", object()):
        with pytest.raises(UnsupportedArtifactStorage):
            persist_artifact(
                b"bytes", member_path=None, container_sha256=None, using="default"
            )
    assert not SourceArtifact.objects.exists()


@pytest.mark.django_db
def test_database_failure_leaves_verified_unregistered_blob_for_retry(artifact_storage):
    from apps.ingestion.models import SourceArtifact
    from apps.ingestion.services.source_artifacts import persist_artifact

    data = b"survives registration crash"
    digest = sha256(data).hexdigest()
    key = f"source_artifacts/v1/sha256/{digest[:2]}/{digest}"
    with patch(
        "django.db.models.query.QuerySet.get_or_create",
        side_effect=RuntimeError("synthetic registration crash"),
    ):
        with pytest.raises(RuntimeError):
            persist_artifact(
                data, member_path=None, container_sha256=None, using="default"
            )
    assert artifact_storage.open(key).read() == data
    assert not SourceArtifact.objects.exists()
    ref = persist_artifact(
        data, member_path=None, container_sha256=None, using="default"
    )
    assert ref.storage_key == key
    assert SourceArtifact.objects.count() == 1


@pytest.mark.django_db
def test_artifact_occurrence_and_revision_rows_are_immutable(artifact_storage):
    from django.core.exceptions import ValidationError

    from apps.ingestion.models import (
        ExtractionRevision,
        SourceArtifact,
        SourceArtifactOccurrence,
    )
    from apps.ingestion.services.source_artifacts import (
        persist_artifact,
        persist_occurrence,
    )

    ref = persist_artifact(
        b"original", member_path=None, container_sha256=None, using="default"
    )
    occurrence = persist_occurrence(
        ref,
        original_filename="notes.txt",
        content_type="text/plain",
        archive_lineage=(),
        using="default",
    )
    row = SourceArtifact.objects.get(sha256=ref.sha256)
    row.size_bytes = 999
    with pytest.raises(ValidationError):
        row.save()
    with pytest.raises(ValidationError):
        SourceArtifactOccurrence.objects.filter(pk=occurrence.occurrence_id).update(
            original_filename="changed"
        )
    revision = ExtractionRevision.objects.create(
        occurrence_id=occurrence.occurrence_id,
        parser_key="fake-v1",
        canonical_text="",
        canonical_sha256=sha256(b"").hexdigest(),
        ordered_elements=[],
        quality="unsupported",
        issue_codes=["unavailable"],
        manifest_sha256=sha256(b"manifest").hexdigest(),
    )
    revision.canonical_text = "changed"
    with pytest.raises(ValidationError):
        revision.save()


@pytest.mark.django_db
def test_occurrence_rejects_unregistered_or_inconsistent_lineage(artifact_storage):
    from apps.ingestion.services.source_artifacts import (
        SourceArtifactIntegrityError,
        persist_artifact,
        persist_occurrence,
    )
    from lib.parsers.source_contracts import ArchiveMemberStepV1

    ref = persist_artifact(
        b"member", member_path="notes.txt", container_sha256="0" * 64, using="default"
    )
    with pytest.raises(SourceArtifactIntegrityError):
        persist_occurrence(
            ref,
            original_filename="notes.txt",
            content_type="text/plain",
            archive_lineage=(ArchiveMemberStepV1("0" * 64, "notes.txt", 0),),
            using="default",
        )


@pytest.mark.django_db
@pytest.mark.parametrize(
    "ordinal,path",
    [
        (-1, "notes.txt"),
        (True, "notes.txt"),
        (0, "../notes.txt"),
        (0, "folder/./notes.txt"),
        (0, "folder\\notes.txt"),
    ],
)
def test_occurrence_requires_canonical_paths_and_nonnegative_integer_ordinals(
    artifact_storage, ordinal, path
):
    from apps.ingestion.services.source_artifacts import (
        SourceArtifactIntegrityError,
        persist_artifact,
        persist_occurrence,
    )
    from lib.parsers.source_contracts import ArchiveMemberStepV1

    container = persist_artifact(
        b"parent", member_path=None, container_sha256=None, using="default"
    )
    ref = persist_artifact(
        b"child",
        member_path="notes.txt",
        container_sha256=container.sha256,
        using="default",
    )
    with pytest.raises(SourceArtifactIntegrityError):
        persist_occurrence(
            ref,
            original_filename="notes.txt",
            content_type="text/plain",
            archive_lineage=(ArchiveMemberStepV1(container.sha256, path, ordinal),),
            using="default",
        )


@pytest.mark.django_db
def test_payload_binding_rejects_mismatched_source_bytes(artifact_storage):
    from apps.ingestion.models import SourceArtifactOccurrence
    from apps.ingestion.services.source_artifacts import (
        SourceArtifactIntegrityError,
        bind_payload_source,
    )
    from aquillm.ingestion.parsers import extract_text_payloads

    payload = extract_text_payloads("notes.txt", b"original")[0]
    payload.source_bytes = b"tampered"
    with pytest.raises(SourceArtifactIntegrityError):
        bind_payload_source(payload, using="default")
    assert not SourceArtifactOccurrence.objects.exists()


@pytest.mark.django_db
def test_database_rejects_direct_mutation_of_immutable_row(artifact_storage):
    from django.db import DatabaseError, connection, transaction

    from apps.ingestion.models import SourceArtifact
    from apps.ingestion.services.source_artifacts import persist_artifact

    ref = persist_artifact(
        b"immutable", member_path=None, container_sha256=None, using="default"
    )
    row = SourceArtifact.objects.get(sha256=ref.sha256)
    with pytest.raises(DatabaseError), transaction.atomic():
        with connection.cursor() as cursor:
            cursor.execute(
                "UPDATE aquillm_sourceartifact SET size_bytes = 100 WHERE id = %s",
                [row.pk],
            )
    row.refresh_from_db()
    assert row.size_bytes == len(b"immutable")


@pytest.mark.django_db
def test_occurrence_rejects_registered_bytes_not_present_at_claimed_location(
    artifact_storage,
):
    from apps.ingestion.models import SourceArtifactOccurrence
    from apps.ingestion.services.source_artifacts import (
        SourceArtifactIntegrityError,
        persist_artifact,
        persist_occurrence,
    )
    from lib.parsers.source_contracts import ArchiveMemberStepV1

    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("notes.txt", b"actual member")
    container = persist_artifact(
        stream.getvalue(), member_path=None, container_sha256=None, using="default"
    )
    ref = persist_artifact(
        b"false member",
        member_path="notes.txt",
        container_sha256=container.sha256,
        using="default",
    )
    with pytest.raises(SourceArtifactIntegrityError):
        persist_occurrence(
            ref,
            original_filename="notes.txt",
            content_type="text/plain",
            archive_lineage=(ArchiveMemberStepV1(container.sha256, "notes.txt", 0),),
            using="default",
        )
    assert not SourceArtifactOccurrence.objects.exists()


@pytest.mark.django_db
@pytest.mark.parametrize("existing", [None, b"member bytes", b"corrupt blob"])
def test_s3_conditional_put_and_existing_412_verify_real_digest(monkeypatch, existing):
    from types import SimpleNamespace

    from botocore.exceptions import ClientError
    from storages.backends.s3 import S3Storage

    from apps.ingestion.models import SourceArtifact
    from apps.ingestion.services.source_artifacts import (
        SourceArtifactIntegrityError,
        persist_artifact,
    )

    data = b"member bytes"
    digest = sha256(data).hexdigest()
    key = f"objects/source_artifacts/v1/sha256/{digest[:2]}/{digest}"
    blobs = {key: existing} if existing is not None else {}

    class ConditionalClient:
        meta = SimpleNamespace(
            service_model=SimpleNamespace(
                operation_model=lambda _name: SimpleNamespace(
                    input_shape=SimpleNamespace(members={"IfNoneMatch": object()})
                )
            )
        )

        def put_object(self, *, Bucket, Key, Body, IfNoneMatch, **_kwargs):
            if IfNoneMatch != "*":
                raise AssertionError("nonconditional write")
            if Key in blobs:
                raise ClientError(
                    {
                        "Error": {"Code": "PreconditionFailed"},
                        "ResponseMetadata": {"HTTPStatusCode": 412},
                    },
                    "PutObject",
                )
            blobs[Key] = Body

    client = ConditionalClient()
    monkeypatch.setattr(
        S3Storage,
        "connection",
        property(lambda self: SimpleNamespace(meta=SimpleNamespace(client=client))),
    )
    storage = S3Storage(
        bucket_name="synthetic",
        access_key="fake",
        secret_key="fake",
        location="objects",
    )
    monkeypatch.setattr(
        storage,
        "_open",
        lambda name, mode="rb": io.BytesIO(blobs[storage._normalize_name(name)]),
    )
    with patch("apps.ingestion.services.source_artifacts.default_storage", storage):
        if existing is not None and existing != data:
            with pytest.raises(SourceArtifactIntegrityError):
                persist_artifact(
                    data, member_path=None, container_sha256=None, using="default"
                )
            assert blobs[key] == existing
            assert not SourceArtifact.objects.exists()
        else:
            ref = persist_artifact(
                data, member_path=None, container_sha256=None, using="default"
            )
            assert blobs[key] == data
            assert ref.sha256 == digest
            assert SourceArtifact.objects.get(sha256=digest).size_bytes == len(data)


@pytest.mark.django_db
@pytest.mark.parametrize("proof", ["persisted", "transient"])
def test_raw_nul_archive_name_cannot_prove_truncated_member(artifact_storage, proof):
    from apps.ingestion.models import SourceArtifactOccurrence
    from apps.ingestion.services.source_artifacts import (
        SourceArtifactIntegrityError,
        bind_payload_source,
        persist_artifact,
        persist_occurrence,
    )
    from aquillm.ingestion.types import ExtractedTextPayload
    from lib.parsers.source_contracts import ArchiveMemberStepV1, source_artifact_ref

    data = b"synthetic notes"
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("notes.txt!suffix", data)
    raw = stream.getvalue()
    assert raw.count(b"notes.txt!suffix") == 2
    crafted = raw.replace(b"notes.txt!suffix", b"notes.txt\x00suffix")
    digest = sha256(crafted).hexdigest()
    lineage = (ArchiveMemberStepV1(digest, "notes.txt", 0),)
    if proof == "persisted":
        persist_artifact(
            crafted, member_path=None, container_sha256=None, using="default"
        )
        artifact = persist_artifact(
            data, member_path="notes.txt", container_sha256=digest, using="default"
        )
        with pytest.raises(SourceArtifactIntegrityError):
            persist_occurrence(
                artifact,
                original_filename="notes.txt",
                content_type="text/plain",
                archive_lineage=lineage,
                using="default",
            )
    else:
        payload = ExtractedTextPayload(
            title="notes",
            normalized_type="txt",
            full_text="synthetic notes",
            source_artifact=source_artifact_ref(
                data, member_path="notes.txt", container_sha256=digest
            ),
            source_bytes=data,
            source_filename="notes.txt",
            source_content_type="text/plain",
            archive_lineage=lineage,
            source_containers=(("crafted.zip", crafted, "application/zip"),),
        )
        with pytest.raises(SourceArtifactIntegrityError):
            bind_payload_source(payload, using="default")
    assert not SourceArtifactOccurrence.objects.exists()
