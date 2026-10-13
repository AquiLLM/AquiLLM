"""test source artifact storage: synthetic regression coverage."""

import io
from hashlib import sha256
from unittest.mock import patch
import pytest
from .source_artifact_support import artifact_storage


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
