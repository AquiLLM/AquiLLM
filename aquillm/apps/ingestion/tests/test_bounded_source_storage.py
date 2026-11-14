"""Bounded source capture must never enter Django's spooling S3File path."""
import io
import time
from hashlib import sha256
from types import SimpleNamespace

import pytest
from botocore.config import Config
from botocore.exceptions import ClientError
from storages.backends.s3 import S3Storage

from apps.ingestion.services.source_artifacts import SourceArtifactIntegrityError, UnsupportedArtifactStorage
from .bounded_source_storage_support import DATA, storage_service


class Body(io.BytesIO):
    def __init__(self, data):
        super().__init__(data)
        self.requests = []

    def read(self, size=-1):
        assert 0 < size <= 65536, "source body read must have a finite allowance"
        self.requests.append(size)
        return super().read(size)


class S3Transport:
    """Only the remote transport is fake; backend configuration stays native."""
    def __init__(self, data=DATA):
        self.blobs = {"private/attached/current.pdf": data}
        self.calls = []
        self.clients = []
        self.bodies = []
        self.options = []
        self.head_override = None
        self.final_head_override = None
        self.response_override = None
        self.body_factory = Body
        self.fail_get = False

    def client(self, service, **options):
        assert service == "s3"
        self.options.append(options)
        owner = self

        class Client:
            closed = False
            meta = SimpleNamespace(service_model=SimpleNamespace(operation_model=lambda name:
                SimpleNamespace(input_shape=SimpleNamespace(members={"IfNoneMatch": object()}))))

            def head_object(self, **params):
                owner.calls.append(("head", params))
                current = dict(ContentLength=len(owner.blobs[params["Key"]]), ETag='"etag"', VersionId="version-1")
                if owner.head_override:
                    current.update(owner.head_override)
                if owner.final_head_override and any(kind == "get" for kind, _ in owner.calls):
                    current.update(owner.final_head_override)
                return current

            def get_object(self, **params):
                owner.calls.append(("get", params))
                if owner.fail_get:
                    raise ClientError({"Error": {"Code": "PreconditionFailed", "Message": "private detail"}}, "GetObject")
                body = owner.body_factory(owner.blobs[params["Key"]])
                owner.bodies.append(body)
                result = dict(ContentLength=len(owner.blobs[params["Key"]]), ETag='"etag"', VersionId="version-1", Body=body)
                if owner.response_override:
                    result.update(owner.response_override)
                return result

            def put_object(self, **params):
                owner.calls.append(("put", params))
                assert params["IfNoneMatch"] == "*", "immutable writes must be conditional"
                if params["Key"] in owner.blobs:
                    raise ClientError({"Error": {"Code": "PreconditionFailed"}, "ResponseMetadata": {"HTTPStatusCode": 412}}, "PutObject")
                owner.blobs[params["Key"]] = params["Body"]
                return {}

            def close(self):
                self.closed = True

        client = Client()
        self.clients.append(client)
        return client


@pytest.fixture
def s3(monkeypatch):
    transport = S3Transport()
    config = Config(signature_version="s3v4", s3={"addressing_style": "path"}, connect_timeout=87, read_timeout=91, retries={"total_max_attempts": 7})
    storage = S3Storage(bucket_name="private-bucket", access_key="fake", secret_key="fake",
        location="private", region_name="test-region", endpoint_url="https://storage.invalid",
        use_ssl=True, verify="private-ca.pem", client_config=config)
    monkeypatch.setattr(storage, "_create_session", lambda: transport)

    def forbidden(*args, **kwargs):
        pytest.fail("bounded source I/O reached shared-client, storage.open, or storage.save")

    monkeypatch.setattr(storage, "open", forbidden)
    monkeypatch.setattr(storage, "save", forbidden)
    monkeypatch.setattr(S3Storage, "connection", property(forbidden))
    return storage, transport, config


def test_s3_capture_uses_conditional_identity_and_scoped_native_client(s3):
    storage, transport, original = s3
    captured = storage_service().capture_attached_file(storage, "attached/current.pdf", deadline=time.monotonic() + 20)
    assert captured.data == DATA
    assert captured.sha256 == sha256(DATA).hexdigest()
    assert captured.size_bytes == len(DATA)
    assert len(captured.object_identity_sha256) == 64
    assert [kind for kind, _ in transport.calls] == ["head", "get", "head"]
    assert transport.calls[1][1] == {"Bucket": "private-bucket", "Key": "private/attached/current.pdf", "IfMatch": '"etag"', "VersionId": "version-1"}
    options = transport.options[0]
    assert {key: options[key] for key in ("region_name", "endpoint_url", "use_ssl", "verify")} == {
        "region_name": "test-region", "endpoint_url": "https://storage.invalid", "use_ssl": True, "verify": "private-ca.pem"}
    bounded = options["config"]
    assert bounded.signature_version == "s3v4" and bounded.s3 == {"addressing_style": "path"}
    assert 0 < bounded.connect_timeout <= 3 and 0 < bounded.read_timeout <= 5
    assert bounded.retries == {"total_max_attempts": 1}
    assert storage.client_config is original and original.connect_timeout == 87 and original.read_timeout == 91
    assert all(body.closed for body in transport.bodies) and all(client.closed for client in transport.clients)


@pytest.mark.parametrize("name", ["../private.pdf", "a/../b", "/private.pdf", "C:/private.pdf", "a\\b", "a\0b", ""])
def test_unsafe_reference_never_contacts_s3(s3, name):
    storage, transport, _ = s3
    with pytest.raises(SourceArtifactIntegrityError):
        storage_service().capture_attached_file(storage, name, deadline=time.monotonic() + 20)
    assert transport.options == []


@pytest.mark.parametrize("parameters", [{"RequestPayer": "requester"}, {"SSECustomerKey": "private"}, {"ExpectedBucketOwner": "private"}, {"ChecksumMode": "ENABLED"}])
def test_unreviewed_download_options_fail_before_client_creation(s3, parameters):
    storage, transport, _ = s3
    storage.object_parameters = parameters
    with pytest.raises(UnsupportedArtifactStorage):
        storage_service().capture_attached_file(storage, "attached/current.pdf", deadline=time.monotonic() + 20)
    assert transport.options == []


@pytest.mark.parametrize("bad_head", [{"ContentLength": 32 * 1024 * 1024 + 1}, {"ContentLength": -1}, {"ContentLength": True}, {"ETag": ""}])
def test_invalid_or_oversize_declared_source_is_rejected_before_get(s3, bad_head):
    storage, transport, _ = s3
    transport.head_override = bad_head
    with pytest.raises(SourceArtifactIntegrityError):
        storage_service().capture_attached_file(storage, "attached/current.pdf", deadline=time.monotonic() + 20)
    assert [kind for kind, _ in transport.calls] == ["head"]
    assert all(client.closed for client in transport.clients)


@pytest.mark.parametrize("failure", ["get", "response", "final_head", "truncated", "over_return", "read_error", "deadline"])
def test_capture_rejects_unproven_bytes_and_always_closes(s3, monkeypatch, failure):
    storage, transport, _ = s3
    service = storage_service()
    now = [10.0]
    monkeypatch.setattr(service.time, "monotonic", lambda: now[0])
    if failure == "get":
        transport.fail_get = True
    elif failure == "response":
        transport.response_override = {"VersionId": "version-2"}
    elif failure == "final_head":
        transport.final_head_override = {"ETag": '"changed"'}
    else:
        class FailedBody(Body):
            def read(self, size=-1):
                if failure == "over_return":
                    return b"x" * (size + 1)
                if failure == "read_error":
                    raise OSError("private detail")
                if failure == "deadline":
                    now[0] = 31.0
                return super().read(size)
        transport.body_factory = lambda data: FailedBody(data[:2] if failure == "truncated" else data)
    with pytest.raises(SourceArtifactIntegrityError):
        service.capture_attached_file(storage, "attached/current.pdf", deadline=30.0)
    assert all(body.closed for body in transport.bodies)
    assert all(client.closed for client in transport.clients)


def test_expired_capture_budget_and_unsupported_backend_fail_without_io(s3):
    storage, transport, _ = s3
    with pytest.raises(SourceArtifactIntegrityError):
        storage_service().capture_attached_file(storage, "attached/current.pdf", deadline=time.monotonic() - 1)
    assert transport.options == []
    with pytest.raises(UnsupportedArtifactStorage):
        storage_service().capture_attached_file(object(), "attached/current.pdf", deadline=time.monotonic() + 20)


@pytest.mark.django_db
@pytest.mark.parametrize("existing", [None, DATA, b"corrupt existing bytes"])
def test_bounded_artifact_occurrence_and_revision_use_conditional_no_spool_reads(s3, monkeypatch, existing):
    from apps.ingestion.models import SourceArtifact, ExtractionRevision
    from apps.ingestion.services import source_artifacts, extraction_revisions
    from lib.parsers.source_contracts import ExtractionV1, SourceElementV1

    storage, transport, _ = s3
    monkeypatch.setattr(source_artifacts, "default_storage", storage)
    digest = sha256(DATA).hexdigest()
    key = f"private/source_artifacts/v1/sha256/{digest[:2]}/{digest}"
    if existing is not None:
        transport.blobs[key] = existing
    deadline = time.monotonic() + 20
    if existing is not None and existing != DATA:
        with pytest.raises(SourceArtifactIntegrityError):
            source_artifacts.persist_artifact(DATA, member_path=None, container_sha256=None, using="default", deadline=deadline)
        assert transport.blobs[key] == existing
        assert not SourceArtifact.objects.exists()
        return
    artifact = source_artifacts.persist_artifact(DATA, member_path=None, container_sha256=None, using="default", deadline=deadline)
    occurrence = source_artifacts.persist_occurrence(artifact, original_filename="current.pdf", content_type="application/pdf", archive_lineage=(), using="default", deadline=deadline)
    extraction = ExtractionV1("current-storage-test-v1", artifact, "exact canonical",
        (SourceElementV1("body", "paragraph", 0, 15, None, None, None, None),), "partial", ("structure_unavailable",))
    revision = extraction_revisions.persist_extraction_revision(occurrence, extraction, using="default", deadline=deadline)
    assert ExtractionRevision.objects.get(pk=revision).occurrence_id == occurrence.occurrence_id
    assert transport.blobs[key] == DATA
    assert [kind for kind, _ in transport.calls].count("get") == 3
    assert all(body.closed for body in transport.bodies) and all(client.closed for client in transport.clients)


def test_expired_artifact_deadline_stops_before_storage_or_database(s3, monkeypatch):
    from apps.ingestion.services import source_artifacts
    storage, transport, _ = s3
    monkeypatch.setattr(source_artifacts, "default_storage", storage)
    with pytest.raises(SourceArtifactIntegrityError):
        source_artifacts.persist_artifact(DATA, member_path=None, container_sha256=None, using="default", deadline=time.monotonic() - 1)
    assert transport.options == [] and transport.calls == []


def test_bounded_artifact_verification_uses_declared_ref_size_without_capture_cap(s3, monkeypatch):
    from apps.ingestion.services import source_artifacts
    from lib.parsers.source_contracts import source_artifact_ref
    storage, transport, _ = s3
    monkeypatch.setattr(source_artifacts, "default_storage", storage)
    data = b"x" * (32 * 1024 * 1024 + 1)
    ref = source_artifact_ref(data, member_path=None, container_sha256=None)
    transport.blobs["private/" + ref.storage_key] = data
    source_artifacts._verify_blob(ref, deadline=time.monotonic() + 20)
    assert all(body.closed for body in transport.bodies)
    transport.blobs["private/" + ref.storage_key] = data + b"x"
    with pytest.raises(SourceArtifactIntegrityError):
        source_artifacts._verify_blob(ref, deadline=time.monotonic() + 20)


def test_bounded_archive_occurrence_fails_before_any_archive_read(s3, monkeypatch):
    from apps.ingestion.services import source_artifacts
    from lib.parsers.source_contracts import ArchiveMemberStepV1, source_artifact_ref
    storage, transport, _ = s3
    monkeypatch.setattr(source_artifacts, "default_storage", storage)
    ref = source_artifact_ref(DATA, member_path="member.pdf", container_sha256="a" * 64)
    with pytest.raises(UnsupportedArtifactStorage):
        source_artifacts.persist_occurrence(ref, original_filename="member.pdf", content_type="application/pdf",
            archive_lineage=(ArchiveMemberStepV1("a" * 64, "member.pdf", 0),), using="default", deadline=time.monotonic() + 20)
    assert transport.options == [] and transport.calls == []
