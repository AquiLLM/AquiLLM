"""Bounded, in-memory I/O for an already authorized native attached reference.

Only exact S3Storage and POSIX FileSystemStorage backends are supported. Socket
limits and monotonic checks are not a DNS/filesystem watchdog: the operator's
process must own the outer hard wall limit. No serving client is reconfigured.
"""
from contextlib import contextmanager
import ntpath
import os
from pathlib import Path
import stat
import time

from botocore.config import Config
from botocore.exceptions import ClientError
from boto3.s3.transfer import S3Transfer
from django.core.files.storage import FileSystemStorage
from django.utils.functional import LazyObject, empty
from storages.backends.s3 import S3Storage, _filter_download_params
from storages.utils import clean_name

from lib.parsers.source_contracts import SourceContractError, validate_artifact_ref
from .source_artifacts import SourceArtifactIntegrityError, UnsupportedArtifactStorage
from .source_storage_capture import (
    CapturedSource, CHUNK_BYTES, check_deadline, _capture, _identity_digest, _read_bytes,
)


MAX_CAPTURE_BYTES = 32 * 1024 * 1024


def _backend(storage):
    if isinstance(storage, LazyObject):
        if storage._wrapped is empty:
            storage._setup()
        storage = storage._wrapped
    if type(storage) not in (S3Storage, FileSystemStorage):
        raise UnsupportedArtifactStorage("Unsupported bounded source storage.")
    if type(storage) is FileSystemStorage and os.name != "posix":
        raise UnsupportedArtifactStorage("Bounded local source storage requires POSIX no-follow descriptors.")
    return storage


def _name(name):
    if (type(name) is not str or not name or "\\" in name or "\0" in name
            or name.startswith("/") or ntpath.splitdrive(name)[0]
            or any(part in ("", ".", "..") for part in name.split("/"))):
        raise SourceArtifactIntegrityError("source_storage_unsafe_name")


@contextmanager
def _s3_client(storage, deadline):
    check_deadline(deadline)
    seconds = deadline - time.monotonic()
    config = (storage.client_config or Config()).merge(Config(
        connect_timeout=min(3, max(0.001, seconds)),
        read_timeout=min(5, max(0.001, seconds)),
        retries={"total_max_attempts": 1}))
    client = None
    try:
        client = storage._create_session().client("s3", region_name=storage.region_name,
            use_ssl=storage.use_ssl, endpoint_url=storage.endpoint_url,
            config=config, verify=storage.verify)
        check_deadline(deadline)
        yield client
    finally:
        if client is not None:
            client.close()


def _s3_parameters(storage, name):
    key = storage._normalize_name(clean_name(name))
    parameters = storage.get_object_parameters(key)
    if (_filter_download_params(parameters)
            or set(parameters) - set(S3Transfer.ALLOWED_UPLOAD_ARGS)):
        raise UnsupportedArtifactStorage("Unreviewed bounded source download options.")
    return {"Bucket": storage.bucket_name, "Key": key}


def _s3_identity(response):
    length, etag, version = (response.get(field) for field in ("ContentLength", "ETag", "VersionId"))
    if (type(length) is not int or length < 0 or type(etag) is not str or not etag
            or (version is not None and (type(version) is not str or not version))):
        raise SourceArtifactIntegrityError("source_storage_identity_invalid")
    return length, etag, version


def _read_s3(storage, name, limit, deadline):
    parameters = _s3_parameters(storage, name)
    with _s3_client(storage, deadline) as client:
        before = client.head_object(**parameters)
        check_deadline(deadline)
        identity = _s3_identity(before)
        length, etag, version = identity
        if length > limit:
            raise SourceArtifactIntegrityError("source_storage_oversize")
        get_parameters = dict(parameters, IfMatch=etag)
        if version is not None:
            get_parameters["VersionId"] = version
        body = None
        try:
            check_deadline(deadline)
            response = client.get_object(**get_parameters)
            body = response.get("Body")
            check_deadline(deadline)
            if _s3_identity(response) != identity or body is None:
                raise SourceArtifactIntegrityError("source_storage_changed")
            data = _read_bytes(body.read, length, deadline)
        finally:
            if body is not None:
                body.close()
        check_deadline(deadline)
        after = client.head_object(**parameters)
        check_deadline(deadline)
        if _s3_identity(after) != identity:
            raise SourceArtifactIntegrityError("source_storage_changed")
        return _capture(data, identity)


@contextmanager
def _local_parent(storage, name, deadline, *, create=False):
    """Traverse every root/relative parent with no-follow directory descriptors."""
    root = Path(storage.location)
    target = Path(storage.path(name))
    try:
        relative = target.relative_to(root)
    except ValueError as exc:
        raise SourceArtifactIntegrityError("source_storage_unsafe_name") from exc
    if not root.is_absolute() or not relative.parts:
        raise SourceArtifactIntegrityError("source_storage_unsafe_name")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    check_deadline(deadline)
    descriptor = os.open(root.anchor, flags)
    try:
        parents = [(part, False) for part in root.parts[1:]] + [(part, create) for part in relative.parts[:-1]]
        for part, may_create in parents:
            check_deadline(deadline)
            if may_create:
                try:
                    os.mkdir(part, dir_fd=descriptor)
                except FileExistsError:
                    pass
            next_descriptor = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        check_deadline(deadline)
        yield descriptor, relative.name
    finally:
        os.close(descriptor)


def _local_identity(info):
    return info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _read_local(storage, name, limit, deadline):
    with _local_parent(storage, name, deadline) as (parent, filename):
        descriptor = os.open(filename, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        try:
            check_deadline(deadline)
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
                raise SourceArtifactIntegrityError("source_storage_file_invalid")
            identity = _local_identity(before)
            data = _read_bytes(lambda size: os.read(descriptor, size), before.st_size, deadline)
            if _local_identity(os.fstat(descriptor)) != identity:
                raise SourceArtifactIntegrityError("source_storage_changed")
            # Rewalk the native path: an opened directory may have been moved.
            with _local_parent(storage, name, deadline) as (current_parent, current_name):
                current = os.stat(current_name, dir_fd=current_parent, follow_symlinks=False)
                if _local_identity(current) != identity:
                    raise SourceArtifactIntegrityError("source_storage_changed")
            check_deadline(deadline)
            return _capture(data, identity)
        finally:
            os.close(descriptor)


def _read(storage, name, limit, deadline):
    check_deadline(deadline)
    _name(name)
    storage = _backend(storage)
    try:
        if type(storage) is S3Storage:
            return _read_s3(storage, name, limit, deadline)
        return _read_local(storage, name, limit, deadline)
    except (SourceArtifactIntegrityError, UnsupportedArtifactStorage):
        raise
    except Exception as exc:
        raise SourceArtifactIntegrityError("source_storage_read_failed") from exc


def capture_attached_file(storage, name, *, deadline):
    """Capture bytes for a caller-authorized FileField name, never a source URL."""
    return _read(storage, name, MAX_CAPTURE_BYTES, deadline)


def verify_artifact(storage, ref, *, deadline):
    try:
        validate_artifact_ref(ref)
    except SourceContractError as exc:
        raise SourceArtifactIntegrityError("source_artifact_integrity_failed") from exc
    captured = _read(storage, ref.storage_key, ref.size_bytes, deadline)
    if captured.size_bytes != ref.size_bytes or captured.sha256 != ref.sha256:
        raise SourceArtifactIntegrityError("source_artifact_integrity_failed")


def put_artifact_if_absent(storage, ref, data, *, deadline):
    check_deadline(deadline)
    try:
        validate_artifact_ref(ref, data=data)
    except SourceContractError as exc:
        raise SourceArtifactIntegrityError("source_artifact_integrity_failed") from exc
    _name(ref.storage_key)
    storage = _backend(storage)
    try:
        if type(storage) is S3Storage:
            # Reject options that would make mandatory readback unsupported.
            parameters = _s3_parameters(storage, ref.storage_key)
            write_parameters = storage.get_object_parameters(parameters["Key"])
            with _s3_client(storage, deadline) as client:
                operation = client.meta.service_model.operation_model("PutObject")
                if "IfNoneMatch" not in operation.input_shape.members:
                    raise UnsupportedArtifactStorage("Source storage lacks conditional PutObject support.")
                write_parameters.update(parameters, Body=data, IfNoneMatch="*")
                check_deadline(deadline)
                try:
                    client.put_object(**write_parameters)
                except ClientError as exc:
                    if exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode") != 412:
                        raise
                check_deadline(deadline)
        else:
            with _local_parent(storage, ref.storage_key, deadline, create=True) as (parent, filename):
                try:
                    descriptor = os.open(filename, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_NONBLOCK,
                        0o600, dir_fd=parent)
                except FileExistsError:
                    return
                try:
                    written = 0
                    while written < len(data):
                        check_deadline(deadline)
                        count = os.write(descriptor, data[written:written + CHUNK_BYTES])
                        if count <= 0:
                            raise SourceArtifactIntegrityError("source_storage_write_failed")
                        written += count
                    check_deadline(deadline)
                except BaseException:
                    # Only this successful O_EXCL creation is eligible for
                    # cleanup. Keep its descriptor open so its inode cannot be
                    # reused, and never follow a replacement symlink.
                    try:
                        created = os.fstat(descriptor)
                        current = os.stat(filename, dir_fd=parent, follow_symlinks=False)
                        if (current.st_dev, current.st_ino) == (created.st_dev, created.st_ino):
                            os.unlink(filename, dir_fd=parent)
                    except OSError:
                        # Preserve the write error if cleanup is unavailable.
                        pass
                    raise
                finally:
                    os.close(descriptor)
    except (SourceArtifactIntegrityError, UnsupportedArtifactStorage):
        raise
    except Exception as exc:
        raise SourceArtifactIntegrityError("source_storage_write_failed") from exc
