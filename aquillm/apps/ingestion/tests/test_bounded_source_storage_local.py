"""POSIX containment and immutable-publication failures use real local files."""
from hashlib import sha256
import os
import time

import pytest
from django.core.files.storage import FileSystemStorage

from apps.ingestion.services.source_artifacts import SourceArtifactIntegrityError
from .bounded_source_storage_support import DATA, storage_service


@pytest.mark.skipif(os.name != "posix", reason="descriptor-relative no-follow storage needs POSIX")
def test_local_capture_requires_contained_regular_file_without_symlink_traversal(tmp_path):
    service = storage_service()
    root = tmp_path / "root"
    root.mkdir()
    (root / "current.pdf").write_bytes(DATA)
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(DATA)
    (root / "linked.pdf").symlink_to(outside)
    (root / "linked-dir").symlink_to(tmp_path, target_is_directory=True)
    os.mkfifo(root / "pipe.pdf")
    storage = FileSystemStorage(location=str(root))
    captured = service.capture_attached_file(storage, "current.pdf", deadline=time.monotonic() + 20)
    assert captured.data == DATA and captured.sha256 == sha256(DATA).hexdigest()
    for name in ("../outside.pdf", "linked.pdf", "linked-dir/outside.pdf", "pipe.pdf", "."):
        with pytest.raises(SourceArtifactIntegrityError):
            service.capture_attached_file(storage, name, deadline=time.monotonic() + 20)


@pytest.mark.skipif(os.name != "posix", reason="descriptor-relative no-follow storage needs POSIX")
@pytest.mark.parametrize("failure", ["deadline_before_write", "deadline_after_partial_write", "partial_write_error"])
def test_failed_local_immutable_creation_can_retry_exact_bytes(tmp_path, monkeypatch, failure):
    from lib.parsers.source_contracts import source_artifact_ref
    service = storage_service()
    storage = FileSystemStorage(location=str(tmp_path))
    ref = source_artifact_ref(DATA, member_path=None, container_sha256=None)
    target = tmp_path / ref.storage_key
    actual_open, actual_write = os.open, os.write
    opened = []
    now = [10.0]

    def open_then_expire(path, flags, *args, **kwargs):
        descriptor = actual_open(path, flags, *args, **kwargs)
        if flags & os.O_CREAT and flags & os.O_EXCL:
            opened.append(descriptor)
            if failure == "deadline_before_write":
                now[0] = 20.0
        return descriptor

    def incomplete_write(descriptor, data):
        count = actual_write(descriptor, data[:2])
        if failure == "partial_write_error":
            raise OSError("synthetic write failed after consuming bytes")
        now[0] = 20.0
        return count

    with monkeypatch.context() as injected:
        injected.setattr(service.time, "monotonic", lambda: now[0])
        injected.setattr(service.os, "open", open_then_expire)
        injected.setattr(service.os, "write", incomplete_write)
        with pytest.raises(SourceArtifactIntegrityError):
            service.put_artifact_if_absent(storage, ref, DATA, deadline=20.0)
    assert len(opened) == 1
    with pytest.raises(OSError):
        os.fstat(opened[0])
    assert not target.exists(), "a failed exclusive creation must not poison the immutable key"
    service.put_artifact_if_absent(storage, ref, DATA, deadline=time.monotonic() + 20)
    service.verify_artifact(storage, ref, deadline=time.monotonic() + 20)
    assert target.read_bytes() == DATA


@pytest.mark.skipif(os.name != "posix", reason="descriptor-relative no-follow storage needs POSIX")
@pytest.mark.parametrize("existing", [DATA, b"existing collision must remain untouched"])
def test_local_immutable_collision_preserves_existing_inode_and_bytes(tmp_path, existing):
    from lib.parsers.source_contracts import source_artifact_ref
    service = storage_service()
    storage = FileSystemStorage(location=str(tmp_path))
    ref = source_artifact_ref(DATA, member_path=None, container_sha256=None)
    target = tmp_path / ref.storage_key
    target.parent.mkdir(parents=True)
    target.write_bytes(existing)
    before = target.stat()
    service.put_artifact_if_absent(storage, ref, DATA, deadline=time.monotonic() + 20)
    after = target.stat()
    assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)
    assert target.read_bytes() == existing
    if existing == DATA:
        service.verify_artifact(storage, ref, deadline=time.monotonic() + 20)
    else:
        with pytest.raises(SourceArtifactIntegrityError):
            service.verify_artifact(storage, ref, deadline=time.monotonic() + 20)
    assert target.read_bytes() == existing


@pytest.mark.skipif(os.name != "posix", reason="descriptor-relative no-follow storage needs POSIX")
@pytest.mark.parametrize("replacement_kind", ["file", "symlink"])
def test_failed_local_immutable_write_preserves_replacement_inode(tmp_path, monkeypatch, replacement_kind):
    from lib.parsers.source_contracts import source_artifact_ref
    service = storage_service()
    storage = FileSystemStorage(location=str(tmp_path))
    ref = source_artifact_ref(DATA, member_path=None, container_sha256=None)
    target = tmp_path / ref.storage_key
    other = tmp_path / "replacement.pdf"
    other.write_bytes(b"replacement bytes")
    actual_write = os.write
    replacement_identity = []

    def replace_then_fail(descriptor, data):
        actual_write(descriptor, data[:2])
        old = os.fstat(descriptor)
        target.unlink()
        if replacement_kind == "symlink":
            target.symlink_to(other)
        else:
            target.write_bytes(b"replacement bytes")
        current = target.lstat()
        assert (old.st_dev, old.st_ino) != (current.st_dev, current.st_ino)
        replacement_identity.append((current.st_dev, current.st_ino))
        raise OSError("synthetic write failed after replacement")

    with monkeypatch.context() as injected:
        injected.setattr(service.os, "write", replace_then_fail)
        with pytest.raises(SourceArtifactIntegrityError):
            service.put_artifact_if_absent(storage, ref, DATA, deadline=time.monotonic() + 20)
    current = target.lstat()
    assert (current.st_dev, current.st_ino) == replacement_identity[0]
    assert target.read_bytes() == other.read_bytes() == b"replacement bytes"
    assert target.is_symlink() == (replacement_kind == "symlink")
