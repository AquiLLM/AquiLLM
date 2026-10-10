"""Pure immutable source identity rules; no Django/runtime/provider imports."""

from dataclasses import FrozenInstanceError
from hashlib import sha256

import pytest


def test_artifact_ref_uses_exact_bytes_and_content_addressed_key():
    from lib.parsers.source_contracts import source_artifact_ref

    ref = source_artifact_ref(
        b"abc", member_path="nested/notes.txt", container_sha256="0" * 64
    )
    assert (
        ref.sha256 == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
    assert ref.size_bytes == 3
    assert (
        ref.storage_key
        == "source_artifacts/v1/sha256/ba/"
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
    assert (ref.member_path, ref.container_sha256) == ("nested/notes.txt", "0" * 64)
    with pytest.raises(FrozenInstanceError):
        ref.sha256 = "1" * 64


@pytest.mark.parametrize(
    "path",
    [
        "../x.txt",
        "/x.txt",
        "C:/x.txt",
        "a/../../x.txt",
        "a\\x.txt",
        "a/./x.txt",
        "a//x.txt",
        "x\x00.txt",
        "",
    ],
)
def test_archive_names_require_safe_canonical_posix_paths(path):
    from lib.parsers.source_contracts import SourceContractError, validate_member_path

    with pytest.raises(SourceContractError):
        validate_member_path(path)


def test_occurrence_digest_has_explicit_version_and_ordered_lineage():
    from lib.parsers.source_contracts import (
        ArchiveMemberStepV1,
        occurrence_key,
        source_artifact_ref,
    )

    ref = source_artifact_ref(
        b"abc", member_path="notes.txt", container_sha256="0" * 64
    )
    lineage = (ArchiveMemberStepV1("0" * 64, "notes.txt", 0),)
    manifest = (
        b'{"archive_lineage":[{"archive_ordinal":0,"container_sha256":"'
        b'0000000000000000000000000000000000000000000000000000000000000000",'
        b'"member_path":"notes.txt"}],"artifact_sha256":"'
        b'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",'
        b'"content_type":"text/plain","original_filename":"notes.txt",'
        b'"version":"source-occurrence-v1"}'
    )
    expected = sha256(manifest).hexdigest()
    assert (
        occurrence_key(
            ref,
            original_filename="notes.txt",
            content_type="text/plain",
            archive_lineage=lineage,
        )
        == expected
    )
    assert (
        occurrence_key(
            ref,
            original_filename="notes.txt",
            content_type="text/plain",
            archive_lineage=(ArchiveMemberStepV1("0" * 64, "notes.txt", 1),),
        )
        != expected
    )
    assert (
        occurrence_key(
            ref,
            original_filename="renamed.txt",
            content_type="text/plain",
            archive_lineage=lineage,
        )
        != expected
    )


@pytest.mark.parametrize("ordinal", [-1, True, 0.5, "0"])
def test_lineage_rejects_non_integer_or_negative_ordinals(ordinal):
    from lib.parsers.source_contracts import (
        ArchiveMemberStepV1,
        SourceContractError,
        occurrence_key,
        source_artifact_ref,
    )

    ref = source_artifact_ref(
        b"abc", member_path="notes.txt", container_sha256="0" * 64
    )
    with pytest.raises(SourceContractError):
        occurrence_key(
            ref,
            original_filename="notes.txt",
            content_type="text/plain",
            archive_lineage=(ArchiveMemberStepV1("0" * 64, "notes.txt", ordinal),),
        )


def test_occurrence_unicode_filename_uses_literal_canonical_ascii_wire():
    from lib.parsers.source_contracts import occurrence_key, source_artifact_ref

    ref = source_artifact_ref(b"abc", member_path=None, container_sha256=None)
    # Literal wire locks the finalized D3/D5 convention independently of dumps().
    manifest = (
        b'{"archive_lineage":[],"artifact_sha256":"'
        b'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",'
        b'"content_type":"text/plain","original_filename":"caf\\u00e9.txt",'
        b'"version":"source-occurrence-v1"}'
    )
    assert occurrence_key(
        ref,
        original_filename="café.txt",
        content_type="text/plain",
        archive_lineage=(),
    ) == sha256(manifest).hexdigest()


def test_lineage_must_match_immediate_container_and_member():
    from lib.parsers.source_contracts import (
        ArchiveMemberStepV1,
        SourceContractError,
        occurrence_key,
        source_artifact_ref,
    )

    ref = source_artifact_ref(
        b"abc", member_path="notes.txt", container_sha256="0" * 64
    )
    for lineage in (
        (),
        (ArchiveMemberStepV1("1" * 64, "notes.txt", 0),),
        (ArchiveMemberStepV1("0" * 64, "other.txt", 0),),
    ):
        with pytest.raises(SourceContractError):
            occurrence_key(
                ref,
                original_filename="notes.txt",
                content_type="text/plain",
                archive_lineage=lineage,
            )


def test_nested_lineage_changes_identity_even_with_identical_leaf():
    from lib.parsers.source_contracts import (
        ArchiveMemberStepV1,
        occurrence_key,
        source_artifact_ref,
    )

    ref = source_artifact_ref(
        b"abc", member_path="notes.txt", container_sha256="1" * 64
    )
    leaf = ArchiveMemberStepV1("1" * 64, "notes.txt", 0)
    first = (ArchiveMemberStepV1("0" * 64, "inner.zip", 0), leaf)
    second = (ArchiveMemberStepV1("0" * 64, "inner.zip", 1), leaf)
    keys = [
        occurrence_key(
            ref,
            original_filename="notes.txt",
            content_type="text/plain",
            archive_lineage=lineage,
        )
        for lineage in (first, second)
    ]
    assert keys[0] != keys[1]


def test_supplied_artifact_identity_must_match_exact_bytes():
    from dataclasses import replace

    from lib.parsers.source_contracts import (
        SourceContractError,
        source_artifact_ref,
        validate_artifact_ref,
    )

    ref = source_artifact_ref(b"abc", member_path=None, container_sha256=None)
    validate_artifact_ref(ref, data=b"abc")
    for candidate, data in (
        (ref, b"different"),
        (replace(ref, size_bytes=4), b"abc"),
        (replace(ref, storage_key="public/digest"), b"abc"),
    ):
        with pytest.raises(SourceContractError):
            validate_artifact_ref(candidate, data=data)
