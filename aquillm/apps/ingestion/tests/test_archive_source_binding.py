"""Member bytes, paths and occurrence lineage must survive archive fan-out."""

import io
import warnings
import zipfile
from hashlib import sha256

import pytest
from pypdf import PdfReader

from aquillm.ingestion.parsers import extract_text_payloads
from aquillm.ingestion.types import ExtractionError


def pdf_bytes(width, *, blank=False):
    import fitz
    document = fitz.open()
    page = document.new_page(width=width, height=100)
    if not blank:
        page.insert_text((10, 25), f'PDF {width}', fontsize=8)
    data = document.tobytes()
    document.close()
    return data


def zip_bytes(members):
    output = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, data in members:
                archive.writestr(name, data)
    return output.getvalue()


@pytest.fixture
def fake_pdf(monkeypatch):
    monkeypatch.setattr(
        "aquillm.ingestion.parsers.extract_figure_payloads_for_format",
        lambda *_args: None,
    )


def test_distinct_pdf_members_keep_exact_bytes(fake_pdf):
    first, second = pdf_bytes(100), pdf_bytes(200)
    archive = zip_bytes([("a/paper.pdf", first), ("b/paper.pdf", second)])
    payloads = extract_text_payloads("papers.zip", archive)
    assert len(payloads) == 2
    for payload, member, path in zip(
        payloads, (first, second), ("a/paper.pdf", "b/paper.pdf")
    ):
        assert payload.source_bytes == member
        assert len(PdfReader(io.BytesIO(payload.source_bytes)).pages) == 1
        assert payload.source_artifact.sha256 == sha256(member).hexdigest()
        assert payload.source_artifact.member_path == path
        assert payload.source_artifact.container_sha256 == sha256(archive).hexdigest()
        assert payload.source_artifact.sha256 != sha256(archive).hexdigest()


def test_same_path_same_bytes_have_distinct_ordinals(fake_pdf):
    member = pdf_bytes(100)
    archive = zip_bytes(
        [("paper.pdf", member), ("paper.pdf", member), ("paper.pdf", pdf_bytes(200))]
    )
    payloads = extract_text_payloads("duplicate.zip", archive)
    assert len(payloads) == 3
    assert [p.archive_lineage[-1].archive_ordinal for p in payloads] == [0, 1, 2]
    assert payloads[0].source_artifact.sha256 == payloads[1].source_artifact.sha256
    assert payloads[1].source_artifact.sha256 != payloads[2].source_artifact.sha256


def test_nested_archive_retains_root_to_leaf_lineage(fake_pdf):
    member = pdf_bytes(100)
    inner = zip_bytes([("leaf.pdf", member)])
    outer = zip_bytes([("folder/inner.zip", inner), ("notes.csv", b"a,b\n1,2\n")])
    payloads = extract_text_payloads("outer.zip", outer)
    assert [p.normalized_type for p in payloads] == ["pdf", "tabular"]
    assert [
        (s.container_sha256, s.member_path, s.archive_ordinal)
        for s in payloads[0].archive_lineage
    ] == [
        (sha256(outer).hexdigest(), "folder/inner.zip", 0),
        (sha256(inner).hexdigest(), "leaf.pdf", 0),
    ]


@pytest.mark.parametrize(
    "name",
    [
        "../paper.pdf",
        "/paper.pdf",
        "C:/paper.pdf",
        "folder/../../paper.pdf",
        "folder\\..\\paper.pdf",
    ],
)
def test_unsafe_archive_member_paths_are_rejected(name, fake_pdf):
    with pytest.raises(ExtractionError):
        extract_text_payloads("unsafe.zip", zip_bytes([(name, pdf_bytes(100))]))


def test_member_limit_counts_members_not_derived_figures(monkeypatch, fake_pdf):
    from aquillm.ingestion.types import ExtractedTextPayload

    monkeypatch.setenv("INGEST_ARCHIVE_MAX_FILES", "2")
    monkeypatch.setattr(
        "aquillm.ingestion.parsers.extract_figure_payloads_for_format",
        lambda _name, _data, _format, outputs: outputs.extend(
            [
                ExtractedTextPayload(
                    title="figure",
                    normalized_type="document_figure",
                    full_text="figure",
                )
                for _ in range(3)
            ]
        ),
    )
    payloads = extract_text_payloads(
        "many.zip", zip_bytes([("paper.pdf", pdf_bytes(100)), ("notes.txt", b"notes")])
    )
    assert len(payloads) == 5
    assert payloads[-1].full_text == "notes"


def test_nested_archives_share_expanded_size_limit(monkeypatch):
    inner = zip_bytes([("notes.txt", b"x" * 500)])
    monkeypatch.setenv("INGEST_ARCHIVE_MAX_TOTAL_BYTES", str(len(inner) + 499))
    outputs = extract_text_payloads("outer.zip", zip_bytes([("inner.zip", inner)]))
    assert len(outputs) == 1
    assert outputs[0].metadata['output_error'] == 'archive_byte_limit'
    assert outputs[0].full_text == ''
    assert outputs[0].source_bytes == inner  # The over-budget leaf was never read.


def test_corrupt_member_bytes_fail_crc_validation():
    original = io.BytesIO()
    with zipfile.ZipFile(original, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("notes.txt", b"original member")
    corrupt = original.getvalue().replace(b"original member", b"corrupt! member")
    with pytest.raises(ExtractionError):
        extract_text_payloads("corrupt.zip", corrupt)


def test_raw_nul_archive_name_is_rejected_before_zipinfo_sanitization():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("notes.txt!suffix", b"synthetic notes")
    raw = stream.getvalue()
    assert raw.count(b"notes.txt!suffix") == 2
    crafted = raw.replace(b"notes.txt!suffix", b"notes.txt\x00suffix")
    with zipfile.ZipFile(io.BytesIO(crafted)) as archive:
        entry = archive.infolist()[0]
        assert entry.orig_filename == "notes.txt\x00suffix"
        assert entry.filename == "notes.txt"
    with pytest.raises(ExtractionError):
        extract_text_payloads("crafted.zip", crafted)


def test_canonical_directory_entry_keeps_member_ordinal():
    payload = extract_text_payloads(
        "directory.zip", zip_bytes([("folder/", b""), ("folder/notes.txt", b"notes")])
    )[0]
    assert payload.source_filename == "folder/notes.txt"
    assert payload.archive_lineage[-1].archive_ordinal == 1


@pytest.mark.django_db
def test_nested_duplicate_containers_bind_distinct_occurrences_and_shared_blobs(
    settings,
    tmp_path,
):
    from apps.ingestion.models import SourceArtifact, SourceArtifactOccurrence
    from apps.ingestion.services.source_artifacts import bind_payload_source

    settings.STORAGES = {
        "default": {
            "BACKEND": "django.core.files.storage.FileSystemStorage",
            "OPTIONS": {"location": str(tmp_path)},
        },
        "staticfiles": {
            "BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"
        },
    }
    leaf = b"synthetic notes"
    inner = zip_bytes([("notes.txt", leaf)])
    outer = zip_bytes([("inner.zip", inner), ("inner.zip", inner)])
    payloads = extract_text_payloads("outer.zip", outer)
    occurrences = [
        bind_payload_source(payload, using="default") for payload in payloads
    ]
    assert len(occurrences) == 2
    assert occurrences[0].occurrence_id != occurrences[1].occurrence_id
    assert occurrences[0].occurrence_key != occurrences[1].occurrence_key
    assert occurrences[0].artifact.sha256 == occurrences[1].artifact.sha256
    assert [o.archive_lineage[0].archive_ordinal for o in occurrences] == [0, 1]
    assert SourceArtifact.objects.count() == 3  # root, shared inner, shared leaf
    assert SourceArtifactOccurrence.objects.count() == 5  # root, two inner, two leaf
    assert bind_payload_source(payloads[0], using="default") == occurrences[0]


def test_top_level_image_keeps_source_and_media_identity(monkeypatch):
    monkeypatch.setattr(
        "aquillm.ingestion.parsers.extract_text_from_image",
        lambda *_args, **_kwargs: {
            "extracted_text": "image",
            "provider": "fake",
            "model": "ocr",
        },
    )
    payload = extract_text_payloads("photo.png", b"image bytes")[0]
    assert payload.source_artifact.sha256 == sha256(b"image bytes").hexdigest()
    assert payload.archive_lineage == ()
    assert payload.source_artifact.member_path is None
    assert (payload.provider, payload.model) == ("fake", "ocr")


def test_figure_artifact_digest_is_independent_of_containing_member(
    monkeypatch, fake_pdf
):
    from aquillm.ingestion.types import ExtractedTextPayload

    member = pdf_bytes(100)
    monkeypatch.setattr(
        "aquillm.ingestion.parsers.extract_figure_payloads_for_format",
        lambda _name, _data, _format, outputs: outputs.append(
            ExtractedTextPayload(
                title="figure",
                normalized_type="document_figure",
                full_text="caption",
                modality="image",
                media_bytes=b"extracted image",
                provider="fake",
                model="ocr",
            )
        ),
    )
    figure = extract_text_payloads("archive.zip", zip_bytes([("paper.pdf", member)]))[1]
    assert figure.source_artifact.sha256 == sha256(member).hexdigest()
    assert figure.media_artifact.sha256 == sha256(b"extracted image").hexdigest()
    assert figure.media_artifact.sha256 != figure.source_artifact.sha256
    assert (figure.provider, figure.model) == ("fake", "ocr")
