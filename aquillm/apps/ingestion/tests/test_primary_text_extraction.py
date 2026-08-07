from __future__ import annotations

import fitz
import pytest

from aquillm.ingestion import parsers
from aquillm.ingestion.types import ExtractedTextPayload, ExtractionError, UnsupportedFileTypeError


@pytest.fixture
def one_page_pdf_bytes() -> bytes:
    """Build a tiny valid PDF owned by this test module."""
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Primary PDF text")
    data = document.tobytes()
    document.close()
    return data


def test_pdf_default_path_detects_once_extracts_primary_once_and_appends_figures_once(
    monkeypatch,
    one_page_pdf_bytes,
):
    detect_calls = []
    primary_calls = []
    figure_calls = []
    real_detect = parsers.detect_ingest_type
    real_primary = parsers.extract_primary_text_payload

    def count_detect(filename, content_type=None):
        detect_calls.append((filename, content_type))
        return real_detect(filename, content_type)

    def count_primary(filename, data, *, content_type=None, ingest_type=None):
        primary_calls.append((filename, content_type, ingest_type))
        return real_primary(
            filename,
            data,
            content_type=content_type,
            ingest_type=ingest_type,
        )

    def count_figures(filename, data, source_format, payloads):
        figure_calls.append((filename, data, source_format, payloads[0]))

    monkeypatch.setattr(parsers, "detect_ingest_type", count_detect)
    monkeypatch.setattr(parsers, "extract_primary_text_payload", count_primary)
    monkeypatch.setattr(parsers, "extract_figure_payloads_for_format", count_figures)

    payloads = parsers.extract_text_payloads(
        "paper.pdf",
        one_page_pdf_bytes,
        content_type="application/pdf",
    )

    assert detect_calls == [("paper.pdf", "application/pdf")]
    assert primary_calls == [("paper.pdf", "application/pdf", "document")]
    assert len(figure_calls) == 1
    assert figure_calls[0][:3] == ("paper.pdf", one_page_pdf_bytes, "pdf")
    assert figure_calls[0][3] == payloads[0]
    assert payloads == [
        ExtractedTextPayload(
            title="paper",
            normalized_type="pdf",
            full_text="Primary PDF text",
        )
    ]


def test_pdf_primary_path_uses_real_text_parser_but_never_figure_hook(
    monkeypatch,
    one_page_pdf_bytes,
):
    pdf_text_calls = []
    real_extract_pdf_text = parsers.extract_pdf_text

    def count_pdf_text(data):
        pdf_text_calls.append(data)
        return real_extract_pdf_text(data)

    def fail_detection(*_args, **_kwargs):
        raise AssertionError("supplied ingest_type must prevent type detection")

    def fail_figures(*_args, **_kwargs):
        raise AssertionError("primary-only extraction must not inspect figures")

    monkeypatch.setattr(parsers, "extract_pdf_text", count_pdf_text)
    monkeypatch.setattr(parsers, "detect_ingest_type", fail_detection)
    monkeypatch.setattr(parsers, "extract_figure_payloads_for_format", fail_figures)

    payload = parsers.extract_primary_text_payload(
        "paper.pdf",
        one_page_pdf_bytes,
        content_type="application/pdf",
        ingest_type="document",
    )

    assert pdf_text_calls == [one_page_pdf_bytes]
    assert payload == ExtractedTextPayload(
        title="paper",
        normalized_type="pdf",
        full_text="Primary PDF text",
    )


NON_PDF_CASES = [
    ("scan.png", "image/png", "image_ocr", None),
    ("talk.mp3", "audio/mpeg", "audio_transcript", None),
    ("clip.mp4", "video/mp4", "video_transcript", None),
    ("notes.txt", None, "txt", None),
    ("notes.md", None, "md", None),
    ("notes.doc", None, "doc", None),
    ("notes.rtf", None, "rtf", None),
    ("page.html", None, "html", None),
    ("page.htm", None, "html", None),
    ("report.docx", None, "docx", "docx"),
    ("report.odt", None, "odt", None),
    ("book.epub", None, "epub", "epub"),
    ("table.csv", None, "tabular", None),
    ("table.tsv", None, "tabular", None),
    ("table.xlsx", None, "xlsx", "xlsx"),
    ("table.xls", None, "xls", None),
    ("table.ods", None, "ods", "ods"),
    ("slides.pptx", None, "pptx", "pptx"),
    ("slides.ppt", None, "ppt", None),
    ("slides.odp", None, "odp", None),
    ("data.json", None, "json", None),
    ("data.jsonl", None, "jsonl", None),
    ("data.xml", None, "xml", None),
    ("data.yaml", None, "yaml", None),
    ("data.yml", None, "yaml", None),
    ("captions.vtt", None, "vtt", None),
    ("captions.srt", None, "srt", None),
    ("extensionless", "text/plain", "text", None),
    ("extensionless", "text/markdown", "text", None),
    ("extensionless", "text/html", "text", None),
]


@pytest.mark.parametrize(
    ("filename", "content_type", "normalized_type", "figure_format"),
    NON_PDF_CASES,
)
def test_non_pdf_dispatch_matrix_is_unchanged(
    monkeypatch,
    filename,
    content_type,
    normalized_type,
    figure_format,
):
    figure_calls = []

    def fail_primary(*_args, **_kwargs):
        raise AssertionError("non-PDF dispatch must not use the PDF primary helper")

    monkeypatch.setattr(parsers, "extract_primary_text_payload", fail_primary)
    monkeypatch.setattr(
        parsers,
        "extract_text_from_image",
        lambda *_args, **_kwargs: {"extracted_text": "parsed:image"},
    )
    monkeypatch.setattr(parsers, "transcribe_media_bytes", lambda *_args, **_kwargs: "parsed:media")
    monkeypatch.setattr(parsers, "_read_text_bytes", lambda _data: "parsed:raw")
    monkeypatch.setattr(parsers, "extract_html_text", lambda _data: "parsed:html")
    monkeypatch.setattr(parsers, "extract_docx_text", lambda _data: "parsed:docx")
    monkeypatch.setattr(parsers, "extract_epub_text", lambda _data: "parsed:epub")
    monkeypatch.setattr(parsers, "extract_xlsx_text", lambda _data: "parsed:xlsx")
    monkeypatch.setattr(parsers, "extract_xls_text", lambda _data: "parsed:xls")
    monkeypatch.setattr(parsers, "extract_ods_text", lambda _data: "parsed:ods")
    monkeypatch.setattr(parsers, "extract_csv_text", lambda _data, delimiter: f"parsed:{delimiter!r}")
    monkeypatch.setattr(parsers, "extract_pptx_text", lambda _data: "parsed:pptx")
    monkeypatch.setattr(parsers, "extract_odp_text", lambda _data: "parsed:odp")
    monkeypatch.setattr(parsers, "extract_json_text", lambda _data: "parsed:json")
    monkeypatch.setattr(parsers, "extract_jsonl_text", lambda _data: "parsed:jsonl")
    monkeypatch.setattr(parsers, "extract_xml_text", lambda _data: "parsed:xml")
    monkeypatch.setattr(parsers, "extract_yaml_text", lambda _data: "parsed:yaml")
    monkeypatch.setattr(parsers, "parse_vtt", lambda _stream: [])
    monkeypatch.setattr(parsers, "coalesce_captions", lambda captions: captions)
    monkeypatch.setattr(parsers, "vtt_to_text", lambda _captions: "parsed:vtt")
    monkeypatch.setattr(parsers, "iter_srt_text", lambda _lines: iter(["parsed:srt"]))

    def append_figure(filename, _data, source_format, payloads):
        figure_calls.append((filename, source_format))
        payloads.append(
            ExtractedTextPayload(
                title="figure",
                normalized_type="document_figure",
                full_text="figure",
                modality="image",
            )
        )

    monkeypatch.setattr(parsers, "extract_figure_payloads_for_format", append_figure)

    payloads = parsers.extract_text_payloads(filename, b"source", content_type=content_type)

    assert isinstance(payloads, list)
    assert payloads[0].normalized_type == normalized_type
    if figure_format is None:
        assert len(payloads) == 1
        assert figure_calls == []
    else:
        assert len(payloads) == 2
        assert payloads[1].normalized_type == "document_figure"
        assert figure_calls == [(filename, figure_format)]


def test_archive_dispatch_preserves_depth_and_list_shape(monkeypatch):
    expected = [ExtractedTextPayload(title="nested", normalized_type="txt", full_text="nested")]
    calls = []

    def extract_archive(filename, data, depth):
        calls.append((filename, data, depth))
        return expected

    monkeypatch.setattr(parsers, "extract_archive_payloads", extract_archive)

    assert parsers.extract_text_payloads("bundle.zip", b"zip", depth=2) is expected
    assert calls == [("bundle.zip", b"zip", 2)]


def test_archive_depth_error_is_preserved():
    with pytest.raises(ExtractionError, match="Nested archive depth exceeded"):
        parsers.extract_text_payloads("bundle.zip", b"zip", depth=3)


def test_unsupported_dispatch_error_is_preserved():
    with pytest.raises(UnsupportedFileTypeError, match="Unsupported file type"):
        parsers.extract_text_payloads(
            "program.exe",
            b"binary",
            content_type="application/octet-stream",
        )
