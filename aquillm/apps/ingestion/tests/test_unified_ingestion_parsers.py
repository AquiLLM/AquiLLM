import pytest

from aquillm.ingestion.parsers import detect_ingest_type, extract_text_payloads
from aquillm.ingestion.types import UnsupportedFileTypeError


def test_detect_type_for_known_extensions():
    assert detect_ingest_type("paper.pdf", "application/pdf") == "document"
    assert detect_ingest_type("sheet.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet") == "tabular"
    assert detect_ingest_type("clip.mp4", "video/mp4") == "video"
    assert detect_ingest_type("notes.vtt", "text/vtt") == "transcript"


def test_extract_csv_rows_into_text():
    payloads = extract_text_payloads("data.csv", b"name,score\nalice,9\n")
    assert len(payloads) == 1
    assert "name, score" in payloads[0].full_text
    assert "alice, 9" in payloads[0].full_text


def test_extract_json_to_pretty_text():
    payloads = extract_text_payloads("data.json", b'{"a": 1, "b": ["x"]}')
    assert len(payloads) == 1
    assert '"a": 1' in payloads[0].full_text


def test_extract_image_payload_preserves_media_and_modality(monkeypatch):
    monkeypatch.setattr(
        "aquillm.ingestion.parsers.extract_text_from_image",
        lambda *_args, **_kwargs: {"extracted_text": "hello image", "provider": "qwen"},
    )
    payloads = extract_text_payloads("note.png", b"img-bytes", content_type="image/png")
    assert len(payloads) == 1
    payload = payloads[0]
    assert payload.modality == "image"
    assert payload.full_text == "hello image"
    assert payload.media_bytes == b"img-bytes"
    assert payload.media_filename == "note.png"
    assert payload.media_content_type == "image/png"
    assert payload.provider == "qwen"


def test_extract_audio_payload_preserves_media_and_modality(monkeypatch):
    monkeypatch.setattr(
        "aquillm.ingestion.parsers.transcribe_media_bytes",
        lambda *_args, **_kwargs: "audio transcript",
    )
    payloads = extract_text_payloads("clip.mp3", b"audio-bytes", content_type="audio/mpeg")
    assert len(payloads) == 1
    payload = payloads[0]
    assert payload.modality == "audio"
    assert payload.full_text == "audio transcript"
    assert payload.media_bytes == b"audio-bytes"
    assert payload.media_filename == "clip.mp3"
    assert payload.media_content_type == "audio/mpeg"


def test_unsupported_type_raises():
    with pytest.raises(UnsupportedFileTypeError):
        detect_ingest_type("file.exe", "application/octet-stream")
