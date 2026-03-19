"""Integration tests for frontend code."""

from pathlib import Path


def test_ingest_row_sets_xml_http_request_header():
    repo_root = Path(__file__).resolve().parents[3]
    ingest_row = repo_root / "react" / "src" / "components" / "IngestRow.tsx"
    contents = ingest_row.read_text(encoding="utf-8")
    assert '"X-Requested-With": "XMLHttpRequest"' in contents
