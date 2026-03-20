"""Integration tests for apps.documents.models package exports."""

from pathlib import Path


def test_documents_models_exports_ingestion_helpers():
    repo_root = Path(__file__).resolve().parents[3]
    init_file = repo_root / "aquillm" / "apps" / "documents" / "models" / "__init__.py"
    contents = init_file.read_text(encoding="utf-8")

    assert "def document_modality(" in contents
    assert "def document_has_raw_media(" in contents
    assert "def document_provider_name(" in contents
    assert "def document_provider_model(" in contents
    assert "'document_modality'" in contents
    assert "'document_has_raw_media'" in contents
    assert "'document_provider_name'" in contents
    assert "'document_provider_model'" in contents

