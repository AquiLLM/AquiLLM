import pytest


def test_sanitize_db_text_strips_nul_bytes():
    from aquillm.task_ingest_helpers import sanitize_db_text

    assert sanitize_db_text("abc") == "abc"
    assert sanitize_db_text("a\x00b\x00c") == "abc"


def test_sanitize_db_text_handles_non_string_inputs():
    from aquillm.task_ingest_helpers import sanitize_db_text

    assert sanitize_db_text(None) == ""
    assert sanitize_db_text(123) == "123"

