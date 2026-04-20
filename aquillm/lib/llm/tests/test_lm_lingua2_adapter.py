"""LM-Lingua2 adapter fail-open and gating."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from django.test import override_settings

from lib.llm.optimizations import lm_lingua2_adapter as l2


@override_settings(LM_LINGUA2_ENABLED=False)
def test_disabled_no_change():
    msgs = [{"role": "user", "content": "x" * 5000}]
    assert l2.maybe_compress_openai_style_messages(msgs) is False


@override_settings(LM_LINGUA2_ENABLED=True)
def test_fail_open_on_compress_error():
    msgs = [{"role": "user", "content": "y" * 5000}]
    fake = MagicMock()
    fake.compress_prompt.side_effect = RuntimeError("nope")
    with patch.object(l2, "_get_compressor", return_value=fake):
        assert l2.maybe_compress_openai_style_messages(msgs) is False
    assert len(msgs[0]["content"]) == 5000
