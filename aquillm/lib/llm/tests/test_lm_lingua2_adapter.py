"""LM-Lingua2 adapter fail-open and gating."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from time import sleep
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


@override_settings(LM_LINGUA2_ENABLED=True)
def test_compressor_calls_are_serialized_across_threads():
    state_lock = Lock()
    state = {"active": 0, "peak": 0}

    class _Compressor:
        def compress_prompt(self, *_args, **_kwargs):
            with state_lock:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            sleep(0.05)
            with state_lock:
                state["active"] -= 1
            return {"compressed_prompt": "short"}

    with (
        patch.object(l2, "_get_compressor", return_value=_Compressor()),
        patch.object(l2, "_min_chars", return_value=256),
    ):
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(
                executor.map(
                    l2.compress_plain_text_for_prompt,
                    ("a" * 5000, "b" * 5000),
                )
            )

    assert results == ["short", "short"]
    assert state["peak"] == 1
