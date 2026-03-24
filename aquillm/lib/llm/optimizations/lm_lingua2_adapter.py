"""Optional LM-Lingua2 (llmlingua) extractive compression for long plain-text turns."""
from __future__ import annotations

import logging
from os import getenv
from typing import Any

logger = logging.getLogger(__name__)

_COMPRESSOR: Any | None = None
_COMPRESSOR_FAILED = False


def _enabled() -> bool:
    v = (getenv("LM_LINGUA2_ENABLED", "") or "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    try:
        from django.conf import settings

        return bool(getattr(settings, "LM_LINGUA2_ENABLED", False))
    except Exception:
        return False


def _min_chars() -> int:
    try:
        from django.conf import settings

        return max(256, int(getattr(settings, "PROMPT_COMPRESS_MIN_CHARS", 4000)))
    except Exception:
        pass
    try:
        return max(256, int((getenv("PROMPT_COMPRESS_MIN_CHARS", "4000") or "4000").strip()))
    except Exception:
        return 4000


def _target_tokens() -> int:
    try:
        from django.conf import settings

        return max(64, int(getattr(settings, "PROMPT_COMPRESS_TARGET_TOKENS", 2048)))
    except Exception:
        pass
    try:
        return max(64, int((getenv("PROMPT_COMPRESS_TARGET_TOKENS", "2048") or "2048").strip()))
    except Exception:
        return 2048


def _model_name() -> str:
    try:
        from django.conf import settings

        s = str(getattr(settings, "LM_LINGUA2_MODEL", "") or "").strip()
        if s:
            return s
    except Exception:
        pass
    raw = (getenv("LM_LINGUA2_MODEL", "") or "").strip()
    return raw or "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"


def _device_map() -> str:
    """cuda when available; else cpu. Web containers are usually CPU-only — override with LM_LINGUA2_DEVICE_MAP."""
    raw = (getenv("LM_LINGUA2_DEVICE_MAP", "") or "").strip()
    if raw:
        return raw
    try:
        from django.conf import settings

        s = str(getattr(settings, "LM_LINGUA2_DEVICE_MAP", "") or "").strip()
        if s:
            return s
    except Exception:
        pass
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _get_compressor():
    global _COMPRESSOR, _COMPRESSOR_FAILED
    if _COMPRESSOR_FAILED:
        return None
    if _COMPRESSOR is not None:
        return _COMPRESSOR
    try:
        from llmlingua import PromptCompressor

        _COMPRESSOR = PromptCompressor(
            model_name=_model_name(),
            device_map=_device_map(),
            use_llmlingua2=True,
        )
    except Exception as exc:
        logger.warning("LM-Lingua2 init failed (fail-open): %s", exc)
        _COMPRESSOR_FAILED = True
        return None
    return _COMPRESSOR


def _compress_plain_text(text: str) -> str | None:
    comp = _get_compressor()
    if comp is None or len(text) < _min_chars():
        return None
    try:
        out = comp.compress_prompt(
            [text],
            target_token=_target_tokens(),
            use_context_level_filter=True,
        )
    except Exception as exc:
        logger.warning("LM-Lingua2 compress failed (fail-open): %s", exc)
        return None
    if isinstance(out, dict):
        for key in ("compressed_prompt", "compressed_context", "compressed_prompt_list"):
            val = out.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
            if isinstance(val, list) and val and isinstance(val[0], str) and val[0].strip():
                return val[0].strip()
    return None


def maybe_compress_openai_style_messages(messages: list[dict[str, Any]]) -> bool:
    """
    Compress long string `content` on user/assistant messages. Returns True if any message changed.
    Never drops tool structure; skips non-string and system-injected rows.
    """
    if not _enabled():
        return False
    changed = False
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "")).lower()
        if role not in ("user", "assistant"):
            continue
        content = msg.get("content")
        if not isinstance(content, str) or len(content) < _min_chars():
            continue
        compressed = _compress_plain_text(content)
        if compressed and len(compressed) < len(content):
            msg["content"] = compressed
            changed = True
            logger.info(
                "lm_lingua2 compressed role=%s chars %s -> %s",
                role,
                len(content),
                len(compressed),
            )
    return changed


__all__ = ["maybe_compress_openai_style_messages"]
