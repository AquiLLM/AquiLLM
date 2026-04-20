"""LMCache env wiring in vLLM startup script and compose services."""
from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_VLLM_SH = _ROOT / "deploy" / "scripts" / "vllm_start.sh"
_BASE_YML = _ROOT / "deploy" / "compose" / "base.yml"


def test_vllm_start_sh_handles_lmcache_enabled_block():
    text = _VLLM_SH.read_text(encoding="utf-8")
    assert "LMCACHE_ENABLED" in text
    assert "LMCACHE_EXTRA_ARGS" in text
    assert "parse_vllm_extra_args.py" in text


def test_base_compose_vllm_exports_lmcache_env():
    text = _BASE_YML.read_text(encoding="utf-8")
    assert "LMCACHE_ENABLED=" in text
    assert "LMCACHE_EXTRA_ARGS=" in text
