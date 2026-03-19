"""Integration tests for vLLM extra-args parsing."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def _parse_args(raw: str) -> list[str]:
    repo_root = Path(__file__).resolve().parents[3]
    parser_script = repo_root / "deploy" / "scripts" / "parse_vllm_extra_args.py"
    result = subprocess.run(
        [sys.executable, str(parser_script), raw],
        check=True,
        capture_output=True,
    )
    return [token.decode("utf-8") for token in result.stdout.split(b"\0") if token]


def test_parser_normalizes_escaped_json_values():
    raw = (
        "--speculative-config "
        '\'{\\"method\\":\\"ngram\\",\\"num_speculative_tokens\\":2,\\"prompt_lookup_max\\":3}\' '
        "--model-loader-extra-config "
        '\'{\\"load_in_4bit\\":true,\\"bnb_4bit_quant_type\\":\\"nf4\\"}\''
    )

    parsed = _parse_args(raw)

    assert parsed == [
        "--speculative-config",
        '{"method":"ngram","num_speculative_tokens":2,"prompt_lookup_max":3}',
        "--model-loader-extra-config",
        '{"load_in_4bit":true,"bnb_4bit_quant_type":"nf4"}',
    ]


def test_vllm_start_script_uses_parser_helper():
    repo_root = Path(__file__).resolve().parents[3]
    start_script = repo_root / "deploy" / "scripts" / "vllm_start.sh"
    contents = start_script.read_text(encoding="utf-8")

    assert "/parse_vllm_extra_args.py" in contents
    assert "mapfile -d '' -t extra_args" in contents
