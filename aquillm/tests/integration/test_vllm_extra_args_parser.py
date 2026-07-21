"""Integration tests for vLLM extra-args parsing."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _parse_args(raw: str) -> list[str]:
    repo_root = Path(__file__).resolve().parents[3]
    parser_script = repo_root / "deploy" / "scripts" / "parse_vllm_extra_args.py"
    result = subprocess.run(
        [sys.executable, str(parser_script), raw],
        check=True,
        capture_output=True,
    )
    return [token.decode("utf-8") for token in result.stdout.split(b"\0") if token]


def _bash_path(path: Path) -> str:
    if os.name != "nt":
        return str(path)
    absolute = path.resolve().as_posix()
    drive = absolute[0].lower()
    return f"/mnt/{drive}/{absolute[3:]}"


def _run_vllm_start(tmp_path: Path, **environment: str) -> list[str]:
    repo_root = Path(__file__).resolve().parents[3]
    start_script = _bash_path(repo_root / "deploy" / "scripts" / "vllm_start.sh")
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        '#!/bin/sh\nfor arg in "$@"; do printf "FINAL_ARG=%s\\n" "$arg"; done\n',
        encoding="utf-8",
        newline="\n",
    )
    fake_python_path = _bash_path(fake_python)
    runner = ["wsl"] if os.name == "nt" else []
    subprocess.run(
        [*runner, "chmod", "+x", fake_python_path],
        check=True,
        capture_output=True,
    )
    command = [
        *runner,
        "env",
        "-i",
        "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        f"VLLM_PYTHON_BIN={fake_python_path}",
        *[f"{key}={value}" for key, value in environment.items()],
        "bash",
        start_script,
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return [
        line.removeprefix("FINAL_ARG=")
        for line in result.stdout.splitlines()
        if line.startswith("FINAL_ARG=")
    ]


def test_parser_normalizes_escaped_json_values():
    raw = (
        "--speculative-config "
        '\'{\\"method\\":\\"ngram\\",\\"num_speculative_tokens\\":2,'
        '\\"prompt_lookup_max\\":3}\' '
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


def test_parser_tokenizes_nemotron_scheduler_and_generation_config_args():
    raw = (
        "--enforce-eager --max-num-seqs 1 --max-num-batched-tokens 50000 "
        "--generation-config /opt/aquillm/nemotron-generation-config"
    )

    parsed = _parse_args(raw)

    assert parsed == [
        "--enforce-eager",
        "--max-num-seqs",
        "1",
        "--max-num-batched-tokens",
        "50000",
        "--generation-config",
        "/opt/aquillm/nemotron-generation-config",
    ]


def test_transcribe_service_kind_supplies_nemotron_args_when_env_args_are_absent(
    tmp_path: Path,
):
    final_args = _run_vllm_start(
        tmp_path,
        VLLM_MODEL="nvidia/nemotron-3.5-asr-streaming-0.6b",
        VLLM_SERVED_MODEL_NAME="nemotron-3.5-asr-streaming-0.6b",
        VLLM_SERVICE_KIND="transcribe",
        VLLM_EXTRA_ARGS=" ",
        TRANSCRIBE_VLLM_EXTRA_ARGS="",
    )

    assert final_args[-7:] == [
        "--enforce-eager",
        "--max-num-seqs",
        "1",
        "--max-num-batched-tokens",
        "50000",
        "--generation-config",
        "/opt/aquillm/nemotron-generation-config",
    ]


def test_transcribe_service_kind_supplies_nemotron_args_when_service_args_are_unset(
    tmp_path: Path,
):
    final_args = _run_vllm_start(
        tmp_path,
        VLLM_MODEL="nvidia/nemotron-3.5-asr-streaming-0.6b",
        VLLM_SERVED_MODEL_NAME="nemotron-3.5-asr-streaming-0.6b",
        VLLM_SERVICE_KIND="transcribe",
        VLLM_EXTRA_ARGS="",
    )

    assert final_args[-7:] == [
        "--enforce-eager",
        "--max-num-seqs",
        "1",
        "--max-num-batched-tokens",
        "50000",
        "--generation-config",
        "/opt/aquillm/nemotron-generation-config",
    ]


def test_transcribe_service_kind_preserves_explicit_whisper_rollback_args(
    tmp_path: Path,
):
    final_args = _run_vllm_start(
        tmp_path,
        VLLM_MODEL="openai/whisper-large-v3-turbo",
        VLLM_SERVED_MODEL_NAME="whisper-large-v3-turbo",
        VLLM_SERVICE_KIND="transcribe",
        VLLM_EXTRA_ARGS="",
        TRANSCRIBE_VLLM_EXTRA_ARGS=(
            "--quantization bitsandbytes --load-format bitsandbytes "
            "--max-num-seqs 1 --max-num-batched-tokens 448"
        ),
    )

    assert final_args[-8:] == [
        "--quantization",
        "bitsandbytes",
        "--load-format",
        "bitsandbytes",
        "--max-num-seqs",
        "1",
        "--max-num-batched-tokens",
        "448",
    ]
    assert "/opt/aquillm/nemotron-generation-config" not in final_args


def test_nemotron_fallback_is_not_applied_to_other_service_kinds(tmp_path: Path):
    final_args = _run_vllm_start(
        tmp_path,
        VLLM_MODEL="example/chat-model",
        VLLM_SERVED_MODEL_NAME="chat-model",
        VLLM_SERVICE_KIND="chat",
        VLLM_EXTRA_ARGS="",
        TRANSCRIBE_VLLM_EXTRA_ARGS="",
    )

    assert "--enforce-eager" not in final_args
    assert "/opt/aquillm/nemotron-generation-config" not in final_args


def test_vllm_start_script_uses_parser_helper():
    repo_root = Path(__file__).resolve().parents[3]
    start_script = repo_root / "deploy" / "scripts" / "vllm_start.sh"
    contents = start_script.read_text(encoding="utf-8")

    assert "/parse_vllm_extra_args.py" in contents
    assert "mapfile -d '' -t extra_args" in contents
