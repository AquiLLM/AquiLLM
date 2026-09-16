"""Integration tests for vLLM extra-args parsing."""

from __future__ import annotations

import os
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


def _bash_path(path: Path) -> str:
    if os.name != "nt":
        return str(path)
    absolute = path.resolve().as_posix()
    drive = absolute[0].lower()
    return f"/mnt/{drive}/{absolute[3:]}"


def _run_vllm_start(
    tmp_path: Path,
    *,
    include_parser: bool = True,
    **environment: str,
) -> list[str]:
    repo_root = Path(__file__).resolve().parents[3]
    checked_in_start = repo_root / "deploy" / "scripts" / "vllm_start.sh"
    start_copy = tmp_path / "vllm_start.sh"
    start_source = checked_in_start.read_text(encoding="utf-8")
    if include_parser:
        parser_path = _bash_path(
            repo_root / "deploy" / "scripts" / "parse_vllm_extra_args.py"
        )
        start_source = start_source.replace(
            'parser_script="/parse_vllm_extra_args.py"',
            f'parser_script="{parser_path}"',
        )
    start_copy.write_text(start_source, encoding="utf-8", newline="\n")
    start_script = _bash_path(start_copy)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(exist_ok=True)
    fake_python = fake_bin / "python3"
    fake_python.write_text(
        "#!/bin/sh\n"
        "if [ \"${1:-}\" = '-' ]; then exec /usr/bin/python3 -; fi\n"
        'case "${1:-}" in '
        '*parse_vllm_extra_args.py) exec /usr/bin/python3 "$@" ;; esac\n'
        'case " $* " in\n'
        "  *' --help '*) test -z \"${FAKE_HELP_COUNT_FILE:-}\" || "
        "printf 'help\\n' >> \"$FAKE_HELP_COUNT_FILE\"; "
        'for item in ${FAKE_VLLM_HELP_ARGS:-"--model '
        "--served-model-name --tokenizer --revision --tokenizer-revision "
        "--code-revision --runner --dtype --trust-remote-code "
        "--tensor-parallel-size --gpu-memory-utilization --max-model-len "
        '--api-key --download-dir"}; do '
        "printf '%s\\n' \"$item\"; done; exit 0 ;;\n"
        "esac\n"
        'for arg in "$@"; do printf "FINAL_ARG=%s\\n" "$arg"; done\n',
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
        f"PATH={_bash_path(fake_bin)}:/usr/local/sbin:/usr/local/bin:"
        "/usr/sbin:/usr/bin:/sbin:/bin",
        "VLLM_PYTHON_BIN=python3",
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


def test_vllm_start_uses_supported_log_requests_disable_flag(tmp_path: Path):
    final_args = _run_vllm_start(
        tmp_path,
        VLLM_EXTRA_ARGS="",
        FAKE_VLLM_HELP_ARGS="--no-enable-log-requests",
    )

    assert "--no-enable-log-requests" in final_args
    assert "--disable-log-requests" not in final_args


def test_vllm_start_unsets_deployment_only_vllm_metadata():
    repo_root = Path(__file__).resolve().parents[3]
    script = (repo_root / "deploy/scripts/vllm_start.sh").read_text(
        encoding="utf-8"
    )
    unset_block = script.split(
        "# Avoid vLLM env validation warnings for wrapper-only variables.", 1
    )[1].split('echo "Starting vLLM', 1)[0]

    for variable in (
        "VLLM_BUILD_URL",
        "VLLM_IMAGE_TAG",
        "VLLM_CACHE_PATH",
        "VLLM_BUILD_PIPELINE",
        "VLLM_BUILD_COMMIT",
        "VLLM_GENESIS_BASE_IMAGE",
    ):
        assert variable in unset_block
