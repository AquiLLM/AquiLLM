"""Integration tests for vLLM extra-args parsing."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


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


_REVISION = "4bd860ac4f15ad1897a214615cccc700f8f71818"
_RERANK_EXTRA_ARGS = (
    "--chat-template /templates/qwen3_vl_reranker.jinja "
    "--hf-overrides "
    '\'{"architectures":["Qwen3VLForSequenceClassification"],'
    '"classifier_from_token":["no","yes"],'
    '"is_original_qwen3_reranker":true}\''
)
_EMBED_EXTRA_ARGS = (
    "--quantization bitsandbytes --load-format bitsandbytes "
    "--model-loader-extra-config "
    '\'{"load_in_4bit":true,"bnb_4bit_compute_dtype":"float16",'
    '"bnb_4bit_quant_type":"nf4","bnb_4bit_use_double_quant":true}\' '
    "--hf-overrides "
    "'{\"matryoshka_dimensions\":[1024]}'"
)


def _strict_reranker_environment(**overrides: str) -> dict[str, str]:
    environment = {
        "VLLM_MODEL": "Qwen/Qwen3-VL-Reranker-2B",
        "VLLM_SERVED_MODEL_NAME": "Qwen/Qwen3-VL-Reranker-2B",
        "VLLM_TOKENIZER": "Qwen/Qwen3-VL-Reranker-2B",
        "VLLM_REVISION": _REVISION,
        "VLLM_TOKENIZER_REVISION": _REVISION,
        "VLLM_CODE_REVISION": _REVISION,
        "VLLM_RUNNER": "pooling",
        "VLLM_TASK": "",
        "VLLM_DTYPE": "float16",
        "VLLM_TRUST_REMOTE_CODE": "1",
        "VLLM_TENSOR_PARALLEL_SIZE": "1",
        "VLLM_GPU_MEMORY_UTILIZATION": "0.30",
        "VLLM_MAX_MODEL_LEN": "1024",
        "VLLM_STRICT_PROTECTED_ARGS": "1",
        "VLLM_API_KEY": "EMPTY",
        "VLLM_DOWNLOAD_DIR": "/root/.cache/huggingface/hub",
        "VLLM_EXTRA_ARGS": _RERANK_EXTRA_ARGS,
        "LMCACHE_ENABLED": "0",
    }
    environment.update(overrides)
    return environment


def test_vllm_start_uses_supported_log_requests_disable_flag(tmp_path: Path):
    final_args = _run_vllm_start(
        tmp_path,
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


def test_vllm_start_forwards_model_tokenizer_and_code_revisions(tmp_path: Path):
    final_args = _run_vllm_start(
        tmp_path,
        **_strict_reranker_environment(),
    )

    assert final_args.count("--revision") == 1
    assert final_args.count("--tokenizer-revision") == 1
    assert final_args.count("--code-revision") == 1
    assert final_args.count("--runner") == 1
    assert final_args.count("--dtype") == 1
    assert "--task" not in final_args
    assert final_args[final_args.index("--revision") + 1] == _REVISION
    assert final_args[final_args.index("--tokenizer-revision") + 1] == _REVISION
    assert final_args[final_args.index("--code-revision") + 1] == _REVISION


@pytest.mark.parametrize(
    "overrides",
    (
        {"VLLM_MODEL": "../mutable"},
        {"VLLM_SERVED_MODEL_NAME": "Qwen/other-model"},
        {"VLLM_TOKENIZER": "Qwen/other-tokenizer"},
        {"VLLM_REVISION": "main"},
        {"VLLM_TOKENIZER_REVISION": ""},
        {"VLLM_CODE_REVISION": "A" * 40},
        {"VLLM_RUNNER": ""},
        {"VLLM_DTYPE": ""},
        {"VLLM_TASK": "score"},
        {"VLLM_TRUST_REMOTE_CODE": "0"},
        {"VLLM_TENSOR_PARALLEL_SIZE": "0"},
        {"VLLM_GPU_MEMORY_UTILIZATION": "1.01"},
        {"VLLM_MAX_MODEL_LEN": "0"},
    ),
)
def test_strict_vllm_start_rejects_mutable_or_incomplete_contract(
    tmp_path: Path,
    overrides: dict[str, str],
):
    with pytest.raises(subprocess.CalledProcessError):
        _run_vllm_start(
            tmp_path,
            **_strict_reranker_environment(**overrides),
        )


def test_strict_vllm_start_rejects_unsupported_required_flag(tmp_path: Path):
    supported_without_code_revision = (
        "--model --served-model-name --tokenizer --revision "
        "--tokenizer-revision --runner --dtype --trust-remote-code "
        "--tensor-parallel-size --gpu-memory-utilization --max-model-len "
        "--api-key --download-dir"
    )

    with pytest.raises(subprocess.CalledProcessError):
        _run_vllm_start(
            tmp_path,
            **_strict_reranker_environment(
                FAKE_VLLM_HELP_ARGS=supported_without_code_revision
            ),
        )


def test_strict_vllm_start_reads_supported_options_once(tmp_path: Path):
    help_count = tmp_path / "help-count"

    _run_vllm_start(
        tmp_path,
        **_strict_reranker_environment(FAKE_HELP_COUNT_FILE=_bash_path(help_count)),
    )

    assert help_count.read_text(encoding="utf-8").splitlines() == ["help"]


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"VLLM_PYTHON_BIN": "python"}, "python3"),
        ({"VLLM_API_KEY": "secret"}, "API key"),
        ({"VLLM_DOWNLOAD_DIR": "/tmp/models"}, "download directory"),
        ({"VLLM_EXTRA_ARGS": ""}, "extra-argument"),
        (
            {"VLLM_EXTRA_ARGS": _RERANK_EXTRA_ARGS + " --enforce-eager"},
            "extra arguments",
        ),
    ),
)
def test_strict_vllm_start_rejects_unattested_runtime_inputs(
    tmp_path: Path,
    overrides: dict[str, str],
    message: str,
):
    with pytest.raises(subprocess.CalledProcessError) as error:
        _run_vllm_start(
            tmp_path,
            **_strict_reranker_environment(**overrides),
        )

    assert message in error.value.stderr


def test_strict_python_token_is_rejected_before_hostile_binary_executes(
    tmp_path: Path,
):
    hostile = tmp_path / "hostile-python"
    marker = tmp_path / "hostile-executed"
    hostile.write_text(
        f"#!/bin/sh\nprintf executed > '{_bash_path(marker)}'\nexit 99\n",
        encoding="utf-8",
        newline="\n",
    )
    runner = ["wsl"] if os.name == "nt" else []
    subprocess.run(
        [*runner, "chmod", "+x", _bash_path(hostile)],
        check=True,
        capture_output=True,
    )

    with pytest.raises(subprocess.CalledProcessError):
        _run_vllm_start(
            tmp_path,
            **_strict_reranker_environment(VLLM_PYTHON_BIN=_bash_path(hostile)),
        )

    assert not marker.exists()


@pytest.mark.parametrize("protected_flag", ("--api-key", "--download-dir"))
def test_strict_vllm_start_rejects_auth_and_cache_extra_arg_overrides(
    tmp_path: Path,
    protected_flag: str,
):
    with pytest.raises(subprocess.CalledProcessError):
        _run_vllm_start(
            tmp_path,
            **_strict_reranker_environment(
                VLLM_EXTRA_ARGS=f"{_RERANK_EXTRA_ARGS} {protected_flag} shadow"
            ),
        )


@pytest.mark.parametrize("use_lmcache", (False, True))
def test_strict_vllm_start_rejects_missing_or_failed_extra_args_parser(
    tmp_path: Path,
    use_lmcache: bool,
):
    extra_environment = (
        {
            "LMCACHE_ENABLED": "1",
            "LMCACHE_EXTRA_ARGS": "--kv-transfer-config '{unterminated",
        }
        if use_lmcache
        else {"VLLM_EXTRA_ARGS": "--chat-template 'unterminated"}
    )

    with pytest.raises(subprocess.CalledProcessError):
        _run_vllm_start(
            tmp_path,
            **_strict_reranker_environment(**extra_environment),
        )
    with pytest.raises(subprocess.CalledProcessError):
        _run_vllm_start(
            tmp_path,
            include_parser=False,
            **_strict_reranker_environment(**extra_environment),
        )


@pytest.mark.parametrize("include_parser", (False, True))
def test_strict_parser_failure_happens_before_vllm_help(
    tmp_path: Path,
    include_parser: bool,
):
    help_count = tmp_path / "help-count"
    extra_args = (
        _RERANK_EXTRA_ARGS if not include_parser else "--chat-template 'unterminated"
    )

    with pytest.raises(subprocess.CalledProcessError):
        _run_vllm_start(
            tmp_path,
            include_parser=include_parser,
            **_strict_reranker_environment(
                FAKE_HELP_COUNT_FILE=_bash_path(help_count),
                VLLM_EXTRA_ARGS=extra_args,
            ),
        )

    assert not help_count.exists()


def test_legacy_service_retains_missing_parser_fallback(tmp_path: Path):
    final_args = _run_vllm_start(
        tmp_path,
        include_parser=False,
        VLLM_MODEL="example/legacy-model",
        VLLM_SERVED_MODEL_NAME="legacy-model",
        VLLM_EXTRA_ARGS="--runner pooling --dtype float16",
    )

    assert final_args[-4:] == ["--runner", "pooling", "--dtype", "float16"]


def test_embedding_pooling_preserves_canonical_bitsandbytes_payload(tmp_path: Path):
    loader_config = (
        '{"load_in_4bit":true,"bnb_4bit_compute_dtype":"float16",'
        '"bnb_4bit_quant_type":"nf4","bnb_4bit_use_double_quant":true}'
    )
    final_args = _run_vllm_start(
        tmp_path,
        **_strict_reranker_environment(
            VLLM_MODEL="Qwen/Qwen3-Embedding-4B",
            VLLM_SERVED_MODEL_NAME="Qwen/Qwen3-Embedding-4B",
            VLLM_TOKENIZER="Qwen/Qwen3-Embedding-4B",
            VLLM_TASK="",
            VLLM_GPU_MEMORY_UTILIZATION="0.12",
            VLLM_MAX_MODEL_LEN="2048",
            VLLM_EXTRA_ARGS=_EMBED_EXTRA_ARGS,
        ),
    )

    assert final_args[final_args.index("--quantization") + 1] == "bitsandbytes"
    assert final_args[final_args.index("--load-format") + 1] == "bitsandbytes"
    assert (
        final_args[final_args.index("--model-loader-extra-config") + 1] == loader_config
    )
    assert json.loads(final_args[final_args.index("--hf-overrides") + 1]) == {
        "matryoshka_dimensions": [1024]
    }
    assert final_args.count("--dtype") == 1


def test_reranker_workaround_strips_bitsandbytes_without_shadowing_typed_dtype(
    tmp_path: Path,
):
    final_args = _run_vllm_start(
        tmp_path,
        **_strict_reranker_environment(
            VLLM_STRICT_PROTECTED_ARGS="0",
            VLLM_EXTRA_ARGS=(
                "--quantization bitsandbytes --load-format bitsandbytes "
                "--model-loader-extra-config '{\"load_in_4bit\":true}' "
                "--chat-template /templates/qwen3_vl_reranker.jinja"
            ),
        ),
    )

    assert "--quantization" not in final_args
    assert "--load-format" not in final_args
    assert "--model-loader-extra-config" not in final_args
    assert final_args.count("--dtype") == 1
    assert final_args[final_args.index("--dtype") + 1] == "float16"
    assert "/templates/qwen3_vl_reranker.jinja" in final_args


def test_strict_reranker_rejects_bitsandbytes_before_legacy_sanitizer(
    tmp_path: Path,
):
    with pytest.raises(subprocess.CalledProcessError):
        _run_vllm_start(
            tmp_path,
            **_strict_reranker_environment(
                VLLM_EXTRA_ARGS=(
                    _RERANK_EXTRA_ARGS
                    + " --quantization bitsandbytes --load-format bitsandbytes "
                    "--model-loader-extra-config '{\"load_in_4bit\":true}'"
                )
            ),
        )


@pytest.mark.parametrize(
    "protected_flag",
    (
        "--model",
        "--served-model-name",
        "--tokenizer",
        "--revision",
        "--tokenizer-revision",
        "--code-revision",
        "--trust-remote-code",
        "--no-trust-remote-code",
        "--runner",
        "--dtype",
        "--tensor-parallel-size",
        "--gpu-memory-utilization",
        "--max-model-len",
        "--task",
    ),
)
@pytest.mark.parametrize("equals_form", (False, True))
def test_vllm_start_rejects_protected_extra_arg_overrides(
    tmp_path: Path,
    protected_flag: str,
    equals_form: bool,
):
    raw = f"{protected_flag}=shadow" if equals_form else f"{protected_flag} shadow"

    with pytest.raises(subprocess.CalledProcessError):
        _run_vllm_start(
            tmp_path,
            **_strict_reranker_environment(VLLM_EXTRA_ARGS=raw),
        )


def test_legacy_services_preserve_protected_extra_args_without_strict_flag(
    tmp_path: Path,
):
    final_args = _run_vllm_start(
        tmp_path,
        VLLM_MODEL="example/legacy-model",
        VLLM_SERVED_MODEL_NAME="legacy-model",
        VLLM_EXTRA_ARGS="--runner pooling --dtype float16",
    )

    assert final_args[final_args.index("--runner") + 1] == "pooling"
    assert final_args[final_args.index("--dtype") + 1] == "float16"


def test_vllm_start_script_uses_parser_helper():
    repo_root = Path(__file__).resolve().parents[3]
    start_script = repo_root / "deploy" / "scripts" / "vllm_start.sh"
    contents = start_script.read_text(encoding="utf-8")

    assert "/parse_vllm_extra_args.py" in contents
    assert "parse_extra_args_into" in contents
    assert "mapfile -d '' -t parsed_output" in contents


def test_checked_in_profiles_keep_protected_runner_and_dtype_out_of_extra_args():
    repo_root = Path(__file__).resolve().parents[3]
    env_lines = (repo_root / ".env.example").read_text(encoding="utf-8").splitlines()
    active_extra_lines = [
        line
        for line in env_lines
        if line and not line.startswith("#") and "VLLM_EXTRA_ARGS=" in line
    ]

    assert active_extra_lines
    assert all(
        "--dtype" not in line and "--runner" not in line for line in active_extra_lines
    )
    assert "VLLM_DTYPE=float16" in env_lines
    assert "OCR_VLLM_DTYPE=float16" in env_lines

    profile = (repo_root / "scripts" / "verify_nemotron_asr.ps1").read_text(
        encoding="utf-8"
    )
    for prefix in (
        "PROFILE_MAIN_EXTRA_ARGS=",
        "PROFILE_EMBED_EXTRA_ARGS=",
        "PROFILE_RERANK_EXTRA_ARGS=",
    ):
        line = next(row for row in profile.splitlines() if row.startswith(prefix))
        assert "--dtype" not in line
        assert "--runner" not in line
    assert profile.count("VLLM_RUNNER: pooling") >= 2
    assert profile.count("VLLM_DTYPE: float16") >= 3


def test_repository_forces_shell_scripts_to_lf():
    repo_root = Path(__file__).resolve().parents[3]
    attributes_path = repo_root / ".gitattributes"

    assert attributes_path.exists(), ".gitattributes must define shell line endings"
    attributes = attributes_path.read_text(encoding="utf-8").splitlines()
    assert "*.sh text eol=lf" in attributes
