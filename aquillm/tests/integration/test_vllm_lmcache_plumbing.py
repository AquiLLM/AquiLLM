"""LMCache env wiring in vLLM startup script and compose services."""
from __future__ import annotations

from pathlib import Path
import json
import os
import runpy
import shlex
import shutil
import subprocess
import sys

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_VLLM_SH = _ROOT / "deploy" / "scripts" / "vllm_start.sh"
_BASE_YML = _ROOT / "deploy" / "compose" / "base.yml"


def _bash_path(path: Path) -> str:
    absolute = path.resolve().as_posix()
    return f"/{absolute[0].lower()}/{absolute[3:]}" if os.name == "nt" else absolute


def _bash_executable() -> str:
    if os.name == "nt":
        return "C:/Program Files/Git/bin/bash.exe"
    return shutil.which("bash") or "bash"


def _chmod_executable(path: Path):
    subprocess.run([_bash_executable(), "-c", 'chmod +x "$1"', "--", _bash_path(path)],
                   check=True, capture_output=True)


def _launch(tmp_path, *, include_parser=True, include_profile_helper=True,
            profile_helper_source=None, **environment):
    """Run real Bash/parser/helper; replace only the GPU vLLM endpoint."""
    source = _VLLM_SH.read_text(encoding="utf-8")
    parser = (_ROOT / "deploy/scripts/parse_vllm_extra_args.py"
              if include_parser else tmp_path / "missing-parser.py")
    helper = (_ROOT / "deploy/scripts/kv_cache_config.py"
              if include_profile_helper else tmp_path / "missing-helper.py")
    if profile_helper_source is not None:
        helper = tmp_path / "kv_cache_config.py"
        helper.write_text(profile_helper_source, encoding="utf-8")
    source = source.replace('parser_script="/parse_vllm_extra_args.py"',
                            f'parser_script="{parser.resolve().as_posix()}"')
    source = source.replace('profile_script="/kv_cache_config.py"',
                            f'profile_script="{helper.resolve().as_posix()}"')
    start = tmp_path / "vllm_start.sh"
    start.write_text(source, encoding="utf-8", newline="\n")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(exist_ok=True)
    fake_python = fake_bin / "python3"
    python = shlex.quote(Path(sys.executable).as_posix())
    fake_python.write_text(
        '#!/bin/bash\n'
        'case "${1:-}" in\n'
        f'  -|*parse_vllm_extra_args.py|*kv_cache_config.py) exec {python} "$@" ;;\n'
        'esac\n'
        'case " $* " in\n'
        '  *" --help "*) test -z "${FAKE_HELP_COUNT_FILE:-}" || '
        'printf "help\\n" >> "$FAKE_HELP_COUNT_FILE"; '
        'printf "%s\\n" "${FAKE_VLLM_HELP_ARGS:-'
        '--model --served-model-name --tokenizer --revision --tokenizer-revision '
        '--code-revision --runner --dtype --trust-remote-code --tensor-parallel-size '
        '--gpu-memory-utilization --max-model-len --max-num-seqs --kv-cache-dtype '
        '--kv-cache-memory-bytes --api-key --download-dir}"; exit 0 ;;\n'
        'esac\n'
        'printf "FINAL_PROFILE_ENV=%s\\n" "${KV_CACHE_TARGET_ACTIVE_SEQUENCES:-}"\n'
        'for arg in "$@"; do printf "FINAL_ARG=%s\\n" "$arg"; done\n',
        encoding="utf-8", newline="\n")
    _chmod_executable(fake_python)
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("VLLM_", "KV_CACHE_", "LMCACHE_", "LLM_", "OCR_", "TRANSCRIBE_", "APP_RERANK_", "MEM0_"))}
    env.update({"VLLM_PYTHON_BIN": "python3", "MSYS2_ARG_CONV_EXCL": "*"})
    env.update(environment)
    # Set PATH inside Bash so Windows PATH import cannot reorder the fake endpoint.
    return subprocess.run([_bash_executable(), "-c", 'export PATH="$1:$PATH"; exec bash "$2"',
                           "--", _bash_path(fake_bin), _bash_path(start)],
                          env=env, capture_output=True, text=True)


def _final_args(result):
    return [line.removeprefix("FINAL_ARG=") for line in result.stdout.splitlines()
            if line.startswith("FINAL_ARG=")]


def _run_vllm_start(tmp_path, **kwargs):
    result = _launch(tmp_path, **kwargs)
    result.check_returncode()
    return _final_args(result)


def test_lmcache_connector_is_forwarded_only_when_enabled(tmp_path):
    payload = '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
    args = _run_vllm_start(tmp_path, LMCACHE_ENABLED="1",
                           LMCACHE_EXTRA_ARGS=f"--kv-transfer-config '{payload}'")
    assert args[-2:] == ["--kv-transfer-config", payload]
    args = _run_vllm_start(tmp_path, LMCACHE_ENABLED="0",
                           LMCACHE_EXTRA_ARGS=f"--kv-transfer-config '{payload}'")
    assert "--kv-transfer-config" not in args


@pytest.mark.parametrize("raw", ["--kv-cache-dtype turboquant_k8v4 --max-num-seqs 8",
                                 "--kv-cache-dtype=turboquant_k8v4 --max-num-seqs=8"])
def test_profile_launcher_emits_one_scheduler_and_preserves_mtp(tmp_path, raw):
    mtp = '{"method":"mtp","num_speculative_tokens":3}'
    result = _launch(tmp_path, KV_CACHE_TARGET_ACTIVE_SEQUENCES="8", VLLM_MAX_MODEL_LEN="262144",
                     VLLM_DTYPE="bfloat16", VLLM_EXTRA_ARGS=f"{raw} --speculative-config '{mtp}'")
    assert result.returncode == 0, result.stderr
    args = _final_args(result)
    assert args.count("--max-num-seqs") == 1
    assert args[args.index("--max-num-seqs") + 1] == "8"
    assert args.count("--max-model-len") == 1
    assert args[args.index("--kv-cache-dtype") + 1] == "turboquant_k8v4"
    assert args[args.index("--dtype") + 1] == "bfloat16"
    assert args[args.index("--speculative-config") + 1] == mtp
    assert "FINAL_PROFILE_ENV=\n" in result.stdout
    assert '"full_attention_payload_bytes": 52076478464' in result.stderr


def test_profile_json_never_executes_shell_syntax(tmp_path):
    marker = tmp_path / "injected"
    payload = json.dumps({"method": "mtp", "note": f"$(touch {_bash_path(marker)}); `touch {_bash_path(marker)}`"})
    args = _run_vllm_start(tmp_path, KV_CACHE_TARGET_ACTIVE_SEQUENCES="4",
                           VLLM_EXTRA_ARGS=f"--speculative-config '{payload}'")
    assert args[args.index("--speculative-config") + 1] == payload
    assert not marker.exists()


@pytest.mark.parametrize("environment", [
    {"VLLM_EXTRA_ARGS": "--max-num-seqs 9"},
    {"VLLM_EXTRA_ARGS": "--kv-cache-dtype turboquant_k4v4"},
    {"VLLM_EXTRA_ARGS": "--cpu-offload-gb=1"},
    {"VLLM_EXTRA_ARGS": "--kv-offloading-size 8"},
    {"VLLM_EXTRA_ARGS": "--disable-hybrid-kv-cache-manager"},
    {"LMCACHE_ENABLED": "1", "LMCACHE_EXTRA_ARGS": "--max-num-seqs 9"},
    {"LMCACHE_ENABLED": "1", "LMCACHE_EXTRA_ARGS": "--kv-transfer-config '{}'"},
    {"KV_CACHE_STORAGE_MODE": "mooncake"}, {"KV_CACHE_EXECUTION_MODE": "paged"},
    {"VLLM_MODEL": "google/gemma-3-12b-it"}, {"LLM_CHOICE": "GPT-OSS"},
    {"VLLM_MODEL": "example/EmbeddingMain"},
])
def test_invalid_profile_stops_before_vllm_help_and_startup(tmp_path, environment):
    help_count = tmp_path / "help-count"
    result = _launch(tmp_path, KV_CACHE_TARGET_ACTIVE_SEQUENCES="4",
                     FAKE_HELP_COUNT_FILE=_bash_path(help_count), **environment)
    assert result.returncode != 0
    assert not _final_args(result)
    assert not help_count.exists()


@pytest.mark.parametrize("kwargs", [
    {"include_profile_helper": False},
    {"profile_helper_source": "import sys\nsys.stdout.buffer.write(b'--max-num-seqs\\x008\\x00')\nsys.exit(9)\n"},
    {"include_parser": False},
])
def test_requested_profile_requires_successful_helpers(tmp_path, kwargs):
    result = _launch(tmp_path, KV_CACHE_TARGET_ACTIVE_SEQUENCES="4", VLLM_EXTRA_ARGS="--enforce-eager", **kwargs)
    assert result.returncode != 0
    assert not _final_args(result)


def test_legacy_offloading_still_disables_hybrid_manager(tmp_path):
    args = _run_vllm_start(tmp_path, VLLM_EXTRA_ARGS="--kv-offloading-size 8")
    assert args.count("--disable-hybrid-kv-cache-manager") == 1


@pytest.mark.parametrize("model,kind", [
    ("nvidia/nemotron-3.5-asr-streaming-0.6b", "transcribe"),
    ("openai/whisper-large-v3-turbo", "transcribe"),
    ("Qwen/Qwen2.5-VL-7B-Instruct", "ocr"),
    ("openai/whisper-large-v3-turbo", ""),
    ("Qwen/Qwen3.5-4B", ""),
    ("nvidia/nemotron-3.5-asr-streaming-0.6b", ""),
])
def test_sidecars_ignore_inherited_profiles(tmp_path, model, kind):
    args = _run_vllm_start(tmp_path, VLLM_MODEL=model, VLLM_SERVICE_KIND=kind,
                           VLLM_EXTRA_ARGS="--max-num-seqs 1", KV_CACHE_TARGET_ACTIVE_SEQUENCES="8",
                           KV_CACHE_STORAGE_MODE="mooncake", KV_CACHE_EXECUTION_MODE="paged",
                           LMCACHE_ENABLED="1", LMCACHE_EXTRA_ARGS="--kv-transfer-config '{}'")
    assert args[args.index("--max-num-seqs") + 1] == "1"
    assert "--kv-transfer-config" not in args
    assert "turboquant_k8v4" not in args


@pytest.mark.parametrize("embedding", [False, True])
def test_strict_sidecars_ignore_shared_profile_without_service_kind(tmp_path, embedding):
    fixtures = runpy.run_path(str(Path(__file__).with_name("test_vllm_extra_args_parser.py")))
    environment = fixtures["_strict_reranker_environment"]()
    if embedding:
        environment.update({
            "VLLM_MODEL": "Qwen/Qwen3-Embedding-4B",
            "VLLM_SERVED_MODEL_NAME": "Qwen/Qwen3-Embedding-4B",
            "VLLM_TOKENIZER": "Qwen/Qwen3-Embedding-4B",
            "VLLM_EXTRA_ARGS": fixtures["_EMBED_EXTRA_ARGS"],
        })
    args = _run_vllm_start(tmp_path, **environment, KV_CACHE_TARGET_ACTIVE_SEQUENCES="8",
                           KV_CACHE_STORAGE_MODE="mooncake", KV_CACHE_EXECUTION_MODE="paged")
    assert "--max-num-seqs" not in args
    assert "--kv-transfer-config" not in args
    assert "turboquant_k8v4" not in args


def test_profile_requires_supported_scheduler_flags(tmp_path):
    result = _launch(tmp_path, KV_CACHE_TARGET_ACTIVE_SEQUENCES="4",
                     FAKE_VLLM_HELP_ARGS="--max-model-len --kv-cache-dtype")
    assert result.returncode != 0
    assert "--max-num-seqs" in result.stderr
    assert not _final_args(result)


def test_explicit_vram_budget_is_emitted_in_bytes(tmp_path):
    args = _run_vllm_start(tmp_path, KV_CACHE_VRAM_GIB="24.25")
    assert args[args.index("--kv-cache-memory-bytes") + 1] == "26038239232"


@pytest.mark.parametrize("raw", [
    "--cpu_offload_gb=1", "--hf-overrides.num_hidden_layers=40",
    "--kv-transfer-config.kv_connector=LMCacheConnectorV1",
    "--max_num_seqs=8", "--kv_cache_dtype=turboquant_k4v4",
    "--cpu-offload-g=1", "--hf-over.num_hidden_layers=40",
    "--config /tmp/weight-offload.yaml", "--config=/tmp/weight-offload.yaml",
    "--conf=/tmp/weight-offload.yaml", "-c /tmp/weight-offload.yaml",
])
def test_profile_guard_rejects_downstream_aliases_before_startup(tmp_path, raw):
    help_count = tmp_path / "help-count"
    result = _launch(tmp_path, KV_CACHE_TARGET_ACTIVE_SEQUENCES="4", VLLM_EXTRA_ARGS=raw,
                     FAKE_HELP_COUNT_FILE=_bash_path(help_count))
    assert result.returncode != 0
    assert not _final_args(result)
    assert not help_count.exists()


def test_launcher_removes_matching_underscore_capacity_options(tmp_path):
    args = _run_vllm_start(tmp_path, KV_CACHE_TARGET_ACTIVE_SEQUENCES="4",
                           VLLM_EXTRA_ARGS="--max_num_seqs=4 --max_model_len 131072 --kv_cache_dtype=turboquant_k8v4 "
                           "--speculative-config.method=mtp --compilation_config.cudagraph_mode=PIECEWISE")
    assert args[-6:] == ["--max-num-seqs", "4", "--max-model-len", "131072",
                         "--kv-cache-dtype", "turboquant_k8v4"]
    assert not any(arg.startswith(("--max_num", "--max_model", "--kv_cache")) for arg in args)
    assert "--speculative-config.method=mtp" in args
    assert "--compilation_config.cudagraph_mode=PIECEWISE" in args


def test_profile_unset_preserves_config_and_alias_forwarding(tmp_path):
    args = _run_vllm_start(tmp_path, VLLM_EXTRA_ARGS="--config /tmp/config.yaml --cpu_offload_gb=1")
    assert args[-3:] == ["--config", "/tmp/config.yaml", "--cpu_offload_gb=1"]


@pytest.mark.parametrize("raw,expected", [
    ("--compilation-config '{}'", ["--compilation-config", "{}"]),
    ("-cc '{}'", ["-cc", "{}"]), ("-cc='{}'", ["-cc={}"]),
    ("-cc.mode 0", ["-cc.mode", "0"]), ("-cc.mode=0", ["-cc.mode=0"]),
])
def test_compilation_alias_survives_profile_launcher(tmp_path, raw, expected):
    args = _run_vllm_start(tmp_path, KV_CACHE_TARGET_ACTIVE_SEQUENCES="4",
                           VLLM_EXTRA_ARGS=raw)
    assert args[-len(expected)-6:] == expected + [
        "--max-num-seqs", "4", "--max-model-len", "262144",
        "--kv-cache-dtype", "turboquant_k8v4",
    ]


def test_base_compose_vllm_exports_lmcache_env():
    text = _BASE_YML.read_text(encoding="utf-8")
    assert "LMCACHE_ENABLED=" in text
    assert "LMCACHE_EXTRA_ARGS=" in text
