"""Preflight catches native/transport failures and cannot authorize registration."""
import importlib
import json
import re
import shlex
import socket
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "deploy/scripts"))


@pytest.fixture
def preflight():
    class LazyPreflight:
        def __getattr__(self, name):
            assert (ROOT / "deploy/scripts/kv_storage_preflight.py").is_file(), "preflight missing"
            return getattr(importlib.import_module("kv_storage_preflight"), name)
    return LazyPreflight()


def test_native_import_failure_is_actionable(preflight):
    with pytest.raises(ValueError, match="native.*build"):
        preflight.import_required_native("aquillm_intentionally_absent_module")


def test_identity_mismatch_fails_before_import_or_model_load(preflight):
    with pytest.raises(ValueError, match="torch"):
        preflight.validate_identity({"torch": "2.10.0", "python": "3.12.13"})


def test_cpp_abi_mismatch(preflight):
    from kv_storage_plan import PINS
    actual = dict(PINS, cxx11_abi=False)
    with pytest.raises(ValueError, match="cxx11_abi"):
        preflight.validate_identity(actual)


def test_native_cli_missing_distribution_exits_64_without_traceback():
    # Pin inspection is an unrelated external boundary. Reach the actual CLI
    # handler with only distribution discovery unavailable, as on a base image.
    code = """
import sys
from importlib import metadata
import kv_storage_preflight as preflight
preflight.base_identity = lambda: {}
def missing_distribution(name):
    raise metadata.PackageNotFoundError(name)
preflight.metadata.version = missing_distribution
sys.argv = ['kv_storage_preflight.py', '--native']
raise SystemExit(preflight.main())
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT / "deploy/scripts",
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 64, result.stderr
    assert "LMCache 0.5.5" in result.stderr
    assert "Dockerfile.kv-storage" in result.stderr
    assert "Traceback" not in result.stderr


def test_unreachable_service_fails(preflight):
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        port = reserved.getsockname()[1]
        with pytest.raises(ValueError, match="unreachable"):
            preflight.check_endpoint("127.0.0.1", port)


def test_reachable_tcp_is_explicitly_not_protocol_handshake(preflight):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        assert preflight.check_endpoint("127.0.0.1", listener.getsockname()[1]) == "tcp-only-not-handshake"


def test_ssd_missing_directory_fails_without_creating_it(preflight, tmp_path):
    path = tmp_path / "missing"
    with pytest.raises(ValueError, match="existing"):
        preflight.check_disk(path, 1)
    assert not path.exists()


def test_disk_cap_checked_against_real_free_space(preflight, tmp_path):
    with pytest.raises(ValueError, match="disk"):
        preflight.check_disk(tmp_path, (1 << 63) - 1)


def test_registration_rejects_cpu_tensors_before_readiness(preflight):
    torch = pytest.importorskip("torch")
    from .test_kv_storage_plan import layout as layout_fixture
    record = layout_fixture.__wrapped__()
    tensors = {0: [torch.empty((8, 16, 4, 388), dtype=torch.uint8)]}
    with pytest.raises(ValueError, match="CUDA"):
        preflight.inspect_allocations(record, tensors)


def test_registration_rejects_mismatching_real_tensor_stride(preflight):
    torch = pytest.importorskip("torch")
    from .test_kv_storage_plan import layout as layout_fixture
    record = layout_fixture.__wrapped__()
    tensors = {0: [torch.empty((8, 4, 16, 388), dtype=torch.uint8).transpose(1, 2)]}
    with pytest.raises(ValueError, match="stride"):
        preflight.inspect_allocations(record, tensors)


def build_python_blocks():
    script = ROOT / "deploy/docker/vllm/build_lmcache.sh"
    return re.findall(r"<<'PY'[^\n]*\n(.*?)\nPY", script.read_text(), flags=re.DOTALL)


def test_sdk_include_paths_skip_unused_missing_directory(tmp_path):
    include = tmp_path / "headers"
    include.mkdir()
    (include / "real_client.h").write_text("// fixture\n")
    missing = tmp_path / "legacy-proto"
    record = [{"directory": str(tmp_path), "file": "/fixture/real_client.cpp",
               "command": shlex.join(["g++", "-Iheaders", "-isystem", str(missing), "-Iheaders"])}]
    commands = tmp_path / "compile_commands.json"
    commands.write_text(json.dumps(record))
    # Map the old fixed external file location to our real fixture, without
    # changing extraction logic; the revised block accepts the file as argv.
    bootstrap = """
import pathlib, sys
original_path = pathlib.Path
def fixture_path(value, *parts):
    return original_path(sys.argv[1] if str(value) == '/opt/Mooncake/build/compile_commands.json' else value, *parts)
pathlib.Path = fixture_path
"""
    result = subprocess.run([sys.executable, "-c", bootstrap + build_python_blocks()[0], str(commands)],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(include)
    assert str(missing) in result.stderr


def test_sdk_runtime_rpath_drops_only_stub_directory():
    blocks = build_python_blocks()
    assert len(blocks) >= 2, "SDK runtime path sanitization missing"
    namespace = {"__name__": "rpath_test"}
    exec(blocks[1], namespace)
    sanitize = namespace["runtime_paths_without_stubs"]
    assert sanitize("/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs:$ORIGIN:/keep/stubs2") == "/usr/local/cuda/lib64:$ORIGIN:/keep/stubs2"
    assert sanitize("$ORIGIN") == "$ORIGIN"
