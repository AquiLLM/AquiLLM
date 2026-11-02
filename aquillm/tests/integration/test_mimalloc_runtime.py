"""Exercise the shipped allocator wrapper and native library in a Linux image.

Build deploy/docker/mimalloc/Dockerfile.test and set MIMALLOC_TEST_IMAGE.
No model downloads, databases, or GPU are needed.
"""

import json
import os
import shutil
import subprocess

import pytest


@pytest.fixture(scope="module")
def image():
    name = os.environ.get("MIMALLOC_TEST_IMAGE")
    if not name:
        pytest.skip("set MIMALLOC_TEST_IMAGE to the built allocator smoke image")
    assert shutil.which("docker"), "Docker is required for allocator runtime tests"
    return name


def run(image, *command, environment=None, entrypoint=None):
    args = ["docker", "run", "--rm", "--network=none"]
    for key, value in (environment or {}).items():
        args.extend(["-e", f"{key}={value}"])
    if entrypoint:
        args.extend(["--entrypoint", entrypoint])
    return subprocess.run(
        [*args, image, *command], capture_output=True, text=True, timeout=60
    )


PROBE = """
import ctypes, json, os, subprocess, sys
lib = ctypes.CDLL(None)
active = hasattr(lib, 'mi_version')
if active:
    malloc_address = ctypes.cast(lib.malloc, ctypes.c_void_p).value
    assert malloc_address == ctypes.cast(lib.mi_malloc, ctypes.c_void_p).value
    lib.malloc.argtypes = [ctypes.c_size_t]
    lib.malloc.restype = ctypes.c_void_p
    lib.realloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.realloc.restype = ctypes.c_void_p
    lib.free.argtypes = [ctypes.c_void_p]
    for size in (1, 512, 4096, 1048576):
        ptr = lib.malloc(size)
        assert ptr
        ctypes.memset(ptr, 42, size)
        ptr = lib.realloc(ptr, size * 2)
        assert ptr and ctypes.string_at(ptr, 1) == b'*'
        lib.free(ptr)
child_code = "import ctypes; print(hasattr(ctypes.CDLL(None), 'mi_version'))"
child = subprocess.check_output(
    [sys.executable, '-c', child_code], text=True).strip()
print(json.dumps({
    'active': active, 'child': child, 'pid': os.getpid(), 'argv': sys.argv[1:],
    'pythonmalloc': os.environ.get('PYTHONMALLOC'),
    'preload': os.environ.get('LD_PRELOAD', '')}))
"""


@pytest.mark.parametrize("pythonmalloc", ["default", "malloc"])
def test_real_malloc_override_and_subprocess_inheritance(image, pythonmalloc):
    result = run(
        image,
        "python3",
        "-c",
        PROBE,
        "argument with spaces",
        "",
        environment={"PYTHONMALLOC": pythonmalloc},
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["active"] and data["child"] == "True"
    assert data["pid"] == 1
    assert data["argv"] == ["argument with spaces", ""]
    assert data["pythonmalloc"] == pythonmalloc


def test_system_mode_restores_original_allocator_and_preserves_other_preloads(image):
    result = run(
        image,
        "python3",
        "-c",
        PROBE,
        environment={"AQUILLM_ALLOCATOR": "system", "LD_PRELOAD": "libm.so.6"},
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert not data["active"] and data["child"] == "False"
    assert data["preload"] == "libm.so.6"


def test_mimalloc_preserves_unrelated_preloads_and_nested_launches(image):
    result = run(
        image,
        "/usr/local/bin/aquillm-allocator",
        "python3",
        "-c",
        PROBE,
        environment={"LD_PRELOAD": "libm.so.6"},
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["active"]
    assert data["preload"].count("/opt/mimalloc/lib/libmimalloc.so") == 1
    assert "libm.so.6" in data["preload"]


@pytest.mark.parametrize("allocator", ["typo", ""])
def test_invalid_allocator_fails_before_running_application(image, allocator):
    result = run(
        image, "echo", "application-ran", environment={"AQUILLM_ALLOCATOR": allocator}
    )
    assert result.returncode == 64
    assert "application-ran" not in result.stdout


@pytest.mark.parametrize(
    "preload", ["libjemalloc.so.2", "libtcmalloc.so.4", "/other/libmimalloc.so"]
)
def test_conflicting_allocator_is_rejected(image, preload):
    result = run(image, "echo", "application-ran", environment={"LD_PRELOAD": preload})
    assert result.returncode == 78
    assert "application-ran" not in result.stdout


def test_missing_library_fails_instead_of_silently_using_libc(image):
    result = run(
        image,
        "-c",
        "mv /opt/mimalloc/lib /opt/mimalloc/hidden; "
        "exec /usr/local/bin/aquillm-allocator echo application-ran",
        entrypoint="/bin/sh",
    )
    assert result.returncode == 78
    assert "application-ran" not in result.stdout


def test_library_that_does_not_override_malloc_is_rejected(image):
    result = run(
        image,
        "-c",
        "rm /opt/mimalloc/lib/libmimalloc.so; "
        "ln -s /lib/$(uname -m)-linux-gnu/libm.so.6 "
        "/opt/mimalloc/lib/libmimalloc.so; "
        "exec /usr/local/bin/aquillm-allocator echo application-ran",
        entrypoint="/bin/sh",
    )
    assert result.returncode == 78
    assert "application-ran" not in result.stdout


def test_exec_preserves_application_exit_code(image):
    result = run(image, "sh", "-c", "exit 37")
    assert result.returncode == 37


@pytest.mark.parametrize("pythonmalloc", ["default", "malloc"])
def test_prefork_workers_survive_repeated_allocation_and_recycling(image, pythonmalloc):
    code = """
import ctypes, multiprocessing
def churn(_):
    lib = ctypes.CDLL(None)
    assert hasattr(lib, 'mi_version')
    for _ in range(30):
        values = [bytearray(2048 + i) for i in range(1000)]
        assert len(values[-1]) == 3047
    return True
if __name__ == '__main__':
    with multiprocessing.get_context('fork').Pool(3, maxtasksperchild=2) as pool:
        assert all(pool.map(churn, range(12)))
    print('workers-ok')
"""
    result = run(
        image, "python3", "-c", code, environment={"PYTHONMALLOC": pythonmalloc}
    )
    assert result.returncode == 0, result.stderr
    assert "workers-ok" in result.stdout
