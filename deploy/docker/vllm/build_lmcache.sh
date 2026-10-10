#!/bin/bash
set -euo pipefail
lm_ref=05a013b29da78cf2321b9b46ec5039dde2fb0bb0
lm_tag=v0.5.5
git init /opt/LMCache
git -C /opt/LMCache remote add origin https://github.com/LMCache/LMCache.git
# setuptools_scm needs the real release tag in local refs, not just its commit.
git -C /opt/LMCache fetch --depth 1 origin "refs/tags/${lm_tag}:refs/tags/${lm_tag}"
test "$(git -C /opt/LMCache rev-parse "refs/tags/${lm_tag}^{commit}")" = "$lm_ref"
git -C /opt/LMCache checkout --detach "$lm_ref"
git -C /opt/LMCache submodule update --init --recursive --depth 1
# Upstream installs libraries but not a complete public header SDK. Keep the
# pinned source tree and reuse the real client's actual CMake include closure.
python3 - /opt/Mooncake/build/compile_commands.json <<'PY' > /opt/kv-storage/mooncake-include-paths.txt
import json
from pathlib import Path
import shlex
import sys
commands = json.loads(Path(sys.argv[1]).read_text())
entry = next(c for c in commands if Path(c['file']).name == 'real_client.cpp')
tokens = iter(shlex.split(entry['command']))
includes = []
for token in tokens:
    if token in ('-I', '-isystem'):
        includes.append(next(tokens))
    elif token.startswith('-I'):
        includes.append(token[2:])
existing = []
for value in includes:
    path = Path(value)
    if not path.is_absolute():
        path = Path(entry['directory']) / path
    path = path.resolve()
    if path.is_dir():
        existing.append(path)
    else:
        print(f'Skipping nonexistent upstream include search directory: {path}', file=sys.stderr)
assert existing and any((p / 'real_client.h').is_file() for p in existing), 'Mooncake real_client.h missing from include closure'
print(';'.join(str(p) for p in dict.fromkeys(existing)))
PY
export MOONCAKE_INCLUDE_DIR="$(cat /opt/kv-storage/mooncake-include-paths.txt)"
# Upstream INSTALL_RPATH_USE_LINK_PATH propagates its build-only driver stub
# directory into this installed ELF. Repair it here to preserve the SDK cache.
python3 - /opt/mooncake-sdk/lib/libtransfer_engine.so <<'PY'
from pathlib import PurePosixPath
import subprocess
import sys

def runtime_paths_without_stubs(value):
    return ':'.join(p for p in value.split(':') if 'stubs' not in PurePosixPath(p).parts)

if __name__ == '__main__':
    library = sys.argv[1]
    before = subprocess.check_output(['patchelf', '--print-rpath', library], text=True).strip()
    after = runtime_paths_without_stubs(before)
    if before != after:
        subprocess.run(['patchelf', '--set-rpath', after, library], check=True)
    observed = subprocess.check_output(['patchelf', '--print-rpath', library], text=True).strip()
    assert observed == after and observed == runtime_paths_without_stubs(observed), 'runtime CUDA stub path remains'
    print(f'Installed transfer engine RPATH: {observed}')
PY
python3 -m pip install --no-cache-dir --no-deps -c /opt/kv-storage/constraints.txt grpcio-tools==1.81.1
# The pinned image packages CUDA component headers beside its Python wheels;
# nvcc's toolkit include directory alone does not contain cusparse/cublas.
cuda_component_include=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include
python3 - "$cuda_component_include" <<'PY' > /opt/kv-storage/cuda-component-include.txt
from pathlib import Path
import sys
root = Path(sys.argv[1])
required = ('cusparse.h', 'cublas_v2.h', 'cublasLt.h', 'cusolverDn.h', 'cuda_runtime.h')
missing = [name for name in required if not (root / name).is_file()]
if missing:
    raise RuntimeError(f"Missing pinned CUDA 13 component headers: {', '.join(missing)}")
print(root)
PY
CPATH="${cuda_component_include}${CPATH:+:${CPATH}}" \
    python3 -m pip wheel --no-build-isolation --no-deps -c /opt/kv-storage/constraints.txt /opt/LMCache -w /opt/kv-storage/wheels
python3 -m pip install --no-deps /opt/kv-storage/wheels/lmcache-*.whl
python3 -m pip freeze --all > /opt/kv-storage/python-lock.txt
sha256sum /opt/kv-storage/wheels/*.whl > /opt/kv-storage/wheel-sha256.txt
