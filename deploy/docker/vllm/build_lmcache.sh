#!/bin/bash
set -euo pipefail
lm_ref=05a013b29da78cf2321b9b46ec5039dde2fb0bb0
git init /opt/LMCache
git -C /opt/LMCache remote add origin https://github.com/LMCache/LMCache.git
git -C /opt/LMCache fetch --depth 1 origin "$lm_ref"
test "$(git -C /opt/LMCache rev-parse FETCH_HEAD)" = "$lm_ref"
git -C /opt/LMCache checkout --detach FETCH_HEAD
git -C /opt/LMCache submodule update --init --recursive --depth 1
# Upstream installs libraries but not a complete public header SDK. Keep the
# pinned source tree and reuse the real client's actual CMake include closure.
python3 - <<'PY' > /opt/kv-storage/mooncake-include-paths.txt
import json
from pathlib import Path
import shlex
commands = json.loads(Path('/opt/Mooncake/build/compile_commands.json').read_text())
entry = next(c for c in commands if c['file'].endswith('/real_client.cpp'))
tokens = iter(shlex.split(entry['command']))
includes = []
for token in tokens:
    if token in ('-I', '-isystem'):
        includes.append(next(tokens))
    elif token.startswith('-I'):
        includes.append(token[2:])
assert includes and all(Path(p).is_dir() for p in includes), 'incomplete Mooncake include closure'
print(';'.join(dict.fromkeys(includes)))
PY
export MOONCAKE_INCLUDE_DIR="$(cat /opt/kv-storage/mooncake-include-paths.txt)"
python3 -m pip install --no-cache-dir --no-deps -c /opt/kv-storage/constraints.txt grpcio-tools==1.81.1
python3 -m pip wheel --no-build-isolation --no-deps -c /opt/kv-storage/constraints.txt /opt/LMCache -w /opt/kv-storage/wheels
python3 -m pip install --no-deps /opt/kv-storage/wheels/lmcache-*.whl
python3 -m pip freeze --all > /opt/kv-storage/python-lock.txt
sha256sum /opt/kv-storage/wheels/*.whl > /opt/kv-storage/wheel-sha256.txt
