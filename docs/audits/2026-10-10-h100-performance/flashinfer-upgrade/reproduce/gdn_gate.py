import os
from pathlib import Path
import subprocess
import sys

import sndr.plugin
sndr.plugin.register()
root = '/workspace/deploy/vllm_plugins/h100_kernels'
if '--ceiling' in sys.argv:
    command = [sys.executable, '/evidence/raw_gdn_ceiling.py','--include-fp16-api','--include-pack-overhead']
elif '--benchmark' in sys.argv:
    command = [sys.executable, root+'/benchmarks/gdn.py','--benchmark','--repeats','200',
               '--gate-stride','96' if '--stride96' in sys.argv else '48']
else:
    command = [sys.executable,'-m','pytest','-c',root+'/pyproject.toml',
               root+'/tests/gpu/test_gdn_adapter.py','-q','-s','-x','-o','cache_dir=/tmp/pytest-cache']
    if '--baseline-stack' in sys.argv or '--native-image' in sys.argv:
        command += ['-k', 'not real_alias_installation']
raise SystemExit(subprocess.call(command))
