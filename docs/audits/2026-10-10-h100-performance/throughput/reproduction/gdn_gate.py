"""Execute actual installed post-Genesis GDN in an isolated test process."""
import sys
import os
import json
import subprocess
worker = '--worker' in sys.argv
model = os.environ.get('VLLM_MODEL', os.environ.get('AQUILLM_H100_MODEL_ID'))
assert model == 'hampsonw/Qwen3.6-27B-AWQ-BF16-INT4-mtp-bf16'
sys.argv = [sys.argv[0], '--model', model, '--speculative-config',
            '{"method":"mtp","num_speculative_tokens":4}']
import sndr.plugin
sndr.plugin.register()
if not worker:
    # Genesis applies source patches; a fresh interpreter sees rewritten aliases.
    raise SystemExit(subprocess.call([sys.executable,__file__,'--worker']))
import inspect
import hashlib
from pathlib import Path
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as module
from aquillm_vllm_h100.gdn_mixed import install_gdn_mixed
assert getattr(module.QwenGatedDeltaNetAttention._forward_core,'_aquillm_h100_gdn_mixed',None)
assert 'num_accepted_tokens' in inspect.signature(module.causal_conv1d_fn).parameters
print(json.dumps({'causal_conv1d_signature':str(inspect.signature(module.causal_conv1d_fn)),
                  'causal_conv1d_file_sha256':hashlib.sha256(Path(inspect.getsourcefile(module.causal_conv1d_fn)).read_bytes()).hexdigest()}),flush=True)
import pytest
root = '/workspace/deploy/vllm_plugins/h100_kernels'
raise SystemExit(pytest.main(['-c',root+'/pyproject.toml',
    root+'/tests/gpu/test_gdn_mixed_gpu.py','-q','-x',
    '-o','cache_dir=/tmp/pytest-cache']))
