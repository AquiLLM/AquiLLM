import hashlib
import importlib.metadata as md
import inspect
import json
import os
import sys
from pathlib import Path
import subprocess

out = Path('/evidence')
prefix = os.environ.get('AQUILLM_EVIDENCE_PREFIX','')
import sndr.plugin
sndr.plugin.register()
if os.environ.get('AQUILLM_PROBE_PATCHED') != '1':
    # Genesis patches disk while importing some pre-patch modules in the API
    # process. A fresh worker sees the patched sources, as in vLLM serving.
    raise SystemExit(subprocess.call([sys.executable, __file__],
                     env={**os.environ, 'AQUILLM_PROBE_PATCHED': '1'}))
import torch
import cutlass
from flashinfer import gdn_decode
from aquillm_vllm_h100.compatibility import verify_runtime
verify_runtime()
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as qwen
from vllm.v1.attention.backends.turboquant_attn import TurboQuantAttentionImpl

functions = {'gdn_mtp': gdn_decode.gated_delta_rule_mtp,
             'vllm_fused_gating': qwen.fused_sigmoid_gating_delta_rule_update,
             'turboquant_prefill': TurboQuantAttentionImpl._prefill_attention}
sources = {}
for name, function in functions.items():
    source = inspect.getsource(inspect.unwrap(function))
    (out/(prefix+name+'.py')).write_text(source)
    sources[name] = {'module': function.__module__, 'signature': str(inspect.signature(function)),
                     'sha256': hashlib.sha256(source.encode()).hexdigest()}
record = {'packages': {n: md.version(n) for n in ['vllm','torch','triton','flashinfer-python','flashinfer-cubin','flashinfer-jit-cache','nvidia-cutlass-dsl','apache-tvm-ffi']},
          'gpu': torch.cuda.get_device_name(), 'sources': sources}
(out/(prefix+'gpu-imports.json')).write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record, indent=2), flush=True)
if os.environ.get('AQUILLM_IMPORT_PROBE_ONLY') == '1':
    raise SystemExit(0)
command = ['python3', '-m', 'pytest', '-c', '/workspace/deploy/vllm_plugins/h100_kernels/pyproject.toml',
           '/workspace/deploy/vllm_plugins/h100_kernels/tests/gpu/test_actual_store_roundtrip.py',
           '/workspace/deploy/vllm_plugins/h100_kernels/tests/gpu/test_mtp_fused.py',
           '/workspace/deploy/vllm_plugins/h100_kernels/tests/gpu/test_verifier_adapter.py',
           '/workspace/deploy/vllm_plugins/h100_kernels/tests/gpu/test_sampler_distribution.py', '-q']
with (out/(prefix+'existing-gpu-tests.log')).open('w') as log:
    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
print((out/(prefix+'existing-gpu-tests.log')).read_text()[-10000:], flush=True)
raise SystemExit(result.returncode)
