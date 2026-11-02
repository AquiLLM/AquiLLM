"""Read-only five-second NVML/container sampling, with no model requests."""
import json
from pathlib import Path
import socket
import subprocess
import sys
import time

assert socket.gethostname() == 'aquillm-dev2'
out = Path('/tmp/h100-fi-upgrade')
prefix = sys.argv[1]
path = out/(prefix+'-memory.jsonl')
deadline = time.monotonic()+3600
with path.open('x') as capture:
    while time.monotonic() < deadline:
        restored = out/(prefix+'-restored.json')
        if restored.exists():
            break
        try:
            info = json.loads(subprocess.check_output(['docker','inspect','compose-vllm-1'],text=True))[0]
            top = subprocess.check_output(['docker','top','compose-vllm-1','-eo','pid'],text=True)
            pids = {int(line.strip()) for line in top.splitlines()[1:] if line.strip().isdigit()}
            raw = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
            used = sum(int(memory.strip()) for line in raw.splitlines() for pid,memory in [line.split(',')]
                       if int(pid.strip()) in pids and memory.strip().isdigit())
            gpu = subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],text=True)
            row = dict(timestamp=time.time(),image=info['Image'],healthy=info['State'].get('Health',{}).get('Status')=='healthy',
                model_process_gpu_mib=used,gpu_memory_mib=[list(map(int,line.split(','))) for line in gpu.splitlines()],
                sampling_interval_seconds=5)
        except (subprocess.CalledProcessError,ValueError,KeyError) as error:
            row = dict(timestamp=time.time(),sample_error=type(error).__name__)
        capture.write(json.dumps(row)+'\n');capture.flush()
        time.sleep(5)
print(json.dumps({'memory_capture':str(path),'restored_observed':restored.exists()}))
