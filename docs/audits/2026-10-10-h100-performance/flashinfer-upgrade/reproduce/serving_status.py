"""Print safe experiment status without configuration or request contents."""
import json
from pathlib import Path
import subprocess
import sys

prefix = sys.argv[1]
out = Path('/tmp/h100-fi-upgrade')
info = json.loads(subprocess.check_output(['docker','inspect','compose-vllm-1'],text=True))[0]
result = dict(image=info['Image'],healthy=info['State'].get('Health',{}).get('Status'),
              running=info['State']['Running'],started_at=info['State']['StartedAt'],
              restarts=info['RestartCount'])
result['evidence'] = [p.name for p in sorted(out.glob(prefix+'-*.jsonl')) if not p.name.endswith('-memory.jsonl')]
log = subprocess.check_output(['docker','logs','--tail','200','compose-vllm-1'],text=True,stderr=subprocess.STDOUT)
result['routes'] = [line for line in log.splitlines() if 'AQUILLM_H100 route_exercised ' in line]
result['errors'] = [line for line in log.splitlines() if any(token in line for token in ('ERROR','SystemExit','Error:'))][-10:]
print(json.dumps(result,indent=2))
