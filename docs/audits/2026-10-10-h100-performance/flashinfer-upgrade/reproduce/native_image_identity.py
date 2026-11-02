import json
from pathlib import Path
import socket
import subprocess

assert socket.gethostname() == 'aquillm-dev2'
out = Path('/tmp/h100-fi-upgrade')
identity = json.loads((out/'build.json').read_text())
assert identity['recipe'] == 'Dockerfile.native-gdn-baseline'
base, candidate = [json.loads(subprocess.check_output(
    ['docker','image','inspect',identity[k]],text=True))[0] for k in ('base','image')]
keys = ('Cmd','Entrypoint','Env','User','WorkingDir','Healthcheck','ExposedPorts')
config = {k:base['Config'].get(k) == candidate['Config'].get(k) for k in keys}
assert all(config.values()), config
code = ('import json,pathlib; p=pathlib.Path("/opt/native-gdn-baseline-evidence"); '
        'print(json.dumps({k:json.loads((p/(k+".json")).read_text()) for k in ("before","after")}))')
manifests = json.loads(subprocess.check_output(
    ['docker','run','--rm','--entrypoint','python3',identity['image'],'-c',code],text=True))
record = dict(identity=identity,image_configuration_unchanged=config,manifests=manifests)
(out/'native-baseline-image-identity.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'image':identity['image'],'configuration_unchanged':all(config.values()),
                  'packages_before':len(manifests['before']['packages']),
                  'packages_after':len(manifests['after']['packages']),
                  'plugins':manifests['after']['plugins']}))
