"""Archive only this experiment's synthetic JSON evidence, never service config."""
from pathlib import Path
import re
import sys
from zipfile import ZIP_DEFLATED, ZipFile

root = Path('/tmp/h100-fi-upgrade')
prefix = sys.argv[1]
if not re.fullmatch(r'(?:fi0618|native-fp16)-[a-z0-9-]+', prefix):
    raise ValueError('Unexpected experiment prefix')
paths = sorted(p for p in root.glob(prefix+'-*') if p.suffix in ('.json','.jsonl'))
if not paths:
    raise RuntimeError('Missing evidence')
archive = root/(prefix+'-evidence.zip')
with ZipFile(archive,'w',compression=ZIP_DEFLATED) as output:
    for path in paths:
        output.write(path,path.name)
print({'archive':str(archive),'files':len(paths)})
