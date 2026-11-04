"""Archive only named synthetic evidence artifacts, never recovery or environment."""
import argparse
from pathlib import Path
import re
import tarfile

parser=argparse.ArgumentParser()
parser.add_argument('--prefix',required=True)
args=parser.parse_args()
assert re.fullmatch('[a-z0-9-]+',args.prefix)
root=Path('/tmp/h100-throughput')
assert (root/(args.prefix+'-restored.json')).exists(), 'Run has not restored baseline'
files=sorted(p for p in root.glob(args.prefix+'-*') if p.is_file() and p.suffix in ('.json','.jsonl','.log'))
target=root/(args.prefix+'.tar.gz')
assert not target.exists()
with tarfile.open(target,'w:gz') as archive:
    for path in files: archive.add(path,arcname=path.name)
print({'files':len(files),'compressed_bytes':target.stat().st_size})
