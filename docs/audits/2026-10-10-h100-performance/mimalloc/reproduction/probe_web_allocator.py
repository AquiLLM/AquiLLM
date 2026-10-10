"""Read-only serving Python mappings and process memory; no secret values."""
import json
import os
from pathlib import Path
import sys

expected=sys.argv[1]
assert expected in ('system','mimalloc')
records=[]
for directory in Path('/proc').iterdir():
    if not directory.name.isdigit() or int(directory.name)==os.getpid():
        continue
    try:
        executable=Path(os.readlink(directory/'exe')).name
        if 'python' not in executable.lower():
            continue
        role='web' if directory.name=='1' else 'python_worker'
        env=dict(item.split('=',1) for item in (directory/'environ').read_bytes().decode(errors='replace').split('\0') if '=' in item)
        libraries=sorted({line.split()[-1] for line in (directory/'maps').read_text().splitlines()
                          if any(name in line for name in ('mimalloc','jemalloc','tcmalloc'))})
        assert bool(libraries)==(expected=='mimalloc')
        assert all('/opt/mimalloc/' in library for library in libraries)
        if role=='web':
            assert env.get('AQUILLM_ALLOCATOR')==expected and env.get('PYTHONMALLOC')=='default'
        else:
            assert env.get('AQUILLM_ALLOCATOR') in (None,expected)
            assert env.get('PYTHONMALLOC') in (None,'default')
        status={key:value.strip() for line in (directory/'status').read_text().splitlines()
                if ':' in line for key,value in [line.split(':',1)] if key in ('VmRSS','VmHWM','Threads')}
        memory={key:value.strip() for line in (directory/'smaps_rollup').read_text().splitlines()
                if ':' in line for key,value in [line.split(':',1)] if key in ('Rss','Pss','Private_Dirty')}
        records.append(dict(pid=int(directory.name),role=role,configured_allocator=expected,
                            observed_env_allocator=env.get('AQUILLM_ALLOCATOR'),
                            observed_env_pythonmalloc=env.get('PYTHONMALLOC'),libraries=libraries,
                            status=status,memory=memory))
    except (FileNotFoundError,ProcessLookupError):
        continue
assert any(row['role']=='web' for row in records)
print(json.dumps(records,indent=2))
