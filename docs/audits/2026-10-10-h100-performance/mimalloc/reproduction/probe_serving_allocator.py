import json
import os
from pathlib import Path
import sys

expected = sys.argv[1]
assert expected in ('system','mimalloc')
records=[]
for directory in Path('/proc').iterdir():
    if not directory.name.isdigit() or int(directory.name)==os.getpid():
        continue
    try:
        cmd = (directory/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        comm = (directory/'comm').read_text().strip()
        role = 'api' if directory.name=='1' else ('engine' if 'EngineCore' in cmd or 'EngineCore' in comm else None)
        if role is None:
            continue
        env=dict(item.split('=',1) for item in (directory/'environ').read_bytes().decode(errors='replace').split('\0') if '=' in item)
        libraries=sorted({line.split()[-1] for line in (directory/'maps').read_text().splitlines()
                          if any(name in line for name in ('mimalloc','jemalloc','tcmalloc'))})
        status={key:value.strip() for line in (directory/'status').read_text().splitlines()
                if ':' in line for key,value in [line.split(':',1)] if key in ('VmRSS','VmHWM','Threads')}
        rollup={key:value.strip() for line in (directory/'smaps_rollup').read_text().splitlines()
                if ':' in line for key,value in [line.split(':',1)] if key in ('Rss','Pss','Private_Dirty')}
        # EngineCore calls setproctitle, which relocates/overwrites the original
        # environment region exposed by /proc. A subprocess reproduction proved
        # its os.environ stays intact while /proc/environ loses these keys.
        if role=='api':
            assert env.get('AQUILLM_ALLOCATOR')==expected
            assert env.get('PYTHONMALLOC')=='default'
        else:
            assert env.get('AQUILLM_ALLOCATOR') in (None,expected)
            assert env.get('PYTHONMALLOC') in (None,'default')
        assert bool(libraries)==(expected=='mimalloc')
        assert all('/opt/mimalloc/' in item for item in libraries)
        records.append(dict(pid=int(directory.name),role=role,configured_allocator=expected,
                            observed_env_allocator=env.get('AQUILLM_ALLOCATOR'),
                            observed_env_pythonmalloc=env.get('PYTHONMALLOC'),
                            process_environment_reliable=role=='api',
                            libraries=libraries,status=status,memory=rollup))
    except (FileNotFoundError,ProcessLookupError):
        continue
assert any(r['role']=='api' for r in records) and any(r['role']=='engine' for r in records)
print(json.dumps(records,indent=2))
