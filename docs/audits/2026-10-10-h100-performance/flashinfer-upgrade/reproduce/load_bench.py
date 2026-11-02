"""Fixed synthetic queued-load comparison; preserves deployed max-num-seqs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import statistics
import time

from serve_bench import request, stream_request, speculation_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--label',required=True)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    base, model = 'http://127.0.0.1:8000','qwen3.6:27b-mtp-awq'
    with request(base,'/tokenize',{'model':model,'prompt':'A synthetic performance note: describe the numbered project stages and their order. ','add_special_tokens':False}) as response:
        seed = json.load(response)['tokens']
    with args.output.open('w') as out:
        for concurrency in (2,4):
            payloads = []
            for i in range(concurrency*2):
                length = (512,8192,36864,512)[i%4]
                payloads.append(dict(model=model,prompt=(seed*(length//len(seed)+1))[:length],max_tokens=256,
                    temperature=0,seed=17,ignore_eos=True,stream=True,stream_options={'include_usage':True}))
            before = speculation_metrics(base)
            started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                results = list(executor.map(lambda x:stream_request(base,x),payloads))
            seconds = time.perf_counter()-started
            complete = all(r['complete'] and r['output_tokens']==256 for r in results)
            row = dict(label=args.label,concurrency=concurrency,requests=len(results),complete=complete,
                error=None if complete else 'invalid_request',seconds=seconds,
                output_tokens_per_second=sum(r['output_tokens'] or 0 for r in results)/seconds if complete else None,
                median_ttft_seconds=statistics.median(r['ttft_seconds'] for r in results) if complete else None,
                input_sha256=hashlib.sha256(json.dumps(payloads,sort_keys=True).encode()).hexdigest(),
                before=before,after=speculation_metrics(base),results=results)
            out.write(json.dumps(row)+'\n'); out.flush()
            print(json.dumps({k:v for k,v in row.items() if k not in ('results','before','after')}),flush=True)
            if not complete: raise SystemExit(1)


if __name__=='__main__':main()
