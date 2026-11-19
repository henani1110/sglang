import concurrent.futures, hashlib, json, time, urllib.request
from pathlib import Path
import sys
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
base = 'http://127.0.0.1:30000'
def run(i):
    payload = {'text': 'Explain why request bursts affect language model streaming. '+str(i), 'sampling_params': {'temperature': 0, 'max_new_tokens': 128, 'ignore_eos': True}, 'stream': True}
    start=time.perf_counter(); events=[]; done=False
    req=urllib.request.Request(base+'/generate', data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=240) as r:
            for line in r:
                if not line.startswith(b'data:'): continue
                data=line[5:].strip()
                if data==b'[DONE]': done=True; break
                ev=json.loads(data); events.append({'elapsed_s':time.perf_counter()-start,'data':ev})
        counts=[e['data'].get('meta_info',{}).get('completion_tokens',0) for e in events]
        result={'id':i, 'payload':payload,'events':events,'done':done,'elapsed_s':time.perf_counter()-start,'completion_tokens':counts[-1] if counts else 0,'counts_monotonic':all(a<=b for a,b in zip(counts,counts[1:]))}
        result['ok']=done and result['completion_tokens']==128 and result['counts_monotonic']
    except Exception as e: result={'id':i,'error':repr(e),'ok':False}
    return result
single=[run(0)]
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool: concurrent=list(pool.map(run,range(1,9)))
results=single+concurrent
(out/'requests.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in results))
summary={'requests':len(results),'passed':sum(r['ok'] for r in results),'output_tokens':sum(r.get('completion_tokens',0) for r in results),'ttft_s':[next((e['elapsed_s'] for e in r.get('events',[]) if e['data'].get('meta_info',{}).get('completion_tokens',0)>0),None) for r in results]}
(out/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary))
if summary['passed']!=9: sys.exit(1)
