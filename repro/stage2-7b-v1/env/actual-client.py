"""Explicit workload generation and native SSE replay. Never count warmup as measurement."""
import argparse, asyncio, hashlib, json, random, time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('action',choices=['make','run']);p.add_argument('--file',required=True);p.add_argument('--out');p.add_argument('--model',default='/home/chx/models/Qwen3-0.6B');p.add_argument('--seed',type=int,default=42);p.add_argument('--count',type=int,default=16);p.add_argument('--input',type=int,default=512);p.add_argument('--output',type=int,default=1024);p.add_argument('--rate',type=float,default=0);p.add_argument('--speed',type=float,default=20);args=p.parse_args();f=Path(args.file)
if args.action=='make':
 import numpy as np
 from transformers import AutoTokenizer
 from sglang.benchmark.datasets.random import sample_random_requests
 random.seed(args.seed);np.random.seed(args.seed)
 tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
 rows=sample_random_requests(args.input,args.output,args.count,0.2,tok,'',return_text=False)
 rng=random.Random(args.seed+100000);offset=0.;requests=[]
 for i,r in enumerate(rows):
  if i and args.rate: offset+=rng.expovariate(args.rate)
  requests.append({'id':i,'arrival_s':offset,'input_ids':r.prompt,'max_new_tokens':int(r.output_len)})
 f.parent.mkdir(parents=True,exist_ok=True);f.write_text(json.dumps({'seed':args.seed,'model':args.model,'range_ratio':0.2,'input_cap':args.input,'output_cap':args.output,'rate':args.rate,'speed':args.speed,'requests':requests},ensure_ascii=False))
 print(json.dumps({'file':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'requests':len(requests)}));raise SystemExit
import aiohttp
work=json.loads(f.read_text());out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
async def main():
 begin=time.perf_counter(); wall_begin=time.time(); tag=out.name
 async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0),timeout=aiohttp.ClientTimeout(total=600),read_bufsize=10*1024**2) as session:
  async def one(r):
   await asyncio.sleep(max(0,begin+r['arrival_s']-time.perf_counter()));send=time.perf_counter();events=[];done=False
   payload={'input_ids':r['input_ids'],'sampling_params':{'temperature':0,'ignore_eos':True,'max_new_tokens':r['max_new_tokens'],'output_speed':work['speed']},'stream':True,'rid':tag+'-'+str(r['id'])}
   try:
    async with session.post('http://127.0.0.1:30000/generate',json=payload) as resp:
     resp.raise_for_status()
     async for line in resp.content:
      if not line.startswith(b'data:'):continue
      b=line[5:].strip()
      if b==b'[DONE]':done=True;break
      received=time.perf_counter()-begin;ev=json.loads(b);events.append({'time_s':received,'data':ev})
    counts=[e['data'].get('meta_info',{}).get('completion_tokens',0) for e in events]
    ok=done and bool(counts) and counts[-1]==r['max_new_tokens'] and all(a<=b for a,b in zip(counts,counts[1:]))
    return {'id':r['id'],'arrival_s':r['arrival_s'],'send_s':send-begin,'end_s':time.perf_counter()-begin,'requested_output':r['max_new_tokens'],'done':done,'ok':ok,'events':events}
   except Exception as e:return {'id':r['id'],'arrival_s':r['arrival_s'],'send_s':send-begin,'error':repr(e),'ok':False,'events':events}
  results=await asyncio.gather(*(one(r) for r in work['requests']))
 (out/'requests.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in results))
 summary={'start_unix_s':wall_begin,'workload_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'requests':len(results),'passed':sum(r['ok'] for r in results),'wall_s':time.perf_counter()-begin}
 (out/'run.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
 if summary['passed']!=summary['requests']:raise SystemExit(1)
asyncio.run(main())
