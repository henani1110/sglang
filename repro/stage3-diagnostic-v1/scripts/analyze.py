import argparse,json,math,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('runs',nargs='+');p.add_argument('--speed',type=float,default=20);p.add_argument('--out',required=True);a=p.parse_args()
def percentile(xs,q):
 xs=sorted(xs); k=(len(xs)-1)*q; lo=int(k); return xs[lo]*(1-(k-lo))+xs[min(lo+1,len(xs)-1)]*(k-lo)
rows=[]
for name in a.runs:
 root=Path(name);reqs=[json.loads(x) for x in (root/'requests.jsonl').read_text().splitlines()];ok=[r for r in reqs if r['ok']];ttft=[];token_ttft=[];tokens=0;effective=0.;stall=0.;active=0.;latencies=[];per_req=[]
 for r in ok:
  ts=[];old=0
  for e in r['events']:
   count=e['data'].get('meta_info',{}).get('completion_tokens',0)
   ts.extend([e['time_s']]*(count-old));old=count
  if not ts: continue
  visible=next((e['time_s'] for e in r['events'] if e['data'].get('text','')!=''),ts[0])
  ttft.append(visible-r['send_s']);token_ttft.append(ts[0]-r['send_s']);latencies.append(r['end_s']-r['send_s']);tokens+=len(ts);buf=0.;prev=ts[0];st=0.;eff=0.
  for t in ts:
   delta=t-prev;st+=max(delta-buf/a.speed,0);buf=max(buf-delta*a.speed,0)
   # Weight the newly arrived token using the buffer including that token.
   buf+=1;fraction=buf/len(ts);eff+=1 if fraction<=.1 else max((.2-fraction)/.1,0);prev=t
  stall+=st;active+=ts[-1]-ts[0];effective+=eff
  per_req.append({'id':r['id'],'ttft_s':visible-r['send_s'],'token_ttft_s':ts[0]-r['send_s'],'tokens':len(ts),'stall_s':st,'generation_window_s':ts[-1]-ts[0],'effective_tokens':eff,'max_client_arrival_delay_s':r['send_s']-r['arrival_s']})
 duration=max(r['end_s'] for r in ok)-min(r['send_s'] for r in ok) if ok else 0
 row={'run':root.name,'requested':len(reqs),'completed':len(ok),'duration_s':duration,'output_tokens':tokens,'raw_output_tok_s':tokens/duration if duration else None,'effective_tok_s':effective/duration if duration else None,'token_ttft_mean_s':statistics.mean(token_ttft),'ttft_mean_s':statistics.mean(ttft) if ttft else None,'ttft_p95_s':percentile(ttft,.95) if ttft else None,'ttft_p99_s':percentile(ttft,.99) if ttft else None,'stall_total_s':stall,'active_request_total_s':active,'stall_fraction':stall/active if active else 0,'e2e_mean_s':statistics.mean(latencies) if latencies else None,'workload_sha256':json.loads((root/'run.json').read_text()).get('workload_sha256')}
 row['start_unix_s']=json.loads((root/'run.json').read_text()).get('start_unix_s')
 (root/'analysis-requests.json').write_text(json.dumps(per_req,indent=2));rows.append(row)
Path(a.out).write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
