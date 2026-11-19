import csv,json,statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path('/home/chx/projects/tokenflow-repro-paper-candidate/repro/stage2-7b-v2'); colors={'baseline':'#2166ac','tokenflow':'#d6604d'}
excluded=set(json.loads((ROOT/'env/operator-overlap-retakes.json').read_text())['excluded_original_runs'])
runtime=[]
reference=json.loads((ROOT/'env/common-normalized-reference.json').read_text())
for group,mode in [(1,'baseline'),(2,'tokenflow'),(3,'baseline'),(4,'tokenflow'),(5,'baseline')]:
 info=json.loads((ROOT/'env'/f'group{group}-{mode}-server-info.json').read_text())
 assert {k:info.get(k) for k in reference}==reference,(group,'common runtime mismatch')
 assert info['enable_hierarchical_cache']==(mode=='tokenflow')
 assert info['kv_offload_policy']==('paper_v1' if mode=='tokenflow' else 'default')
 if mode=='tokenflow':assert info['hicache_size']==8
 runtime.append({'group':group,'mode':mode,'actual_common_settings_equal':True,'feature_switches_verified':True})
(ROOT/'env/runtime-settings-audit.json').write_text(json.dumps(runtime,indent=2))
rows=[];request_cache={};output_sequences={}
def read_requests(name):
 if name not in request_cache:
  request_cache.clear()
  request_cache[name]=[json.loads(s) for s in (ROOT/'raw'/name/'requests.jsonl').read_text().splitlines()]
 return request_cache[name]
for f in sorted((ROOT/'raw').glob('*/analysis.json')):
 if f.parent.name in excluded:continue
 row=json.loads(f.read_text())[0];req=read_requests(row['run']);assert [r['id'] for r in req]==list(range(len(req)));output_sequences[row['run']]=[r['events'][-1]['data']['output_ids'] for r in req];start=min(r['send_s'] for r in req);end=max(r['end_s'] for r in req);anchor=row['start_unix_s']
 resource=ROOT/'raw'/f"group{row['server_group']}-{row['mode']}-resources.csv"
 samples=list(csv.DictReader(resource.open()));selected=[r for r in samples if anchor+start<=float(r['unix_s'])<=anchor+end]
 if selected:
  for col in ['gpu1_mib','host_used_gib','gpu1_temp_c','cpu_percent']:
   row[col+'_peak']=max(float(r[col]) for r in selected)
  row['gpu1_util_mean']=statistics.mean(float(r['gpu1_util']) for r in selected)
  row['gpu1_sm_mhz_min']=min(float(r['gpu1_sm_mhz']) for r in selected)
 row['retractions_sum']=sum(r['events'][-1]['data']['meta_info'].get('total_retractions',0) for r in req)
 row['client_arrival_delay_max_s']=max(r['send_s']-r['arrival_s'] for r in req)
 # Keep cache-report fields verbatim in requests; aggregate only unambiguous host/cpu keys.
 details=[r['events'][-1]['data']['meta_info'].get('cached_tokens_details') for r in req]
 row['cache_report_request_count']=sum(isinstance(d,dict) for d in details)
 row['reported_host_cached_tokens']=sum(sum(v for k,v in d.items() if k.lower() in ['host','cpu'] and isinstance(v,(int,float))) for d in details if isinstance(d,dict)) if any(isinstance(d,dict) and any(k.lower() in ['host','cpu'] for k in d) for d in details) else None
 per=json.loads((f.parent/'analysis-requests.json').read_text());q=sorted(per,key=lambda r:r['id']);n=max(1,len(q)//4)
 row['first_quarter_ttft_mean_s']=statistics.mean(r['ttft_s'] for r in q[:n]);row['last_quarter_ttft_mean_s']=statistics.mean(r['ttft_s'] for r in q[-n:]);rows.append(row)
assert len(rows)==24 and sum(r['completed'] for r in rows)==2280
(ROOT/'summary-selected.json').write_text(json.dumps(rows,indent=2))
(ROOT/'summary-enriched.json').write_text(json.dumps(rows,indent=2))
fields=list(dict.fromkeys(k for r in rows for k in r))
with (ROOT/'summary-enriched.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
cases=[c for c in ['E1','E2','P2','P4'] if any(r['case']==c for r in rows)];pairs=[]
for case in cases:
 for seed in sorted({r['seed'] for r in rows if r['case']==case}):
  match={r['mode']:r for r in rows if r['case']==case and r['seed']==seed}
  if set(match)!={'baseline','tokenflow'}:continue
  b,t=match['baseline'],match['tokenflow'];assert b['workload_sha256']==t['workload_sha256'];assert b['output_tokens']==t['output_tokens']
  mismatch=[i for i,(x,y) in enumerate(zip(output_sequences[b['run']],output_sequences[t['run']])) if x!=y]
  pairs.append({'case':case,'seed':seed,'token_mismatch_count':len(mismatch),'mismatched_request_ids':mismatch,'ttft_mean_reduction_percent':100*(1-t['ttft_mean_s']/b['ttft_mean_s']),'ttft_p99_reduction_percent':100*(1-t['ttft_p99_s']/b['ttft_p99_s']),'raw_throughput_change_percent':100*(t['raw_output_tok_s']/b['raw_output_tok_s']-1),'effective_throughput_change_percent':100*(t['effective_tok_s']/b['effective_tok_s']-1),'stall_fraction_change':t['stall_fraction']-b['stall_fraction']})
(ROOT/'paired-changes.json').write_text(json.dumps(pairs,indent=2))
fig,axes=plt.subplots(2,2,figsize=(10,7),squeeze=False)
for ax,case in zip(axes.flat,cases):
 for mode in ['baseline','tokenflow']:
  for i,r in enumerate([r for r in rows if r['case']==case and r['mode']==mode]):
   vals=sorted(x['ttft_s'] for x in json.loads((ROOT/'raw'/r['run']/'analysis-requests.json').read_text()));ax.plot(vals,np.arange(1,len(vals)+1)/len(vals),color=colors[mode],alpha=.65,label=mode if i==0 else None)
 ax.set(title=case,xlabel='Client TTFT to nonempty text (s)',ylabel='Request CDF');ax.grid(alpha=.2);ax.legend()
fig.suptitle('Each curve is one run; three paired seeds per scenario');fig.tight_layout();fig.savefig(ROOT/'plots/ttft-cdf.png',dpi=180);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11,4));x=np.arange(len(cases));width=.34
for ax,field,title in zip(axes,['raw_output_tok_s','effective_tok_s'],['Raw output throughput','Consumption-weighted throughput']):
 for j,mode in enumerate(['baseline','tokenflow']):
  groups=[[r[field] for r in rows if r['case']==c and r['mode']==mode] for c in cases];means=[statistics.mean(g) for g in groups];errs=np.array([[m-min(g) for m,g in zip(means,groups)],[max(g)-m for m,g in zip(means,groups)]])
  ax.bar(x+(j-.5)*width,means,width,yerr=errs,capsize=3,color=colors[mode],label=mode)
 ax.set_xticks(x,cases);ax.set(title=title,ylabel='Output tokens / measurement second');ax.grid(axis='y',alpha=.2);ax.legend()
fig.suptitle('Mean and full range across paired runs; 20 token/s consumption');fig.tight_layout();fig.savefig(ROOT/'plots/throughput.png',dpi=180);plt.close(fig)
# Trace one identical request from a representative burst, without pooling bursts.
case='E2' if 'E2' in cases else cases[0];br=next(r for r in rows if r['case']==case and r['mode']=='baseline');tr=next(r for r in rows if r['case']==case and r['mode']=='tokenflow' and r['seed']==br['seed']);bp=json.loads((ROOT/'raw'/br['run']/'analysis-requests.json').read_text());rid=max(bp,key=lambda r:r['ttft_s'])['id'];fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True)
for row in [br,tr]:
 req=next(r for r in read_requests(row['run']) if r['id']==rid);times=[];old=0
 for e in req['events']:
  n=e['data']['meta_info'].get('completion_tokens',0);times.extend([e['time_s']]*(n-old));old=n
 buf=0.;prev=times[0];buffers=[];stalls=[]
 for t in times:
  dt=t-prev
  if dt>buf/20:stalls.append((prev+buf/20,t))
  buf=max(buf-20*dt,0)+1;buffers.append(buf);prev=t
 c=colors[row['mode']];axes[0].step(times,np.arange(1,len(times)+1),where='post',color=c,label=row['mode']);axes[0].plot([times[0],times[-1]],[1,1+20*(times[-1]-times[0])],ls='--',color=c,alpha=.5);axes[1].plot(times,buffers,color=c,label=row['mode'])
 for lo,hi in stalls:
  if hi>lo:axes[1].axvspan(lo,hi,color=c,alpha=.12)
axes[0].set(ylabel='Generated tokens',title=f'{case}, seed {br["seed"]}, identical request {rid}; dashed: 20 token/s demand');axes[1].set(ylabel='Unconsumed token buffer',xlabel='Time since measurement start (s)')
for ax in axes:ax.grid(alpha=.2);ax.legend()
fig.tight_layout();fig.savefig(ROOT/'plots/request-buffer.png',dpi=180);plt.close(fig)
lines=['# 7B阶段二v2结果（实际公共参数已核对）','','数值为每轮指标在配对重复中的均值；P99先逐轮计算，不拼接为一次大突发。','', '|场景|系统|轮数|每轮请求数|平均TTFT(s)|每轮P99均值(s)|原始tok/s|加权有效tok/s|断流比例|','|---|---|---:|---:|---:|---:|---:|---:|---:|']
for case in cases:
 for mode in ['baseline','tokenflow']:
  group=[r for r in rows if r['case']==case and r['mode']==mode];mean=lambda k:statistics.mean(r[k] for r in group)
  lines.append(f'|{case}|{mode}|{len(group)}|{group[0]["requested"]}|{mean("ttft_mean_s"):.3f}|{mean("ttft_p99_s"):.3f}|{mean("raw_output_tok_s"):.1f}|{mean("effective_tok_s"):.1f}|{mean("stall_fraction"):.3%}|')
lines+=['','指标：TTFT为请求实际发送至首个非空文本SSE；另存token计数TTFT。消费从首个正token计数开始，以20token/s模拟；断流比例分母为各请求首末生成token之间时间之和。有效吞吐用10%/20%响应长度缓冲阈值作分段线性token计权，再除以相同测量窗口；这是事件精度重实现，不等同作者离散时间图中的valid_throughput。','', '两系统实际确定性均开启、piecewise CUDA graph共同关闭；9个短请求跨系统逐token一致。两轮与归档重叠的原始测量保留但排除，使用预先标记的同负载-clean重测替代，主统计仍为24轮/2280请求。逐请求时间戳、完整补丁和资源样本均保留。该结果不能归为未经适配作者artifact的原图复现。']
(ROOT/'结果摘要.md').write_text('\n'.join(lines)+'\n');print(json.dumps({'runs':len(rows),'pairs':len(pairs),'cases':cases,'plots':3}))
