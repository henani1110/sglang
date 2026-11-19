import argparse, csv, datetime, hashlib, json, os, signal, socket, subprocess, sys, threading, time, urllib.request
from pathlib import Path
from zoneinfo import ZoneInfo
ROOT=Path('/home/chx/projects/tokenflow-repro-paper-candidate'); os.chdir(ROOT)
PYTHON=str(ROOT/'.venv/bin/python'); BASE=ROOT/'repro/stage2-7b-v1'; MODEL=ROOT/'repro/models/Qwen2-7B-Instruct'
CASES={'E1':(60,512,1024,0),'E2':(80,1024,2048,0),'P2':(120,512,1024,2),'P4':(120,512,1024,4)}
p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('--cases',nargs='+',default=list(CASES));p.add_argument('--seeds',nargs='+',type=int,default=[101,102,103]);a=p.parse_args()
for d in ['workloads','raw','logs','env','plots']: (BASE/d).mkdir(parents=True,exist_ok=True)
def event(kind,**kw):
 row={'time':datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'event':kind,**kw}
 with (BASE/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 print(json.dumps(row),flush=True)
def digest(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def prepare():
 import numpy as np, random
 from transformers import AutoTokenizer
 from sglang.benchmark.datasets.random import sample_random_requests
 tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
 for case,(n,inp,out,rate) in {**CASES,'warm':(60,512,1024,0)}.items():
  for seed in ([999] if case=='warm' else a.seeds):
   target=BASE/'workloads'/f'{case}-s{seed}.json'
   if target.exists():continue
   random.seed(seed);np.random.seed(seed)
   rows=sample_random_requests(inp,out,n,.2,tok,'',return_text=False);rng=random.Random(seed+100000);offset=0.;req=[]
   for i,r in enumerate(rows):
    if i and rate:offset+=rng.expovariate(rate)
    req.append({'id':i,'arrival_s':offset,'input_ids':r.prompt,'max_new_tokens':int(r.output_len)})
   target.write_text(json.dumps({'seed':seed,'model':str(MODEL),'input_cap':inp,'output_cap':out,'range_ratio':.2,'rate':rate,'speed':20,'requests':req}))
   event('workload_saved',case=case,seed=seed,sha256=digest(target),input_tokens=sum(len(r['input_ids']) for r in req),output_tokens=sum(r['max_new_tokens'] for r in req))
prepare()
if a.action=='prepare':raise SystemExit
manifest=json.loads((ROOT/'repro/env/model-7b-manifest.json').read_text())
event('waiting_for_model')
wait_start=time.monotonic()
while any(not (MODEL/f['path']).exists() or (MODEL/(f['path']+'.aria2')).exists() or (MODEL/f['path']).stat().st_size!=f['size'] for f in manifest['files']):
 if time.monotonic()-wait_start>3600:raise RuntimeError('model download incomplete after one hour')
 time.sleep(10)
for f in manifest['files']:
 actual=digest(MODEL/f['path']);assert actual==f['sha256'],f['path'];event('model_file_verified',file=f['path'],sha256=actual)
(ROOT/'repro/env/model-7b.json').write_text(json.dumps({'repo_id':manifest['repo_id'],'revision':manifest['revision'],'path':str(MODEL),'weights_sha256_verified':True},indent=2))
patch=subprocess.check_output(['git','diff']);identity={'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'patch_sha256':hashlib.sha256(patch).hexdigest(),'launcher_sha256':digest(ROOT/'repro/scripts/launch.sh'),'client_sha256':digest(ROOT/'repro/scripts/replay.py'),'analyzer_sha256':digest(ROOT/'repro/scripts/analyze.py'),'freeze_sha256':digest(ROOT/'repro/env/freeze.txt'),'model_revision':manifest['revision'],'dtype':'bfloat16','attention_backend':'triton','sampling_backend':'pytorch','gpu':1,'mem_fraction_static':.8,'max_running_requests':128,'deterministic':True,'cpu_kv_decimal_gb':8,'sync_cache':True,'write_producer_fence':True,'heavy_kv_verifier':False,'policy_debug_log':False,'trace_log':False}
identity_path=BASE/'env/identity.json'
if identity_path.exists():assert json.loads(identity_path.read_text())==identity,'source or environment changed'
else:identity_path.write_text(json.dumps(identity,indent=2));(BASE/'env/repro.patch').write_bytes(patch)
proc=None;server_mode=None;group=0;monitor_stop=None;monitor_thread=None
class Monitor:
 def __init__(self,path,parent):self.path=path;self.parent=parent;self.stop=threading.Event();self.thread=threading.Thread(target=self.run,daemon=True)
 def run(self):
  import psutil
  with self.path.open('w') as f:
   w=csv.writer(f);w.writerow(['unix_s','gpu0_mib','gpu0_util','gpu1_mib','gpu1_util','gpu1_watts','gpu1_temp_c','gpu1_sm_mhz','gpu1_mem_mhz','host_used_gib','host_available_gib','cpu_percent'])
   while not self.stop.is_set():
    try:
     raw=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu,power.draw,temperature.gpu,clocks.sm,clocks.mem','--format=csv,noheader,nounits'],text=True,timeout=5);g=[x.split(',') for x in raw.strip().splitlines()];m=psutil.virtual_memory();w.writerow([time.time(),g[0][0],g[0][1],g[1][0],g[1][1],g[1][2],g[1][3],g[1][4],g[1][5],(m.total-m.available)/2**30,m.available/2**30,psutil.cpu_percent()]);f.flush()
    except Exception as e:event('monitor_warning',error=repr(e))
    self.stop.wait(1)
 def finish(self):self.stop.set();self.thread.join(timeout=10)
monitor=None

def request(path,method='GET'):
 req=urllib.request.Request('http://127.0.0.1:30000'+path,method=method,data=b'' if method=='POST' else None)
 return urllib.request.urlopen(req,timeout=10).read()
def stop_server():
 global proc,server_mode,monitor
 if proc is None:return
 if proc.poll() is None:
  os.killpg(proc.pid,signal.SIGTERM)
  try:proc.wait(timeout=30)
  except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)
 if monitor:monitor.finish();monitor=None
 event('server_stopped',mode=server_mode,exit_code=proc.returncode)
 proc=None;server_mode=None
 time.sleep(3)
def replay(work,out,log):
 with log.open('x') as f:subprocess.run([PYTHON,'-u','repro/scripts/replay.py','run','--file',str(work),'--out',str(out)],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=1800)
 result=json.loads((out/'run.json').read_text());assert result['passed']==result['requests'];assert result['workload_sha256']==digest(work)
 return result
def start_server(mode):
 global proc,server_mode,group,monitor
 if server_mode==mode and proc and proc.poll() is None:return
 stop_server();group+=1
 with socket.socket() as sock:
  assert sock.connect_ex(('127.0.0.1',30000))!=0,'port 30000 already occupied'
 gpu_uuid=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
 assert not any(gpu_uuid in l for l in apps.splitlines()),'GPU1 already has another compute process'
 env=os.environ.copy()
 for k in ['MAX_TOTAL_TOKENS','OFFLOAD_INTERVAL']:env.pop(k,None)
 env.update({'MODEL_PATH':str(MODEL),'MAX_RUNNING':'128','MEM_FRACTION':'.80','DETERMINISTIC':'1','NO_OVERLAP':'0','POLICY_DEBUG':'0','SGLANG_ENABLE_SYNC_CACHE':'1' if mode=='tokenflow' else '0','TOKENFLOW_REPRO_FENCE_WRITES':'1','TOKENFLOW_REPRO_TRACE':'0','TOKENFLOW_REPRO_VERIFY_KV':'0','TOKENFLOW_REPRO_SNAPSHOT_WRITES':'0'})
 tag=f'group{group}-{mode}';(BASE/'env'/f'{tag}.json').write_text(json.dumps({k:env[k] for k in ['MODEL_PATH','MAX_RUNNING','MEM_FRACTION','DETERMINISTIC','NO_OVERLAP','POLICY_DEBUG','SGLANG_ENABLE_SYNC_CACHE','TOKENFLOW_REPRO_FENCE_WRITES','TOKENFLOW_REPRO_TRACE','TOKENFLOW_REPRO_VERIFY_KV']},indent=2))
 log=(BASE/'logs'/f'{tag}-server.log').open('x');proc=subprocess.Popen(['bash','repro/scripts/launch.sh',mode],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close();server_mode=mode
 monitor=Monitor(BASE/'raw'/f'{tag}-resources.csv',proc.pid);monitor.thread.start();event('server_starting',mode=mode,pid=proc.pid,group=group)
 for i in range(600):
  if proc.poll() is not None:raise RuntimeError(f'{mode} startup failed, see group{group} log')
  try:request('/health');break
  except Exception:time.sleep(1)
 else:raise RuntimeError('server startup timeout')
 (BASE/'env'/f'{tag}-server-info.json').write_bytes(request('/server_info'))
 event('server_ready',mode=mode,group=group)
 # Nine short correctness requests and full-load warmup are excluded from performance.
 with (BASE/'logs'/f'{tag}-smoke.log').open('x') as f:subprocess.run([PYTHON,'repro/scripts/smoke_stream.py',str(BASE/'raw'/f'{tag}-smoke')],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=300)
 replay(BASE/'workloads/warm-s999.json',BASE/'raw'/f'{tag}-warm',BASE/'logs'/f'{tag}-warm.log');event('warmup_completed',mode=mode,group=group)
try:
 for seed in a.seeds:
  for mode in (['baseline','tokenflow'] if seed%2 else ['tokenflow','baseline']):
   todo=[case for case in a.cases if not (BASE/'raw'/f'{case}-{mode}-s{seed}'/'run.json').exists()]
   if not todo:continue
   start_server(mode)
   for case in todo:
    assert proc.poll() is None,'server exited'
    assert hashlib.sha256(subprocess.check_output(['git','diff'])).hexdigest()==identity['patch_sha256'],'source changed during matrix'
    request('/flush_cache','POST');work=BASE/'workloads'/f'{case}-s{seed}.json';tag=f'{case}-{mode}-s{seed}';out=BASE/'raw'/tag
    event('measurement_start',tag=tag,group=group,workload_sha256=digest(work))
    result=replay(work,out,BASE/'logs'/f'{tag}-client.log')
    subprocess.run([PYTHON,'repro/scripts/analyze.py',str(out),'--out',str(out/'analysis.json')],stdout=subprocess.DEVNULL,check=True)
    row=json.loads((out/'analysis.json').read_text())[0];row.update({'case':case,'mode':mode,'seed':seed,'server_group':group});(out/'analysis.json').write_text(json.dumps([row],indent=2));event('measurement_complete',**row)
 stop_server()
 all_rows=[]
 for f in sorted((BASE/'raw').glob('*/analysis.json')):all_rows+=json.loads(f.read_text())
 (BASE/'summary.json').write_text(json.dumps(all_rows,indent=2))
 with (BASE/'summary.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
 event('matrix_complete',runs=len(all_rows))
except Exception as e:
 event('failed',error=repr(e));raise
finally:stop_server()
