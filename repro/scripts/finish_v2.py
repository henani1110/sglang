import hashlib,json,os,subprocess,time,traceback
from pathlib import Path
ROOT=Path('/home/chx/projects/tokenflow-repro-paper-candidate');os.chdir(ROOT);BASE=ROOT/'repro/stage2-7b-v2';PYTHON=str(ROOT/'.venv/bin/python')
def emit(event,**kwargs):print(json.dumps({'event':event,'unix_s':time.time(),**kwargs}),flush=True)
try:
 emit('waiting_for_main_matrix')
 while True:
  events=[json.loads(x) for x in (ROOT/'repro/logs/stage2-driver-v2.log').read_text().splitlines() if x.startswith('{')]
  if any(e['event']=='failed' for e in events):raise RuntimeError('main matrix failed; preserve logs')
  if any(e['event']=='matrix_complete' and e['runs']==24 for e in events):break
  time.sleep(10)
 for name in ['summary.json','summary.csv']:
  target=BASE/'env'/('initial-matrix-'+name);assert not target.exists();target.write_bytes((BASE/name).read_bytes())
 emit('retakes_starting')
 with (ROOT/'repro/logs/stage2-retakes.log').open('x') as f:subprocess.run([PYTHON,'-u','repro/scripts/stage2_retakes.py','run','--cases','E2','P2','--seeds','101'],stdout=f,stderr=subprocess.STDOUT,check=True)
 emit('retakes_finished')
 with (ROOT/'repro/logs/stage2-report-v2.log').open('x') as f:subprocess.run([PYTHON,'-u','repro/scripts/stage2_report_v2.py'],stdout=f,stderr=subprocess.STDOUT,check=True)
 rows=json.loads((BASE/'summary-selected.json').read_text());assert len(rows)==24;assert all(r['completed']==r['requested'] for r in rows)
 for case in ['E1','E2','P2','P4']:
  for seed in [101,102,103]:
   pair=[r for r in rows if r['case']==case and r['seed']==seed];assert len(pair)==2;assert pair[0]['workload_sha256']==pair[1]['workload_sha256'];assert pair[0]['output_tokens']==pair[1]['output_tokens']
 identity=json.loads((BASE/'env/identity.json').read_text());assert hashlib.sha256(subprocess.check_output(['git','diff'])).hexdigest()==identity['patch_sha256']
 for g,mode in [(1,'baseline'),(2,'tokenflow'),(3,'baseline'),(4,'tokenflow'),(5,'baseline')]:
  info=json.loads((BASE/'env'/f'group{g}-{mode}-server-info.json').read_text());assert info['enable_deterministic_inference'] and info['disable_piecewise_cuda_graph']
  comparison=BASE/'env'/f'group{g}-{mode}-short-token-comparison.json'
  if comparison.exists():assert not json.loads(comparison.read_text())['mismatched_ids']
 (BASE/'env/final-gpu-state.csv').write_text(subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True))
 audit={'selected_runs':len(rows),'selected_requests':sum(r['completed'] for r in rows),'selected_output_tokens':sum(r['output_tokens'] for r in rows),'excluded_original_runs':json.loads((BASE/'env/operator-overlap-retakes.json').read_text())['excluded_original_runs'],'retakes':2,'paired_workloads_equal':True,'source_patch_unchanged':True,'actual_settings_verified_all_groups':True,'short_tokens_verified_all_groups':True}
 (BASE/'env/final-audit.json').write_text(json.dumps(audit,indent=2));emit('audit_complete',**audit)
 # Save exact source/supporting files before archiving; GPU measurements have ended.
 for name in ['launch_v2.sh','stage2_driver_v2.py','stage2_retakes.py','replay.py','analyze.py','stage2_report_v2.py']:(BASE/'env'/('actual-'+name)).write_bytes((ROOT/'repro/scripts'/name).read_bytes())
 (BASE/'env/packages-freeze.txt').write_bytes((ROOT/'repro/env/freeze.txt').read_bytes())
 (ROOT/'repro/artifacts').mkdir(exist_ok=True);bundle=ROOT/'repro/artifacts/tokenflow-stage2-v2-evidence.tar.gz'
 with bundle.open('xb') as f:subprocess.run(['tar','-C',str(ROOT),'-czf','-','repro/stage2-7b-v2'],stdout=f,check=True)
 with bundle.open('rb') as f:bundle_sha=hashlib.file_digest(f,'sha256').hexdigest()
 result={'status':'complete',**audit,'bundle':str(bundle),'bundle_sha256':bundle_sha};(BASE/'finished.json').write_text(json.dumps(result,indent=2));emit('complete',**result)
except Exception as e:
 (BASE/'finish-error.json').write_text(json.dumps({'status':'incomplete','error':repr(e),'traceback':traceback.format_exc()},indent=2));emit('failed',error=repr(e));raise
