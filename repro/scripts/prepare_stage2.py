from pathlib import Path
root=Path('/home/chx/projects/tokenflow-repro-paper-candidate')
p=root/'repro/scripts/launch.sh';s=p.read_text();s=s.replace('--model-path /home/chx/models/Qwen3-0.6B','--model-path "${MODEL_PATH:-/home/chx/models/Qwen3-0.6B}"').replace('--mem-fraction-static 0.80','--mem-fraction-static "${MEM_FRACTION:-0.80}"');p.write_text(s)
p=root/'repro/scripts/replay.py';s=p.read_text();s=s.replace('begin=time.perf_counter(); tag=out.name','begin=time.perf_counter(); wall_begin=time.time(); tag=out.name').replace("summary={'workload_sha256'", "summary={'start_unix_s':wall_begin,'workload_sha256'");p.write_text(s)
p=root/'repro/scripts/analyze.py';s=p.read_text();old=" (root/'analysis-requests.json').write_text";new=" row['start_unix_s']=json.loads((root/'run.json').read_text()).get('start_unix_s')\n"+old;assert s.count(old)==1;s=s.replace(old,new);p.write_text(s)
