import json,datetime
from pathlib import Path
root=Path('/home/chx/projects/tokenflow-repro-paper-candidate');base=root/'repro/stage2-7b-v2'
events=[json.loads(x) for x in (root/'repro/logs/stage2-driver-v2.log').read_text().splitlines() if x.startswith('{')]
a=datetime.datetime.fromisoformat('2026-10-28T15:06:14.222351+08:00').timestamp();b=datetime.datetime.fromisoformat('2026-10-28T15:07:09.004579+08:00').timestamp();over=[]
for e in events:
 if e['event']!='measurement_complete':continue
 r=json.loads((base/'raw'/e['run']/'run.json').read_text());start=r['start_unix_s'];end=start+e['duration_s']
 if start<b and end>a:over.append(e['run'])
record={'reason':'Task-owned v1 evidence archive ran concurrently; retain originals and retake before final selection.','archive_start_unix_s':a,'archive_end_unix_s':b,'excluded_original_runs':over,'replacement_suffix':'-clean'}
(base/'env/operator-overlap-retakes.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
manifest=json.loads((root/'repro/env/model-7b-manifest.json').read_text());stats={}
for f in manifest['files']:
 s=(root/'repro/models/Qwen2-7B-Instruct'/f['path']).stat();stats[f['path']]={'size':s.st_size,'mtime_ns':s.st_mtime_ns}
(base/'env/model-file-stats.json').write_text(json.dumps(stats,indent=2))
