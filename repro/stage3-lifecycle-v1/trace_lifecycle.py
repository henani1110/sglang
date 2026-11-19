"""One instrumented E2 replay to correlate request divergence with KV events."""
import datetime, hashlib, json, os, signal, socket, subprocess, time
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path('/home/chx/projects/tokenflow-repro-paper-candidate')
os.chdir(ROOT)
V2 = ROOT / 'repro/stage2-7b-v2'
BASE = ROOT / 'repro/stage3-lifecycle-v1'
TAG = 'tokenflow-lifecycle-r3'
PY = str(ROOT / '.venv/bin/python')
MODEL = ROOT / 'repro/models/Qwen2-7B-Instruct'
OUT = BASE / 'raw' / TAG
if OUT.exists():
    raise RuntimeError(f'refusing to overwrite {OUT}')
OUT.mkdir(parents=True)

def emit(kind, **kw):
    row = {'time': datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), 'event': kind, **kw}
    with (BASE / 'events.jsonl').open('a') as f:
        f.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)

def request(path, method='GET'):
    import urllib.request
    req = urllib.request.Request('http://127.0.0.1:30000' + path, method=method,
                                 data=b'' if method == 'POST' else None)
    return urllib.request.urlopen(req, timeout=15).read()

identity = json.loads((V2 / 'env/identity.json').read_text())
assert hashlib.sha256(subprocess.check_output(['git', 'diff'])).hexdigest() == identity['patch_sha256']
PATCH = BASE / 'lifecycle-trace.patch'
patch_applied = False

def restore_source():
    global patch_applied
    if patch_applied:
        subprocess.run(['git', 'apply', '-R', str(PATCH)], check=True)
        actual = hashlib.sha256(subprocess.check_output(['git', 'diff'])).hexdigest()
        if actual != identity['patch_sha256']:
            raise RuntimeError(f'candidate diff not restored: {actual}')
        patch_applied = False
        emit('diagnostic_patch_reverted', restored_diff_sha256=actual)

import atexit
def on_term(signum, frame):
    raise SystemExit(f'interrupted by signal {signum}; cleanup will run')
atexit.register(restore_source)
signal.signal(signal.SIGTERM, on_term)
subprocess.run(['git', 'apply', str(PATCH)], check=True)
patch_applied = True
instrumented_hash = hashlib.sha256(subprocess.check_output(['git', 'diff'])).hexdigest()
(BASE / 'env' / 'lifecycle-instrumentation.json').write_text(json.dumps({
    'base_diff_sha256': identity['patch_sha256'],
    'instrumented_diff_sha256': instrumented_hash,
    'patch': str(PATCH),
    'source_files': ['python/sglang/srt/managers/scheduler.py', 'python/sglang/srt/managers/schedule_batch.py']
}, indent=2))
subprocess.run([PY, '-m', 'py_compile',
                'python/sglang/srt/managers/scheduler.py',
                'python/sglang/srt/managers/schedule_batch.py'], check=True)
for name in ['E2-s101.json', 'warm-s999.json']:
    target = BASE / 'workloads' / name
    assert target.exists()
gpu = subprocess.check_output(['nvidia-smi', '-i', '1', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader'], text=True)
assert not any(gpu in line for line in apps.splitlines())
with socket.socket() as s:
    assert s.connect_ex(('127.0.0.1', 30000)) != 0

env = os.environ.copy()
for k in ['MAX_TOTAL_TOKENS', 'OFFLOAD_INTERVAL']:
    env.pop(k, None)
env.update({'MODEL_PATH': str(MODEL), 'MAX_RUNNING': '128', 'MEM_FRACTION': '.80',
            'DETERMINISTIC': '1', 'NO_OVERLAP': '0', 'POLICY_DEBUG': '1',
            'SGLANG_ENABLE_SYNC_CACHE': '1', 'TOKENFLOW_REPRO_FENCE_WRITES': '1',
            'TOKENFLOW_REPRO_TRACE': '1', 'TOKENFLOW_REPRO_VERIFY_KV': '0',
            'TOKENFLOW_REPRO_SNAPSHOT_WRITES': '0', 'RUN_TAG': TAG})
(BASE / 'env' / f'{TAG}-env.json').write_text(json.dumps(
    {k: env[k] for k in ['MODEL_PATH', 'MAX_RUNNING', 'MEM_FRACTION', 'DETERMINISTIC', 'NO_OVERLAP',
                        'POLICY_DEBUG', 'SGLANG_ENABLE_SYNC_CACHE', 'TOKENFLOW_REPRO_FENCE_WRITES',
                        'TOKENFLOW_REPRO_TRACE', 'TOKENFLOW_REPRO_VERIFY_KV']}, indent=2))
log = (BASE / 'logs' / f'{TAG}-server.log').open('x')
proc = subprocess.Popen(['bash', str(BASE / 'scripts/launch_diagnostic.sh'), 'tokenflow'], env=env,
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
log.close()

def stop():
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)
    emit('server_stopped', returncode=proc.returncode)

try:
    emit('trace_start', tag=TAG, trace=True, policy_debug=True, kv_verify=False, instrumented_diff_sha256=instrumented_hash)
    for _ in range(600):
        if proc.poll() is not None:
            raise RuntimeError('TokenFlow server exited; inspect trace server log')
        try:
            request('/health')
            break
        except Exception:
            time.sleep(1)
    else:
        raise RuntimeError('server startup timeout')
    info = json.loads(request('/server_info'))
    (BASE / 'env' / f'{TAG}-server-info.json').write_text(json.dumps(info, indent=2))
    ref = json.loads((V2 / 'env/common-normalized-reference.json').read_text())
    assert {k: info.get(k) for k in ref} == ref
    assert info.get('enable_hierarchical_cache') and info.get('kv_offload_policy') == 'paper_v1'
    smoke = OUT / 'smoke'
    subprocess.run([PY, 'repro/scripts/smoke_stream.py', str(smoke)], check=True, timeout=300)
    (BASE / 'logs' / f'{TAG}-warm.log').parent.mkdir(parents=True, exist_ok=True)
    with (BASE / 'logs' / f'{TAG}-warm.log').open('x') as f:
        subprocess.run([PY, 'repro/scripts/replay.py', 'run', '--file', str(BASE / 'workloads/warm-s999.json'),
                        '--out', str(OUT / 'warm')], stdout=f, stderr=subprocess.STDOUT,
                       check=True, timeout=1800)
    request('/flush_cache', 'POST')
    begin = time.time()
    emit('e2_measurement_start', tag=TAG)
    with (BASE / 'logs' / f'{TAG}-client.log').open('x') as f:
        subprocess.run([PY, 'repro/scripts/replay.py', 'run', '--file', str(BASE / 'workloads/E2-s101.json'),
                        '--out', str(OUT)], stdout=f, stderr=subprocess.STDOUT,
                       check=True, timeout=1800)
    run = json.loads((OUT / 'run.json').read_text())
    run.update(start_unix_s=begin, tag=TAG)
    (OUT / 'run.json').write_text(json.dumps(run, indent=2))
    subprocess.run([PY, 'repro/scripts/analyze.py', str(OUT), '--out', str(OUT / 'analysis.json')],
                   check=True, stdout=subprocess.DEVNULL)
    emit('trace_measurement_complete', completed=run['passed'], requests=run['requests'],
         output_tokens=sum(x['max_new_tokens'] for x in
                           json.loads((BASE / 'workloads/E2-s101.json').read_text())['requests']))
finally:
    stop()
    restore_source()
