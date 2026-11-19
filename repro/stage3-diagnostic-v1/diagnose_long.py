"""Replays exact E2-s101 twice per system without changing frozen v2 data."""
import datetime, hashlib, json, os, shutil, signal, socket, subprocess, time
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path('/home/chx/projects/tokenflow-repro-paper-candidate')
os.chdir(ROOT)
V2 = ROOT / 'repro/stage2-7b-v2'
BASE = ROOT / 'repro/stage3-diagnostic-v1'
PY = str(ROOT / '.venv/bin/python')
MODEL = ROOT / 'repro/models/Qwen2-7B-Instruct'
if BASE.exists() and any(p.name != 'diagnose_long.py' for p in BASE.iterdir()):
    raise RuntimeError(f'refusing to overwrite nonempty diagnostic directory {BASE}')
for d in ['raw', 'logs', 'env', 'scripts', 'workloads']:
    (BASE / d).mkdir(parents=True, exist_ok=True)

def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

identity = json.loads((V2 / 'env/identity.json').read_text())
assert hashlib.sha256(subprocess.check_output(['git', 'diff'])).hexdigest() == identity['patch_sha256']
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == identity['commit']
for name in ['E2-s101.json', 'warm-s999.json']:
    shutil.copy2(V2 / 'workloads' / name, BASE / 'workloads' / name)
work = BASE / 'workloads/E2-s101.json'
old_work_sha = json.loads((V2 / 'raw/E2-baseline-s101/run.json').read_text())['workload_sha256']
assert sha_file(work) == old_work_sha
for name in ['replay.py', 'analyze.py', 'smoke_stream.py']:
    shutil.copy2(ROOT / 'repro/scripts' / name, BASE / 'scripts' / name)
launcher = (ROOT / 'repro/scripts/launch_v2.sh').read_text()
old_path = 'repro/stage2-7b-v2/env/${RUN_TAG:-$mode}-command.txt'
assert old_path in launcher
(BASE / 'scripts/launch_diagnostic.sh').write_text(
    launcher.replace(old_path, 'repro/stage3-diagnostic-v1/env/${RUN_TAG:-$mode}-command.txt'))
os.chmod(BASE / 'scripts/launch_diagnostic.sh', 0o755)
stats = json.loads((V2 / 'env/model-file-stats.json').read_text())
for filename, expected in stats.items():
    st = (MODEL / filename).stat()
    assert st.st_size == expected['size'] and st.st_mtime_ns == expected['mtime_ns']

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

proc = None
mode = None
group = 0

def stop():
    global proc, mode
    if proc is None:
        return
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)
    emit('server_stopped', mode=mode, returncode=proc.returncode)
    proc, mode = None, None
    time.sleep(3)

def start(m, tag):
    global proc, mode, group
    stop()
    group += 1
    with socket.socket() as s:
        assert s.connect_ex(('127.0.0.1', 30000)) != 0, 'port 30000 is occupied'
    gpu_uuid = subprocess.check_output(
        ['nvidia-smi', '-i', '1', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    apps = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader'], text=True)
    assert not any(gpu_uuid in line for line in apps.splitlines()), 'GPU1 has another compute process'
    assert hashlib.sha256(subprocess.check_output(['git', 'diff'])).hexdigest() == identity['patch_sha256']
    env = os.environ.copy()
    for k in ['MAX_TOTAL_TOKENS', 'OFFLOAD_INTERVAL']:
        env.pop(k, None)
    env.update({'MODEL_PATH': str(MODEL), 'MAX_RUNNING': '128', 'MEM_FRACTION': '.80',
                'DETERMINISTIC': '1', 'NO_OVERLAP': '0', 'POLICY_DEBUG': '0',
                'SGLANG_ENABLE_SYNC_CACHE': '1' if m == 'tokenflow' else '0',
                'TOKENFLOW_REPRO_FENCE_WRITES': '1', 'TOKENFLOW_REPRO_TRACE': '0',
                'TOKENFLOW_REPRO_VERIFY_KV': '0', 'TOKENFLOW_REPRO_SNAPSHOT_WRITES': '0',
                'RUN_TAG': tag})
    with (BASE / 'env' / f'{tag}-env.json').open('w') as f:
        json.dump({k: env[k] for k in ['MODEL_PATH', 'MAX_RUNNING', 'MEM_FRACTION', 'DETERMINISTIC',
                                       'NO_OVERLAP', 'POLICY_DEBUG', 'SGLANG_ENABLE_SYNC_CACHE',
                                       'TOKENFLOW_REPRO_FENCE_WRITES', 'TOKENFLOW_REPRO_TRACE',
                                       'TOKENFLOW_REPRO_VERIFY_KV']}, f, indent=2)
    log = (BASE / 'logs' / f'{tag}-server.log').open('x')
    proc = subprocess.Popen(['bash', str(BASE / 'scripts/launch_diagnostic.sh'), m], env=env,
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    log.close()
    mode = m
    emit('server_starting', mode=m, pid=proc.pid, group=group)
    for _ in range(600):
        if proc.poll() is not None:
            raise RuntimeError(f'{m} server died; inspect {tag}-server.log')
        try:
            request('/health')
            break
        except Exception:
            time.sleep(1)
    else:
        raise RuntimeError('server startup timeout')
    info = json.loads(request('/server_info'))
    (BASE / 'env' / f'{tag}-server-info.json').write_text(json.dumps(info, indent=2))
    ref = json.loads((V2 / 'env/common-normalized-reference.json').read_text())
    assert {k: info.get(k) for k in ref} == ref, f'{tag}: common runtime settings differ'
    assert info.get('enable_hierarchical_cache') == (m == 'tokenflow')
    if m == 'tokenflow':
        assert info.get('kv_offload_policy') == 'paper_v1' and info.get('hicache_size') == 8
    emit('settings_verified', tag=tag, version=info.get('version'),
         chunked_prefill_size=info.get('chunked_prefill_size'),
         hierarchical_cache=info.get('enable_hierarchical_cache'), policy=info.get('kv_offload_policy'))
    smoke = BASE / 'raw' / f'{tag}-smoke'
    smoke.mkdir()
    with (BASE / 'logs' / f'{tag}-smoke.log').open('x') as f:
        subprocess.run([PY, str(BASE / 'scripts/smoke_stream.py'), str(smoke)],
                       stdout=f, stderr=subprocess.STDOUT, check=True, timeout=300)
    actual = [json.loads(x)['events'][-1]['data']['output_ids']
              for x in (smoke / 'requests.jsonl').read_text().splitlines()]
    expected = json.loads((V2 / 'env/short-token-reference.json').read_text())
    mismatches = [i for i, (a, b) in enumerate(zip(actual, expected)) if a != b]
    assert len(actual) == len(expected) and not mismatches, (tag, 'short output mismatch', mismatches)
    emit('short_smoke_passed', tag=tag, requests=len(actual))
    warm = BASE / 'raw' / f'{tag}-warm'
    warm.mkdir()
    with (BASE / 'logs' / f'{tag}-warm.log').open('x') as f:
        subprocess.run([PY, str(BASE / 'scripts/replay.py'), 'run', '--file',
                        str(V2 / 'workloads/warm-s999.json'), '--out', str(warm)],
                       stdout=f, stderr=subprocess.STDOUT, check=True, timeout=1800)
    emit('warmup_passed', tag=tag)

def run_case(m, rep):
    tag = f'{m}-r{rep}'
    start(m, tag)
    request('/flush_cache', 'POST')
    out = BASE / 'raw' / tag
    out.mkdir()
    begin = time.time()
    emit('e2_measurement_start', tag=tag, workload_sha256=sha_file(work))
    with (BASE / 'logs' / f'{tag}-client.log').open('x') as f:
        subprocess.run([PY, str(BASE / 'scripts/replay.py'), 'run', '--file', str(work), '--out', str(out)],
                       stdout=f, stderr=subprocess.STDOUT, check=True, timeout=1800)
    run_path = out / 'run.json'
    summary = json.loads(run_path.read_text())
    summary.update(start_unix_s=begin, diagnostic_tag=tag, workload_sha256=sha_file(work))
    run_path.write_text(json.dumps(summary, indent=2))
    analysis = BASE / 'env' / f'{tag}-analysis.json'
    subprocess.run([PY, str(BASE / 'scripts/analyze.py'), str(out), '--out', str(analysis)],
                   check=True, stdout=subprocess.DEVNULL)
    row = json.loads(analysis.read_text())[0]
    emit('e2_measurement_complete', tag=tag, completed=row['completed'],
         output_tokens=row['output_tokens'], duration_s=row['duration_s'],
         mean_ttft_s=row['ttft_mean_s'], p99_ttft_s=row['ttft_p99_s'])

def compare(a, b):
    result = []
    for x, y in zip(a, b):
        ax = x['events'][-1]['data']['output_ids']
        by = y['events'][-1]['data']['output_ids']
        if ax != by:
            first = next((i for i, (u, v) in enumerate(zip(ax, by)) if u != v), min(len(ax), len(by)))
            result.append({'id': x['id'], 'length_a': len(ax), 'length_b': len(by),
                           'first_difference_zero_based': first})
    return result

try:
    emit('diagnostic_start', commit=identity['commit'], patch_sha256=identity['patch_sha256'],
         workload_sha256=sha_file(work), original_workload_sha256=old_work_sha)
    order = [('baseline', 1), ('tokenflow', 1), ('baseline', 2), ('tokenflow', 2)]
    for m, rep in order:
        run_case(m, rep)
    stop()
    runs = {f'{m}-r{rep}': [json.loads(x) for x in
             (BASE / 'raw' / f'{m}-r{rep}' / 'requests.jsonl').read_text().splitlines()]
            for m, rep in order}
    audit = {}
    for m in ['baseline', 'tokenflow']:
        old = [json.loads(x) for x in (V2 / 'raw' / f'E2-{m}-s101' / 'requests.jsonl').read_text().splitlines()]
        audit[m] = {'repeat_mismatches': compare(runs[f'{m}-r1'], runs[f'{m}-r2']),
                    'original_vs_r1_mismatches': compare(old, runs[f'{m}-r1']),
                    'original_vs_r2_mismatches': compare(old, runs[f'{m}-r2'])}
    audit['cross_system'] = {'r1': compare(runs['baseline-r1'], runs['tokenflow-r1']),
                             'r2': compare(runs['baseline-r2'], runs['tokenflow-r2'])}
    (BASE / 'token-difference-audit.json').write_text(json.dumps(audit, indent=2))
    emit('diagnostic_complete', baseline_repeat_mismatches=len(audit['baseline']['repeat_mismatches']),
         tokenflow_repeat_mismatches=len(audit['tokenflow']['repeat_mismatches']),
         cross_system_r1=len(audit['cross_system']['r1']), cross_system_r2=len(audit['cross_system']['r2']))
except Exception as e:
    emit('diagnostic_failed', error=repr(e))
    raise
finally:
    stop()
