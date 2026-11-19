"""Compare cold-prefill and radix-prefix reuse logits on identical contexts."""
import datetime, hashlib, json, os, signal, socket, subprocess, time
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path('/home/chx/projects/tokenflow-repro-paper-candidate')
os.chdir(ROOT)
V2 = ROOT / 'repro/stage2-7b-v2'
BASE = ROOT / 'repro/stage3-diagnostic-v1'
PY = str(ROOT / '.venv/bin/python')
MODEL = ROOT / 'repro/models/Qwen2-7B-Instruct'
OUT = BASE / 'cache-route-probes.json'
if OUT.exists():
    raise RuntimeError(f'refusing to overwrite {OUT}')
identity = json.loads((V2 / 'env/identity.json').read_text())
assert hashlib.sha256(subprocess.check_output(['git', 'diff'])).hexdigest() == identity['patch_sha256']
workload = json.loads((BASE / 'workloads/E2-s101.json').read_text())
baseline_rows = [json.loads(x) for x in
                 (BASE / 'raw/baseline-r1/requests.jsonl').read_text().splitlines()]
prompt = next(r['input_ids'] for r in workload['requests'] if r['id'] == 15)
generated = baseline_rows[15]['events'][-1]['data']['output_ids']
prefixes = {n: prompt + generated[:n] for n in [1022, 1023, 1024]}
proc = None

def emit(kind, **kw):
    row = {'time': datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), 'event': kind, **kw}
    with (BASE / 'cache-route-probes.jsonl').open('a') as f:
        f.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)

def http(path, method='GET', body=None):
    import urllib.request
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request('http://127.0.0.1:30000' + path, method=method,
                                    data=data, headers={'Content-Type': 'application/json'} if body else {})
    return urllib.request.urlopen(request, timeout=300).read()

def stop():
    global proc
    if proc is not None:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)
        emit('server_stopped', returncode=proc.returncode)
        proc = None
        time.sleep(3)

def start(mode, route):
    global proc
    with socket.socket() as s:
        assert s.connect_ex(('127.0.0.1', 30000)) != 0
    gpu = subprocess.check_output(['nvidia-smi', '-i', '1', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader'], text=True)
    assert not any(gpu in line for line in apps.splitlines())
    tag = f'cache-route-{mode}-{route}'
    env = os.environ.copy()
    for k in ['MAX_TOTAL_TOKENS', 'OFFLOAD_INTERVAL']:
        env.pop(k, None)
    env.update({'MODEL_PATH': str(MODEL), 'MAX_RUNNING': '128', 'MEM_FRACTION': '.80',
                'DETERMINISTIC': '1', 'NO_OVERLAP': '0', 'POLICY_DEBUG': '0',
                'SGLANG_ENABLE_SYNC_CACHE': '1' if mode == 'tokenflow' else '0',
                'TOKENFLOW_REPRO_FENCE_WRITES': '1', 'TOKENFLOW_REPRO_TRACE': '0',
                'TOKENFLOW_REPRO_VERIFY_KV': '0', 'TOKENFLOW_REPRO_SNAPSHOT_WRITES': '0',
                'RUN_TAG': tag})
    log = (BASE / 'logs' / f'{tag}-server.log').open('x')
    proc = subprocess.Popen(['bash', str(BASE / 'scripts/launch_diagnostic.sh'), mode], env=env,
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    log.close()
    emit('server_starting', mode=mode, route=route, pid=proc.pid)
    for _ in range(600):
        if proc.poll() is not None:
            raise RuntimeError(f'{tag} server exited')
        try:
            http('/health')
            break
        except Exception:
            time.sleep(1)
    else:
        raise RuntimeError('startup timeout')
    info = json.loads(http('/server_info'))
    ref = json.loads((V2 / 'env/common-normalized-reference.json').read_text())
    assert {k: info.get(k) for k in ref} == ref
    assert info.get('enable_hierarchical_cache') == (mode == 'tokenflow')
    if mode == 'tokenflow':
        assert info.get('kv_offload_policy') == 'paper_v1'
    http('/flush_cache', 'POST')

def query(mode, route, n):
    payload = {'input_ids': prefixes[n], 'sampling_params': {'temperature': 0, 'max_new_tokens': 1,
                'ignore_eos': True}, 'return_logprob': True, 'top_logprobs_num': 10,
                'logprob_start_len': len(prefixes[n]) - 1, 'stream': False,
                'rid': f'cache-route-{mode}-{route}-{n}'}
    resp = json.loads(http('/generate', 'POST', payload))
    meta = resp.get('meta_info', {})
    return {'mode': mode, 'route': route, 'prefix_output_tokens': n,
            'output_ids': resp.get('output_ids'), 'output_token_logprobs': meta.get('output_token_logprobs'),
            'output_top_logprobs': meta.get('output_top_logprobs')}

results = []
try:
    for mode in ['baseline', 'tokenflow']:
        for route in ['cold', 'cached']:
            start(mode, route)
            counts = [1024] if route == 'cold' else [1022, 1023, 1024]
            for n in counts:
                row = query(mode, route, n)
                results.append(row)
                emit('cache_route_probe', **row)
            stop()
    OUT.write_text(json.dumps({'request_id': 15, 'prompt_tokens': len(prompt),
                               'results': results,
                               'note': 'cold starts with the full prefix; cached route extends 1022, then 1023, then 1024 output tokens.'}, indent=2))
    emit('cache_route_probes_complete', output=str(OUT))
finally:
    stop()
