"""Compare logits after identical prefixes just before/at the first divergence."""
import datetime, hashlib, json, os, signal, socket, subprocess, time
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path('/home/chx/projects/tokenflow-repro-paper-candidate')
os.chdir(ROOT)
V2 = ROOT / 'repro/stage2-7b-v2'
BASE = ROOT / 'repro/stage3-diagnostic-v1'
PY = str(ROOT / '.venv/bin/python')
MODEL = ROOT / 'repro/models/Qwen2-7B-Instruct'
OUT = BASE / 'prefix-logit-probes.json'
if OUT.exists():
    raise RuntimeError(f'refusing to overwrite {OUT}')
identity = json.loads((V2 / 'env/identity.json').read_text())
assert hashlib.sha256(subprocess.check_output(['git', 'diff'])).hexdigest() == identity['patch_sha256']
workload = json.loads((BASE / 'workloads/E2-s101.json').read_text())
baseline_rows = [json.loads(x) for x in
                 (BASE / 'raw/baseline-r1/requests.jsonl').read_text().splitlines()]
probe_specs = []
for rid, counts in [(15, [1022, 1023, 1024]), (78, [786])]:
    prompt = next(r['input_ids'] for r in workload['requests'] if r['id'] == rid)
    generated = baseline_rows[rid]['events'][-1]['data']['output_ids']
    assert len(generated) == (1260 if rid == 15 else 1718)
    for n in counts:
        probe_specs.append({'request_id': rid, 'prefix_output_tokens': n,
                            'input_ids': prompt + generated[:n]})
proc = None

def emit(kind, **kw):
    row = {'time': datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), 'event': kind, **kw}
    with (BASE / 'prefix-probes.jsonl').open('a') as f:
        f.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)

def http_request(path, method='GET', body=None):
    import urllib.request
    data = None if body is None else json.dumps(body).encode()
    req0 = urllib.request.Request('http://127.0.0.1:30000' + path, method=method,
                                  data=data, headers={'Content-Type': 'application/json'} if body else {})
    return urllib.request.urlopen(req0, timeout=300).read()

def stop():
    global proc
    if proc and proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)
    if proc:
        emit('server_stopped', returncode=proc.returncode)
        proc = None

def probe(mode):
    global proc
    with socket.socket() as s:
        assert s.connect_ex(('127.0.0.1', 30000)) != 0
    gpu = subprocess.check_output(['nvidia-smi', '-i', '1', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader'], text=True)
    assert not any(gpu in line for line in apps.splitlines())
    env = os.environ.copy()
    for k in ['MAX_TOTAL_TOKENS', 'OFFLOAD_INTERVAL']:
        env.pop(k, None)
    tag = f'prefix-probe-{mode}-r2'
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
    emit('server_starting', mode=mode, pid=proc.pid)
    for _ in range(600):
        if proc.poll() is not None:
            raise RuntimeError(f'{mode} startup failed')
        try:
            http_request('/health')
            break
        except Exception:
            time.sleep(1)
    else:
        raise RuntimeError('server startup timeout')
    info = json.loads(http_request('/server_info'))
    ref = json.loads((V2 / 'env/common-normalized-reference.json').read_text())
    assert {k: info.get(k) for k in ref} == ref
    assert info.get('enable_hierarchical_cache') == (mode == 'tokenflow')
    if mode == 'tokenflow':
        assert info.get('kv_offload_policy') == 'paper_v1'
    results = []
    http_request('/flush_cache', 'POST')
    for spec in probe_specs:
        n = spec['prefix_output_tokens']
        payload = {'input_ids': spec['input_ids'], 'sampling_params': {'temperature': 0, 'max_new_tokens': 1,
                    'ignore_eos': True}, 'return_logprob': True, 'top_logprobs_num': 10,
                    'logprob_start_len': len(spec['input_ids']) - 1, 'stream': False,
                    'rid': f'prefix-probe-{mode}-{spec["request_id"]}-{n}'}
        resp = json.loads(http_request('/generate', 'POST', payload))
        meta = resp.get('meta_info', {})
        row = {'mode': mode, 'request_id': spec['request_id'],
               'prefix_output_tokens': n, 'input_tokens': len(spec['input_ids']),
               'output_ids': resp.get('output_ids'),
               'output_token_logprobs': meta.get('output_token_logprobs'),
               'output_top_logprobs': meta.get('output_top_logprobs'),
               'input_top_logprobs': meta.get('input_top_logprobs'),
               'meta_keys': sorted(meta.keys())}
        results.append(row)
        emit('prefix_probe_done', mode=mode, request_id=spec['request_id'], prefix_output_tokens=n,
             generated=resp.get('output_ids'), top_logprobs=meta.get('output_top_logprobs'))
    return results

try:
    all_results = {}
    for mode in ['baseline', 'tokenflow']:
        all_results[mode] = probe(mode)
        stop()
    result = {'request_ids': [15, 78], 'prefix_probes': [
                  {'request_id': x['request_id'], 'prefix_output_tokens': x['prefix_output_tokens']}
                  for x in probe_specs], 'source_run': 'baseline-r1', 'results': all_results,
              'note': 'Top log probabilities at identical prefixes; standalone probes do not exercise pause/resume.'}
    OUT.write_text(json.dumps(result, indent=2))
    emit('prefix_probes_complete', output=str(OUT))
finally:
    stop()
