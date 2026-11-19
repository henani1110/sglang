#!/usr/bin/env python3
"""Buffer engine-result/API-yield trace rows and record per-request token indexes."""
from __future__ import annotations
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKUP = ROOT / "artifacts" / "original_files" / "timeline_buffer_prepatch"
API_BATCH_SIZE = 512

def main() -> None:
    package = Path(subprocess.check_output(
        [sys.executable, "-c", "import vllm,pathlib;print(pathlib.Path(vllm.__file__).parent)"],
        text=True,
    ).strip().splitlines()[-1])
    if "vl-stage-split-20261106-191642" not in str(package):
        raise RuntimeError(f"refusing non-isolated package {package}")
    path = package / "vl_pilot_hook.py"
    source = path.read_text()
    backup = BACKUP / "vllm" / "vl_pilot_hook.py"
    backup.parent.mkdir(parents=True, exist_ok=True)
    if not backup.exists():
        shutil.copy2(path, backup)
    marker = "_SERVER_EVENTS_LOCK = threading.Lock()\n"
    if source.count(marker) != 1:
        raise RuntimeError("server event lock marker mismatch")
    if "_ENGINE_EVENTS: list" not in source:
        source = source.replace(
            marker,
            marker
            + "_ENGINE_EVENTS: list[dict[str, Any]] = []\n"
            + "_ENGINE_EVENTS_LOCK = threading.Lock()\n"
            + "_ENGINE_TOKEN_COUNTS: dict[str, int] = {}\n"
            + "_API_TOKEN_COUNTS: dict[str, int] = {}\n",
            1,
        )

    server_fn = '''def record_server_token_event(
    request_id: str,
    token_ids: list[int] | tuple[int, ...],
    prompt_token_ids: list[int] | tuple[int, ...] | None,
    finish: bool,
) -> None:
    if os.environ.get("VL_PILOT_PROBE") != "1":
        return
    ids = [int(token) for token in token_ids]
    now = time.perf_counter_ns()
    with _SERVER_EVENTS_LOCK:
        token_start = _API_TOKEN_COUNTS.get(request_id, 0)
        _API_TOKEN_COUNTS[request_id] = token_start + len(ids)
        _SERVER_EVENTS.append({
            "kind": "api_yield",
            "request_id": request_id,
            "time_ns": now,
            "token_index_start": token_start if ids else None,
            "token_index_end": token_start + len(ids) - 1 if ids else None,
            "token_count": len(ids),
            "token_ids": ids,
            "finish": bool(finish),
        })
        if not finish and len(_SERVER_EVENTS) < __API_BATCH_SIZE__:
            return
        rows = list(_SERVER_EVENTS)
        _SERVER_EVENTS.clear()
    with _ENGINE_EVENTS_LOCK:
        rows.extend(_ENGINE_EVENTS)
        _ENGINE_EVENTS.clear()
    rows.sort(key=lambda event: int(event["time_ns"]))
    path = os.environ.get("VL_PILOT_EVENT_FILE")
    if path and rows:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = "".join(
            json.dumps(event, sort_keys=True, separators=(",", ":")) + "\\\\n"
            for event in rows
        ).encode()
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, payload)
        finally:
            os.close(fd)


'''.replace("__API_BATCH_SIZE__", str(API_BATCH_SIZE))
    pattern = re.compile(r"def record_server_token_event\(.*?(?=\ndef record_engine_result_ready)", re.S)
    source, count = pattern.subn(server_fn.rstrip() + "\n", source, count=1)
    if count != 1:
        raise RuntimeError(f"server event function replacement count {count}")

    engine_fn = '''def record_engine_result_ready(request_id: str, prompt_idx: int, result: Any) -> None:
    if os.environ.get("VL_PILOT_PROBE") != "1":
        return
    outputs = getattr(result, "outputs", ()) or ()
    now = time.perf_counter_ns()
    with _ENGINE_EVENTS_LOCK:
        for output in outputs:
            ids = [int(token) for token in (getattr(output, "token_ids", ()) or ())]
            if not ids:
                continue
            output_index = int(getattr(output, "index", 0))
            request_key = f"{request_id}-{int(prompt_idx)}-{output_index}"
            token_start = _ENGINE_TOKEN_COUNTS.get(request_key, 0)
            _ENGINE_TOKEN_COUNTS[request_key] = token_start + len(ids)
            _ENGINE_EVENTS.append({
                "kind": "engine_result_ready_to_api",
                "request_id": request_id,
                "engine_request_id": f"{request_id}-{int(prompt_idx)}",
                "output_index": output_index,
                "prompt_idx": int(prompt_idx),
                "time_ns": now,
                "token_index_start": token_start,
                "token_index_end": token_start + len(ids) - 1,
                "token_ids": ids,
                "finish_reason": getattr(output, "finish_reason", None),
            })


'''
    pattern = re.compile(r"def record_engine_result_ready\(.*?(?=\ndef record_scheduler_event)", re.S)
    source, count = pattern.subn(engine_fn.rstrip() + "\n", source, count=1)
    if count != 1:
        raise RuntimeError(f"engine event function replacement count {count}")
    path.write_text(source)
    print(f"buffered_timeline_applied={path}")

if __name__ == "__main__":
    main()
