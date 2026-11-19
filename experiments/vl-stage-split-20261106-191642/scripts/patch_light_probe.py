#!/usr/bin/env python3
"""Reduce timeline writes to first output per request for the A-stage probe."""
from __future__ import annotations
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKUP = ROOT / "artifacts" / "original_files" / "timeline_light_prepatch"

def main() -> None:
    package = Path(subprocess.check_output(
        [sys.executable, "-c", "import vllm,pathlib;print(pathlib.Path(vllm.__file__).parent)"],
        text=True,
    ).strip().splitlines()[-1])
    if "vl-stage-split-20261106-191642" not in str(package):
        raise RuntimeError(f"refusing non-isolated package {package}")
    hook = package / "vl_pilot_hook.py"
    text = hook.read_text()
    marker = "_SERVER_EVENTS_LOCK = threading.Lock()\n"
    if text.count(marker) != 1:
        raise RuntimeError("server event lock marker mismatch")
    if "_ENGINE_FIRST_READY" not in text:
        backup = BACKUP / "vllm" / "vl_pilot_hook.py"
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(hook, backup)
        text = text.replace(
            marker,
            marker + "_ENGINE_FIRST_READY: set[str] = set()\n_ENGINE_FIRST_READY_LOCK = threading.Lock()\n",
            1,
        )
    start = text.index("def record_engine_result_ready(")
    end = text.index("\ndef record_scheduler_event(", start)
    function = '''def record_engine_result_ready(request_id: str, prompt_idx: int, result: Any) -> None:
    if os.environ.get("VL_PILOT_PROBE") != "1":
        return
    request_key = f"{request_id}-{int(prompt_idx)}"
    outputs = getattr(result, "outputs", ()) or ()
    token_outputs = [
        output for output in outputs
        if getattr(output, "token_ids", None)
    ]
    if not token_outputs:
        return
    with _ENGINE_FIRST_READY_LOCK:
        if request_key in _ENGINE_FIRST_READY:
            return
        _ENGINE_FIRST_READY.add(request_key)
    _append_event({
        "kind": "engine_result_ready_to_api",
        "request_id": request_id,
        "engine_request_id": request_key,
        "prompt_idx": int(prompt_idx),
        "time_ns": time.perf_counter_ns(),
        "token_ids": [
            int(token)
            for output in token_outputs
            for token in (getattr(output, "token_ids", ()) or ())
        ],
    })


'''
    text = text[:start] + function + text[end + 1:]
    hook.write_text(text)
    print(f"light_probe_applied={hook}")

if __name__ == "__main__":
    main()
