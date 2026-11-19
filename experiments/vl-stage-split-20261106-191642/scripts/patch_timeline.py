#!/usr/bin/env python3
"""Patch only the isolated vLLM copy with low-overhead timeline events."""
from __future__ import annotations
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKUP = ROOT / "artifacts" / "original_files" / "timeline_prepatch"
PACKAGE: Path

def replace_once(path: Path, old: str, new: str) -> None:
    source = path.read_text()
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"expected exactly one anchor in {path}, found {count}")
    relative = path.relative_to(PACKAGE)
    backup = BACKUP / relative
    backup.parent.mkdir(parents=True, exist_ok=True)
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(source.replace(old, new, 1))

def package_dir() -> Path:
    value = subprocess.check_output(
        [sys.executable, "-c", "import vllm,pathlib;print(pathlib.Path(vllm.__file__).parent)"],
        text=True,
    ).strip().splitlines()[-1]
    return Path(value)

def main() -> None:
    global PACKAGE
    PACKAGE = package_dir()
    if "vl-stage-split-20261106-191642" not in str(PACKAGE):
        raise RuntimeError(f"refusing to patch non-isolated package: {PACKAGE}")

    hook = PACKAGE / "vl_pilot_hook.py"
    replace_once(
        hook,
        "def _append_event(event: dict[str, Any]) -> None:\n",
        "def _append_event(event: dict[str, Any]) -> None:\n"
        '    if os.environ.get("VL_PILOT_PROBE") != "1":\n'
        "        return\n",
    )
    replace_once(
        hook,
        '    row = {\n        "kind": "server_token_ready",\n',
        '    if os.environ.get("VL_PILOT_PROBE") != "1":\n'
        "        return\n"
        '    row = {\n        "kind": "api_yield",\n',
    )
    engine_fn = '''
def record_engine_result_ready(request_id: str, prompt_idx: int, result: Any) -> None:
    if os.environ.get("VL_PILOT_PROBE") != "1":
        return
    outputs = getattr(result, "outputs", ()) or ()
    _append_event({
        "kind": "engine_result_ready_to_api",
        "request_id": request_id,
        "engine_request_id": f"{request_id}-{int(prompt_idx)}",
        "prompt_idx": int(prompt_idx),
        "time_ns": time.perf_counter_ns(),
        "outputs": [
            {
                "index": int(getattr(output, "index", 0)),
                "token_ids": [int(token) for token in (getattr(output, "token_ids", ()) or ())],
                "finish_reason": getattr(output, "finish_reason", None),
            }
            for output in outputs
        ],
    })


'''
    replace_once(
        hook,
        "def record_scheduler_event(scheduler_output: Any) -> None:\n",
        engine_fn + "def record_scheduler_event(scheduler_output: Any) -> None:\n",
    )
    replace_once(
        hook,
        'def record_scheduler_event(scheduler_output: Any) -> None:\n    if not os.environ.get("VL_PILOT_SCHED_FILE"):\n',
        'def record_scheduler_event(scheduler_output: Any) -> None:\n'
        '    if os.environ.get("VL_PILOT_PROBE") != "1":\n'
        "        return\n"
        '    if not os.environ.get("VL_PILOT_SCHED_FILE"):\n',
    )
    replace_once(
        hook,
        '    mode = str(state.get("mode", "NORMAL")).upper()\n    keys = _MM_KEYS.get()\n',
        '    mode = str(state.get("mode", "NORMAL")).upper()\n'
        '    if os.environ.get("VL_PILOT_PROBE") != "1" and mode in {"NORMAL", "R"}:\n'
        "        return compute()\n"
        "    keys = _MM_KEYS.get()\n",
    )

    serving = PACKAGE / "entrypoints" / "openai" / "completion" / "serving.py"
    replace_once(
        serving,
        "from vllm.vl_pilot_hook import record_server_token_event\n",
        "from vllm.vl_pilot_hook import (\n"
        "    record_engine_result_ready,\n"
        "    record_server_token_event,\n"
        ")\n",
    )
    replace_once(
        serving,
        "            async for prompt_idx, res in result_generator:\n                last_res = res\n",
        "            async for prompt_idx, res in result_generator:\n"
        "                record_engine_result_ready(request_id, prompt_idx, res)\n"
        "                last_res = res\n",
    )
    print(f"patched_package={PACKAGE}")
    print(f"timeline_backup={BACKUP}")

if __name__ == "__main__":
    main()
