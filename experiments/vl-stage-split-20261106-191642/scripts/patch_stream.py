#!/usr/bin/env python3
"""Add buffered server-side token-ready timestamps to the isolated vLLM copy."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKUP_DIR = ROOT / "artifacts" / "original_files"


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one patch anchor in {path}, found {count}")
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup = BACKUP_DIR / path.name
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(text.replace(old, new, 1))


def main() -> None:
    out = subprocess.check_output(
        [sys.executable, "-c", "import vllm,pathlib;print(pathlib.Path(vllm.__file__).parent)"],
        text=True,
    ).strip().splitlines()[-1]
    path = Path(out) / "entrypoints" / "openai" / "completion" / "serving.py"
    replace_once(
        path,
        "from vllm.logger import init_logger\n",
        "from vllm.logger import init_logger\nfrom vllm.vl_pilot_hook import record_server_token_event\n",
    )
    replace_once(
        path,
        '                    response_json = chunk.model_dump_json(exclude_unset=True)\n'
        '                    yield f"data: {response_json}\\n\\n"\n',
        '                    response_json = chunk.model_dump_json(exclude_unset=True)\n'
        '                    record_server_token_event(\n'
        '                        request_id,\n'
        '                        as_list(output.token_ids),\n'
        '                        prompt_token_ids_to_return,\n'
        '                        finish_reason is not None,\n'
        '                    )\n'
        '                    yield f"data: {response_json}\\n\\n"\n',
    )
    print(f"patched_stream={path}")


if __name__ == "__main__":
    main()
