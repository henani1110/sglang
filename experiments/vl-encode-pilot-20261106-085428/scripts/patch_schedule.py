#!/usr/bin/env python3
"""Add scheduler-iteration evidence and flush helpers to the pilot vLLM copy."""
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
    backup = BACKUP_DIR / path.name
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(text.replace(old, new, 1))


def main() -> None:
    out = subprocess.check_output(
        [sys.executable, "-c", "import vllm,pathlib;print(pathlib.Path(vllm.__file__).parent)"],
        text=True,
    ).strip().splitlines()[-1]
    package = Path(out)
    runner = package / "v1" / "worker" / "gpu_model_runner.py"
    replace_once(
        runner,
        "from vllm.vl_pilot_hook import flush_cuda_events, mm_keys_context\n",
        "from vllm.vl_pilot_hook import (\n"
        "    flush_cuda_events,\n"
        "    flush_scheduler_events,\n"
        "    mm_keys_context,\n"
        "    record_scheduler_event,\n"
        ")\n",
    )
    replace_once(
        runner,
        """    def pilot_flush_visual_events(self) -> list[dict[str, object]]:
        return flush_cuda_events()
""",
        """    def pilot_flush_visual_events(self) -> dict[str, object]:
        return {
            "cuda_events": flush_cuda_events(),
            "scheduler_flushed": flush_scheduler_events(),
        }
""",
    )
    replace_once(
        runner,
        """        if self.execute_model_state is not None:
            raise RuntimeError(
                "State error: sample_tokens() must be called "
                "after execute_model() returns None."
            )

        # If ngram_gpu is used, we need to copy the scheduler_output to avoid
""",
        """        if self.execute_model_state is not None:
            raise RuntimeError(
                "State error: sample_tokens() must be called "
                "after execute_model() returns None."
            )

        record_scheduler_event(scheduler_output)

        # If ngram_gpu is used, we need to copy the scheduler_output to avoid
""",
    )
    print(f"patched_schedule={runner}")


if __name__ == "__main__":
    main()
