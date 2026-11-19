#!/usr/bin/env python3
"""Add an opt-in independent multimodal encoder compute budget to run overlay."""
from __future__ import annotations

import difflib
import shutil
import sys
from pathlib import Path


MARKER = "VL_MAX_ENCODER_INPUT_TOKENS"
ANCHOR = """        self.max_num_encoder_input_tokens = (
            mm_budget.encoder_compute_budget if mm_budget else 0
        )
"""
INSERT = ANCHOR + """        # Optional C-stage dual-budget control. The LM token budget remains
        # scheduler_config.max_num_batched_tokens; this independent limit only
        # constrains visual encoder admission on the same scheduler instance.
        encoder_budget_override = os.environ.get("VL_MAX_ENCODER_INPUT_TOKENS")
        if encoder_budget_override is not None:
            self.max_num_encoder_input_tokens = int(encoder_budget_override)
            if self.max_num_encoder_input_tokens <= 0:
                raise ValueError("VL_MAX_ENCODER_INPUT_TOKENS must be positive")
            logger.info(
                "Using independent multimodal encoder compute budget: %d",
                self.max_num_encoder_input_tokens,
            )
"""


def patch(path: Path, backup: Path, diff_path: Path) -> None:
    source = path.read_text()
    if MARKER in source:
        print("dual encoder budget patch already present")
        return
    if "import os\n" not in source:
        raise SystemExit("expected os import from the existing run-scoped S patch")
    if source.count(ANCHOR) != 1:
        raise SystemExit(f"expected one encoder-budget anchor, got {source.count(ANCHOR)}")
    updated = source.replace(ANCHOR, INSERT, 1)
    backup.parent.mkdir(parents=True, exist_ok=True)
    diff_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, backup)
    diff_path.write_text("".join(difflib.unified_diff(
        source.splitlines(keepends=True),
        updated.splitlines(keepends=True),
        fromfile=str(backup),
        tofile=str(path),
    )))
    compile(updated, str(path), "exec")
    path.write_text(updated)
    print(f"patched {path}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("usage: patch_dual_encoder_budget.py SCHEDULER BACKUP DIFF")
    patch(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
