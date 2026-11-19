#!/usr/bin/env python3
"""Apply the opt-in E -> D -> P scheduler experiment to the run overlay."""
from __future__ import annotations

import difflib
import shutil
import sys
from pathlib import Path


INSERTION = '''        # Run-scoped, opt-in split: execute one new visual encoder input while
        # only already-decoding text requests consume LM/KV tokens. The visual
        # request remains newly admitted with its encoder input scheduled, but
        # its prefill tokens are deferred to a later scheduler iteration.
        stage_split_encoder_only_req_ids: set[str] = set()
        if (
            self.use_v2_model_runner
            and os.environ.get("VL_STAGE_SPLIT_MODE") == "S"
            and not self.is_encoder_decoder
            and len(scheduled_new_reqs) == 1
            and not scheduled_resumed_reqs
            and len(scheduled_encoder_inputs) == 1
        ):
            split_request = scheduled_new_reqs[0]
            split_req_id = split_request.request_id
            split_encoder_ids = scheduled_encoder_inputs.get(split_req_id, [])
            other_scheduled_ids = [
                req_id
                for req_id, count in num_scheduled_tokens.items()
                if req_id != split_req_id and count > 0
            ]
            other_scheduled_reqs = [self.requests.get(req_id) for req_id in other_scheduled_ids]
            only_inflight_text_decode = bool(other_scheduled_reqs) and all(
                request is not None
                and not request.has_encoder_inputs
                and request.num_computed_tokens >= request.num_prompt_tokens
                and request.num_output_tokens > 0
                for request in other_scheduled_reqs
            )
            if (
                split_request.has_encoder_inputs
                and split_request.num_computed_tokens == 0
                and split_request.num_output_tokens == 0
                and split_request.num_output_placeholders == 0
                and len(split_encoder_ids) == 1
                and split_req_id not in scheduled_spec_decode_tokens
                and only_inflight_text_decode
            ):
                deferred_prefill_tokens = num_scheduled_tokens.pop(split_req_id, 0)
                if deferred_prefill_tokens > 0:
                    token_budget += deferred_prefill_tokens
                    input_budget += deferred_prefill_tokens + draft_slots
                    prefill_admitted = False
                    stage_split_encoder_only_req_ids.add(split_req_id)

        encoder_input_update_schedule = dict(num_scheduled_tokens)
        encoder_input_update_schedule.update(
            {req_id: 0 for req_id in stage_split_encoder_only_req_ids}
        )

'''


def patch(path: Path, backup: Path, diff_path: Path) -> None:
    source = path.read_text()
    if "stage_split_encoder_only_req_ids" in source:
        print("scheduler split patch already present")
        return
    if "import itertools\n" not in source:
        raise SystemExit("scheduler import anchor not found")
    marker = "        # Check if the scheduling constraints are satisfied.\n"
    if source.count(marker) != 1:
        raise SystemExit(f"expected one schedule marker, got {source.count(marker)}")
    call_old = '''            encoder_input_updates=self._get_encoder_input_updates(
                num_scheduled_tokens, new_reqs_data, scheduled_encoder_inputs
            ),'''.replace("+", "")
    call_new = '''            encoder_input_updates=self._get_encoder_input_updates(
                encoder_input_update_schedule, new_reqs_data, scheduled_encoder_inputs
            ),'''
    if source.count(call_old) != 1:
        raise SystemExit(f"expected one encoder update call, got {source.count(call_old)}")
    updated = source.replace("import itertools\n", "import itertools\nimport os\n", 1)
    updated = updated.replace(marker, INSERTION + marker, 1)
    updated = updated.replace(call_old, call_new, 1)
    backup.parent.mkdir(parents=True, exist_ok=True)
    diff_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, backup)
    delta = difflib.unified_diff(
        source.splitlines(keepends=True),
        updated.splitlines(keepends=True),
        fromfile=str(backup),
        tofile=str(path),
    )
    diff_path.write_text("".join(delta))
    path.write_text(updated)
    compile(updated, str(path), "exec")
    print(f"patched {path}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("usage: patch_scheduler_split.py SCHEDULER BACKUP DIFF")
    patch(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
