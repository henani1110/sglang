#!/usr/bin/env python3
"""Apply the D-validated N/S scheduler split to the isolated E overlay."""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path

EXPECTED_SOURCE_SHA256 = "d76cc22257550972137df9c893505d0bd0f5d482374691e6eca4372772de501a"
IMPORT_ANCHOR = "import itertools\nimport time\n"
IMPORT_REPLACEMENT = "import itertools\nimport os\nimport time\n\nfrom vllm.vl_pilot_hook import record_scheduler_backlog_snapshot\n"
SPLIT_ANCHOR = "        # Check if the scheduling constraints are satisfied.\n"
ENCODER_UPDATE_ANCHOR = """            encoder_input_updates=self._get_encoder_input_updates(
                num_scheduled_tokens, new_reqs_data, scheduled_encoder_inputs
            ),"""
ENCODER_UPDATE_REPLACEMENT = """            encoder_input_updates=self._get_encoder_input_updates(
                encoder_input_update_schedule, new_reqs_data, scheduled_encoder_inputs
            ),"""
BACKLOG_RETURN_ANCHOR = '''        with record_function_or_nullcontext("schedule: update_after_schedule"):
            self._update_after_schedule(scheduler_output)
        return scheduler_output
'''
BACKLOG_RETURN_REPLACEMENT = '''        with record_function_or_nullcontext("schedule: update_after_schedule"):
            self._update_after_schedule(scheduler_output)
        if os.environ.get("VL_PILOT_BACKLOG_SNAPSHOT") == "1":
            snapshot_observer_started_ns = time.perf_counter_ns()
            waiting_by_id = {
                request.request_id: request
                for queue in (self.waiting, self.kv_holding_waiting)
                for request in queue
            }
            waiting_by_id.update(
                {request.request_id: request for request in self.deferred_waiting}
            )
            waiting_encoder = [
                request
                for request in waiting_by_id.values()
                if request.has_encoder_inputs
                and request.num_computed_tokens < request.num_prompt_tokens
            ]
            running_encoder_prefills = [
                request
                for request in self.running
                if request.has_encoder_inputs
                and request.num_computed_tokens < request.num_prompt_tokens
            ]
            active_text_decodes = sum(
                not request.has_encoder_inputs
                and request.num_computed_tokens >= request.num_prompt_tokens
                and request.num_output_tokens > 0
                for request in self.running
            )
            snapshot_row = {
                "kind": "scheduler_backlog_snapshot",
                "time_ns": time.perf_counter_ns(),
                "scheduler_step": int(self.current_step),
                "waiting_requests": int(len(self.waiting) + len(self.kv_holding_waiting)),
                "deferred_waiting_requests": int(len(self.deferred_waiting)),
                "running_requests": int(len(self.running)),
                "waiting_encoder_requests": int(len(waiting_encoder)),
                "running_encoder_prefill_requests": int(len(running_encoder_prefills)),
                "active_text_decodes": int(active_text_decodes),
                "scheduled_encoder_requests": int(len(scheduled_encoder_inputs)),
                "scheduled_encoder_inputs": int(
                    sum(len(input_ids) for input_ids in scheduled_encoder_inputs.values())
                ),
                "scheduled_encoder_token_budget": int(self.max_num_encoder_input_tokens),
            }
            snapshot_row["observer_state_build_ns"] = int(
                time.perf_counter_ns() - snapshot_observer_started_ns
            )
            snapshot_row["previous_snapshot_observer_total_ns"] = int(
                getattr(self, "_e_last_backlog_snapshot_total_ns", 0)
            )
            record_scheduler_backlog_snapshot(snapshot_row)
            self._e_last_backlog_snapshot_total_ns = int(
                time.perf_counter_ns() - snapshot_observer_started_ns
            )
        return scheduler_output
'''
LOGICAL_SOURCE_PATH = "vllm/v1/core/sched/scheduler.py"
LOGICAL_OUTPUT_PATH = "overlay/vllm/v1/core/sched/scheduler.py"
LOGICAL_SNAPSHOT_PATH = "source_snapshot/scheduler_before_e_split.py"
LOGICAL_DIFF_PATH = "patches/scheduler_split_e.diff"

SPLIT_BLOCK = """        # E-only opt-in S split: run one new image encoder input while
        # currently decoding text requests use this LM step; defer image prefill.
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
            other_scheduled_reqs = [
                self.requests.get(req_id) for req_id in other_scheduled_ids
            ]
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

"""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--diff", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    source_bytes = args.source.read_bytes()
    source_sha = sha256(source_bytes)
    if source_sha != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            f"unexpected vLLM scheduler source sha256={source_sha}; "
            f"expected={EXPECTED_SOURCE_SHA256}"
        )
    source = source_bytes.decode("utf-8")
    if "stage_split_encoder_only_req_ids" in source:
        raise RuntimeError("E scheduler S split already exists in source")
    if source.count(IMPORT_ANCHOR) != 1:
        raise RuntimeError("unexpected E scheduler import anchor")
    if source.count(SPLIT_ANCHOR) != 1:
        raise RuntimeError("unexpected E scheduler constraint anchor")
    if source.count(ENCODER_UPDATE_ANCHOR) != 1:
        raise RuntimeError("unexpected E encoder input update call")
    if source.count(BACKLOG_RETURN_ANCHOR) != 1:
        raise RuntimeError("unexpected E scheduler return anchor")

    patched = source.replace(IMPORT_ANCHOR, IMPORT_REPLACEMENT, 1)
    patched = patched.replace(SPLIT_ANCHOR, SPLIT_BLOCK + SPLIT_ANCHOR, 1)
    patched = patched.replace(
        ENCODER_UPDATE_ANCHOR, ENCODER_UPDATE_REPLACEMENT, 1
    )
    patched = patched.replace(BACKLOG_RETURN_ANCHOR, BACKLOG_RETURN_REPLACEMENT, 1)
    patched_bytes = patched.encode("utf-8")

    args.snapshot.parent.mkdir(parents=True, exist_ok=True)
    if args.snapshot.exists():
        if sha256(args.snapshot.read_bytes()) != EXPECTED_SOURCE_SHA256:
            raise RuntimeError(f"snapshot already exists with unexpected content: {args.snapshot}")
    else:
        args.snapshot.write_bytes(source_bytes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.diff.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(patched_bytes)
    diff_text = "".join(
        difflib.unified_diff(
            source.splitlines(True),
            patched.splitlines(True),
            fromfile=f"a/{LOGICAL_SOURCE_PATH}",
            tofile=f"b/{LOGICAL_OUTPUT_PATH}",
        )
    )
    args.diff.write_text(diff_text, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "mode": "E-isolated-D-validated-S-split",
        "source_path": LOGICAL_SOURCE_PATH,
        "source_sha256": source_sha,
        "output_path": LOGICAL_OUTPUT_PATH,
        "output_sha256": sha256(patched_bytes),
        "snapshot_path": LOGICAL_SNAPSHOT_PATH,
        "diff_path": LOGICAL_DIFF_PATH,
        "diff_sha256": sha256(diff_text.encode("utf-8")),
        "split_condition": "one new image request, one encoder input, only in-flight text decode",
        "scope": "N remains unchanged when VL_STAGE_SPLIT_MODE is unset or not S; optional VL_PILOT_BACKLOG_SNAPSHOT emits count-only state by nonblocking UDP and never activates content pruning",
    }
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
