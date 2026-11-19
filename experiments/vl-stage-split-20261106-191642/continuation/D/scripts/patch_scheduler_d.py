#!/usr/bin/env python3
"""Build a D-only adaptive scheduler overlay from the frozen source snapshot."""
from __future__ import annotations

import argparse
import difflib
import hashlib
from pathlib import Path

IMPORT_ANCHOR = "import itertools\nimport os\nimport time\n"
IMPORT_REPLACEMENT = "import hashlib\nimport itertools\nimport json\nimport os\nimport time\n"
INIT_ANCHOR = "        self.use_v2_model_runner = vllm_config.use_v2_model_runner\n"
INIT_REPLACEMENT = INIT_ANCHOR + """        # D adaptive policy state, opt-in and isolated from A/B/C.
        self._d_last_text_token_ns: dict[str, int] = {}
        self._d_action_file = None
        self._d_profile = None
        self._d_profile_sha256 = None
        if os.environ.get("VL_STAGE_SPLIT_MODE") == "A":
            if not self.use_v2_model_runner:
                raise RuntimeError("D policy A requires the validated V2 model runner")
            profile_path = os.environ.get("VL_STAGE_SPLIT_PROFILE")
            decision_path = os.environ.get("VL_STAGE_SPLIT_DECISIONS")
            if not profile_path or not decision_path:
                raise RuntimeError("A requires profile and decision-log paths")
            with open(profile_path, encoding="utf-8") as profile_file:
                profile_bytes = profile_file.read()
            self._d_profile = json.loads(profile_bytes)
            self._d_profile_sha256 = hashlib.sha256(profile_bytes.encode()).hexdigest()
            os.makedirs(os.path.dirname(decision_path) or ".", exist_ok=True)
            self._d_action_file = open(decision_path, "a", encoding="utf-8", buffering=1)
            logger.info("D policy A loaded frozen profile sha256=%s", self._d_profile_sha256)
"""
TOKEN_ANCHOR = """                new_token_ids, stopped = self._update_request_with_output(
                    request, new_token_ids, is_stale=output_is_stale
                )
"""
TOKEN_REPLACEMENT = TOKEN_ANCHOR + """                if (
                    not output_is_stale
                    and new_token_ids
                    and not request.has_encoder_inputs
                ):
                    self._d_last_text_token_ns[req_id] = time.monotonic_ns()
"""
ADAPTIVE_BLOCK = """        # D policy A chooses N/S per newly admitted image request.
        if (
            self.use_v2_model_runner
            and os.environ.get("VL_STAGE_SPLIT_MODE") == "A"
            and not self.is_encoder_decoder
        ):
            adaptive_images = [req for req in scheduled_new_reqs if req.has_encoder_inputs]
            adaptive_image_ids = {req.request_id for req in adaptive_images}
            other_scheduled_ids = [
                req_id for req_id, count in num_scheduled_tokens.items()
                if count > 0 and req_id not in adaptive_image_ids
            ]
            other_scheduled_reqs = [self.requests.get(req_id) for req_id in other_scheduled_ids]
            only_inflight_text_decode = bool(other_scheduled_reqs) and all(
                req is not None
                and not req.has_encoder_inputs
                and req.num_computed_tokens >= req.num_prompt_tokens
                and req.num_output_tokens > 0
                for req in other_scheduled_reqs
            )
            now_ns = time.monotonic_ns()
            active_texts = [
                (req, self._d_last_text_token_ns.get(req.request_id))
                for req in self.requests.values()
                if not req.has_encoder_inputs
                and req.status == RequestStatus.RUNNING
                and req.num_output_tokens > 0
                and not req.is_finished()
                and self._d_last_text_token_ns.get(req.request_id) is not None
            ]
            active_text_ages = [
                (req.request_id, max(0.0, (now_ns - last_ns) / 1e6))
                for req, last_ns in active_texts
            ]
            lane_bucket = 2 if len(active_text_ages) <= 4 else 6

            for image_req in adaptive_images:
                req_id = image_req.request_id
                encoder_ids = scheduled_encoder_inputs.get(req_id, [])
                feature_tokens = sum(
                    image_req.get_num_encoder_embeds(i)
                    for i in range(image_req.num_encoder_inputs)
                )
                image_grid = 512 if feature_tokens <= 256 else 768
                profile_key = f"{image_grid}:{lane_bucket}"
                profile_cell = self._d_profile["cells"].get(profile_key)
                if profile_cell is None:
                    raise RuntimeError(f"frozen D profile has no cell {profile_key}")
                e_hat_ms = float(profile_cell["e_hat_ms"])
                p_hat_ms = float(profile_cell["p_hat_ms"])
                decode_step_hat_ms = float(profile_cell["decode_step_hat_ms"])
                split_gap_hat_ms = float(profile_cell["split_gap_hat_ms"])
                image_age_ms = max(0.0, (time.time() - image_req.arrival_time) * 1000.0)
                text_deadline_base = (
                    min(30.0 - age_ms for _, age_ms in active_text_ages)
                    if active_text_ages else 1e6
                )
                d_slack_n = text_deadline_base - e_hat_ms - p_hat_ms - decode_step_hat_ms
                d_slack_s = text_deadline_base - e_hat_ms - decode_step_hat_ms
                i_slack_n = 55.0 - image_age_ms - e_hat_ms - p_hat_ms
                i_slack_s = 55.0 - image_age_ms - e_hat_ms - split_gap_hat_ms - p_hat_ms

                if d_slack_n >= 0 and i_slack_n >= 0:
                    selected_action = "N"
                elif d_slack_s >= 0 and i_slack_s >= 0:
                    selected_action = "S"
                else:
                    cost_n = max(0.0, -d_slack_n) / 30.0 + max(0.0, -i_slack_n) / 55.0
                    cost_s = max(0.0, -d_slack_s) / 30.0 + max(0.0, -i_slack_s) / 55.0
                    selected_action = "S" if cost_s < cost_n else "N"

                eligible = (
                    bool(encoder_ids)
                    and len(adaptive_images) == 1
                    and len(scheduled_encoder_inputs) == 1
                    and len(encoder_ids) == 1
                    and image_req.num_computed_tokens == 0
                    and image_req.num_output_tokens == 0
                    and image_req.num_output_placeholders == 0
                    and req_id not in scheduled_spec_decode_tokens
                    and not scheduled_resumed_reqs
                    and only_inflight_text_decode
                )
                reason = "eligible"
                if not eligible:
                    selected_action = "N"
                    reason = "S-ineligible:requires-one-new-image-and-only-inflight-text-decode"

                decision = {
                    "kind": "d_policy_decision",
                    "time_ns": now_ns,
                    "request_id": req_id,
                    "selected_action": selected_action,
                    "eligible_for_split": eligible,
                    "reason": reason,
                    "image_grid": image_grid,
                    "encoder_feature_tokens": feature_tokens,
                    "active_text_request_ids": [req_id for req_id, _ in active_text_ages],
                    "active_text_ages_ms": [age for _, age in active_text_ages],
                    "image_queue_age_ms": image_age_ms,
                    "e_hat_ms": e_hat_ms,
                    "p_hat_ms": p_hat_ms,
                    "decode_step_hat_ms": decode_step_hat_ms,
                    "split_gap_hat_ms": split_gap_hat_ms,
                    "d_slack_n_ms": d_slack_n if active_text_ages else None,
                    "d_slack_s_ms": d_slack_s if active_text_ages else None,
                    "i_slack_n_ms": i_slack_n,
                    "i_slack_s_ms": i_slack_s,
                    "profile_sha256": self._d_profile_sha256,
                }
                self._d_action_file.write(
                    json.dumps(decision, sort_keys=True, separators=(",", ":")) + "\\n"
                )
                self._d_action_file.flush()

                if selected_action == "S":
                    deferred_prefill_tokens = num_scheduled_tokens.pop(req_id, 0)
                    if deferred_prefill_tokens > 0:
                        token_budget += deferred_prefill_tokens
                        input_budget += deferred_prefill_tokens + draft_slots
                        stage_split_encoder_only_req_ids.add(req_id)
"""
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diff", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.read_text()
    original = source
    if "d_policy_decision" in source:
        raise RuntimeError("D adaptive scheduler patch is already present")
    if source.count(IMPORT_ANCHOR) != 1:
        raise RuntimeError("unexpected scheduler import anchor")
    source = source.replace(IMPORT_ANCHOR, IMPORT_REPLACEMENT, 1)
    if source.count(INIT_ANCHOR) != 1:
        raise RuntimeError("unexpected scheduler V2 init anchor")
    source = source.replace(INIT_ANCHOR, INIT_REPLACEMENT, 1)
    if source.count(TOKEN_ANCHOR) != 1:
        raise RuntimeError("unexpected scheduler output update anchor")
    source = source.replace(TOKEN_ANCHOR, TOKEN_REPLACEMENT, 1)
    split_anchor = "        encoder_input_update_schedule = dict(num_scheduled_tokens)\n"
    if source.count(split_anchor) != 1:
        raise RuntimeError("unexpected scheduler split insertion anchor")
    source = source.replace(split_anchor, ADAPTIVE_BLOCK + split_anchor, 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.diff.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(source)
    diff_text = "".join(difflib.unified_diff(
        original.splitlines(True), source.splitlines(True),
        fromfile=str(args.source), tofile=str(args.output)
    ))
    args.diff.write_text(diff_text)
    print("output=" + str(args.output))
    print("diff=" + str(args.diff))
    print("sha256=" + hashlib.sha256(source.encode()).hexdigest())


if __name__ == "__main__":
    main()

