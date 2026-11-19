#!/usr/bin/env python3
"""Fit the frozen D E/P/slack profile from independent control-image runs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSONL {path}:{line_no}: {exc}") from exc
    return rows


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        raise ValueError("percentile requires at least one finite value")
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def request_steps(scheduler_rows: list[dict[str, Any]], response_id: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    encoder_step = None
    prefill_step = None
    for row in scheduler_rows:
        if row.get("kind") != "scheduler_step":
            continue
        encoder_ids = row.get("scheduled_encoder_inputs", {})
        token_ids = row.get("num_scheduled_tokens", {})
        has_encoder = any(str(key).startswith(response_id) for key in encoder_ids)
        has_tokens = any(
            str(key).startswith(response_id) and int(value) > 0
            for key, value in token_ids.items()
        )
        if has_encoder and encoder_step is None:
            encoder_step = row
        if has_tokens and prefill_step is None:
            prefill_step = row
        if encoder_step is not None and prefill_step is not None:
            break
    return encoder_step, prefill_step


def cuda_durations(events: list[dict[str, Any]], phase: str) -> list[float]:
    values: list[float] = []
    for row in events:
        if row.get("kind") != "cuda_event_flush" or row.get("phase") != phase:
            continue
        response = row.get("response", {})
        for result in response.get("results", []):
            for event in result.get("cuda_events", []):
                value = float(event.get("cuda_elapsed_ms", 0.0))
                if value > 0 and math.isfinite(value):
                    values.append(value)
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output-profile", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--required-calibration-seeds", type=int, default=4)
    args = parser.parse_args()

    index = json.loads(args.index.read_text(encoding="utf-8"))
    runs = list(index.get("runs", []))
    cells: dict[str, dict[str, Any]] = {}
    run_errors: list[dict[str, Any]] = []
    valid_by_seed: dict[tuple[int, int, str, str, int], dict[str, Any]] = {}
    expected_keys = {
        (grid, lanes, arrival, policy, seed)
        for grid in (512, 768)
        for lanes in (2, 6)
        for arrival in ("uniform", "burst")
        for policy in ("N", "S")
        for seed in range(300, 304)
    }
    for run in runs:
        run_id = str(run.get("run_id", "unknown"))
        try:
            seed_key = (
                int(run["image_size"]),
                int(run["text_streams"]),
                str(run["arrival_mode"]),
                str(run["policy"]),
                int(run["seed"]),
            )
            result_path = Path(run["result_path"])
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            run_errors.append({"run_id": run_id, "reason": "result_missing_or_invalid", "detail": str(exc)})
            continue
        if seed_key not in expected_keys:
            run_errors.append({"run_id": run_id, "reason": "outside_preregistered_calibration_grid", "key": list(seed_key)})
            continue
        if result.get("status") != "completed" or int(run.get("process_exit", 1)) != 0:
            run_errors.append({"run_id": run_id, "reason": "run_not_successfully_completed", "status": result.get("status"), "process_exit": run.get("process_exit")})
            continue
        if run.get("analysis_eligible") is not True:
            run_errors.append({"run_id": run_id, "reason": "trace_not_analysis_eligible", "trace_qa": run.get("trace_qa")})
            continue
        missing_events = [
            field for field in ("client_events_path", "scheduler_events_path", "server_events_path")
            if not Path(run.get(field, "")).is_file() or Path(run.get(field, "")).stat().st_size == 0
        ]
        if missing_events:
            run_errors.append({"run_id": run_id, "reason": "event_files_missing_or_empty", "fields": missing_events})
            continue
        if seed_key in valid_by_seed:
            run_errors.append({"run_id": run_id, "reason": "duplicate_success_excluded", "key": list(seed_key), "included_run_id": valid_by_seed[seed_key]["run_id"]})
            continue
        valid_by_seed[seed_key] = run

    missing_keys = sorted(expected_keys - set(valid_by_seed))
    if missing_keys:
        incomplete = {
            "status": "incomplete",
            "expected_runs": len(expected_keys),
            "valid_unique_runs": len(valid_by_seed),
            "missing_seed_policy_keys": [list(key) for key in missing_keys],
            "run_errors": run_errors,
            "indexed_runs": len(runs),
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(incomplete, sort_keys=True, ensure_ascii=False, indent=2) + "\n")
        raise RuntimeError(f"calibration analysis requires all 64 successful seed-policy runs; found {len(valid_by_seed)}/64; see {args.report}")

    for run in valid_by_seed.values():
        run_id = run["run_id"]
        key = f"{int(run['image_size'])}:{int(run['text_streams'])}"
        cell = cells.setdefault(
            key,
            {
                "e_samples": [],
                "p_samples": [],
                "decode_samples": [],
                "split_samples": [],
                "run_ids": [],
                "by_mode": {},
            },
        )
        result_path = Path(run["result_path"])
        client_path = Path(run["client_events_path"])
        scheduler_path = Path(run["scheduler_events_path"])
        server_path = Path(run["server_events_path"])
        result = json.loads(result_path.read_text(encoding="utf-8"))
        client_rows = read_jsonl(client_path)
        scheduler_rows = read_jsonl(scheduler_path)
        server_rows = read_jsonl(server_path)
        cell["run_ids"].append(run_id)
        mode = str(run["arrival_mode"])
        mode_bucket = cell["by_mode"].setdefault(mode, {"e_samples": 0, "p_samples": 0, "split_samples": 0})

        e_samples = cuda_durations(client_rows, "measurement")
        cell["e_samples"].extend(e_samples)
        mode_bucket["e_samples"] += len(e_samples)

        phase_end = next(
            (int(row["time_ns"]) for row in reversed(client_rows)
             if row.get("kind") == "phase_end" and row.get("phase") == "measurement"),
            None,
        )
        measure_start = result.get("measurement_start_ns")
        text_response_ids = {
            str(row["request_id"])
            for row in client_rows
            if row.get("kind") == "text_response" and row.get("request_id")
        }
        decode_times_by_request: dict[str, list[int]] = {request_id: [] for request_id in text_response_ids}
        for step in scheduler_rows:
            if step.get("kind") != "scheduler_step":
                continue
            stamp = int(step["time_ns"])
            if measure_start is not None and stamp < int(measure_start):
                continue
            if phase_end is not None and stamp > phase_end:
                continue
            for request_id, count in step.get("num_scheduled_tokens", {}).items():
                if int(count) != 1:
                    continue
                for response_id in text_response_ids:
                    if str(request_id).startswith(response_id):
                        decode_times_by_request[response_id].append(stamp)
                        break
        text_gaps: list[float] = []
        for stamps in decode_times_by_request.values():
            ordered = sorted(set(stamps))
            text_gaps.extend((after - before) / 1e6 for before, after in zip(ordered, ordered[1:]) if after > before)
        cell["decode_samples"].extend(text_gaps)

        measurement_images = [
            row for row in client_rows
            if row.get("kind") == "image_response"
            and row.get("phase") == "measurement"
            and not row.get("error")
            and row.get("request_id")
            and row.get("actual_send_ns")
            and row.get("first_token_ns")
        ]
        for image in measurement_images:
            encoder_step, prefill_step = request_steps(scheduler_rows, str(image["request_id"]))
            if encoder_step is None or prefill_step is None:
                run_errors.append(
                    {"run_id": run_id, "request_id": image.get("request_id"), "reason": "scheduler_request_unmatched"}
                )
                continue
            response_id = str(image["request_id"])
            engine_rows = [
                row for row in server_rows
                if row.get("kind") == "engine_result_ready_to_api"
                and str(row.get("request_id", "")) == response_id
                and int(row.get("token_index_start", -1)) == 0
            ]
            api_rows = [
                row for row in server_rows
                if row.get("kind") == "api_yield"
                and str(row.get("request_id", "")) == response_id
                and int(row.get("token_index_start", -1)) == 0
            ]
            if not engine_rows:
                run_errors.append(
                    {"run_id": run_id, "request_id": image.get("request_id"), "reason": "engine_first_token_unmatched"}
                )
                continue
            if not api_rows:
                run_errors.append(
                    {"run_id": run_id, "request_id": image.get("request_id"), "reason": "api_first_token_unmatched"}
                )
                continue
            engine_first = min(engine_rows, key=lambda row: int(row["time_ns"]))
            api_first = min(api_rows, key=lambda row: int(row["time_ns"]))
            client_token_ids = [int(token) for token in image.get("token_ids", [])]
            engine_token_ids = [int(token) for token in engine_first.get("token_ids", [])]
            api_token_ids = [int(token) for token in api_first.get("token_ids", [])]
            if (
                not client_token_ids
                or not engine_token_ids
                or not api_token_ids
                or engine_token_ids[0] != client_token_ids[0]
                or api_token_ids[0] != client_token_ids[0]
            ):
                run_errors.append(
                    {
                        "run_id": run_id,
                        "request_id": response_id,
                        "reason": "first_token_id_mismatch",
                        "client_token_id": client_token_ids[0] if client_token_ids else None,
                        "engine_token_id": engine_token_ids[0] if engine_token_ids else None,
                        "api_token_id": api_token_ids[0] if api_token_ids else None,
                    }
                )
                continue
            engine_ready_ns = int(engine_first["time_ns"])
            api_yield_ns = int(api_first["time_ns"])
            client_first_token_ns = int(image["first_token_ns"])
            if not engine_ready_ns <= api_yield_ns <= client_first_token_ns:
                run_errors.append(
                    {
                        "run_id": run_id,
                        "request_id": response_id,
                        "reason": "first_token_event_order_invalid",
                        "engine_ready_ns": engine_ready_ns,
                        "api_yield_ns": api_yield_ns,
                        "client_first_token_ns": client_first_token_ns,
                    }
                )
                continue
            queue_ms = (int(encoder_step["time_ns"]) - int(image["actual_send_ns"])) / 1e6
            if queue_ms < -2.0:
                run_errors.append(
                    {"run_id": run_id, "request_id": image.get("request_id"), "reason": "clock_domain_mismatch", "queue_ms": queue_ms}
                )
                continue
            engine_after_prefill_start_ms = (engine_ready_ns - int(prefill_step["time_ns"])) / 1e6
            sample = {
                "run_id": run_id,
                "mode": mode,
                "policy": run["policy"],
                "queue_ms": max(0.0, queue_ms),
                "engine_after_prefill_start_ms": engine_after_prefill_start_ms,
                "engine_ready_ns": engine_ready_ns,
                "api_yield_ns": api_yield_ns,
                "client_first_token_ns": client_first_token_ns,
                "encoder_step_ns": int(encoder_step["time_ns"]),
                "prefill_step_ns": int(prefill_step["time_ns"]),
            }
            cell.setdefault("image_samples", []).append(sample)
            mode_bucket["p_samples"] += 1
            if run["policy"] == "S":
                cell["split_samples"].append(
                    {
                        "run_id": run_id,
                        "mode": mode,
                        "gap_ms_before_e": (int(prefill_step["time_ns"]) - int(encoder_step["time_ns"])) / 1e6,
                    }
                )
                mode_bucket["split_samples"] += 1

    profile_cells: dict[str, dict[str, Any]] = {}
    report_cells: dict[str, Any] = {}
    expected_profile_keys = {f"{grid}:{lanes}" for grid in (512, 768) for lanes in (2, 6)}
    missing = sorted(expected_profile_keys - set(cells))
    if missing:
        raise RuntimeError(f"calibration index is missing profile cells: {missing}")

    for key in sorted(expected_profile_keys):
        cell = cells[key]
        grid, lanes = (int(value) for value in key.split(":"))
        if len(cell["run_ids"]) < args.required_calibration_seeds * 4:
            raise RuntimeError(
                f"cell {key} has {len(cell['run_ids'])} completed runs; expected at least "
                f"{args.required_calibration_seeds * 4} runs (both modes and N/S)"
            )
        e_values = cell["e_samples"]
        if not e_values:
            raise RuntimeError(f"cell {key} has no measurement CUDA event samples")
        e_hat = percentile(e_values, 0.75)
        image_samples = cell.get("image_samples", [])
        p_values = [
            max(0.0, sample["engine_after_prefill_start_ms"] - e_hat)
            for sample in image_samples if sample["policy"] == "N"
        ]
        split_values = [
            max(0.0, sample["gap_ms_before_e"] - e_hat)
            for sample in cell["split_samples"]
        ]
        d_values = cell["decode_samples"]
        if not p_values or not split_values or not d_values:
            raise RuntimeError(
                f"cell {key} lacks required P/split/D samples: P={len(p_values)}, "
                f"split={len(split_values)}, D={len(d_values)}"
            )
        profile_cells[key] = {
            "e_hat_ms": e_hat,
            "p_hat_ms": percentile(p_values, 0.75),
            "decode_step_hat_ms": percentile(d_values, 0.75),
            "split_gap_hat_ms": percentile(split_values, 0.75),
            "grid": grid,
            "active_text_streams": lanes,
        }
        report_cells[key] = {
            **profile_cells[key],
            "sample_counts": {
                "E_cuda_events": len(e_values),
                "P_after_prefill": len(p_values),
                "decode_intertoken_gaps": len(d_values),
                "split_gap": len(split_values),
            },
            "run_ids": cell["run_ids"],
            "by_arrival_mode": cell["by_mode"],
            "E_samples_ms": e_values,
            "P_samples_ms": p_values,
            "decode_samples_ms": d_values,
            "split_samples_ms": split_values,
        }

    profile = {
        "schema_version": 1,
        "stage": "D independent calibration",
        "frozen": True,
        "estimator": {
            "quantile": 0.75,
            "E": "75th percentile CUDA-event visual-forward duration",
            "P": "75th percentile engine first-token-ready time after scheduler prefill start, minus E; scheduler queue and API/SSE delay excluded",
            "decode_step": "75th percentile positive per-request scheduler-step interval while that text request decodes",
            "split_gap": "75th percentile S prefill-step minus E-step interval, minus E; includes concurrent D and scheduler handoff",
        },
        "decision_deadlines_ms": {"text_gap": 30.0, "image_ttft": 55.0},
        "calibration_index": str(args.index),
        "calibration_run_ids": sorted(run_id for cell in cells.values() for run_id in cell["run_ids"]),
        "cells": profile_cells,
    }
    args.output_profile.parent.mkdir(parents=True, exist_ok=True)
    profile_bytes = (json.dumps(profile, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode()
    args.output_profile.write_bytes(profile_bytes)
    profile_sha = hashlib.sha256(profile_bytes).hexdigest()
    args.output_profile.with_suffix(args.output_profile.suffix + ".sha256").write_text(profile_sha + "  " + args.output_profile.name + "\n")

    report = {
        "status": "frozen",
        "profile_path": str(args.output_profile),
        "profile_sha256": profile_sha,
        "cells": report_cells,
        "run_errors": run_errors,
        "completed_runs": sum(len(cell["run_ids"]) for cell in cells.values()),
        "indexed_runs": len(runs),
        "valid_unique_seed_policy_runs": len(valid_by_seed),
        "limitations": [
            "P is an operational post-E engine estimate conditioned on the calibration serving path, not an isolated GEMM duration.",
            "The calibration is a geometry-control profile and must remain fixed during the holdout matrix.",
            "Per-request decisions and observed outcomes will be reported separately; no formal holdout data is used to refit this profile.",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "frozen", "profile": str(args.output_profile), "sha256": profile_sha, "cells": len(profile_cells), "runs": len(runs)}))


if __name__ == "__main__":
    main()
