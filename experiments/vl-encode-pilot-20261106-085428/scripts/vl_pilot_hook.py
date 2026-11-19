"""Small opt-in hook shared by the vLLM runner and Qwen3-VL model copy."""
from __future__ import annotations

import contextvars
import hashlib
import json
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

import torch

_MM_KEYS: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
    "vl_pilot_mm_keys", default=()
)
_FEATURES: dict[str, torch.Tensor] = {}
_FEATURES_LOCK = threading.Lock()
_EVENTS: list[tuple[torch.cuda.Event, torch.cuda.Event, dict[str, Any]]] = []
_EVENTS_LOCK = threading.Lock()
_SERVER_EVENTS: list[dict[str, Any]] = []
_SERVER_EVENTS_LOCK = threading.Lock()
_SCHEDULER_EVENTS: list[dict[str, Any]] = []
_SCHEDULER_EVENTS_LOCK = threading.Lock()


@contextmanager
def mm_keys_context(keys: list[str] | tuple[str, ...]) -> Iterator[None]:
    token = _MM_KEYS.set(tuple(str(key) for key in keys))
    try:
        yield
    finally:
        _MM_KEYS.reset(token)


def _control() -> dict[str, Any]:
    path = os.environ.get("VL_PILOT_CONTROL_FILE")
    if not path:
        return {"mode": "NORMAL", "verify": False, "diagnostic": False}
    try:
        value = json.loads(Path(path).read_text())
        if isinstance(value, dict):
            return value
    except (OSError, json.JSONDecodeError):
        pass
    return {"mode": "NORMAL", "verify": False, "diagnostic": False}


def _feature_path(key: str) -> Path:
    if not key or len(key) > 512 or "/" in key or "\\" in key:
        raise RuntimeError(f"invalid vLLM multimodal cache key: {key!r}")
    root = os.environ.get("VL_PILOT_FEATURE_DIR")
    if not root:
        raise RuntimeError("VL_PILOT_FEATURE_DIR is not set")
    filename = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return Path(root) / f"{filename}.pt"


def _load_feature(key: str, device: torch.device) -> torch.Tensor:
    with _FEATURES_LOCK:
        feature = _FEATURES.get(key)
        if feature is None:
            path = _feature_path(key)
            if not path.is_file():
                raise RuntimeError(f"visual feature miss for mm_hash={key}")
            feature = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(feature, torch.Tensor):
                raise RuntimeError(f"invalid feature file for mm_hash={key}")
            feature = feature.to(device=device, non_blocking=True).contiguous()
            _FEATURES[key] = feature
        elif feature.device != device:
            feature = feature.to(device=device, non_blocking=True).contiguous()
            _FEATURES[key] = feature
    return feature


def _feature_key(state: dict[str, Any], mm_hash: str) -> str:
    """Use the fixed pilot dataset index across request-local vLLM hashes."""
    image_index = state.get("image_index")
    if not isinstance(image_index, int) or image_index < 0:
        raise RuntimeError(
            f"pilot mode requires a non-negative image_index; got {image_index!r} "
            f"for mm_hash={mm_hash}"
        )
    return f"pilot-image-{image_index}"


def _append_event(event: dict[str, Any]) -> None:
    path = os.environ.get("VL_PILOT_EVENT_FILE")
    if not path:
        return
    line = json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a") as handle:
        handle.write(line)


def record_server_token_event(
    request_id: str,
    token_ids: list[int] | tuple[int, ...],
    prompt_token_ids: list[int] | tuple[int, ...] | None,
    finish: bool,
) -> None:
    row = {
        "kind": "server_token_ready",
        "request_id": request_id,
        "time_ns": time.perf_counter_ns(),
        "token_ids": [int(token) for token in token_ids],
    }
    if prompt_token_ids is not None:
        row["prompt_token_ids"] = [int(token) for token in prompt_token_ids]
    with _SERVER_EVENTS_LOCK:
        _SERVER_EVENTS.append(row)
        if not finish and len(_SERVER_EVENTS) < 128:
            return
        rows = list(_SERVER_EVENTS)
        _SERVER_EVENTS.clear()
    path = os.environ.get("VL_PILOT_EVENT_FILE")
    if path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = "".join(
            json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
            for event in rows
        ).encode()
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, payload)
        finally:
            os.close(fd)


def record_scheduler_event(scheduler_output: Any) -> None:
    if not os.environ.get("VL_PILOT_SCHED_FILE"):
        return
    row = {
        "kind": "scheduler_step",
        "time_ns": time.perf_counter_ns(),
        "total_num_scheduled_tokens": int(scheduler_output.total_num_scheduled_tokens),
        "num_scheduled_tokens": {
            str(key): int(value)
            for key, value in scheduler_output.num_scheduled_tokens.items()
        },
        "scheduled_encoder_inputs": {
            str(key): [int(index) for index in value]
            for key, value in scheduler_output.scheduled_encoder_inputs.items()
        },
        "scheduled_spec_decode_tokens": {
            str(key): [int(token) for token in value]
            for key, value in scheduler_output.scheduled_spec_decode_tokens.items()
        },
        "finished_req_ids": sorted(str(key) for key in scheduler_output.finished_req_ids),
        "free_encoder_mm_hashes": [str(key) for key in scheduler_output.free_encoder_mm_hashes],
    }
    with _SCHEDULER_EVENTS_LOCK:
        _SCHEDULER_EVENTS.append(row)
        if len(_SCHEDULER_EVENTS) < 96:
            return
        rows = list(_SCHEDULER_EVENTS)
        _SCHEDULER_EVENTS.clear()
    path = os.environ["VL_PILOT_SCHED_FILE"]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
        for event in rows
    ).encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, payload)
    finally:
        os.close(fd)


def pilot_visual_forward(
    model: Any,
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor | list[list[int]],
    compute: Callable[[], torch.Tensor],
) -> torch.Tensor:
    state = _control()
    mode = str(state.get("mode", "NORMAL")).upper()
    keys = _MM_KEYS.get()
    if mode not in {"NORMAL", "CAPTURE", "VERIFY", "R", "B"}:
        raise RuntimeError(f"unknown VL pilot mode: {mode}")
    key = keys[0] if len(keys) == 1 else None
    if mode != "NORMAL" and key is None:
        raise RuntimeError(f"expected one visual mm_hash, got {keys!r}")
    feature_key = _feature_key(state, key) if mode in {"CAPTURE", "VERIFY", "B"} and key else None

    start_ns = time.perf_counter_ns()
    event_start = event_end = None
    if state.get("diagnostic", False):
        event_start = torch.cuda.Event(enable_timing=True)
        event_end = torch.cuda.Event(enable_timing=True)
        event_start.record()
        torch.cuda.nvtx.range_push(f"vl_pilot_visual_{mode}")

    profiler = None
    trace_path = None
    if state.get("torch_profile", False):
        trace_root = os.environ.get("VL_PILOT_TRACE_DIR")
        if not trace_root:
            raise RuntimeError("VL_PILOT_TRACE_DIR is not set for torch profiler capture")
        trace_label = f"{state.get('episode_id', 'episode')}-{mode}-{state.get('image_index', 'unknown')}"
        trace_name = hashlib.sha256(trace_label.encode("utf-8")).hexdigest()[:20]
        trace_path = Path(trace_root) / f"{trace_name}.json"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=False,
            with_stack=False,
        )
        profiler.start()

    if mode == "B":
        assert feature_key is not None
        output = _load_feature(feature_key, pixel_values.device)
        cache_hit = True
        check: dict[str, Any] = {"pass": True, "kind": "resident_feature"}
    else:
        output = compute()
        cache_hit = False
        check = {"pass": None}
        if mode == "CAPTURE":
            assert feature_key is not None
            path = _feature_path(feature_key)
            path.parent.mkdir(parents=True, exist_ok=True)
            cpu_feature = output.detach().to("cpu").contiguous()
            torch.save(cpu_feature, path)
            feature_sha256 = hashlib.sha256(
                cpu_feature.view(torch.uint8).numpy().tobytes()
            ).hexdigest()
            with _FEATURES_LOCK:
                _FEATURES[feature_key] = output.detach().contiguous()
            check = {"pass": True, "kind": "capture", "feature_sha256": feature_sha256}
        elif mode == "VERIFY":
            assert feature_key is not None
            reference = _load_feature(feature_key, output.device)
            delta = (output.float() - reference.float()).abs()
            max_abs = float(delta.max().item()) if delta.numel() else 0.0
            exact = bool(torch.equal(output, reference))
            close = bool(torch.allclose(output, reference, rtol=1e-3, atol=1e-3))
            check = {"pass": close, "bitwise_equal": exact, "max_abs_diff": max_abs}
            if not close:
                raise RuntimeError(f"R feature mismatch for mm_hash={key}: {check}")

    if event_end is not None:
        torch.cuda.nvtx.range_pop()
        event_end.record()
        with _EVENTS_LOCK:
            _EVENTS.append((event_start, event_end, {"mode": mode, "mm_hash": key}))

    if profiler is not None and trace_path is not None:
        profiler.stop()
        profiler.export_chrome_trace(str(trace_path))

    _append_event(
        {
            "kind": "visual_forward",
            "mode": mode,
            "mm_hash": key,
            "feature_key": feature_key,
            "image_index": state.get("image_index"),
            "torch_profile_trace": str(trace_path) if trace_path else None,
            "time_ns": start_ns,
            "cpu_elapsed_ms": (time.perf_counter_ns() - start_ns) / 1e6,
            "pixel_shape": list(pixel_values.shape),
            "pixel_dtype": str(pixel_values.dtype),
            "pixel_device": str(pixel_values.device),
            "grid_shape": list(grid_thw.shape) if isinstance(grid_thw, torch.Tensor) else None,
            "output_shape": list(output.shape),
            "output_dtype": str(output.dtype),
            "output_device": str(output.device),
            "native_cache_hit": cache_hit,
            "check": check,
        }
    )
    return output


def flush_cuda_events() -> list[dict[str, Any]]:
    torch.cuda.synchronize()
    with _EVENTS_LOCK:
        events = list(_EVENTS)
        _EVENTS.clear()
    return [
        {
            **metadata,
            "cuda_elapsed_ms": start.elapsed_time(end),
        }
        for start, end, metadata in events
    ]


def flush_scheduler_events() -> int:
    path = os.environ.get("VL_PILOT_SCHED_FILE")
    if not path:
        return 0
    with _SCHEDULER_EVENTS_LOCK:
        rows = list(_SCHEDULER_EVENTS)
        _SCHEDULER_EVENTS.clear()
    if not rows:
        return 0
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
        for event in rows
    ).encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, payload)
    finally:
        os.close(fd)
    return len(rows)
