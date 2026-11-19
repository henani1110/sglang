"""Small opt-in hook shared by the vLLM runner and Qwen3-VL model copy."""
from __future__ import annotations

import atexit
import contextvars
from queue import Empty, Full, Queue
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
_ENGINE_EVENTS: list[dict[str, Any]] = []
_ENGINE_EVENTS_LOCK = threading.Lock()
_ENGINE_TOKEN_COUNTS: dict[str, int] = {}
_API_TOKEN_COUNTS: dict[str, int] = {}
_SCHEDULER_EVENTS: list[dict[str, Any]] = []
_SCHEDULER_EVENTS_LOCK = threading.Lock()


class _AsyncJSONLWriter:
    """Bounded, non-blocking producer queue with one background file writer."""

    def __init__(self, path: str, max_queue: int):
        self.path = Path(path)
        self.queue: Queue[tuple[str, Any]] = Queue(maxsize=max_queue)
        self.counter_lock = threading.Lock()
        self.submitted = 0
        self.written = 0
        self.dropped = 0
        self.write_errors = 0
        self.last_error: str | None = None
        self.thread = threading.Thread(
            target=self._run,
            name=f"vl-pilot-jsonl-{self.path.name}",
            daemon=True,
        )
        self.thread.start()

    def submit_many(self, rows: list[dict[str, Any]]) -> None:
        for index, row in enumerate(rows):
            try:
                self.queue.put_nowait(("row", row))
            except Full:
                with self.counter_lock:
                    self.dropped += len(rows) - index
                return
            with self.counter_lock:
                self.submitted += 1

    def flush(self, timeout_s: float = 20.0) -> bool:
        done = threading.Event()
        try:
            self.queue.put(("barrier", done), timeout=timeout_s)
        except Full:
            return False
        return done.wait(timeout_s)

    def close(self, timeout_s: float = 30.0) -> bool:
        if self.thread.is_alive():
            try:
                self.queue.put(("close", None), timeout=min(timeout_s, 10.0))
            except Full:
                return False
            self.thread.join(timeout_s)
        return not self.thread.is_alive()

    def _write_rows(self, fd: int | None, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        try:
            if fd is None:
                raise OSError("async JSONL file did not open")
            payload = "".join(
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                for row in rows
            ).encode("utf-8")
            view = memoryview(payload)
            while view:
                count = os.write(fd, view)
                if count <= 0:
                    raise OSError("short write to async JSONL file")
                view = view[count:]
            with self.counter_lock:
                self.written += len(rows)
        except Exception as exc:
            with self.counter_lock:
                self.write_errors += len(rows)
                self.last_error = f"{type(exc).__name__}: {exc}"

    def _write_summary(self) -> None:
        with self.counter_lock:
            summary = {
                "path": str(self.path),
                "submitted": self.submitted,
                "written": self.written,
                "dropped": self.dropped,
                "write_errors": self.write_errors,
                "queue_capacity": self.queue.maxsize,
                "last_error": self.last_error,
                "closed_cleanly": True,
            }
        target = self.path.with_name(self.path.name + ".async-meta.json")
        temporary = target.with_suffix(target.suffix + ".tmp")
        try:
            temporary.write_text(json.dumps(summary, indent=2) + "\n")
            os.replace(temporary, target)
        except OSError as exc:
            with self.counter_lock:
                self.last_error = f"summary {type(exc).__name__}: {exc}"

    def _run(self) -> None:
        fd: int | None = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        except OSError as exc:
            with self.counter_lock:
                self.last_error = f"open {type(exc).__name__}: {exc}"

        pending: list[dict[str, Any]] = []
        should_close = False
        while not should_close:
            try:
                kind, payload = self.queue.get(timeout=0.1)
            except Empty:
                if pending:
                    self._write_rows(fd, pending)
                    pending.clear()
                continue

            if kind == "row":
                pending.append(payload)
                if len(pending) >= 512:
                    self._write_rows(fd, pending)
                    pending.clear()
                continue

            if pending:
                self._write_rows(fd, pending)
                pending.clear()
            if fd is not None:
                try:
                    os.fsync(fd)
                except OSError as exc:
                    with self.counter_lock:
                        self.write_errors += 1
                        self.last_error = f"fsync {type(exc).__name__}: {exc}"

            if kind == "barrier":
                payload.set()
            elif kind == "close":
                should_close = True

        if fd is not None:
            os.close(fd)
        self._write_summary()


_ASYNC_JSONL_WRITERS: dict[str, _AsyncJSONLWriter] = {}
_ASYNC_JSONL_WRITERS_LOCK = threading.Lock()


def _async_jsonl_writer(path: str) -> _AsyncJSONLWriter:
    with _ASYNC_JSONL_WRITERS_LOCK:
        writer = _ASYNC_JSONL_WRITERS.get(path)
        if writer is None:
            capacity = int(os.environ.get("VL_PILOT_ASYNC_QUEUE", "65536"))
            writer = _AsyncJSONLWriter(path, max_queue=capacity)
            _ASYNC_JSONL_WRITERS[path] = writer
        return writer


def _submit_jsonl_rows(path: str | None, rows: list[dict[str, Any]]) -> bool:
    if not path or not rows or os.environ.get("VL_PILOT_ASYNC_LOG") != "1":
        return False
    _async_jsonl_writer(path).submit_many(rows)
    return True


def _flush_async_jsonl(path: str | None) -> bool:
    if not path:
        return True
    with _ASYNC_JSONL_WRITERS_LOCK:
        writer = _ASYNC_JSONL_WRITERS.get(path)
    return True if writer is None else writer.flush()


def _close_async_jsonl_writers() -> None:
    with _ASYNC_JSONL_WRITERS_LOCK:
        writers = list(_ASYNC_JSONL_WRITERS.values())
    for writer in writers:
        writer.close()


atexit.register(_close_async_jsonl_writers)


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
    if os.environ.get("VL_PILOT_PROBE") != "1":
        return
    path = os.environ.get("VL_PILOT_EVENT_FILE")
    if not path:
        return
    if _submit_jsonl_rows(path, [event]):
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
    if os.environ.get("VL_PILOT_PROBE") != "1":
        return
    ids = [int(token) for token in token_ids]
    now = time.perf_counter_ns()
    with _SERVER_EVENTS_LOCK:
        token_start = _API_TOKEN_COUNTS.get(request_id, 0)
        _API_TOKEN_COUNTS[request_id] = token_start + len(ids)
        _SERVER_EVENTS.append({
            "kind": "api_yield",
            "request_id": request_id,
            "time_ns": now,
            "token_index_start": token_start if ids else None,
            "token_index_end": token_start + len(ids) - 1 if ids else None,
            "token_count": len(ids),
            "token_ids": ids,
            "finish": bool(finish),
        })
        if not finish and len(_SERVER_EVENTS) < 512:
            return
        rows = list(_SERVER_EVENTS)
        _SERVER_EVENTS.clear()
    with _ENGINE_EVENTS_LOCK:
        rows.extend(_ENGINE_EVENTS)
        _ENGINE_EVENTS.clear()
    rows.sort(key=lambda event: int(event["time_ns"]))
    path = os.environ.get("VL_PILOT_EVENT_FILE")
    if path and rows and _submit_jsonl_rows(path, rows):
        return
    if path and rows:
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
def record_engine_result_ready(request_id: str, prompt_idx: int, result: Any) -> None:
    if os.environ.get("VL_PILOT_PROBE") != "1":
        return
    outputs = getattr(result, "outputs", ()) or ()
    now = time.perf_counter_ns()
    with _ENGINE_EVENTS_LOCK:
        for output in outputs:
            ids = [int(token) for token in (getattr(output, "token_ids", ()) or ())]
            if not ids:
                continue
            output_index = int(getattr(output, "index", 0))
            request_key = f"{request_id}-{int(prompt_idx)}-{output_index}"
            token_start = _ENGINE_TOKEN_COUNTS.get(request_key, 0)
            _ENGINE_TOKEN_COUNTS[request_key] = token_start + len(ids)
            _ENGINE_EVENTS.append({
                "kind": "engine_result_ready_to_api",
                "request_id": request_id,
                "engine_request_id": f"{request_id}-{int(prompt_idx)}",
                "output_index": output_index,
                "prompt_idx": int(prompt_idx),
                "time_ns": now,
                "token_index_start": token_start,
                "token_index_end": token_start + len(ids) - 1,
                "token_ids": ids,
                "finish_reason": getattr(output, "finish_reason", None),
            })

def record_scheduler_event(scheduler_output: Any) -> None:
    if os.environ.get("VL_PILOT_PROBE") != "1":
        return
    path = os.environ.get("VL_PILOT_SCHED_FILE")
    if not path:
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
    if _submit_jsonl_rows(path, rows):
        return
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
    if os.environ.get("VL_PILOT_PROBE") != "1" and mode in {"NORMAL", "R"}:
        return compute()
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
    if _submit_jsonl_rows(path, rows):
        if not _flush_async_jsonl(path):
            raise RuntimeError("async scheduler trace did not flush within 20 seconds")
        return len(rows)
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
