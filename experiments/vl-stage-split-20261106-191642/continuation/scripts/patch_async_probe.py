#!/usr/bin/env python3
"""Patch the isolated vLLM trace hook to defer JSONL I/O to a bounded queue."""
from __future__ import annotations

import argparse
import difflib
import shutil
from pathlib import Path


ASYNC_HELPERS = r'''

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
'''


APPEND_EVENT = r'''def _append_event(event: dict[str, Any]) -> None:
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


'''


SERVER_EVENT = r'''def record_server_token_event(
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


'''


SCHEDULER_EVENT = r'''def record_scheduler_event(scheduler_output: Any) -> None:
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


'''


FLUSH_SCHEDULER = r'''def flush_scheduler_events() -> int:
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
'''


def replace_function(source: str, name: str, replacement: str, next_name: str | None) -> str:
    start_marker = f"def {name}("
    start = source.find(start_marker)
    if start < 0:
        raise RuntimeError(f"missing function {name}")
    if next_name is None:
        return source[:start] + replacement
    next_marker = f"\ndef {next_name}("
    end = source.find(next_marker, start)
    if end < 0:
        raise RuntimeError(f"missing end marker {next_name} after {name}")
    return source[:start] + replacement.rstrip() + source[end:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()

    hook = (args.package / "vl_pilot_hook.py").resolve()
    run_root = args.run_root.resolve()
    if "vl-stage-split-20261106-191642" not in str(hook):
        raise RuntimeError(f"refusing to patch package outside this run: {hook}")
    if not hook.is_file():
        raise FileNotFoundError(hook)

    source = hook.read_text()
    if "class _AsyncJSONLWriter" in source:
        raise RuntimeError("async JSONL patch is already present")
    original = source

    if "import atexit\n" not in source:
        source = source.replace(
            "import contextvars\n",
            "import atexit\nimport contextvars\nfrom queue import Empty, Full, Queue\n",
            1,
        )
    marker = "_SCHEDULER_EVENTS_LOCK = threading.Lock()\n"
    if source.count(marker) != 1:
        raise RuntimeError("probe hook lock anchor is not unique")
    source = source.replace(marker, marker + ASYNC_HELPERS, 1)

    source = replace_function(source, "_append_event", APPEND_EVENT, "record_server_token_event")
    source = replace_function(source, "record_server_token_event", SERVER_EVENT, "record_engine_result_ready")
    source = replace_function(source, "record_scheduler_event", SCHEDULER_EVENT, "pilot_visual_forward")
    source = replace_function(source, "flush_scheduler_events", FLUSH_SCHEDULER, None)

    backup = run_root / "artifacts/original_files/continuation_prepatch/vllm/vl_pilot_hook.py"
    backup.parent.mkdir(parents=True, exist_ok=True)
    if backup.exists():
        raise RuntimeError(f"refusing to overwrite existing source backup: {backup}")
    shutil.copy2(hook, backup)

    patch_dir = run_root / "continuation/patches"
    patch_dir.mkdir(parents=True, exist_ok=True)
    diff = "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            source.splitlines(keepends=True),
            fromfile=str(hook) + ".pre-async",
            tofile=str(hook),
        )
    )
    (patch_dir / "vl_pilot_hook.async-jsonl.diff").write_text(diff)
    hook.write_text(source)
    print(f"patched={hook}")
    print(f"backup={backup}")
    print(f"diff={patch_dir / 'vl_pilot_hook.async-jsonl.diff'}")


if __name__ == "__main__":
    main()
