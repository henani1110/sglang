#!/usr/bin/env python3
"""Add E stage events, process provenance and durable trace flushing."""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path

EXPECTED_SOURCE_SHA256 = "34221e50b43a8c72b50baa0f56d34e391988ae2086ed9d67520cd882bcdf2189"
ANCHOR = "def flush_cuda_events() -> list[dict[str, Any]]:\n"
IMPORT_ANCHOR = "import os\nimport threading\n"
IMPORT_REPLACEMENT = "import os\nimport socket\nimport threading\n"
MM_KEYS_ANCHOR = '''_MM_KEYS: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
    "vl_pilot_mm_keys", default=()
)
'''
MM_KEYS_REPLACEMENT = MM_KEYS_ANCHOR + '''_MM_REQUEST_IDS: contextvars.ContextVar[dict[str, tuple[str, ...]]] = contextvars.ContextVar(
    "vl_pilot_mm_request_ids", default={}
)
'''
MM_KEYS_CONTEXT_ANCHOR = '''@contextmanager
def mm_keys_context(keys: list[str] | tuple[str, ...]) -> Iterator[None]:
    token = _MM_KEYS.set(tuple(str(key) for key in keys))
    try:
        yield
    finally:
        _MM_KEYS.reset(token)
'''
MM_KEYS_CONTEXT_REPLACEMENT = MM_KEYS_CONTEXT_ANCHOR + '''

@contextmanager
def mm_request_ids_context(mapping: dict[str, list[str] | tuple[str, ...]]) -> Iterator[None]:
    normalized = {
        str(key): tuple(sorted({str(request_id) for request_id in request_ids}))
        for key, request_ids in mapping.items()
    }
    token = _MM_REQUEST_IDS.set(normalized)
    try:
        yield
    finally:
        _MM_REQUEST_IDS.reset(token)


def mm_request_ids_for_hashes(mm_hashes: list[str] | tuple[str, ...]) -> list[str]:
    mapping = _MM_REQUEST_IDS.get()
    return sorted(
        {
            request_id
            for mm_hash in mm_hashes
            for request_id in mapping.get(str(mm_hash), ())
        }
    )


def current_mm_keys() -> list[str]:
    return list(_MM_KEYS.get())
'''
VISUAL_KEYS_ANCHOR = '''    key = keys[0] if len(keys) == 1 else None
    if mode != "NORMAL" and key is None:
        raise RuntimeError(f"expected one visual mm_hash, got {keys!r}")
    feature_key = _feature_key(state, key) if mode in {"CAPTURE", "VERIFY", "B"} and key else None
'''
VISUAL_KEYS_REPLACEMENT = '''    key = keys[0] if len(keys) == 1 else None
    if mode != "NORMAL" and key is None:
        raise RuntimeError(f"expected one visual mm_hash, got {keys!r}")
    request_ids = sorted(
        {
            request_id
            for mm_hash in keys
            for request_id in _MM_REQUEST_IDS.get().get(str(mm_hash), ())
        }
    )
    feature_key = _feature_key(state, key) if mode in {"CAPTURE", "VERIFY", "B"} and key else None
'''
CUDA_META_ANCHOR = '''            _EVENTS.append((event_start, event_end, {"mode": mode, "mm_hash": key}))
'''
CUDA_META_REPLACEMENT = '''            _EVENTS.append((event_start, event_end, {
                "mode": mode,
                "mm_hash": key,
                "mm_hashes": list(keys),
                "request_ids": request_ids,
            }))
'''
VISUAL_ROW_ANCHOR = '''            "mm_hash": key,
            "feature_key": feature_key,
'''
VISUAL_ROW_REPLACEMENT = '''            "mm_hash": key,
            "mm_hashes": list(keys),
            "request_ids": request_ids,
            "feature_key": feature_key,
'''
SCHEDULER_SIGNATURE_ANCHOR = "def record_scheduler_event(scheduler_output: Any) -> None:\n"
SCHEDULER_SIGNATURE_REPLACEMENT = '''def _pilot_block_ids_by_group(block_ids: Any) -> list[list[int]] | None:
    if block_ids is None:
        return None
    return [[int(block_id) for block_id in group] for group in block_ids]


def _pilot_kv_cache_groups(kv_cache_config: Any | None) -> dict[str, Any] | None:
    if kv_cache_config is None:
        return None
    groups = []
    for group_index, group in enumerate(kv_cache_config.kv_cache_groups):
        spec = group.kv_cache_spec
        try:
            page_size_bytes = int(spec.page_size_bytes)
        except (AttributeError, TypeError, NotImplementedError):
            page_size_bytes = None
        try:
            block_size = int(spec.block_size)
        except (AttributeError, TypeError, NotImplementedError):
            block_size = None
        layer_count = len(group.layer_names)
        groups.append({
            "group_index": group_index,
            "spec_type": type(spec).__name__,
            "layer_names": [str(name) for name in group.layer_names],
            "layer_count": layer_count,
            "block_size_tokens": block_size,
            "page_size_bytes_per_layer": page_size_bytes,
            "bytes_per_logical_block": (
                page_size_bytes * layer_count if page_size_bytes is not None else None
            ),
            "dtype": str(getattr(spec, "dtype", None)),
        })
    return {
        "num_blocks": int(kv_cache_config.num_blocks),
        "kv_cache_layout": getattr(kv_cache_config, "kv_cache_layout", None),
        "kv_tp_replicas": int(getattr(kv_cache_config, "kv_tp_replicas", 1)),
        "groups": groups,
    }


def record_scheduler_event(
    scheduler_output: Any, kv_cache_config: Any | None = None
) -> None:
'''
SCHEDULER_ROW_ANCHOR = '''        "finished_req_ids": sorted(str(key) for key in scheduler_output.finished_req_ids),
        "free_encoder_mm_hashes": [str(key) for key in scheduler_output.free_encoder_mm_hashes],
    }
'''
SCHEDULER_ROW_REPLACEMENT = '''        "finished_req_ids": sorted(str(key) for key in scheduler_output.finished_req_ids),
        "free_encoder_mm_hashes": [str(key) for key in scheduler_output.free_encoder_mm_hashes],
        "preempted_req_ids": sorted(
            str(key) for key in (scheduler_output.preempted_req_ids or set())
        ),
        "kv_cache_groups": _pilot_kv_cache_groups(kv_cache_config),
        "scheduled_new_reqs": [
            {
                "request_id": str(req.req_id),
                "block_ids_by_group": _pilot_block_ids_by_group(req.block_ids),
                "num_computed_tokens": int(req.num_computed_tokens),
                "prompt_len": int(req.prompt_len),
            }
            for req in scheduler_output.scheduled_new_reqs
        ],
        "scheduled_cached_reqs": [
            {
                "request_id": str(request_id),
                "new_block_ids_by_group": _pilot_block_ids_by_group(
                    cached.new_block_ids[index]
                ),
                "resumed": str(request_id) in cached.resumed_req_ids,
                "num_computed_tokens": int(cached.num_computed_tokens[index]),
                "num_output_tokens": int(cached.num_output_tokens[index]),
            }
            for cached in [scheduler_output.scheduled_cached_reqs]
            for index, request_id in enumerate(cached.req_ids)
        ],
    }
'''
FLUSH_BODY_ANCHOR = (
    "def flush_cuda_events() -> list[dict[str, Any]]:\n"
    "    torch.cuda.synchronize()\n"
)
FLUSH_BODY_REPLACEMENT = FLUSH_BODY_ANCHOR + "    flush_async_event_log_with_summary()\n"
META_SUMMARY_ANCHOR = '        target = self.path.with_name(self.path.name + ".async-meta.json")\n'
META_SUMMARY_REPLACEMENT = '        target = self.path.with_name(f"{self.path.name}.async-meta.{os.getpid()}.json")\n'
APPEND_EVENT_ANCHOR = (
    "    if not path:\n"
    "        return\n"
    "    if _submit_jsonl_rows(path, [event]):\n"
)
APPEND_EVENT_REPLACEMENT = (
    "    if not path:\n"
    "        return\n"
    "    event = {\n"
    "        **event,\n"
    "        \"trace_process_pid\": os.getpid(),\n"
    "        \"trace_async_jsonl_enabled\": os.environ.get(\"VL_PILOT_ASYNC_LOG\"),\n"
    "    }\n"
    "    if _submit_jsonl_rows(path, [event]):\n"
)
INSERTION = """_BACKLOG_SNAPSHOT_SOCKET: socket.socket | None = None
_BACKLOG_SNAPSHOT_SOCKET_LOCK = threading.Lock()


def record_scheduler_backlog_snapshot(snapshot: dict[str, Any]) -> None:
    # Send a compact scheduler state packet to the local diagnostic broker.
    global _BACKLOG_SNAPSHOT_SOCKET
    if os.environ.get("VL_PILOT_PROBE") != "1":
        return
    port_text = os.environ.get("VL_PILOT_BACKLOG_UDP_PORT")
    if not port_text:
        return
    try:
        port = int(port_text)
        if not 1 <= port <= 65535:
            return
        packet = json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode()
        if len(packet) > 1400:
            return
        sock = _BACKLOG_SNAPSHOT_SOCKET
        if sock is None:
            with _BACKLOG_SNAPSHOT_SOCKET_LOCK:
                sock = _BACKLOG_SNAPSHOT_SOCKET
                if sock is None:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    sock.setblocking(False)
                    _BACKLOG_SNAPSHOT_SOCKET = sock
        sock.sendto(packet, ("127.0.0.1", port))
    except (OSError, TypeError, ValueError, OverflowError):
        return


def record_stage_cuda_event(
    start: torch.cuda.Event,
    end: torch.cuda.Event,
    metadata: dict[str, Any],
) -> None:
    with _EVENTS_LOCK:
        _EVENTS.append((start, end, metadata))


_STAGE_DIAGNOSTICS_LOCK = threading.Lock()
_STAGE_DIAGNOSTICS_COUNT = 0


def record_stage_diagnostic(metadata: dict[str, Any]) -> None:
    global _STAGE_DIAGNOSTICS_COUNT
    if os.environ.get("VL_PILOT_PROBE") != "1":
        return
    with _STAGE_DIAGNOSTICS_LOCK:
        if _STAGE_DIAGNOSTICS_COUNT >= 65536:
            return
        _STAGE_DIAGNOSTICS_COUNT += 1
    _append_event({
        "kind": "stage_probe_diagnostic",
        "time_ns": time.perf_counter_ns(),
        **metadata,
    })


def record_queue_event(
    phase: str,
    request_id: str,
    *,
    time_ns_override: int | None = None,
    **metadata: Any,
) -> None:
    # Append a low-volume API/engine queue timing event when enabled.
    if os.environ.get("VL_PILOT_QUEUE_PROBE") != "1":
        return
    path = os.environ.get("VL_PILOT_EVENT_FILE")
    if not path:
        return
    observed_at_ns = time.perf_counter_ns()
    row = {
        "kind": "queue_timing",
        "phase": str(phase),
        "request_id": str(request_id),
        "time_ns": int(time_ns_override or observed_at_ns),
        "observed_at_ns": observed_at_ns,
        "trace_process_pid": os.getpid(),
        "trace_async_jsonl_enabled": os.environ.get("VL_PILOT_ASYNC_LOG"),
        **metadata,
    }
    if _submit_jsonl_rows(path, [row]):
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, payload)
    finally:
        os.close(fd)


def flush_async_event_log_with_summary() -> None:
    path = os.environ.get("VL_PILOT_EVENT_FILE")
    if not path or os.environ.get("VL_PILOT_ASYNC_LOG") != "1":
        return
    writer = _async_jsonl_writer(path)
    if not writer.flush():
        raise RuntimeError("async E event log did not flush within 20 seconds")
    with writer.counter_lock:
        snapshot = {
            "submitted": writer.submitted,
            "written": writer.written,
            "dropped": writer.dropped,
            "write_errors": writer.write_errors,
            "queue_capacity": writer.queue.maxsize,
            "last_error": writer.last_error,
        }
    writer.submit_many(
        [
            {
                "kind": "async_jsonl_process_flush_summary",
                "time_ns": time.perf_counter_ns(),
                "trace_process_pid": os.getpid(),
                "writer_snapshot": snapshot,
            }
        ]
    )
    if not writer.flush():
        raise RuntimeError("async E event log summary did not flush within 20 seconds")


"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--diff", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    # Always patch from the immutable E copy, not from the destination which is
    # overwritten below. This makes repeated service launches deterministic.
    source_bytes = args.snapshot.read_bytes() if args.snapshot.exists() else args.source.read_bytes()
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    if source_sha != EXPECTED_SOURCE_SHA256:
        raise SystemExit(f"unexpected copied E pilot hook SHA-256: {source_sha}")
    source = source_bytes.decode("utf-8")
    if (
        source.count(IMPORT_ANCHOR) != 1
        or source.count(ANCHOR) != 1
        or source.count(MM_KEYS_ANCHOR) != 1
        or source.count(MM_KEYS_CONTEXT_ANCHOR) != 1
        or source.count(VISUAL_KEYS_ANCHOR) != 1
        or source.count(CUDA_META_ANCHOR) != 1
        or source.count(VISUAL_ROW_ANCHOR) != 1
        or source.count(SCHEDULER_SIGNATURE_ANCHOR) != 1
        or source.count(SCHEDULER_ROW_ANCHOR) != 1
        or source.count(FLUSH_BODY_ANCHOR) != 1
        or source.count(META_SUMMARY_ANCHOR) != 1
        or source.count(APPEND_EVENT_ANCHOR) != 1
    ):
        raise SystemExit("expected one CUDA flush, summary and event append anchor")
    patched = source.replace(IMPORT_ANCHOR, IMPORT_REPLACEMENT, 1)
    patched = patched.replace(MM_KEYS_ANCHOR, MM_KEYS_REPLACEMENT, 1)
    patched = patched.replace(MM_KEYS_CONTEXT_ANCHOR, MM_KEYS_CONTEXT_REPLACEMENT, 1)
    patched = patched.replace(VISUAL_KEYS_ANCHOR, VISUAL_KEYS_REPLACEMENT, 1)
    patched = patched.replace(CUDA_META_ANCHOR, CUDA_META_REPLACEMENT, 1)
    patched = patched.replace(VISUAL_ROW_ANCHOR, VISUAL_ROW_REPLACEMENT, 1)
    patched = patched.replace(SCHEDULER_SIGNATURE_ANCHOR, SCHEDULER_SIGNATURE_REPLACEMENT, 1)
    patched = patched.replace(SCHEDULER_ROW_ANCHOR, SCHEDULER_ROW_REPLACEMENT, 1)
    patched = patched.replace(META_SUMMARY_ANCHOR, META_SUMMARY_REPLACEMENT, 1)
    patched = patched.replace(APPEND_EVENT_ANCHOR, APPEND_EVENT_REPLACEMENT, 1)
    patched = patched.replace(FLUSH_BODY_ANCHOR, FLUSH_BODY_REPLACEMENT, 1)
    patched = patched.replace(ANCHOR, INSERTION + ANCHOR, 1)

    args.snapshot.parent.mkdir(parents=True, exist_ok=True)
    if not args.snapshot.exists():
        args.snapshot.write_bytes(source_bytes)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    args.diff.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.destination.write_text(patched, encoding="utf-8")
    args.diff.write_text(
        "".join(
            difflib.unified_diff(
                source.splitlines(keepends=True),
                patched.splitlines(keepends=True),
                fromfile="copied pilot hook.py",
                tofile="E overlay pilot hook.py",
            )
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "component": "vllm.vl_pilot_hook.record_stage_cuda_event",
        "source_sha256": source_sha,
        "patched_sha256": hashlib.sha256(patched.encode()).hexdigest(),
        "source_provenance": "copy of D overlay's async measurement hook; D files were not modified",
        "change": "append stage CUDA events and bounded diagnostics with process provenance; map visual encoder event hashes to all scheduled request IDs; log KV block allocation and runtime page geometry; expose an opt-in nonblocking local UDP scheduler-backlog snapshot; flush worker traces and write summaries per process",
    }
    temp = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
    temp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temp, args.manifest)
    print(json.dumps({"status": "patched", **manifest}, ensure_ascii=False))


if __name__ == "__main__":
    main()
