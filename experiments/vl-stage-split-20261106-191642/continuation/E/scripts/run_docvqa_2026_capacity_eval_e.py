#!/usr/bin/env python3
"""Run the calibration-only Qwen3-VL-4B capacity reference on DocVQA-2026."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def port_is_closed(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        connection.settimeout(1)
        return connection.connect_ex((host, port)) != 0


def endpoint_ready(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def gpu_memory() -> list[dict[str, str]] | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    rows: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 4:
            rows.append({"index": parts[0], "name": parts[1], "memory_used": parts[2], "memory_total": parts[3]})
    return rows


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stop_process_group(process: subprocess.Popen[bytes]) -> dict[str, Any]:
    result: dict[str, Any] = {"pid": process.pid, "signals_sent": [], "return_code": None, "stopped": False}
    if process.poll() is not None:
        result["return_code"] = process.returncode
        result["stopped"] = True
        return result
    try:
        os.killpg(process.pid, signal.SIGINT)
        result["signals_sent"].append("SIGINT")
    except ProcessLookupError:
        pass
    try:
        result["return_code"] = process.wait(timeout=60)
        result["stopped"] = True
        return result
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGTERM)
        result["signals_sent"].append("SIGTERM")
    except ProcessLookupError:
        pass
    try:
        result["return_code"] = process.wait(timeout=30)
        result["stopped"] = True
    except subprocess.TimeoutExpired:
        result["return_code"] = process.poll()
    return result


def start_memory_sampler(path: Path, interval_ms: int) -> subprocess.Popen[bytes] | None:
    if interval_ms <= 0:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    output = path.open("wb")
    output.write(b"timestamp,index,memory.used [MiB],memory.total [MiB]\n")
    output.flush()
    process = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=timestamp,index,memory.used,memory.total", "--format=csv,noheader", "-lms", str(interval_ms)],
        stdin=subprocess.DEVNULL,
        stdout=output,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    # Keep the stream alive until the sampler exits; attach it for deterministic cleanup.
    setattr(process, "_codex_output_stream", output)
    return process


def stop_memory_sampler(process: subprocess.Popen[bytes] | None) -> dict[str, Any] | None:
    if process is None:
        return None
    try:
        process.terminate()
        return_code = process.wait(timeout=5)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            process.kill()
        except ProcessLookupError:
            pass
        return_code = process.wait(timeout=5)
    stream = getattr(process, "_codex_output_stream", None)
    if stream is not None:
        stream.close()
    return {"pid": process.pid, "return_code": return_code, "stopped": process.poll() is not None}


def import_evaluator(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("docvqa_2026_pinned_official_evaluator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import official evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def request_one(
    *,
    candidate: dict[str, Any],
    document: dict[str, Any],
    image_root: Path,
    edge: int,
    endpoint: str,
    model: str,
    prompt_prefix: str,
    evaluator: Any,
    evaluator_sha256: str,
    candidate_manifest_sha256: str,
    document_manifest_sha256: str,
    seed: int,
    max_tokens: int,
    timeout: int,
    run_id: str,
    prompt_template: str,
    keep_ratio: float,
    attempt: int,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    image_sha_records: list[dict[str, Any]] = []
    for page in document["pages"]:
        relative_path = Path(page["relative_path"])
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"unsafe page path in manifest: {page['relative_path']}")
        image_path = (image_root / relative_path).resolve()
        try:
            image_path.relative_to(image_root.resolve())
        except ValueError as exc:
            raise ValueError(f"page path escaped images root: {image_path}") from exc
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        content.append({"type": "image_url", "image_url": {"url": image_path.as_uri()}})
        image_sha_records.append({"page_index": page["page_index"], "sha256_png": page["sha256_png"]})

    prompt = prompt_prefix + "\n\nQuestion: " + str(candidate["question"])
    content.append({"type": "text", "text": prompt})
    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "seed": seed,
        "max_tokens": max_tokens,
        "stream": False,
        "mm_processor_kwargs": {
            "min_pixels": edge * edge,
            "max_pixels": edge * edge,
            "images_kwargs": {"e_keep_ratio": keep_ratio},
        },
    }
    if prompt_template in {"concise-stopline", "best-effort-stopline"}:
        body["stop"] = ["\n"]
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started_ns = time.monotonic_ns()
    status: int | None = None
    response_body = b""
    error: str | None = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            response_body = response.read()
    except urllib.error.HTTPError as exc:
        status = exc.code
        response_body = exc.read()
        error = f"HTTPError: {exc.code}"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        error = f"{type(exc).__name__}: {exc}"
        response_body = str(exc).encode("utf-8", errors="replace")
    elapsed_ms = (time.monotonic_ns() - started_ns) / 1e6
    try:
        response_json = json.loads(response_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        response_json = None

    choices = (response_json or {}).get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}
    raw_prediction = message.get("content") if isinstance(message, dict) else None
    if not isinstance(raw_prediction, str):
        raw_prediction = ""
    finish_reason = choice.get("finish_reason")
    response_complete = status == 200 and bool(raw_prediction) and finish_reason == "stop" and error is None
    is_correct: bool | None = None
    extracted_answer: str | None = None
    if raw_prediction:
        is_correct, extracted_answer = evaluator.evaluate_docvqa_prediction(raw_prediction, candidate["answer"])

    response_choices = (response_json or {}).get("choices") or []
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "attempt": attempt,
        "sample_id": candidate["sample_id"],
        "quality_split": candidate["quality_split"],
        "prompt_template": prompt_template,
        "doc_id": candidate["doc_id"],
        "doc_category": candidate["doc_category"],
        "page_count": len(document["pages"]),
        "page_manifest_sha256": sha256_bytes(json.dumps(image_sha_records, sort_keys=True, separators=(",", ":")).encode("utf-8")),
        "processor_edge": edge,
        "keep_ratio": keep_ratio,
        "arm": "full_token_keep_all" if keep_ratio == 1.0 else "fixed_keep_ratio",
        "model": model,
        "seed": seed,
        "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "document_manifest_sha256": document_manifest_sha256,
        "evaluator_sha256": evaluator_sha256,
        "request_sha256": sha256_bytes(payload),
        "http_status": status,
        "elapsed_client_ms": elapsed_ms,
        "error": error,
        "prompt_tokens": ((response_json or {}).get("usage") or {}).get("prompt_tokens"),
        "completion_tokens": ((response_json or {}).get("usage") or {}).get("completion_tokens"),
        "finish_reason": finish_reason,
        "response_complete": response_complete,
        "raw_prediction": raw_prediction,
        "extracted_answer": extracted_answer,
        "official_score_correct": is_correct,
        "source_answer": candidate["answer"],
        "rights_status": candidate.get("rights_status"),
        "local_eval_scope_only": True,
        "redistributable": False,
        "formal_quality_run": False,
        "response_json_choices_count": len(response_choices),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge", type=int, choices=(512,), required=True)
    parser.add_argument("--attempt", default="a1")
    parser.add_argument("--split", choices=("calibration",), default="calibration")
    parser.add_argument(
        "--prompt-template",
        choices=("concise",),
        default="concise",
    )
    parser.add_argument("--server-script", type=Path, required=True)
    parser.add_argument("--documents-jsonl", type=Path, required=True)
    parser.add_argument("--candidates-jsonl", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument(
        "--images-root",
        type=Path,
        required=True,
        help=(
            "root directory containing manifest relative_path entries "
            "(normally the dataset root, not its images/ subdirectory)"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="qwen3vl4b")
    parser.add_argument("--keep-ratio", type=float, default=1.0)
    parser.add_argument("--port", type=int, default=8285)
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--start-timeout", type=int, default=900)
    parser.add_argument("--request-timeout", type=int, default=3600)
    parser.add_argument("--memory-sample-interval-ms", type=int, default=250)
    parser.add_argument("--retry-incomplete", type=int, default=0, help="retry failed/incomplete rows after the first full pass")
    args = parser.parse_args()

    if args.retry_incomplete < 0:
        raise ValueError("retry-incomplete must be non-negative")
    if not 0.0 < args.keep_ratio <= 1.0:
        raise ValueError("keep-ratio must be finite and in (0, 1]")
    if args.keep_ratio != 1.0:
        raise ValueError("the 4B capacity reference is frozen to full visual-token retention")
    if not args.attempt.replace("-", "").replace("_", "").replace(".", "").isalnum():
        raise ValueError("attempt may contain only letters, digits, dot, underscore, or hyphen")
    if not port_is_closed("127.0.0.1", args.port):
        raise RuntimeError(f"port {args.port} is already open; refusing to replace another service")

    candidates = load_jsonl(args.candidates_jsonl)
    if args.split != "all":
        candidates = [row for row in candidates if row["quality_split"] == args.split]
    if not candidates:
        raise ValueError("no candidate rows selected")
    documents = {row["doc_id"]: row for row in load_jsonl(args.documents_jsonl)}
    for candidate in candidates:
        if candidate["doc_id"] not in documents:
            raise ValueError(f"missing document manifest for {candidate['doc_id']}")

    image_root = args.images_root.resolve()
    checked_page_paths: set[Path] = set()
    missing_page_paths: list[str] = []
    for doc_id in sorted({row["doc_id"] for row in candidates}):
        for page in documents[doc_id]["pages"]:
            relative_path = Path(page["relative_path"])
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(f"unsafe page path in manifest: {page['relative_path']}")
            image_path = (image_root / relative_path).resolve()
            try:
                image_path.relative_to(image_root)
            except ValueError as exc:
                raise ValueError(f"page path escaped images root: {image_path}") from exc
            if image_path not in checked_page_paths:
                checked_page_paths.add(image_path)
                if not image_path.is_file():
                    missing_page_paths.append(str(image_path))
    if missing_page_paths:
        preview = "; ".join(missing_page_paths[:5])
        suffix = " …" if len(missing_page_paths) > 5 else ""
        raise FileNotFoundError(
            f"image preflight failed: {len(missing_page_paths)} missing page file(s): {preview}{suffix}"
        )

    e_root = Path(__file__).resolve().parents[1]
    pinned_site_packages = e_root / "env" / "docvqa-parse" / "site-packages"
    if pinned_site_packages.is_dir():
        # The SSD parse environment intentionally uses a system Python and
        # keeps its locked packages in this isolated, non-default directory.
        sys.path.insert(0, str(pinned_site_packages))
    evaluator = import_evaluator(args.evaluator)
    if args.prompt_template == "official":
        prompt_prefix = evaluator.get_evaluation_prompt()
    elif args.prompt_template == "concise":
        prompt_prefix = (
            "You answer questions using all pages of one document, in their original order. "
            "Use only evidence in the document. If the answer is not present, answer Unknown. "
            "Do not explain or show reasoning. Return exactly one line in this format: "
            "FINAL ANSWER: <answer>."
        )
    elif args.prompt_template == "concise-stopline":
        prompt_prefix = (
            "Use all document pages in their original order to answer the question. "
            "If the answer is absent, write Unknown. Do not explain. "
            "Output only one short line: FINAL ANSWER: <answer>."
        )
    else:
        prompt_prefix = (
            "Read all document pages in order and answer using evidence from the document. "
            "Give the shortest exact answer that the question asks for. Do not explain, "
            "repeat the question, or enumerate alternatives. If the question asks for multiple "
            "items, provide only the requested items. Do not answer Unknown merely because you "
            "are uncertain: provide the best-supported answer whenever any page contains relevant "
            "evidence. Use Unknown only if the answer is not present or recoverable after checking "
            "the document. Output exactly one line in this format: FINAL ANSWER: <answer>"
        )
    candidate_manifest_sha256 = sha256_file(args.candidates_jsonl)
    document_manifest_sha256 = sha256_file(args.documents_jsonl)
    evaluator_sha256 = sha256_file(args.evaluator)
    max_model_len = 50_000
    # Qwen3-VL-4B has 36 layers, 8 KV heads, head_dim=128, and BF16 KV.
    # 8e9 bytes fits the audited 45.3k-token longest calibration document.
    kv_cache_bytes = 8_000_000_000
    keep_label = f"k{round(args.keep_ratio * 100):02d}"
    arm = "full_token_keep_all" if args.keep_ratio == 1.0 else f"fixed_keep_{keep_label}"
    run_id = f"docvqa-2026-capacity-qwen3vl4b-edge{args.edge}-{keep_label}-{args.split}-{args.prompt_template}-{args.attempt}-s{args.seed}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_jsonl = args.output_dir / "predictions.jsonl"
    summary_path = args.output_dir / "manager-summary.json"
    server_log = args.output_dir / "server-supervisor.log"
    memory_log = args.output_dir / "nvidia-smi-memory.csv"
    base_url = f"http://127.0.0.1:{args.port}/v1"

    existing_rows: list[dict[str, Any]] = load_jsonl(result_jsonl) if result_jsonl.exists() else []
    successful_ids = {row["sample_id"] for row in existing_rows if row.get("http_status") == 200 and row.get("response_complete")}
    pending = [row for row in candidates if row["sample_id"] not in successful_ids]
    environment = os.environ.copy()
    environment.update({
        "VL_PILOT_PORT": str(args.port),
        "VL_PILOT_LABEL": run_id,
        "VL_PILOT_MAX_MODEL_LEN": str(max_model_len),
        "VL_PILOT_MAX_NUM_SEQS": "1",
        "VL_PILOT_MAX_NUM_BATCHED_TOKENS": "4096",
        "VL_PILOT_KV_CACHE_MEMORY_BYTES": str(kv_cache_bytes),
        "VL_PILOT_MAX_MEDIA_REQUEST_DECODE_BYTES": "1073741824",
        "VL_PILOT_PROBE": "0",
        "VL_PILOT_QUEUE_PROBE": "0",
        "VL_IMAGE_KEEP_RATIO": "1.0",
    })
    summary: dict[str, Any] = {
        "run_id": run_id,
        "dataset": "VLR-CVC/DocVQA-2026",
        "model_checkpoint": "Qwen/Qwen3-VL-4B-Instruct",
        "model_revision": "ebb281ec70b05090aa6165b016eac8ec08e71b17",
        "capacity_reference_only": True,
        "runner_python_executable": sys.executable,
        "runner_python_version": sys.version,
        "split": args.split,
        "prompt_template": args.prompt_template,
        "keep_ratio": args.keep_ratio,
        "edge": args.edge,
        "arm": arm,
        "smoke_only": False,
        "formal_quality_run": False,
        "local_eval_scope_only": True,
        "redistributable": False,
        "candidate_count_selected": len(candidates),
        "pending_at_start": len(pending),
        "previously_completed": len(candidates) - len(pending),
        "attempt": args.attempt,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "max_model_len": max_model_len,
        "kv_cache_memory_bytes": kv_cache_bytes,
        "memory_sample_interval_ms": args.memory_sample_interval_ms,
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "document_manifest_sha256": document_manifest_sha256,
        "evaluator_sha256": evaluator_sha256,
        "images_root": str(image_root),
        "image_preflight_checked_files": len(checked_page_paths),
        "result_jsonl": str(result_jsonl),
        "server_log": str(server_log),
        "memory_log": str(memory_log),
        "gpu_memory_before": gpu_memory(),
        "candidate_errors": [],
    }
    process: subprocess.Popen[bytes] | None = None
    sampler: subprocess.Popen[bytes] | None = None
    server_stream = server_log.open("wb")
    try:
        process = subprocess.Popen(
            ["bash", str(args.server_script)],
            stdin=subprocess.DEVNULL,
            stdout=server_stream,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
        )
        summary["server_pid"] = process.pid
        ready_url = base_url + "/models"
        deadline = time.monotonic() + args.start_timeout
        while time.monotonic() < deadline:
            if endpoint_ready(ready_url):
                summary["server_ready"] = True
                break
            if process.poll() is not None:
                summary["server_ready"] = False
                summary["server_return_code_before_requests"] = process.returncode
                break
            time.sleep(2)
        else:
            summary["server_ready"] = False
            summary["server_start_timeout"] = True

        if summary.get("server_ready"):
            sampler = start_memory_sampler(memory_log, args.memory_sample_interval_ms)
            prompt_url = base_url + "/chat/completions"
            pass_index = 0
            for retry_pass in range(args.retry_incomplete + 1):
                if retry_pass == 0:
                    todo = pending
                else:
                    seen = load_jsonl(result_jsonl) if result_jsonl.exists() else []
                    now_success = {row["sample_id"] for row in seen if row.get("http_status") == 200 and row.get("response_complete")}
                    todo = [row for row in candidates if row["sample_id"] not in now_success]
                    if not todo:
                        break
                for candidate in todo:
                    pass_index += 1
                    try:
                        record = request_one(
                            candidate=candidate,
                            document=documents[candidate["doc_id"]],
                            image_root=image_root,
                            edge=args.edge,
                            endpoint=prompt_url,
                            model=args.model,
                            prompt_prefix=prompt_prefix,
                            evaluator=evaluator,
                            evaluator_sha256=evaluator_sha256,
                            candidate_manifest_sha256=candidate_manifest_sha256,
                            document_manifest_sha256=document_manifest_sha256,
                            seed=args.seed,
                            max_tokens=args.max_tokens,
                            timeout=args.request_timeout,
                            run_id=run_id,
                            prompt_template=args.prompt_template,
                            keep_ratio=args.keep_ratio,
                            attempt=retry_pass + 1,
                        )
                    except Exception as exc:
                        record = {
                            "created_utc": datetime.now(timezone.utc).isoformat(),
                            "run_id": run_id,
                            "attempt": retry_pass + 1,
                            "sample_id": candidate["sample_id"],
                            "quality_split": candidate["quality_split"],
                            "doc_id": candidate["doc_id"],
                            "doc_category": candidate["doc_category"],
                            "page_count": candidate["page_count"],
                            "processor_edge": args.edge,
                            "keep_ratio": args.keep_ratio,
                            "http_status": None,
                            "response_complete": False,
                            "official_score_correct": None,
                            "error": f"{type(exc).__name__}: {exc}",
                            "formal_quality_run": False,
                            "local_eval_scope_only": True,
                            "redistributable": False,
                        }
                    with result_jsonl.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                    if not record.get("response_complete"):
                        summary["candidate_errors"].append({"sample_id": candidate["sample_id"], "attempt": retry_pass + 1, "error": record.get("error"), "http_status": record.get("http_status")})
                    print(json.dumps({
                        "state": "candidate_finished",
                        "edge": args.edge,
                        "split": candidate["quality_split"],
                        "sample_id": candidate["sample_id"],
                        "pages": candidate["page_count"],
                        "http_status": record.get("http_status"),
                        "complete": record.get("response_complete"),
                        "score": record.get("official_score_correct"),
                        "elapsed_ms": round(record.get("elapsed_client_ms", 0), 1),
                        "progress": f"{pass_index}/{len(todo)}",
                    }, ensure_ascii=False), flush=True)
                    if not record.get("response_complete"):
                        time.sleep(1)
            latest_rows = load_jsonl(result_jsonl) if result_jsonl.exists() else []
            latest_by_id: dict[str, dict[str, Any]] = {}
            for row in latest_rows:
                latest_by_id[row["sample_id"]] = row
            selected = [latest_by_id[row["sample_id"]] for row in candidates if row["sample_id"] in latest_by_id]
            summary["completed_unique"] = sum(row.get("http_status") == 200 and row.get("response_complete") for row in selected)
            summary["scored_unique"] = sum(row.get("official_score_correct") is not None for row in selected)
            summary["correct_unique"] = sum(row.get("official_score_correct") is True for row in selected)
            summary["incorrect_unique"] = sum(row.get("official_score_correct") is False for row in selected)
            summary["missing_unique"] = len(candidates) - len(selected)
            summary["split_counts"] = {
                split: {
                    "selected": sum(row["quality_split"] == split for row in candidates),
                    "completed": sum(row.get("quality_split") == split and row.get("http_status") == 200 and row.get("response_complete") for row in selected),
                    "correct": sum(row.get("quality_split") == split and row.get("official_score_correct") is True for row in selected),
                }
                for split in ("calibration", "heldout")
            }
        else:
            summary["completed_unique"] = 0
            summary["scored_unique"] = 0
            summary["correct_unique"] = 0
            summary["incorrect_unique"] = 0
            summary["missing_unique"] = len(candidates)
    except BaseException as exc:
        summary["manager_exception"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        summary["memory_sampler_shutdown"] = stop_memory_sampler(sampler)
        if process is not None:
            summary["shutdown"] = stop_process_group(process)
        server_stream.close()
        summary["port_closed_after_shutdown"] = port_is_closed("127.0.0.1", args.port)
        summary["gpu_memory_after"] = gpu_memory()
        summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(summary_path, summary)
        print(json.dumps({
            "state": "shutdown_complete",
            "edge": args.edge,
            "completed_unique": summary.get("completed_unique"),
            "selected": summary.get("candidate_count_selected"),
            "port_closed": summary["port_closed_after_shutdown"],
            "gpu_memory_after": summary["gpu_memory_after"],
            "summary": str(summary_path),
        }, ensure_ascii=False), flush=True)

    return 0 if summary.get("completed_unique") == len(candidates) and summary.get("port_closed_after_shutdown") else 1


if __name__ == "__main__":
    raise SystemExit(main())
