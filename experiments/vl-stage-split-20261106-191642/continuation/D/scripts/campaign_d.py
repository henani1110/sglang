#!/usr/bin/env python3
"""Open-loop D-stage campaign client with paired image/text traces."""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import httpx
from transformers import AutoTokenizer


QUESTION = "Describe the image in one concise sentence."
TEXT_BASES = [
    "Explain how a city library can make its public computers easier to use while keeping them reliable.",
    "Compare two reasonable ways to organize a small team's weekly engineering notes and decisions.",
    "Describe a careful process for checking whether a home energy estimate matches actual usage.",
    "Give a concise plan for teaching a beginner to read a map and estimate walking time.",
    "Outline practical steps for diagnosing an intermittent network delay in a small office.",
    "Explain how a research team can make a repeated systems experiment easier to reproduce.",
]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, payload)
    finally:
        os.close(fd)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def build_prompts(model_path: Path) -> list[list[int]]:
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False
    )
    prompts: list[list[int]] = []
    bos = tokenizer.bos_token_id
    for index, base in enumerate(TEXT_BASES):
        content_ids = tokenizer.encode(base + f" Reference case {index}:", add_special_tokens=False)
        filler = tokenizer.encode(
            " The system should be clear, useful, and easy to maintain.",
            add_special_tokens=False,
        )
        while len(content_ids) < 256:
            content_ids.extend(filler)
        ids = ([int(bos)] + content_ids[:255]) if bos is not None else content_ids[:256]
        if len(ids) != 256:
            raise RuntimeError(f"text prompt {index} did not produce exactly 256 tokens")
        prompts.append(ids)
    return prompts


def load_images(manifest_path: Path, image_size: int) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text())
    if int(manifest.get("selected_count", 0)) < 32 or manifest.get("status") != "complete":
        raise RuntimeError("D holdout image manifest must be complete with at least 32 images")
    result = []
    for row in manifest["images"]:
        processed = row["processed"][str(image_size)]
        path = Path(processed["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        data = path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        if sha != processed["sha256"]:
            raise RuntimeError(f"processed image hash mismatch: {path}")
        result.append(
            {
                "image_id": row["image_id"],
                "path": str(path),
                "sha256": sha,
                "bytes": data,
                "title": row["title"],
            }
        )
    return result


def load_control_images(directory: Path) -> list[dict[str, Any]]:
    result = []
    for index in range(4):
        path = directory / f"geometry-{index:02d}.png"
        data = path.read_bytes()
        result.append(
            {
                "image_id": f"geometry-{index:02d}",
                "path": str(path),
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": data,
                "title": f"existing geometric mechanism control {index}",
            }
        )
    return result


def planned_offsets(mode: str, duration_s: int) -> list[float]:
    if mode == "uniform":
        return [float(t) for t in range(0, duration_s, 10)]
    if mode == "burst":
        offsets = []
        for t in range(0, duration_s, 2):
            offsets.extend([float(t), float(t) + 0.1])
        return offsets
    raise ValueError(f"unknown arrival mode: {mode}")


class Campaign:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.base = args.base_url.rstrip("/")
        self.run_id = args.run_id
        self.seed = int(args.seed)
        if args.dataset_manifest:
            self.images = load_images(Path(args.dataset_manifest), args.image_size)
        elif args.control_image_dir:
            self.images = load_control_images(Path(args.control_image_dir))
        else:
            raise ValueError("provide --dataset-manifest for holdout or --control-image-dir for calibration")
        self.prompts = build_prompts(Path(args.model_path))
        self.image_data_urls = [
            "data:image/png;base64," + base64.b64encode(row["bytes"]).decode("ascii")
            for row in self.images
        ]
        self.events_path = Path(args.client_events)
        self.results: list[dict[str, Any]] = []
        self.stop_text = asyncio.Event()
        self.ready_events = [asyncio.Event() for _ in range(args.text_streams)]
        self.sequence = [0] * args.text_streams
        self.measurement_image_sends = 0
        self.measure_start_ns: int | None = None
        self.phase_start_ns: dict[str, int] = {}
        self.rng = random.Random(self.seed)
        self.image_order = list(range(len(self.images)))
        self.rng.shuffle(self.image_order)
        self.http = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10, read=90, write=30, pool=20),
            limits=httpx.Limits(max_connections=96, max_keepalive_connections=48),
        )

    async def close(self) -> None:
        await self.http.aclose()

    async def reset_caches(self) -> None:
        for route in ("/reset_mm_cache", "/reset_encoder_cache"):
            response = await self.http.post(self.base + route)
            if response.status_code != 200:
                raise RuntimeError(f"{route} failed: HTTP {response.status_code}: {response.text[:500]}")

    async def flush_cuda_events(self, phase: str) -> None:
        response = await self.http.post(
            self.base + "/collective_rpc",
            json={"method": "pilot_flush_visual_events", "timeout": 30},
        )
        response.raise_for_status()
        append_jsonl(
            self.events_path,
            {
                "kind": "cuda_event_flush",
                "run_id": self.run_id,
                "phase": phase,
                "time_ns": time.monotonic_ns(),
                "response": response.json(),
            },
        )

    def set_control(self) -> None:
        path = Path(self.args.control_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "mode": "NORMAL",
            "arm": self.args.policy,
            "episode_id": self.run_id,
            "image_index": None,
            "verify": False,
            "diagnostic": bool(self.args.calibration_diagnostic),
            "torch_profile": False,
        }
        atomic_json(path, payload)

    async def stream_request(
        self,
        endpoint: str,
        payload: dict[str, Any],
        *,
        kind: str,
        label: str,
        phase: str,
        lane: int | None = None,
        sequence: int | None = None,
        planned_ns: int | None = None,
        actual_send_ns: int | None = None,
        ready: asyncio.Event | None = None,
        image: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        start_ns = time.monotonic_ns()
        token_ids: list[int] = []
        token_times_ns: list[int] = []
        prompt_ids: list[int] | None = None
        response_id = None
        finish_reason = None
        error = None
        first_token_ns = None
        try:
            async with self.http.stream("POST", self.base + endpoint, json=payload) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode(errors="replace")
                    raise RuntimeError(f"HTTP {response.status_code}: {body[:800]}")
                if ready is not None:
                    ready.set()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    raw = line[6:]
                    if raw == "[DONE]":
                        break
                    now = time.monotonic_ns()
                    try:
                        obj = json.loads(raw)
                    except json.JSONDecodeError:
                        append_jsonl(
                            self.events_path,
                            {"kind": "sse_parse_error", "run_id": self.run_id, "phase": phase, "time_ns": now, "label": label},
                        )
                        continue
                    response_id = response_id or obj.get("id")
                    choices = obj.get("choices") or []
                    if choices:
                        choice = choices[0]
                        ids = choice.get("token_ids")
                        if ids is None:
                            ids = choice.get("output_token_ids")
                        if isinstance(ids, list):
                            values = [int(value) for value in ids]
                            if values:
                                first_token_ns = first_token_ns or now
                                token_ids.extend(values)
                                token_times_ns.extend([now] * len(values))
                        if choice.get("finish_reason") is not None:
                            finish_reason = choice.get("finish_reason")
                        if prompt_ids is None and isinstance(choice.get("prompt_token_ids"), list):
                            prompt_ids = [int(value) for value in choice["prompt_token_ids"]]
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if ready is not None:
                ready.set()

        end_ns = time.monotonic_ns()
        row: dict[str, Any] = {
            "kind": f"{kind}_response",
            "run_id": self.run_id,
            "phase": phase,
            "label": label,
            "lane": lane,
            "sequence": sequence,
            "request_id": response_id,
            "planned_ns": planned_ns,
            "actual_send_ns": actual_send_ns,
            "started_ns": start_ns,
            "first_token_ns": first_token_ns,
            "finished_ns": end_ns,
            "token_ids": token_ids,
            "token_times_ns": token_times_ns,
            "prompt_token_ids": prompt_ids,
            "finish_reason": finish_reason,
            "error": error,
        }
        if image is not None:
            row.update(
                image_id=image["image_id"],
                image_sha256=image["sha256"],
                image_title=image["title"],
                image_size=self.args.image_size,
            )
        append_jsonl(self.events_path, row)
        self.results.append(row)
        return row

    async def text_lane(self, lane: int) -> None:
        while not self.stop_text.is_set():
            sequence = self.sequence[lane]
            self.sequence[lane] += 1
            prompt = self.prompts[(lane + sequence) % len(self.prompts)]
            label = f"{self.run_id}-text-l{lane}-r{sequence}"
            payload = {
                "model": self.args.model_name,
                "prompt": prompt,
                "max_tokens": self.args.text_tokens,
                "temperature": 0,
                "ignore_eos": True,
                "stream": True,
                "return_token_ids": True,
                "user": label,
            }
            await self.stream_request(
                "/v1/completions",
                payload,
                kind="text",
                label=label,
                phase="background",
                lane=lane,
                sequence=sequence,
                ready=self.ready_events[lane] if sequence == 0 else None,
            )

    async def image_request(
        self,
        *,
        image_slot: int,
        phase: str,
        offset_s: float,
        planned_ns: int,
    ) -> dict[str, Any]:
        image_index = self.image_order[image_slot % len(self.image_order)]
        image = self.images[image_index]
        data_url = self.image_data_urls[image_index]
        label = f"{self.run_id}-{phase}-image-{image_slot:04d}"
        payload = {
            "model": self.args.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": QUESTION},
                    ],
                }
            ],
            "max_tokens": self.args.image_tokens,
            "temperature": 0,
            "ignore_eos": True,
            "stream": True,
            "return_token_ids": True,
            "user": label,
        }
        actual_send_ns = time.monotonic_ns()
        if phase == "measurement":
            self.measurement_image_sends += 1
        append_jsonl(
            self.events_path,
            {
                "kind": "image_request_send",
                "run_id": self.run_id,
                "phase": phase,
                "image_slot": image_slot,
                "image_id": image["image_id"],
                "image_sha256": image["sha256"],
                "request_label": label,
                "planned_ns": planned_ns,
                "planned_offset_s": offset_s,
                "actual_send_ns": actual_send_ns,
            },
        )
        return await self.stream_request(
            "/v1/chat/completions",
            payload,
            kind="image",
            label=label,
            phase=phase,
            sequence=image_slot,
            planned_ns=planned_ns,
            actual_send_ns=actual_send_ns,
            image=image,
        )

    async def launch_phase_images(self, phase: str, start_ns: int, duration_s: int, first_slot: int) -> list[asyncio.Task]:
        tasks: list[asyncio.Task] = []
        offsets = planned_offsets(self.args.arrival_mode, duration_s)
        for local_index, offset_s in enumerate(offsets):
            planned_ns = start_ns + int(offset_s * 1e9)
            delay = (planned_ns - time.monotonic_ns()) / 1e9
            if delay > 0:
                await asyncio.sleep(delay)
            slot = first_slot + local_index
            tasks.append(
                asyncio.create_task(
                    self.image_request(
                        image_slot=slot,
                        phase=phase,
                        offset_s=offset_s,
                        planned_ns=planned_ns,
                    )
                )
            )
        return tasks

    async def run(self) -> dict[str, Any]:
        started_ns = time.monotonic_ns()
        status = "failed"
        error = None
        text_tasks: list[asyncio.Task] = []
        image_tasks: list[asyncio.Task] = []
        try:
            self.set_control()
            await self.reset_caches()
            text_tasks = [
                asyncio.create_task(self.text_lane(lane))
                for lane in range(self.args.text_streams)
            ]
            await asyncio.wait_for(
                asyncio.gather(*(event.wait() for event in self.ready_events)),
                timeout=30,
            )
            warmup_start_ns = time.monotonic_ns()
            self.phase_start_ns["warmup"] = warmup_start_ns
            append_jsonl(
                self.events_path,
                {"kind": "phase_start", "run_id": self.run_id, "phase": "warmup", "time_ns": warmup_start_ns},
            )
            warmup_tasks = await self.launch_phase_images(
                "warmup", warmup_start_ns, self.args.warmup_seconds, 0
            )
            image_tasks.extend(warmup_tasks)
            warmup_image_results = await asyncio.gather(*warmup_tasks, return_exceptions=True)
            append_jsonl(
                self.events_path,
                {
                    "kind": "warmup_drain_complete",
                    "run_id": self.run_id,
                    "time_ns": time.monotonic_ns(),
                    "image_requests": len(warmup_tasks),
                    "image_exceptions": sum(isinstance(row, Exception) for row in warmup_image_results),
                },
            )
            if self.args.calibration_diagnostic:
                await self.flush_cuda_events("warmup")
            measure_start_ns = time.monotonic_ns()
            self.measure_start_ns = measure_start_ns
            self.phase_start_ns["measurement"] = self.measure_start_ns
            append_jsonl(
                self.events_path,
                {
                    "kind": "phase_start",
                    "run_id": self.run_id,
                    "phase": "measurement",
                    "time_ns": self.measure_start_ns,
                    "planned_time_ns": measure_start_ns,
                },
            )
            measurement_tasks = await self.launch_phase_images(
                "measurement",
                measure_start_ns,
                self.args.measure_seconds,
                len(warmup_tasks),
            )
            image_tasks.extend(measurement_tasks)
            measurement_end_ns = measure_start_ns + int(self.args.measure_seconds * 1e9)
            delay = (measurement_end_ns - time.monotonic_ns()) / 1e9
            if delay > 0:
                await asyncio.sleep(delay)
            self.stop_text.set()
            append_jsonl(
                self.events_path,
                {"kind": "phase_end", "run_id": self.run_id, "phase": "measurement", "time_ns": time.monotonic_ns()},
            )
            try:
                await asyncio.wait_for(asyncio.gather(*text_tasks, return_exceptions=True), timeout=120)
            except asyncio.TimeoutError:
                for task in text_tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*text_tasks, return_exceptions=True)
            try:
                image_results = await asyncio.wait_for(
                    asyncio.gather(*image_tasks, return_exceptions=True), timeout=120
                )
            except asyncio.TimeoutError:
                for task in image_tasks:
                    if not task.done():
                        task.cancel()
                image_results = await asyncio.gather(*image_tasks, return_exceptions=True)
            if self.args.calibration_diagnostic:
                await self.flush_cuda_events("measurement")
            status = "completed"
            return {
                "status": status,
                "run_id": self.run_id,
                "policy": self.args.policy,
                "arrival_mode": self.args.arrival_mode,
                "image_size": self.args.image_size,
                "text_streams": self.args.text_streams,
                "seed": self.seed,
                "block": self.args.block,
                "warmup_seconds": self.args.warmup_seconds,
                "measurement_seconds": self.args.measure_seconds,
                "started_ns": started_ns,
                "finished_ns": time.monotonic_ns(),
                "measurement_start_ns": self.measure_start_ns,
                "text_requests": sum(1 for row in self.results if row["kind"] == "text_response"),
                "image_requests": sum(1 for row in self.results if row["kind"] == "image_response"),
                "image_errors": sum(1 for row in self.results if row["kind"] == "image_response" and row.get("error")),
                "text_errors": sum(1 for row in self.results if row["kind"] == "text_response" and row.get("error")),
                "measurement_images_planned": len(planned_offsets(self.args.arrival_mode, self.args.measure_seconds)),
                "measurement_images_sent": self.measurement_image_sends,
                "measurement_image_responses": sum(
                    1 for row in self.results if row["kind"] == "image_response" and row.get("phase") == "measurement"
                ),
                "measurement_image_errors": sum(
                    1 for row in self.results
                    if row["kind"] == "image_response" and row.get("phase") == "measurement" and row.get("error")
                ),
                "image_task_exceptions": sum(isinstance(row, Exception) for row in image_results),
                "dataset_image_count": len(self.images),
                "dataset_manifest": str(self.args.dataset_manifest or self.args.control_image_dir),
            }
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            append_jsonl(
                self.events_path,
                {"kind": "campaign_failure", "run_id": self.run_id, "time_ns": time.monotonic_ns(), "error": error},
            )
            return {
                "status": status,
                "run_id": self.run_id,
                "policy": self.args.policy,
                "seed": self.seed,
                "started_ns": started_ns,
                "finished_ns": time.monotonic_ns(),
                "error": error,
            }
        finally:
            self.stop_text.set()
            for task in text_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*text_tasks, return_exceptions=True)
            await self.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8271")
    parser.add_argument("--model-name", default="qwen3vlpilot")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--control-image-dir", type=Path)
    parser.add_argument("--image-size", type=int, choices=[512, 768], required=True)
    parser.add_argument("--text-streams", type=int, choices=[2, 6], required=True)
    parser.add_argument("--arrival-mode", choices=["uniform", "burst"], required=True)
    parser.add_argument("--policy", choices=["N", "F", "S", "A", "U"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--block", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--text-tokens", type=int, default=192)
    parser.add_argument("--image-tokens", type=int, default=16)
    parser.add_argument("--warmup-seconds", type=int, default=60)
    parser.add_argument("--measure-seconds", type=int, default=180)
    parser.add_argument("--control-file", type=Path, required=True)
    parser.add_argument("--client-events", type=Path, required=True)
    parser.add_argument("--results-file", type=Path, required=True)
    parser.add_argument("--calibration-diagnostic", action="store_true")
    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()
    atomic_json(
        Path(args.results_file),
        {
            "status": "running",
            "run_id": args.run_id,
            "policy": args.policy,
            "seed": args.seed,
            "block": args.block,
            "image_size": args.image_size,
            "text_streams": args.text_streams,
            "arrival_mode": args.arrival_mode,
            "started_ns": time.monotonic_ns(),
        },
    )
    campaign = Campaign(args)
    result = await campaign.run()
    atomic_json(Path(args.results_file), result)
    print(json.dumps(result, sort_keys=True, ensure_ascii=False))
    if result.get("status") != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(async_main())

