#!/usr/bin/env python3
"""Run the fixed text/image trace against the isolated vLLM server."""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import math
import os
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
from transformers import AutoTokenizer


QUESTION = "Describe the geometric pattern in one short sentence."
IMAGE_OUTPUT_TOKENS = 16
DEFAULT_TEXT_TOKENS = 192
TEXT_BASES = [
    "Explain how a city library can make its public computers easier to use while keeping them reliable.",
    "Compare two reasonable ways to organize a small team's weekly engineering notes and decisions.",
    "Describe a careful process for checking whether a home energy estimate matches actual usage.",
    "Give a concise plan for teaching a beginner to read a map and estimate walking time.",
    "Outline practical steps for diagnosing an intermittent network delay in a small office.",
    "Explain how a research team can make a repeated systems experiment easier to reproduce.",
]
ARM_ORDERS = [
    ["T", "R", "B"],
    ["R", "B", "T"],
    ["B", "T", "R"],
    ["T", "B", "R"],
    ["B", "R", "T"],
    ["R", "T", "B"],
]


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n")
    os.replace(temporary, path)


def build_prompts(model_path: Path) -> list[list[int]]:
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False
    )
    prompts: list[list[int]] = []
    bos = tokenizer.bos_token_id
    for index, base in enumerate(TEXT_BASES):
        text = base + f" Reference case {index}:"
        content_ids = tokenizer.encode(text, add_special_tokens=False)
        filler = tokenizer.encode(
            " The system should be clear, useful, and easy to maintain.",
            add_special_tokens=False,
        )
        while len(content_ids) < 256:
            content_ids.extend(filler)
        if bos is not None:
            ids = [int(bos)] + content_ids[:255]
        else:
            ids = content_ids[:256]
        if len(ids) != 256:
            raise RuntimeError(f"prompt {index} is not 256 tokens")
        prompts.append(ids)
    return prompts


def decode_sse_ids(obj: dict[str, Any], endpoint: str) -> tuple[list[int], list[int] | None, str | None]:
    try:
        choice = obj["choices"][0]
    except (KeyError, IndexError, TypeError):
        return [], None, None
    ids = choice.get("token_ids")
    if ids is None:
        ids = choice.get("output_token_ids")
    prompt_ids = choice.get("prompt_token_ids")
    text = choice.get("text")
    if text is None:
        delta = choice.get("delta") or {}
        text = delta.get("content")
    return (
        [int(item) for item in ids] if isinstance(ids, list) else [],
        [int(item) for item in prompt_ids] if isinstance(prompt_ids, list) else None,
        text if isinstance(text, str) else None,
    )


class Pilot:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.base = args.base_url.rstrip("/")
        self.client_events = Path(args.client_events)
        self.control = Path(args.control_file)
        self.image_dir = Path(args.image_dir)
        self.images = [self.image_dir / f"geometry-{i:02d}.png" for i in range(4)]
        self.image_bytes = [path.read_bytes() for path in self.images]
        self.image_data_urls = [
            "data:image/png;base64," + base64.b64encode(value).decode("ascii")
            for value in self.image_bytes
        ]
        self.image_hashes = [hashlib.sha256(value).hexdigest() for value in self.image_bytes]
        self.prompts = build_prompts(Path(args.model_path))
        self.text_streams = int(getattr(args, "text_streams", 4))
        if self.text_streams not in (2, 4, 6):
            raise ValueError(f"text_streams must be 2, 4, or 6, got {self.text_streams}")
        self.results: list[dict[str, Any]] = []
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5, read=15, write=30, pool=10),
            limits=httpx.Limits(max_connections=24, max_keepalive_connections=12),
        )
        self.total_text_tokens = args.text_tokens
        self.arrival_offset_ms = args.arrival_offset_ms

    async def close(self) -> None:
        await self.client.aclose()

    def set_mode(
        self,
        mode: str,
        *,
        arm: str,
        episode_id: str,
        image_index: int | None = None,
        verify: bool = False,
        diagnostic: bool = False,
        torch_profile: bool = False,
    ) -> None:
        atomic_json(
            self.control,
            {
                "mode": mode,
                "arm": arm,
                "episode_id": episode_id,
                "image_index": image_index,
                "verify": verify,
                "diagnostic": diagnostic,
                "torch_profile": torch_profile,
            },
        )

    def validate_episode(
        self,
        row: dict[str, Any],
        *,
        expected_text_tokens: int,
        require_image: bool,
    ) -> list[str]:
        errors: list[str] = []
        if row.get("error"):
            errors.append(str(row["error"]))
        texts = row.get("text", [])
        if len(texts) != self.text_streams:
            errors.append(f"expected {self.text_streams} text streams, got {len(texts)}")
        for lane, item in enumerate(texts):
            if item.get("error"):
                errors.append(f"text lane {lane}: {item['error']}")
            if item.get("token_count") != expected_text_tokens:
                errors.append(
                    f"text lane {lane}: expected {expected_text_tokens} tokens, "
                    f"got {item.get('token_count')}"
                )
            if len(item.get("prompt_token_ids") or []) != 256:
                errors.append(f"text lane {lane}: prompt token count is not 256")
        trigger_counts = row.get("text_token_counts_at_trigger", [])
        if len(trigger_counts) != self.text_streams or trigger_counts[0] < 16 or any(
            count <= 0 for count in trigger_counts[1:]
        ):
            errors.append(f"invalid trigger counts: {trigger_counts}")
        if not all(row.get("active_text_streams_at_anchor", [])):
            errors.append(f"not all {self.text_streams} text streams were active at the anchor")
        image = row.get("image")
        if require_image:
            if image is None:
                errors.append("image request is missing")
            else:
                if image.get("error"):
                    errors.append(f"image request: {image['error']}")
                if image.get("token_count") != IMAGE_OUTPUT_TOKENS:
                    errors.append(
                        f"image output expected {IMAGE_OUTPUT_TOKENS} tokens, "
                        f"got {image.get('token_count')}"
                    )
        row["validation_errors"] = errors
        return errors

    async def reset_caches(self) -> None:
        for route in ("/reset_mm_cache", "/reset_encoder_cache"):
            response = await self.client.post(self.base + route)
            if response.status_code != 200:
                raise RuntimeError(f"{route} returned {response.status_code}: {response.text[:500]}")

    async def flush_cuda_events(self, episode_id: str) -> dict[str, Any]:
        response = await self.client.post(
            self.base + "/collective_rpc",
            json={"method": "pilot_flush_visual_events", "timeout": 20},
        )
        response.raise_for_status()
        value = response.json()
        row = {"kind": "cuda_event_flush", "episode_id": episode_id, "time_ns": time.monotonic_ns(), "response": value}
        append_jsonl(Path(self.args.diagnostics_file), [row])
        return row

    async def read_stream(
        self,
        endpoint: str,
        payload: dict[str, Any],
        *,
        request_label: str,
        lane: int | None = None,
        ready: asyncio.Event | None = None,
        on_token: Any = None,
    ) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        output_ids: list[int] = []
        prompt_ids: list[int] | None = None
        response_id: str | None = None
        text_parts: list[str] = []
        error: str | None = None
        started_ns = time.monotonic_ns()
        response = None
        try:
            async with self.client.stream("POST", self.base + endpoint, json=payload) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode(errors="replace")
                    raise RuntimeError(f"HTTP {response.status_code}: {body[:1000]}")
                if ready is not None:
                    ready.set()
                chunk_index = 0
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    raw = line[6:]
                    if raw == "[DONE]":
                        rows.append({"kind": "sse_done", "time_ns": time.monotonic_ns(), "raw": raw})
                        break
                    now = time.monotonic_ns()
                    try:
                        obj = json.loads(raw)
                    except json.JSONDecodeError:
                        rows.append({"kind": "sse_parse_error", "time_ns": now, "raw": raw})
                        continue
                    response_id = response_id or obj.get("id")
                    ids, this_prompt_ids, text = decode_sse_ids(obj, endpoint)
                    if prompt_ids is None and this_prompt_ids is not None:
                        prompt_ids = this_prompt_ids
                    if ids:
                        output_ids.extend(ids)
                        if on_token is not None:
                            trigger_yield = on_token(len(ids), now)
                            if trigger_yield:
                                await asyncio.sleep(0)
                    if text:
                        text_parts.append(text)
                    rows.append(
                        {
                            "kind": "sse_chunk",
                            "time_ns": now,
                            "chunk_index": chunk_index,
                            "request_id": response_id,
                            "lane": lane,
                            "token_ids": ids,
                            "prompt_token_ids": this_prompt_ids,
                            "text": text,
                            "finish_reason": (obj.get("choices") or [{}])[0].get("finish_reason"),
                            "raw": raw,
                        }
                    )
                    chunk_index += 1
        except Exception as exc:  # captured as trial evidence
            error = f"{type(exc).__name__}: {exc}"
            if ready is not None:
                ready.set()
        return {
            "label": request_label,
            "request_id": response_id,
            "started_ns": started_ns,
            "finished_ns": time.monotonic_ns(),
            "token_ids": output_ids,
            "prompt_token_ids": prompt_ids,
            "text": "".join(text_parts),
            "events": rows,
            "error": error,
        }

    async def request_image(
        self,
        image_index: int,
        *,
        episode_id: str,
        max_tokens: int = IMAGE_OUTPUT_TOKENS,
        on_send: Any = None,
    ) -> dict[str, Any]:
        image_path = self.images[image_index % len(self.images)]
        normalized_index = image_index % len(self.images)
        url = self.image_data_urls[normalized_index]
        label = f"{episode_id}-image"
        body = {
            "model": self.args.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": url}},
                        {"type": "text", "text": QUESTION},
                    ],
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0,
            "ignore_eos": True,
            "stream": True,
            "return_token_ids": True,
            "user": label,
        }
        send_ns = time.monotonic_ns()
        if on_send is not None:
            on_send(send_ns)
        result = await self.read_stream(
            "/v1/chat/completions", body, request_label=label
        )
        result["send_ns"] = send_ns
        result["image_index"] = image_index % len(self.images)
        result["image_sha256"] = self.image_hashes[normalized_index]
        return result

    async def run_image_only(
        self,
        image_index: int,
        *,
        mode: str,
        episode_id: str,
        max_tokens: int = IMAGE_OUTPUT_TOKENS,
        diagnostic: bool = False,
    ) -> dict[str, Any]:
        self.set_mode(
            mode,
            arm=mode,
            episode_id=episode_id,
            image_index=image_index % len(self.images),
            verify=(mode == "VERIFY"),
            diagnostic=diagnostic,
        )
        await self.reset_caches()
        result = await self.request_image(image_index, episode_id=episode_id, max_tokens=max_tokens)
        append_jsonl(self.client_events, result.pop("events"))
        result["mode"] = mode
        return result

    async def run_episode(
        self,
        *,
        block: int | str,
        arm: str,
        trial: int,
        image_index: int,
        diagnostic: bool = False,
        text_tokens: int | None = None,
        torch_profile: bool = False,
    ) -> dict[str, Any]:
        episode_id = f"{block}-{arm}-{trial:02d}"
        # C-stage strategy labels describe scheduling policy, while the visual
        # hook mode must remain NORMAL so all four policies execute full online E.
        mode = "NORMAL" if arm in {"T", "W", "N", "F", "S", "U"} else arm
        episode_text_tokens = self.total_text_tokens if text_tokens is None else text_tokens
        self.set_mode(
            mode,
            arm=arm,
            episode_id=episode_id,
            image_index=image_index % len(self.images),
            diagnostic=diagnostic,
            torch_profile=torch_profile,
        )
        await self.reset_caches()
        start_ns = time.monotonic_ns()
        ready_events = [asyncio.Event() for _ in range(self.text_streams)]
        trigger_event = asyncio.Event()
        token_counts = [0] * self.text_streams
        trigger_box: dict[str, Any] = {}
        lane_result: list[dict[str, Any] | None] = [None] * self.text_streams

        def on_token(lane: int):
            def callback(count: int, time_ns: int) -> bool:
                token_counts[lane] += count
                if (
                    not trigger_event.is_set()
                    and token_counts[0] >= 16
                    and all(count_value > 0 for count_value in token_counts[1:])
                ):
                    trigger_box["time_ns"] = time_ns
                    trigger_box["token_counts"] = list(token_counts)
                    trigger_event.set()
                    return True
                return False

            return callback

        async def start_text(lane: int) -> None:
            label = f"{episode_id}-text{lane}"
            body = {
                "model": self.args.model_name,
                "prompt": self.prompts[lane],
                "max_tokens": episode_text_tokens,
                "temperature": 0,
                "ignore_eos": True,
                "stream": True,
                "return_token_ids": True,
                "user": label,
            }
            lane_result[lane] = await self.read_stream(
                "/v1/completions",
                body,
                request_label=label,
                lane=lane,
                ready=ready_events[lane],
                on_token=on_token(lane),
            )

        tasks = [asyncio.create_task(start_text(lane)) for lane in range(self.text_streams)]
        image_result_task: asyncio.Task | None = None
        try:
            await asyncio.wait_for(trigger_event.wait(), timeout=30)
            trigger_ns = int(trigger_box["time_ns"])
            trigger_counts = list(trigger_box["token_counts"])
            target_anchor_ns = trigger_ns + int(self.arrival_offset_ms * 1e6)
            remaining_ns = target_anchor_ns - time.monotonic_ns()
            if remaining_ns > 0:
                await asyncio.sleep(remaining_ns / 1e9)
            anchor_box: dict[str, Any] = {}
            client_events: list[dict[str, Any]] = [
                {}
            ]
            if arm != "T":
                sent_event = asyncio.Event()

                def on_image_send(sent_ns: int) -> None:
                    anchor_box["time_ns"] = sent_ns
                    anchor_box["token_counts"] = list(token_counts)
                    anchor_box["active_text_streams"] = [not task.done() for task in tasks]
                    sent_event.set()

                image_result_task = asyncio.create_task(
                    self.request_image(image_index, episode_id=episode_id, on_send=on_image_send)
                )
                await sent_event.wait()
                anchor_ns = anchor_box["time_ns"]
                counts_at_anchor = anchor_box["token_counts"]
                active_at_anchor = anchor_box["active_text_streams"]
                client_events = [
                    {
                        "kind": "episode_anchor",
                        "episode_id": episode_id,
                        "block": block,
                        "arm": arm,
                        "time_ns": anchor_ns,
                        "trigger_ns": trigger_ns,
                        "arrival_offset_ms": (anchor_ns - trigger_ns) / 1e6,
                        "text_token_counts_at_trigger": trigger_counts,
                        "text_token_counts_at_anchor": counts_at_anchor,
                        "active_text_streams_at_anchor": active_at_anchor,
                        "image_index": image_index % len(self.images),
                    },
                    {
                        "kind": "image_request_send",
                        "episode_id": episode_id,
                        "arm": arm,
                        "time_ns": anchor_ns,
                        "trigger_ns": trigger_ns,
                        "arrival_offset_ms": (anchor_ns - trigger_ns) / 1e6,
                        "text_token_counts_at_anchor": counts_at_anchor,
                        "active_text_streams_at_anchor": active_at_anchor,
                        "image_index": image_index % len(self.images),
                    },
                ]
            else:
                anchor_ns = max(target_anchor_ns, time.monotonic_ns())
                counts_at_anchor = list(token_counts)
                active_at_anchor = [not task.done() for task in tasks]
                client_events = [
                    {
                        "kind": "episode_anchor",
                        "episode_id": episode_id,
                        "block": block,
                        "arm": arm,
                        "time_ns": anchor_ns,
                        "trigger_ns": trigger_ns,
                        "arrival_offset_ms": (anchor_ns - trigger_ns) / 1e6,
                        "text_token_counts_at_trigger": trigger_counts,
                        "text_token_counts_at_anchor": counts_at_anchor,
                        "active_text_streams_at_anchor": active_at_anchor,
                        "image_index": image_index % len(self.images),
                    }
                ]
            gather_targets = tasks + (
                [image_result_task] if image_result_task is not None else []
            )
            gathered = await asyncio.wait_for(
                asyncio.gather(*gather_targets, return_exceptions=True), timeout=30
            )
            text_results = gathered[: len(tasks)]
            if image_result_task is not None:
                image_result = gathered[-1]
                if isinstance(image_result, Exception):
                    image_result = {
                        "label": f"{episode_id}-image",
                        "error": f"{type(image_result).__name__}: {image_result}",
                        "image_index": image_index % len(self.images),
                        "image_sha256": self.image_hashes[image_index % len(self.images)],
                        "token_ids": [],
                        "events": [],
                    }
                image_events = image_result.pop("events")
                for event in image_events:
                    event.update(episode_id=episode_id, arm=arm, block=block, trial=trial)
                client_events.extend(image_events)
            else:
                image_result = None
            for lane, task_result in enumerate(text_results):
                if isinstance(task_result, Exception):
                    lane_result[lane] = {
                        "label": f"{episode_id}-text{lane}",
                        "error": f"{type(task_result).__name__}: {task_result}",
                    }
            for lane, result in enumerate(lane_result):
                if result is not None:
                    text_events = result.pop("events")
                    for event in text_events:
                        event.update(episode_id=episode_id, arm=arm, block=block, trial=trial)
                    client_events.extend(text_events)
            append_jsonl(self.client_events, client_events)
            row = {
                "episode_id": episode_id,
                "block": block,
                "arm": arm,
                "trial": trial,
                "image_index": image_index % len(self.images),
                "start_ns": start_ns,
                "trigger_ns": trigger_ns,
                "anchor_ns": anchor_ns,
                "end_ns": time.monotonic_ns(),
                "arrival_offset_ms": (anchor_ns - trigger_ns) / 1e6,
                "planned_arrival_offset_ms": self.arrival_offset_ms,
                "text_token_counts_at_trigger": trigger_counts,
                "text_token_counts_at_anchor": counts_at_anchor,
                "active_text_streams_at_anchor": active_at_anchor,
                "text_output_token_target": episode_text_tokens,
                "text": [
                    {
                        key: result.get(key)
                        for key in ("label", "request_id", "started_ns", "finished_ns", "error", "prompt_token_ids")
                    }
                    | {"token_count": len(result.get("token_ids", [])), "token_ids": result.get("token_ids", [])}
                    for result in lane_result
                    if result is not None
                ],
                "image": None
                if image_result is None
                else {
                    key: image_result.get(key)
                    for key in ("label", "request_id", "started_ns", "finished_ns", "error", "image_index", "image_sha256")
                }
                | {"token_count": len(image_result.get("token_ids", [])), "token_ids": image_result.get("token_ids", [])},
                "diagnostic": diagnostic,
            }
            self.results.append(row)
            return row
        except Exception as exc:
            for task in tasks:
                if not task.done():
                    task.cancel()
            if image_result_task is not None and not image_result_task.done():
                image_result_task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if image_result_task is not None:
                await asyncio.gather(image_result_task, return_exceptions=True)
            row = {
                "episode_id": episode_id,
                "block": block,
                "arm": arm,
                "trial": trial,
                "image_index": image_index % len(self.images),
                "start_ns": start_ns,
                "end_ns": time.monotonic_ns(),
                "error": f"{type(exc).__name__}: {exc}",
                "token_counts": token_counts,
                "diagnostic": diagnostic,
            }
            self.results.append(row)
            append_jsonl(self.client_events, [{"kind": "episode_failure", **row}])
            return row

    async def save_results(self, status: str) -> None:
        path = Path(self.args.results_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "status": status,
                    "model": self.args.model_name,
                    "text_prompt_token_lengths": [len(ids) for ids in self.prompts],
                    "text_output_tokens": self.total_text_tokens,
                    "active_text_streams": self.text_streams,
                    "image_output_tokens": IMAGE_OUTPUT_TOKENS,
                    "arrival_offset_ms": self.arrival_offset_ms,
                    "episodes": self.results,
                },
                indent=2,
            )
            + "\n"
        )
        os.replace(temporary, path)

    async def preflight(self) -> int:
        # Confirm a pure-text reference and the streaming token-ID contract.
        text_probe = await self.run_episode(
            block="preflight-text", arm="T", trial=0, image_index=0
        )
        text_ok = (
            not self.validate_episode(
                text_probe, expected_text_tokens=self.total_text_tokens, require_image=False
            )
            and text_probe.get("text_token_counts_at_trigger", [0])[0] >= 16
            and all(
                count > 0
                for count in text_probe.get("text_token_counts_at_trigger", [])[1:]
            )
            and all(text_probe.get("active_text_streams_at_anchor", []))
            and all(
                item.get("token_count") == self.total_text_tokens
                and item.get("prompt_token_ids") == self.prompts[index]
                for index, item in enumerate(text_probe.get("text", []))
            )
        )
        if not text_ok:
            await self.save_results("preflight_failed_text_stream_or_token_ids")
            return 2

        captures = []
        for image_index in range(4):
            result = await self.run_image_only(
                image_index, mode="CAPTURE", episode_id=f"preflight-capture-{image_index}", max_tokens=1
            )
            captures.append(result)
        if any(result.get("error") or not result.get("token_ids") for result in captures):
            await self.save_results("preflight_failed_feature_capture")
            return 3

        equivalence = []
        for image_index in range(4):
            r_result = await self.run_image_only(
                image_index, mode="VERIFY", episode_id=f"preflight-verify-R-{image_index}"
            )
            b_result = await self.run_image_only(
                image_index, mode="B", episode_id=f"preflight-verify-B-{image_index}"
            )
            match = r_result.get("token_ids") == b_result.get("token_ids")
            equivalence.append(
                {
                    "image_index": image_index,
                    "R_error": r_result.get("error"),
                    "B_error": b_result.get("error"),
                    "R_token_count": len(r_result.get("token_ids", [])),
                    "B_token_count": len(b_result.get("token_ids", [])),
                    "token_ids_match": match,
                    "R_token_ids": r_result.get("token_ids"),
                    "B_token_ids": b_result.get("token_ids"),
                }
            )
        Path(self.args.equivalence_file).write_text(json.dumps(equivalence, indent=2) + "\n")
        if any(
            item["R_error"] or item["B_error"] or not item["token_ids_match"]
            for item in equivalence
        ):
            await self.save_results("preflight_failed_R_B_output_equivalence")
            return 4

        for arm, image_index in (("R", 0), ("B", 1)):
            episode = await self.run_episode(
                block="preflight-diagnostic",
                arm=arm,
                trial=image_index,
                image_index=image_index,
                diagnostic=True,
            )
            if episode.get("error") or any(item.get("error") for item in episode.get("text", [])):
                await self.save_results("preflight_failed_mixed_diagnostic")
                return 5
            if self.validate_episode(
                episode, expected_text_tokens=self.total_text_tokens, require_image=True
            ):
                await self.save_results("preflight_failed_mixed_diagnostic_validation")
                return 10
            if (
                episode.get("image") is None
                or not all(episode.get("active_text_streams_at_anchor", []))
                or episode.get("arrival_offset_ms", 0) > self.arrival_offset_ms + 25
            ):
                await self.save_results("preflight_failed_mixed_timing_or_inflight")
                return 8
            await self.flush_cuda_events(str(episode["episode_id"]))

        await self.save_results("preflight_passed")
        print(json.dumps({"text_probe": text_probe, "equivalence": equivalence}, indent=2))
        return 0

    async def formal(self) -> int:
        if self.args.blocks != 6 or self.args.episodes_per_arm != 16:
            raise RuntimeError("formal pilot is fixed at six blocks and 16 episodes per arm")
        for block, order in enumerate(ARM_ORDERS, start=1):
            warmup = await self.run_episode(
                block=block,
                arm="W",
                trial=0,
                image_index=0,
                text_tokens=64,
            )
            if self.validate_episode(
                warmup, expected_text_tokens=64, require_image=True
            ):
                await self.flush_cuda_events(f"formal-block-{block}-warmup-failure")
                await self.save_results("formal_stopped_on_warmup_failure")
                return 9
            for arm in order:
                arm_started = time.monotonic()
                for trial in range(self.args.episodes_per_arm):
                    if time.monotonic() - arm_started >= 180:
                        await self.flush_cuda_events(f"formal-block-{block}-{arm}-budget-stop")
                        await self.save_results("formal_stopped_arm_block_180s_budget")
                        return 6
                    row = await self.run_episode(
                        block=block,
                        arm=arm,
                        trial=trial,
                        image_index=trial % len(self.images),
                    )
                    if self.validate_episode(
                        row,
                        expected_text_tokens=self.total_text_tokens,
                        require_image=(arm != "T"),
                    ):
                        await self.flush_cuda_events(f"formal-block-{block}-{arm}-episode-failure")
                        await self.save_results("formal_stopped_on_episode_failure")
                        return 7
                print(f"completed block={block} arm={arm} elapsed_s={time.monotonic()-arm_started:.1f}", flush=True)
        await self.flush_cuda_events("formal-final")
        await self.save_results("formal_completed")
        return 0

    async def microdiag(self) -> int:
        capture = await self.run_image_only(
            0,
            mode="CAPTURE",
            episode_id="microdiag-cache-warmup",
            max_tokens=1,
        )
        if capture.get("error") or not capture.get("token_ids"):
            await self.save_results("microdiag_stopped_on_cache_warmup_failure")
            return 12
        self.results.append(
            {
                "episode_id": "microdiag-cache-warmup",
                "block": "microdiag",
                "arm": "CAPTURE",
                "trial": -1,
                "image_index": 0,
                "image_sha256": capture.get("image_sha256"),
                "output_token_count": len(capture.get("token_ids", [])),
                "mode": capture.get("mode"),
            }
        )
        probes = (("R", False, False), ("R", True, False), ("R", True, True), ("B", True, True))
        for trial, (arm, diagnostic, torch_profile) in enumerate(probes):
            row = await self.run_episode(
                block="microdiag",
                arm=arm,
                trial=trial,
                image_index=0,
                diagnostic=diagnostic,
                torch_profile=torch_profile,
            )
            if self.validate_episode(
                row, expected_text_tokens=self.total_text_tokens, require_image=True
            ):
                await self.flush_cuda_events(f"microdiag-{arm}-{trial}-failure")
                await self.save_results("microdiag_stopped_on_episode_failure")
                return 11
            await self.flush_cuda_events(f"microdiag-{arm}-{trial}")
        await self.save_results("microdiag_completed")
        return 0

    async def secondary(self) -> int:
        if self.args.episodes_per_arm != 16:
            raise RuntimeError("secondary repeat is fixed at 16 episodes per arm")
        orders = (("R", "B", "T"), ("B", "R", "T"))
        for block, order in enumerate(orders, start=1):
            warmup = await self.run_episode(
                block=block,
                arm="W",
                trial=0,
                image_index=0,
                text_tokens=64,
            )
            if self.validate_episode(warmup, expected_text_tokens=64, require_image=True):
                await self.flush_cuda_events(f"secondary-block-{block}-warmup-failure")
                await self.save_results("secondary_stopped_on_warmup_failure")
                return 13
            for arm in order:
                arm_started = time.monotonic()
                for trial in range(self.args.episodes_per_arm):
                    if time.monotonic() - arm_started >= 180:
                        await self.flush_cuda_events(f"secondary-block-{block}-{arm}-budget-stop")
                        await self.save_results("secondary_stopped_arm_block_180s_budget")
                        return 14
                    row = await self.run_episode(
                        block=block,
                        arm=arm,
                        trial=trial,
                        image_index=trial % len(self.images),
                    )
                    if self.validate_episode(
                        row,
                        expected_text_tokens=self.total_text_tokens,
                        require_image=(arm != "T"),
                    ):
                        await self.flush_cuda_events(f"secondary-block-{block}-{arm}-episode-failure")
                        await self.save_results("secondary_stopped_on_episode_failure")
                        return 15
                print(
                    f"completed secondary_block={block} arm={arm} "
                    f"elapsed_s={time.monotonic()-arm_started:.1f}",
                    flush=True,
                )
        await self.flush_cuda_events("secondary-final")
        await self.save_results("secondary_completed")
        return 0


async def main_async() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "formal", "microdiag", "secondary"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8261")
    parser.add_argument("--model-name", default="qwen3vlpilot")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--control-file", required=True)
    parser.add_argument("--client-events", required=True)
    parser.add_argument("--diagnostics-file", required=True)
    parser.add_argument("--equivalence-file", required=True)
    parser.add_argument("--results-file", required=True)
    parser.add_argument("--text-tokens", type=int, default=DEFAULT_TEXT_TOKENS)
    parser.add_argument("--arrival-offset-ms", type=float, default=5.0)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--episodes-per-arm", type=int, default=16)
    args = parser.parse_args()
    pilot = Pilot(args)
    try:
        health = await pilot.client.get(args.base_url.rstrip("/") + "/health")
        health.raise_for_status()
        if args.mode == "preflight":
            return await pilot.preflight()
        if args.mode == "formal":
            return await pilot.formal()
        if args.mode == "microdiag":
            return await pilot.microdiag()
        return await pilot.secondary()
    finally:
        if not Path(args.results_file).exists():
            await pilot.save_results("interrupted")
        await pilot.close()


if __name__ == "__main__":
    try:
        code = asyncio.run(main_async())
    except Exception as exc:
        print(f"fatal: {type(exc).__name__}: {exc}", flush=True)
        raise
    raise SystemExit(code)
