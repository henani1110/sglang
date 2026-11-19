#!/usr/bin/env python3
"""Create the paired pilot traces and preserve request-level SSE evidence."""

import argparse
import asyncio
import hashlib
import json
import random
import time
from pathlib import Path


UNIFORM_LONG_SLOTS = {3, 8, 13, 18, 23, 28, 33, 38}
BURST_LONG_SLOTS = {8, 9, 10, 11, 28, 29, 30, 31}
REQUESTS_PER_TRACE = 40
SHORT_COUNT = 32
SHORT_TOKENS = 512
LONG_TOKENS = 6144
OUTPUT_TOKENS = 256


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_traces(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    dataset_path = Path(args.dataset)
    dataset = json.loads(dataset_path.read_text())
    rows = []
    for row in dataset:
        turns = row.get("conversations", row.get("conversation", []))
        if len(turns) >= 2 and turns[0].get("value", "").strip():
            rows.append(turns[0]["value"])

    random.Random(args.seed).shuffle(rows)
    if len(rows) < REQUESTS_PER_TRACE:
        raise ValueError(f"ShareGPT has only {len(rows)} usable prompts")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    requests = []
    for index, text in enumerate(rows[:REQUESTS_PER_TRACE]):
        length = SHORT_TOKENS if index < SHORT_COUNT else LONG_TOKENS
        tokens = tokenizer.encode(text, add_special_tokens=True)
        if not tokens:
            raise ValueError(f"empty tokenized prompt at row {index}")
        prompt_ids = (tokens * ((length + len(tokens) - 1) // len(tokens)))[:length]
        requests.append(
            {
                "id": index,
                "class": "short" if index < SHORT_COUNT else "long",
                "input_ids": prompt_ids,
                "expected_input_tokens": length,
                "max_new_tokens": OUTPUT_TOKENS,
            }
        )

    traces = {
        "uniform": UNIFORM_LONG_SLOTS,
        "long_bursts": BURST_LONG_SLOTS,
    }
    for trace_name, long_slots in traces.items():
        short_requests = iter(requests[:SHORT_COUNT])
        long_requests = iter(requests[SHORT_COUNT:])
        scheduled = []
        for slot in range(1, REQUESTS_PER_TRACE + 1):
            request = next(long_requests) if slot in long_slots else next(short_requests)
            scheduled.append({**request, "arrival_s": (slot - 1) / args.rate})

        payload = {
            "seed": args.seed,
            "source_dataset": str(dataset_path),
            "source_dataset_sha256": sha256_file(dataset_path),
            "model_path": str(args.model),
            "tokenizer_revision": args.revision,
            "rate_per_second": args.rate,
            "trace": trace_name,
            "requests": scheduled,
        }
        output = Path(args.output) / f"{trace_name}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, separators=(",", ":")))
        print(json.dumps({"trace": str(output), "sha256": sha256_file(output)}))


async def replay(args: argparse.Namespace) -> None:
    import aiohttp

    trace_path = Path(args.trace)
    trace = json.loads(trace_path.read_text())
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    requests = trace["requests"]
    if args.smoke:
        by_id = {request["id"]: request for request in requests}
        requests = [
            {**by_id[index], "arrival_s": 0.0}
            for index in (0, 32, 1, 33)
        ]
    timeout = aiohttp.ClientTimeout(total=args.timeout, sock_read=args.no_progress_timeout)
    connector = aiohttp.TCPConnector(limit=64, limit_per_host=64)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        async def one(request: dict) -> dict:
            target = started + request["arrival_s"]
            await asyncio.sleep(max(0.0, target - time.perf_counter()))
            sent = time.perf_counter()
            events = []
            done = False
            status = None
            error = None
            payload = {
                "input_ids": request["input_ids"],
                "sampling_params": {
                    "temperature": 0,
                    "ignore_eos": True,
                    "max_new_tokens": request["max_new_tokens"],
                },
                "stream": True,
                "rid": f"{output_dir.name}-{request['id']}",
            }
            try:
                async with session.post(args.url, json=payload) as response:
                    status = response.status
                    response.raise_for_status()
                    async for line in response.content:
                        if not line.startswith(b"data:"):
                            continue
                        body = line[5:].strip()
                        if body == b"[DONE]":
                            done = True
                            break
                        event = json.loads(body)
                        events.append(
                            {
                                "time_s": time.perf_counter() - started,
                                "data": event,
                            }
                        )
            except Exception as exc:
                error = repr(exc)
            ended = time.perf_counter()

            metadata = [
                event["data"].get("meta_info", {})
                for event in events
            ]
            counts = [item.get("completion_tokens", 0) for item in metadata]
            final_event = events[-1]["data"] if events else {}
            output_text = final_event.get("text", "")
            count_monotonic = all(
                previous <= current for previous, current in zip(counts, counts[1:])
            )
            actual_input = metadata[-1].get("prompt_tokens") if metadata else None
            passed = (
                error is None
                and status == 200
                and done
                and bool(counts)
                and counts[-1] == request["max_new_tokens"]
                and count_monotonic
                and actual_input == request["expected_input_tokens"]
                and bool(output_text)
            )
            return {
                "id": request["id"],
                "class": request["class"],
                "expected_input_tokens": request["expected_input_tokens"],
                "actual_prompt_tokens": actual_input,
                "arrival_s": request["arrival_s"],
                "send_s": sent - started,
                "send_lateness_s": sent - target,
                "end_s": ended - started,
                "first_token_s": next(
                    (
                        event["time_s"] - (sent - started)
                        for event in events
                        if event["data"].get("meta_info", {}).get("completion_tokens", 0) > 0
                    ),
                    None,
                ),
                "completion_tokens": counts[-1] if counts else 0,
                "count_monotonic": count_monotonic,
                "done": done,
                "status": status,
                "error": error,
                "ok": passed,
                "final_text": output_text,
                "events": events,
            }

        results = await asyncio.gather(*(one(request) for request in requests))

    requests_path = output_dir / "requests.jsonl"
    requests_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in results)
    )
    summary = {
        "trace_path": str(trace_path),
        "trace_sha256": sha256_file(trace_path),
        "requests": len(results),
        "passed": sum(item["ok"] for item in results),
        "wall_s": time.perf_counter() - started,
        "requests_path": str(requests_path),
    }
    (output_dir / "run.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("make")
    make.add_argument("--model", required=True)
    make.add_argument("--dataset", required=True)
    make.add_argument("--revision", required=True)
    make.add_argument("--output", required=True)
    make.add_argument("--seed", type=int, default=40902026)
    make.add_argument("--rate", type=float, default=2.0)
    run = commands.add_parser("run")
    run.add_argument("--trace", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--url", default="http://127.0.0.1:30000/generate")
    run.add_argument("--timeout", type=float, default=90)
    run.add_argument("--no-progress-timeout", type=float, default=30)
    run.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.command == "make":
        make_traces(args)
    else:
        asyncio.run(replay(args))


if __name__ == "__main__":
    main()
