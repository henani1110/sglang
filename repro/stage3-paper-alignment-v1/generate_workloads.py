#!/usr/bin/env python3
"""Generate paired paper-description and author-README-style TokenFlow workloads.

Run inside the candidate repository's venv with the local model tokenizer:
  python /path/to/generate_workloads.py --model-path repro/models/Qwen2-7B-Instruct \
    --sharegpt-json /path/to/ShareGPT_V3_unfiltered_cleaned_split.json \
    --out repro/stage3-paper-alignment-v1/workloads

The PDF does not state normal-distribution sigma/truncation or Poisson request count.
The `paper-normal-adaptation` suite therefore makes those assumptions explicit;
`author-range-ratio-02` follows the cached public SGLang sampler exactly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


CASES = {
    "E1": {"count": 60, "input_mean": 512, "output_mean": 1024, "rate": 0.0},
    "E2": {"count": 80, "input_mean": 1024, "output_mean": 2048, "rate": 0.0},
    # The paper gives these arrival rates, but no request count; 120 is the
    # explicit local adaptation used in the completed v2 runs.
    "P2": {"count": 120, "input_mean": 512, "output_mean": 1024, "rate": 2.0},
    "P4": {"count": 120, "input_mean": 512, "output_mean": 1024, "rate": 4.0},
}
SHAREGPT_REVISION = "044ca94aec8d8cdee04973000738431161247677"
SHAREGPT_SHA256 = "35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def author_range_lengths(rng: np.random.RandomState, mean: int, n: int, ratio: float) -> list[int]:
    low = max(int(mean * ratio), 1)
    return rng.randint(low, mean + 1, size=n).astype(int).tolist()


def normal_adaptation_lengths(
    rng: np.random.RandomState, inputs: list[int], outputs: list[int], max_context: int
) -> tuple[list[int], list[int]]:
    xs: list[int] = []
    ys: list[int] = []
    for mu_in, mu_out in zip(inputs, outputs):
        for _ in range(10000):
            x = int(round(float(rng.normal(mu_in, 0.2 * mu_in))))
            y = int(round(float(rng.normal(mu_out, 0.2 * mu_out))))
            if x > 0 and y > 0 and x + y <= max_context:
                xs.append(x)
                ys.append(y)
                break
        else:
            raise RuntimeError("normal rejection sampler exceeded 10000 attempts")
    return xs, ys


def load_sharegpt_pairs(path: Path) -> list[tuple[str, str]]:
    """Load the same first-two-turn pairs used by SGLang's random sampler."""
    with path.open() as f:
        data = json.load(f)
    pairs = []
    for row in data:
        turns = row.get("conversations", row.get("conversation", []))
        if len(turns) >= 2:
            pairs.append((turns[0]["value"], turns[1]["value"]))
    return pairs


def load_sharegpt_prompts(pairs: list[tuple[str, str]], tokenizer, seed: int, count: int) -> list[list[int]]:
    """Mirror SGLang sample_random_requests(random_sample=True, return_text=False)."""
    pairs = list(pairs)
    random.seed(seed)
    random.shuffle(pairs)
    prompts = []
    for prompt, _completion in pairs:
        ids = tokenizer.encode(prompt)
        if ids:
            prompts.append(ids)
        if len(prompts) == count:
            break
    if len(prompts) != count:
        raise ValueError(f"ShareGPT corpus yielded {len(prompts)} non-empty prompts; need {count}")
    return prompts


def fit_prompt(ids: list[int], target_len: int) -> list[int]:
    if len(ids) >= target_len:
        return ids[:target_len]
    repeat = (target_len + len(ids) - 1) // len(ids)
    return (ids * repeat)[:target_len]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--sharegpt-json", required=True)
    ap.add_argument("--sharegpt-revision", default=SHAREGPT_REVISION)
    ap.add_argument("--expected-sharegpt-sha256", default=SHAREGPT_SHA256)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[101, 102, 103])
    ap.add_argument("--max-context", type=int, default=32768)
    ap.add_argument("--range-ratio", type=float, default=0.2)
    ap.add_argument("--output-speed", type=int, default=20)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    corpus_path = Path(args.sharegpt_json)
    corpus_sha = sha256(corpus_path)
    if corpus_sha != args.expected_sharegpt_sha256:
        raise ValueError(f"ShareGPT SHA256 mismatch: expected {args.expected_sharegpt_sha256}, got {corpus_sha}")
    sharegpt_pairs = load_sharegpt_pairs(corpus_path)
    prompts_by_seed = {
        seed: load_sharegpt_prompts(sharegpt_pairs, tokenizer, seed, max(spec["count"] for spec in CASES.values()))
        for seed in args.seeds
    }
    rows = []

    for suite in ("paper-normal-adaptation", "author-range-ratio-02"):
        for case, spec in CASES.items():
            for seed in args.seeds:
                rng = np.random.RandomState(seed)
                n = spec["count"]
                prompts = prompts_by_seed[seed][:n]
                if suite == "paper-normal-adaptation":
                    x, y = normal_adaptation_lengths(
                        rng,
                        [spec["input_mean"]] * n,
                        [spec["output_mean"]] * n,
                        args.max_context,
                    )
                    distribution = {
                        "kind": "normal",
                        "sigma_fraction_of_mean": 0.2,
                        "rounding": "Python round then int",
                        "truncation": "reject values <= 0 or input+output > max_context and resample both lengths",
                        "max_context": args.max_context,
                    }
                else:
                    x = author_range_lengths(rng, spec["input_mean"], n, args.range_ratio)
                    y = author_range_lengths(rng, spec["output_mean"], n, args.range_ratio)
                    distribution = {
                        "kind": "uniform_integer_range",
                        "range_ratio": args.range_ratio,
                        "inclusive_min": True,
                        "exclusive_max": False,
                        "source": "TokenFlow cached public branch SGLang sample_random_requests/compute_random_lens",
                    }

                arrival_rng = random.Random(seed + 100000)
                arrival = 0.0
                requests = []
                for i, (in_len, out_len) in enumerate(zip(x, y)):
                    if i and spec["rate"] > 0:
                        arrival += arrival_rng.expovariate(spec["rate"])
                    input_ids = fit_prompt(prompts[i], in_len)
                    requests.append(
                        {
                            "id": i,
                            "arrival_s": arrival,
                            "input_ids": input_ids,
                            "max_new_tokens": int(out_len),
                        }
                    )

                filename = f"{case.lower()}-{suite}-s{seed}.json"
                path = out / filename
                if path.exists():
                    raise FileExistsError(f"refusing to overwrite existing workload: {path}")
                payload = {
                    "schema_version": 1,
                    "suite": suite,
                    "case": case,
                    "seed": seed,
                    "model_path": str(Path(args.model_path).resolve()),
                    "prompt_corpus": "anon8231489123/ShareGPT_Vicuna_unfiltered:ShareGPT_V3_unfiltered_cleaned_split.json",
                    "prompt_corpus_revision": args.sharegpt_revision,
                    "prompt_corpus_sha256": corpus_sha,
                    "request_count": n,
                    "input_mean_target": spec["input_mean"],
                    "output_mean_target": spec["output_mean"],
                    "length_distribution": distribution,
                    "arrival_rate_per_second": spec["rate"],
                    "arrival_generation": "Poisson inter-arrivals via random.Random(seed + 100000); all burst requests arrive at t=0",
                    "prompt_generation": "first user turn from ShareGPT conversations, shuffled by Python random.seed(seed), tokenized locally, then truncated/repeated to the requested input length; mirrors SGLang sample_random_requests(random_sample=True)",
                    "speed": args.output_speed,
                    "output_speed_tokens_per_second": args.output_speed,
                    "requests": requests,
                }
                path.write_text(json.dumps(payload, separators=(",", ":")))
                rows.append(
                    {
                        "file": filename,
                        "sha256": sha256(path),
                        "suite": suite,
                        "case": case,
                        "seed": seed,
                        "request_count": n,
                        "input_mean_actual": statistics.mean(x),
                        "input_std_actual": statistics.pstdev(x),
                        "input_min": min(x),
                        "input_max": max(x),
                        "output_mean_actual": statistics.mean(y),
                        "output_std_actual": statistics.pstdev(y),
                        "output_min": min(y),
                        "output_max": max(y),
                        "last_arrival_s": arrival,
                    }
                )

    index = {
        "generator": "generate_workloads.py",
        "generator_sha256": sha256(Path(__file__)),
        "model_path": str(Path(args.model_path).resolve()),
        "model_revision": "f2826a00ceef68f0f2b946d945ecc0477ce4450c",
        "prompt_corpus": str(corpus_path.resolve()),
        "prompt_corpus_revision": args.sharegpt_revision,
        "prompt_corpus_sha256": corpus_sha,
        "prompt_corpus_sha256": corpus_sha,
        "seeds": args.seeds,
        "suites": ["paper-normal-adaptation", "author-range-ratio-02"],
        "workloads": rows,
    }
    index_path = out / "workload-index.json"
    if index_path.exists():
        raise FileExistsError(f"refusing to overwrite existing index: {index_path}")
    index_path.write_text(json.dumps(index, indent=2))
    print(json.dumps({"index": str(index_path), "workloads": len(rows), "sha256": sha256(index_path)}, indent=2))


if __name__ == "__main__":
    main()
