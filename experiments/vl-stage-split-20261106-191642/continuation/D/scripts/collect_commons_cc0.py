#!/usr/bin/env python3
"""Collect a reproducible, per-file CC0 image set from Wikimedia Commons."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import random
import re
import shutil
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import requests
from PIL import Image, ImageOps


API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "VLStageSplitResearch/1.0 (research dataset; local reproducibility run)"
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
LICENSES = {"CC0", "CC-Zero"}


def plain(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("value", "")
    return html.unescape(re.sub(r"<[^>]*>", " ", str(value or ""))).strip()


def request_page(session: requests.Session, params: dict[str, Any]) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            response = session.get(
                API,
                params=params,
                timeout=(10, 60),
                headers={"Connection": "close", "Accept-Encoding": "identity"},
            )
            response.raise_for_status()
            payload = response.json()
            if "error" in payload:
                raise RuntimeError(f"Commons API error: {payload['error']}")
            return payload
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt == 4:
                break
            print(f"[commons] API retry {attempt + 1}/5: {type(exc).__name__}: {exc}", flush=True)
            time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"Commons API page failed after 5 attempts: {last_error}")


def candidates(session: requests.Session, max_pages: int) -> list[dict[str, Any]]:
    params: dict[str, Any] = {
        "action": "query",
        "generator": "categorymembers",
        "gcmtitle": "Category:CC-Zero",
        "gcmtype": "file",
        "gcmnamespace": 6,
        "gcmlimit": 100,
        "prop": "imageinfo",
        "iiprop": "url|size|extmetadata",
        "iiurlwidth": 2048,
        "format": "json",
    }
    found: list[dict[str, Any]] = []
    seen_titles: set[str] = set()
    for page_number in range(max_pages):
        data = request_page(session, params)
        print(f"[commons] scanned API page {page_number + 1}/{max_pages}", flush=True)
        for page in data.get("query", {}).get("pages", {}).values():
            title = page.get("title", "")
            if title in seen_titles:
                continue
            seen_titles.add(title)
            info = (page.get("imageinfo") or [{}])[0]
            meta = info.get("extmetadata") or {}
            license_name = plain(meta.get("LicenseShortName"))
            license_url = plain(meta.get("LicenseUrl"))
            suffix = Path(unquote(title)).suffix.lower()
            if (
                license_name not in LICENSES
                or "/publicdomain/zero/1.0" not in license_url
                or suffix not in ALLOWED_SUFFIXES
                or int(info.get("width", 0)) < 768
                or int(info.get("height", 0)) < 768
                or max(int(info.get("width", 0)), int(info.get("height", 0)))
                / max(1, min(int(info.get("width", 0)), int(info.get("height", 0))))
                > 2.5
                or int(info.get("size", 0)) <= 0
                or int(info.get("size", 0)) > 30 * 1024 * 1024
                or not info.get("url")
                or not info.get("thumburl")
            ):
                continue
            found.append(
                {
                    "title": title,
                    "description_url": info.get("descriptionurl"),
                    "original_url": info.get("url"),
                    "download_url": info.get("thumburl"),
                    "download_variant": "Wikimedia thumbnail (requested 2048 px long edge)",
                    "thumbnail_width": int(info.get("thumbwidth", 0)),
                    "thumbnail_height": int(info.get("thumbheight", 0)),
                    "width": int(info["width"]),
                    "height": int(info["height"]),
                    "bytes": int(info["size"]),
                    "license": license_name,
                    "license_url": license_url,
                    "artist_credit": plain(meta.get("Artist")),
                    "description": plain(meta.get("ImageDescription")),
                    "date_time_original": plain(meta.get("DateTimeOriginal")),
                }
            )
        more = data.get("continue")
        if not more:
            break
        params.update(more)
        time.sleep(0.15)
    return found


def search_candidates(session: requests.Session, phrases: list[str]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    seen_titles: set[str] = set()
    for phrase in phrases:
        params: dict[str, Any] = {
            "action": "query",
            "generator": "search",
            "gsrsearch": f"{phrase} filetype:bitmap",
            "gsrnamespace": 6,
            "gsrlimit": 50,
            "prop": "imageinfo",
            "iiprop": "url|size|extmetadata",
            "iiurlwidth": 1024,
            "format": "json",
        }
        data = request_page(session, params)
        pages = data.get("query", {}).get("pages", {}).values()
        phrase_count = 0
        for page in pages:
            title = page.get("title", "")
            if title in seen_titles:
                continue
            seen_titles.add(title)
            info = (page.get("imageinfo") or [{}])[0]
            meta = info.get("extmetadata") or {}
            license_name = plain(meta.get("LicenseShortName"))
            license_url = plain(meta.get("LicenseUrl"))
            width = int(info.get("width", 0))
            height = int(info.get("height", 0))
            suffix = Path(unquote(title)).suffix.lower()
            if (
                license_name not in LICENSES
                or "/publicdomain/zero/1.0" not in license_url
                or suffix not in ALLOWED_SUFFIXES
                or min(width, height) < 768
                or max(width, height) / max(1, min(width, height)) > 1.33
                or int(info.get("size", 0)) <= 0
                or int(info.get("size", 0)) > 30 * 1024 * 1024
                or not info.get("url")
                or not info.get("thumburl")
            ):
                continue
            found.append(
                {
                    "title": title,
                    "description_url": info.get("descriptionurl"),
                    "original_url": info.get("url"),
                    "download_url": info.get("thumburl"),
                    "download_variant": "Wikimedia 1024 px thumbnail; source aspect ratio <= 1.33",
                    "thumbnail_width": int(info.get("thumbwidth", 0)),
                    "thumbnail_height": int(info.get("thumbheight", 0)),
                    "width": width,
                    "height": height,
                    "bytes": int(info["size"]),
                    "license": license_name,
                    "license_url": license_url,
                    "artist_credit": plain(meta.get("Artist")),
                    "description": plain(meta.get("ImageDescription")),
                    "date_time_original": plain(meta.get("DateTimeOriginal")),
                    "search_phrase": phrase,
                }
            )
            phrase_count += 1
        print(f"[commons] search {phrase!r}: {phrase_count} eligible CC0 candidates", flush=True)
        time.sleep(0.5)
    return found


def average_hash(image: Image.Image) -> int:
    pixels = list(ImageOps.grayscale(image).resize((8, 8)).tobytes())
    average = sum(pixels) / len(pixels)
    value = 0
    for pixel in pixels:
        value = (value << 1) | int(pixel >= average)
    return value


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def diversify_candidates(pool: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in pool:
        groups.setdefault(str(row.get("search_phrase") or "cc-zero-category"), []).append(row)
    keys = list(groups)
    rng.shuffle(keys)
    for rows in groups.values():
        rng.shuffle(rows)
    ordered: list[dict[str, Any]] = []
    while any(groups.values()):
        for key in keys:
            if groups[key]:
                ordered.append(groups[key].pop())
    return ordered


def write_manifest(
    path: Path,
    *,
    status: str,
    seed: int,
    required: int,
    candidate_count: int,
    selected: list[dict[str, Any]],
    errors: list[dict[str, str]],
) -> None:
    payload = {
        "status": status,
        "source": "Wikimedia Commons API, Category:CC-Zero",
        "api": API,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "collection_seed": seed,
        "required_count": required,
        "candidate_count": candidate_count,
        "selected_count": len(selected),
        "license_rule": "per-file LicenseShortName CC0/CC-Zero and LicenseUrl contains /publicdomain/zero/1.0",
        "images": selected,
        "rejected_downloads": errors,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def save_png(image: Image.Image, path: Path, edge: int) -> str:
    resized = image.resize((edge, edge), Image.Resampling.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    resized.save(path, format="PNG", optimize=True)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--candidate-pages", type=int, default=8)
    parser.add_argument("--search-terms", nargs="*", default=[])
    parser.add_argument("--seed", type=int, default=20261107)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "commons-source-manifest.json"
    attempts_dir = args.out / "acquisition_attempts"
    attempts_dir.mkdir(parents=True, exist_ok=True)
    prior_selected: list[dict[str, Any]] = []
    prior_errors: list[dict[str, str]] = []
    if manifest_path.exists():
        existing_bytes = manifest_path.read_bytes()
        existing_sha = hashlib.sha256(existing_bytes).hexdigest()
        archived_hashes = {
            hashlib.sha256(path.read_bytes()).hexdigest()
            for path in attempts_dir.glob("*.json")
        }
        if existing_sha not in archived_hashes:
            number = 1 + len(list(attempts_dir.glob("collection-attempt-*-partial.json")))
            shutil.copy2(manifest_path, attempts_dir / f"collection-attempt-{number:02d}-partial.json")
        prior = json.loads(existing_bytes)
        if (
            prior.get("status") != "complete"
            and prior.get("collection_seed") == args.seed
            and prior.get("required_count") == args.count
        ):
            prior_selected = list(prior.get("images", []))
            prior_errors = list(prior.get("rejected_downloads", []))

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    pool = candidates(session, args.candidate_pages)
    if args.search_terms:
        search_pool = search_candidates(session, args.search_terms)
        seen = {row["title"] for row in pool}
        pool.extend(row for row in search_pool if row["title"] not in seen)
    rejected_titles = {row.get("title") for row in prior_errors}
    print(f"[commons] candidate pool after per-file CC0/dimension filter: {len(pool)}", flush=True)
    pool = diversify_candidates(pool, args.seed)
    selected: list[dict[str, Any]] = prior_selected
    hashes: set[str] = {str(row.get("source_download_sha256")) for row in selected}
    perceptual: list[int] = [
        int(str(row["perceptual_hash_average_8x8"]), 16)
        for row in selected
        if row.get("perceptual_hash_average_8x8")
    ]
    errors: list[dict[str, str]] = prior_errors
    raw_dir = args.out / "source_downloads"
    image_dirs = {512: args.out / "images" / "512", 768: args.out / "images" / "768"}
    write_manifest(
        manifest_path,
        status="in_progress",
        seed=args.seed,
        required=args.count,
        candidate_count=len(pool),
        selected=selected,
        errors=errors,
    )

    for candidate in pool:
        if len(selected) >= args.count:
            break
        if candidate["title"] in {row["title"] for row in selected}:
            continue
        if candidate["title"] in rejected_titles:
            continue
        title_slug = re.sub(r"[^A-Za-z0-9]+", "_", candidate["title"].removeprefix("File:"))[:72].strip("_")
        try:
            response = None
            for attempt in range(3):
                print(f"[commons] download attempt {attempt + 1}/3: {candidate['title']}", flush=True)
                response = session.get(candidate["download_url"], timeout=(10, 20))
                if response.status_code != 429 and response.status_code < 500:
                    break
                retry_after = response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else min(2**attempt, 8)
                if attempt == 2:
                    break
                time.sleep(min(15.0, max(1.5, delay)))
            assert response is not None
            response.raise_for_status()
            raw = response.content
            if len(raw) > 10 * 1024 * 1024:
                raise ValueError("thumbnail exceeds 10 MiB safety cap")
            raw_sha = hashlib.sha256(raw).hexdigest()
            if raw_sha in hashes:
                raise ValueError("duplicate exact source bytes")
            with Image.open(BytesIO(raw)) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
            if min(image.size) < 768:
                raise ValueError(f"decoded dimensions below 768: {image.size}")
            side = min(image.size)
            left = (image.width - side) // 2
            top = (image.height - side) // 2
            square = image.crop((left, top, left + side, top + side))
            phash = average_hash(square)
            if any(hamming(phash, prior) <= 3 for prior in perceptual):
                raise ValueError("near-duplicate perceptual hash")

            image_id = f"cc0-{len(selected):02d}-{raw_sha[:12]}"
            raw_suffix = Path(unquote(candidate["title"]).removeprefix("File:")).suffix.lower()
            raw_path = raw_dir / f"{image_id}{raw_suffix or '.bin'}"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_bytes(raw)
            processed = {}
            for edge in (512, 768):
                output = image_dirs[edge] / f"{image_id}.png"
                processed[str(edge)] = {
                    "path": str(output),
                    "sha256": save_png(square, output, edge),
                    "width": edge,
                    "height": edge,
                }
            row = {
                **candidate,
                "image_id": image_id,
                "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                "source_download_path": str(raw_path),
                "source_download_sha256": raw_sha,
                "source_download_bytes": len(raw),
                "raw_decoded_size": [image.width, image.height],
                "crop": {"method": "center_square_after_exif_transpose", "box": [left, top, left + side, top + side]},
                "perceptual_hash_average_8x8": f"{phash:016x}",
                "processed": processed,
            }
            selected.append(row)
            hashes.add(raw_sha)
            perceptual.append(phash)
            time.sleep(1.5)
            print(f"[commons] selected {len(selected)}/{args.count}: {candidate['title']}", flush=True)
        except Exception as exc:
            errors.append({"title": candidate["title"], "error": f"{type(exc).__name__}: {exc}"})
            time.sleep(1.5)
            print(f"[commons] rejected: {candidate['title']} ({errors[-1]['error']})", flush=True)
        write_manifest(
            manifest_path,
            status="complete" if len(selected) >= args.count else "in_progress",
            seed=args.seed,
            required=args.count,
            candidate_count=len(pool),
            selected=selected,
            errors=errors,
        )

    print(json.dumps({"candidates": len(pool), "selected": len(selected), "rejected": len(errors), "manifest": str(manifest_path)}))
    if len(selected) < args.count:
        raise SystemExit(f"only collected {len(selected)} of {args.count}; retain partial manifest and extend the candidate scan")


if __name__ == "__main__":
    main()
