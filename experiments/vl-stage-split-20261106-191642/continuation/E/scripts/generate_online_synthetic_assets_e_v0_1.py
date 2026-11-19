#!/usr/bin/env python3
"""Generate deterministic PNG chart assets without network or third-party data."""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import zlib
from pathlib import Path


def chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def render_chart(index: int, size: int = 512) -> bytes:
    pixels = bytearray([255]) * (size * size * 3)

    def rect(x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
        x0, x1 = sorted((max(0, x0), min(size - 1, x1)))
        y0, y1 = sorted((max(0, y0), min(size - 1, y1)))
        row = bytes(color) * (x1 - x0 + 1)
        for y in range(y0, y1 + 1):
            start = (y * size + x0) * 3
            pixels[start : start + len(row)] = row

    # Axes and light grid.
    for y in range(112, 425, 52):
        rect(76, y, 448, y + 1, (225, 229, 234))
    rect(76, 110, 79, 426, (35, 40, 46))
    rect(76, 422, 450, 425, (35, 40, 46))

    count = 6 + index * 2
    color = ((36, 116, 181), (227, 126, 49), (65, 150, 105), (150, 92, 170))[index]
    gap = 8 + index
    plot_width = 344
    bar_width = max(8, (plot_width - (count - 1) * gap) // count)
    base = 420
    for bar in range(count):
        # A deterministic chart shape with progressively denser categories.
        value = 64 + ((bar * 73 + index * 41 + (bar * bar * 11)) % 220)
        x = 92 + bar * (bar_width + gap)
        rect(x, base - value, x + bar_width, base, color)
        if index >= 2:
            rect(x + 2, base - value + 8, x + bar_width - 2, base - value + 10, (255, 255, 255))
            rect(x + 2, base - value + 18, x + bar_width - 2, base - value + 20, (255, 255, 255))

    # A contrasting trend series, useful to the visual encoder but not sourced
    # from any benchmark or external image collection.
    points = []
    for x in range(92, 440, 5):
        y = 160 + ((x * (index + 3) + (x // 17) * 29 + index * 37) % 190)
        points.append((x, y))
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
        err = dx - dy
        while True:
            rect(x0 - 1, y0 - 1, x0 + 1, y0 + 1, (174, 53, 52))
            if x0 == x1 and y0 == y1:
                break
            twice = 2 * err
            if twice > -dy:
                err -= dy
                x0 += sx
            if twice < dx:
                err += dx
                y0 += sy

    raw = b"".join(b"\x00" + bytes(pixels[y * size * 3 : (y + 1) * size * 3]) for y in range(size))
    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()
    if args.size not in (512, 768):
        parser.error("size must be 512 or 768")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for index in range(4):
        name = f"geometry-{index:02d}.png"
        data = render_chart(index, args.size)
        path = args.output_dir / name
        path.write_bytes(data)
        rows.append({"image_id": name.removesuffix(".png"), "path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "synthetic": True})
    manifest = {
        "schema_version": 1,
        "dataset_id": "e-online-synthetic-charts-v0.1",
        "origin": "deterministic local raster generator; no external source or benchmark annotations",
        "size": args.size,
        "selected_count": len(rows),
        "images": rows,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
