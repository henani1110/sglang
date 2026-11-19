#!/usr/bin/env python3
"""Create four fixed 768px RGB geometric PNGs for the pilot trace."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

SEED = 20261106
SIZE = 768
COUNT = 4


def main() -> None:
    import sys

    target = Path(sys.argv[1])
    target.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    records = []
    for image_index in range(COUNT):
        image = Image.new("RGB", (SIZE, SIZE), (248, 248, 244))
        draw = ImageDraw.Draw(image)
        draw.rectangle((22, 22, SIZE - 23, SIZE - 23), outline=(30, 39, 50), width=8)
        draw.line((SIZE // 2, 30, SIZE // 2, SIZE - 30), fill=(200, 205, 207), width=2)
        draw.line((30, SIZE // 2, SIZE - 30, SIZE // 2), fill=(200, 205, 207), width=2)
        palette = [
            (34, 95, 160),
            (216, 90, 59),
            (36, 142, 113),
            (235, 174, 47),
            (118, 89, 160),
        ]
        for shape_index in range(28):
            x0 = rng.randrange(45, SIZE - 180)
            y0 = rng.randrange(45, SIZE - 180)
            extent = rng.randrange(52, 205)
            x1, y1 = min(SIZE - 45, x0 + extent), min(SIZE - 45, y0 + extent)
            color = palette[(image_index + shape_index + rng.randrange(len(palette))) % len(palette)]
            fill = tuple(int((channel + 248) / 2) for channel in color)
            kind = (shape_index + image_index) % 3
            if kind == 0:
                draw.rectangle((x0, y0, x1, y1), fill=fill, outline=color, width=4)
            elif kind == 1:
                draw.ellipse((x0, y0, x1, y1), fill=fill, outline=color, width=4)
            else:
                draw.polygon(
                    ((x0 + extent // 2, y0), (x1, y1), (x0, y1)),
                    fill=fill,
                    outline=color,
                )
        path = target / f"geometry-{image_index:02d}.png"
        image.save(path, format="PNG", optimize=False, compress_level=9)
        payload = path.read_bytes()
        records.append(
            {
                "id": image_index,
                "path": str(path),
                "width": SIZE,
                "height": SIZE,
                "mode": "RGB",
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    (target / "images.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "rule": "Python random.Random(seed), 768x768 RGB, border/cross and 28 seeded colored rectangles/ellipses/triangles; Pillow PNG compression level 9.",
                "images": records,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
