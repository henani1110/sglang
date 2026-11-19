#!/usr/bin/env python3
"""Create immutable 512px and 768px geometry image sets on the SSD."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from PIL import Image


ROOT = Path("/media/chx/Win10/vl-stage-split-20261106-191642")
SOURCE = ROOT / "datasets"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    manifest: dict[str, object] = {"source_dir": str(SOURCE), "sets": {}}
    for size in (512, 768):
        target_dir = ROOT / f"datasets-{size}"
        target_dir.mkdir(parents=True, exist_ok=True)
        entries = []
        for index in range(4):
            name = f"geometry-{index:02d}.png"
            src = SOURCE / name
            dst = target_dir / name
            if not src.is_file():
                raise FileNotFoundError(src)
            if not dst.exists():
                with Image.open(src) as image:
                    image = image.convert("RGB")
                    if image.size != (size, size):
                        image = image.resize((size, size), Image.Resampling.LANCZOS)
                    image.save(dst, format="PNG", optimize=False)
            with Image.open(dst) as actual:
                if actual.size != (size, size):
                    raise ValueError(f"{dst} has size {actual.size}, expected {(size, size)}")
            with Image.open(src) as source_image:
                source_size = list(source_image.size)
            entries.append({
                "file": name,
                "source_size": source_size,
                "output_size": [size, size],
                "sha256": sha256(dst),
            })
        manifest["sets"][str(size)] = {"directory": str(target_dir), "images": entries}
    out = ROOT / "continuation/manifests/C-datasets.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    if out.exists() and out.read_text() != serialized:
        raise FileExistsError(f"refusing to overwrite a different dataset manifest: {out}")
    out.write_text(serialized)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
