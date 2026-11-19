#!/usr/bin/env python3
"""Build a sorted SHA-256 manifest for the complete E workspace."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def excluded(path: Path, output: Path) -> bool:
    rel = path.relative_to(output.parents[1])
    return (
        path == output
        or "__pycache__" in rel.parts
        or path.suffix == ".pyc"
        or path.name == ".DS_Store"
        or path.is_symlink()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("archive/e-v0.10-full-sha256-v0.1.txt"))
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    paths = sorted(
        (path for path in root.rglob("*") if path.is_file() and not excluded(path, output)),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for path in paths:
            stream.write(f"{digest(path)}  {path.relative_to(root).as_posix()}\n")
    total_bytes = sum(path.stat().st_size for path in paths)
    print(f"manifest={output}")
    print(f"files={len(paths)}")
    print(f"bytes={total_bytes}")
    print("excluded=manifest itself, __pycache__, .pyc, .DS_Store, symlinks")


if __name__ == "__main__":
    main()
