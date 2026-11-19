#!/usr/bin/env python3
"""Pinned local implementation of ChartQA paper relaxed accuracy plus diagnostics."""
from __future__ import annotations

import math
import re

YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")


def parse_number(value: str) -> float | None:
    try:
        parsed = float(value.strip())
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def normalize_text(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def relaxed_accuracy(prediction: str, target: str, tolerance: float = 0.05) -> bool:
    """Numeric values use paper's 5% relative tolerance; text uses case-insensitive exact match."""
    p = parse_number(prediction)
    t = parse_number(target)
    if p is not None and t is not None:
        if t == 0:
            return p == 0
        return abs(p - t) / abs(t) <= tolerance
    return normalize_text(prediction) == normalize_text(target)


def exact_match(prediction: str, target: str) -> bool:
    return normalize_text(prediction) == normalize_text(target)


def is_year(target: str) -> bool:
    return bool(YEAR_RE.fullmatch(str(target).strip()))


def year_strict_correct(prediction: str, target: str) -> bool | None:
    if not is_year(target):
        return None
    return normalize_text(prediction) == normalize_text(target)
