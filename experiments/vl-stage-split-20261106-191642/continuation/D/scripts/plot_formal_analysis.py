#!/usr/bin/env python3
"""Render compact, dependency-free SVG figures from the frozen D summaries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape


COLORS = {"N": "#4C78A8", "F": "#F58518", "S": "#E45756", "A": "#54A24B", "U": "#B279A2"}


def t(x: float, y: float, value: str, size: int = 13, anchor: str = "middle", color: str = "#202124", weight: str = "normal") -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" font-size="{size}" text-anchor="{anchor}" fill="{color}" font-weight="{weight}">{escape(value)}</text>'


def line(x1: float, y1: float, x2: float, y2: float, color: str = "#9aa0a6", width: float = 1.0) -> str:
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width:.1f}"/>'


def rect(x: float, y: float, w: float, h: float, color: str, opacity: float = 1.0) -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{color}" fill-opacity="{opacity:.2f}" rx="1.5"/>'


def save_svg(path: Path, width: int, height: int, elements: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(elements)
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n'
        f'<rect width="100%" height="100%" fill="white"/>\n{body}\n</svg>\n',
        encoding="utf-8",
    )


def grouped_goodput(cells: list[dict], output: Path) -> None:
    width, height = 1320, 690
    e = [t(width / 2, 34, "Joint 30/55 ms SLO goodput by load and policy", 20, weight="bold")]
    series = ["N", "F", "S", "A", "U"]
    for panel, grid in enumerate((512, 768)):
        left = 74 + panel * 650
        top, plot_h, plot_w = 80, 470, 570
        bottom = top + plot_h
        e.append(t(left + plot_w / 2, 64, f"{grid} image grid", 16, weight="bold"))
        for tick in range(0, 11, 2):
            y = bottom - plot_h * tick / 10
            e.append(line(left, y, left + plot_w, y, "#e5e7eb"))
            e.append(t(left - 12, y + 4, f"{tick/10:.1f}", 11, "end", "#5f6368"))
        e.append(line(left, top, left, bottom, "#5f6368"))
        e.append(line(left, bottom, left + plot_w, bottom, "#5f6368"))
        subset = [c for c in cells if c["grid"] == grid]
        order = [(2, "uniform"), (2, "burst"), (6, "uniform"), (6, "burst")]
        for group, (lanes, arrival) in enumerate(order):
            cell_x = left + 18 + group * 138
            e.append(t(cell_x + 52, bottom + 22, f"L{lanes} {arrival}", 11))
            bars = [c for c in subset if c["text_streams"] == lanes and c["arrival_mode"] == arrival]
            bars.sort(key=lambda c: series.index(c["policy"]) if c["policy"] in series else 99)
            for j, c in enumerate(bars):
                value = c["metrics_across_seed_runs"]["joint_30_55_goodput_per_s"]["mean"] or 0.0
                x = cell_x + j * 19
                bar_h = max(0.0, min(1.0, value)) * plot_h
                e.append(rect(x, bottom - bar_h, 14, bar_h, COLORS[c["policy"]]))
        if panel == 0:
            e.append(f'<text transform="translate(20 {top + plot_h / 2:.1f}) rotate(-90)" font-family="Arial, sans-serif" font-size="12" text-anchor="middle" fill="#5f6368">goodput (passing images/s)</text>')
    legend_y = 640
    start_x = 445
    for j, policy in enumerate(series):
        x = start_x + j * 90
        e.append(rect(x, legend_y - 12, 13, 13, COLORS[policy]))
        e.append(t(x + 20, legend_y, policy, 12, "start"))
    e.append(t(width / 2, 674, "Arrival conditions are open-loop; bars are means across 12 paired trace seeds per cell.", 11, color="#5f6368"))
    save_svg(output, width, height, e)


def paired_delta(paired: list[dict], output: Path) -> None:
    candidates = ["F", "S", "A"]
    rows = []
    for item in paired:
        if item.get("metric") == "joint_30_55_goodput_per_s" and item.get("policy_vs_N") in candidates:
            rows.append(item)
    rows.sort(key=lambda x: (x["grid"], x["text_streams"], x["arrival_mode"], candidates.index(x["policy_vs_N"])))
    vals = [z for r in rows for z in [*r["bootstrap_95_ci"], r["mean_paired_difference_candidate_minus_N"]]]
    lo, hi = min(vals + [-0.01]), max(vals + [0.01])
    pad = max(0.005, (hi - lo) * 0.08)
    lo -= pad
    hi += pad
    width, height = 1250, 760
    left, right, top, bottom = 280, 1155, 92, 700
    plot_w, plot_h = right - left, bottom - top
    e = [t(width / 2, 36, "Paired change in joint goodput versus native N", 20, weight="bold"),
         t(width / 2, 61, "Point = mean paired difference; line = 95% seed-bootstrap CI (5,000 resamples)", 13, color="#5f6368")]
    for tick in range(6):
        value = lo + (hi - lo) * tick / 5
        x = left + plot_w * tick / 5
        e.append(line(x, top, x, bottom, "#e5e7eb"))
        e.append(t(x, bottom + 22, f"{value:.3f}", 11, color="#5f6368"))
    zero_x = left + plot_w * ((0 - lo) / (hi - lo))
    e.append(line(zero_x, top, zero_x, bottom, "#202124", 1.5))
    e.append(line(left, bottom, right, bottom, "#5f6368"))
    e.append(t(width / 2, 746, "Change in passing images per second; positive favors candidate", 13, color="#5f6368"))
    labels = []
    for grid in (512, 768):
        for lanes in (2, 6):
            for arrival in ("uniform", "burst"):
                labels.append((grid, lanes, arrival))
    group_y = {k: top + i * (plot_h / len(labels)) + 24 for i, k in enumerate(labels)}
    colors = {"F": COLORS["F"], "S": COLORS["S"], "A": COLORS["A"]}
    for i, (grid, lanes, arrival) in enumerate(labels):
        y0 = group_y[(grid, lanes, arrival)]
        if i % 2 == 0:
            e.append(rect(0, y0 - 17, width, 34, "#f7f8fa"))
        e.append(t(left - 16, y0 + 4, f"G{grid} · L{lanes} · {arrival}", 12, "end"))
        for j, candidate in enumerate(candidates):
            r = next(x for x in rows if x["grid"] == grid and x["text_streams"] == lanes and x["arrival_mode"] == arrival and x["policy_vs_N"] == candidate)
            ci0, ci1 = r["bootstrap_95_ci"]
            mean = r["mean_paired_difference_candidate_minus_N"]
            y = y0 + (j - 1) * 8
            x0, x1 = left + plot_w * (ci0 - lo) / (hi - lo), left + plot_w * (ci1 - lo) / (hi - lo)
            xm = left + plot_w * (mean - lo) / (hi - lo)
            e.append(line(x0, y, x1, y, colors[candidate], 2.4))
            e.append(line(x0, y - 4, x0, y + 4, colors[candidate], 1.4))
            e.append(line(x1, y - 4, x1, y + 4, colors[candidate], 1.4))
            e.append(f'<circle cx="{xm:.1f}" cy="{y:.1f}" r="4.0" fill="{colors[candidate]}"/>')
    for j, candidate in enumerate(candidates):
        x = 480 + j * 105
        e.append(line(x, 82, x + 28, 82, colors[candidate], 2.5))
        e.append(t(x + 35, 86, f"{candidate} − N", 12, "start"))
    save_svg(output, width, height, e)


def state_audit(states: list[dict], output: Path) -> None:
    states.sort(key=lambda x: (x["d_slack_bin"], x["i_slack_bin"], x["e_over_p_bin"]))
    width, height = 1320, 610
    left, right, top = 390, 1000, 118
    max_count = max((s["decisions"] for s in states), default=1)
    plot_w = right - left
    e = [t(width / 2, 36, "A policy: observed actions by predicted slack state", 20, weight="bold"),
         t(width / 2, 62, "Each row is a request decision; bars count selected N/S actions.", 13, color="#5f6368")]
    e.append(rect(left, 84, 16, 14, COLORS["N"]))
    e.append(t(left + 23, 96, "selected N", 12, "start"))
    e.append(rect(left + 132, 84, 16, 14, COLORS["S"]))
    e.append(t(left + 155, 96, "selected S", 12, "start"))
    for tick in range(6):
        count = max_count * tick / 5
        x = left + plot_w * tick / 5
        e.append(line(x, top, x, top + len(states) * 66, "#e5e7eb"))
        e.append(t(x, top - 10, f"{count:.0f}", 11, color="#5f6368"))
    for i, s in enumerate(states):
        y = top + i * 66
        if i % 2 == 0:
            e.append(rect(0, y - 20, width, 58, "#f7f8fa"))
        label = f"D:{s['d_slack_bin']} · I:{s['i_slack_bin']} · {s['e_over_p_bin']}"
        e.append(t(left - 18, y + 1, label, 11, "end"))
        n, split = s.get("selected_N", 0), s.get("selected_S", 0)
        n_w, s_w = plot_w * n / max_count, plot_w * split / max_count
        e.append(rect(left, y - 11, n_w, 18, COLORS["N"]))
        e.append(rect(left + n_w, y - 11, s_w, 18, COLORS["S"]))
        rate = (s.get("actual_split", 0) / split) if split else None
        pass_rate = (s.get("joint_30_55_passes", 0) / s["decisions"]) if s.get("decisions") else 0.0
        rate_text = f"S executed {s.get('actual_split', 0)}/{split}; joint-pass {pass_rate:.1%}; trace {s.get('trace_order_valid', 0)}/{s['decisions']}"
        e.append(t(right + 18, y + 2, rate_text, 11, "start", "#3c4043"))
        e.append(t(left, y + 24, f"{s['decisions']} decisions", 10, "start", "#5f6368"))
    e.append(t(width / 2, 590, "Decision requests are descriptive; paired inference uses trace seed, not request count.", 11, color="#5f6368"))
    save_svg(output, width, height, e)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, required=True)
    args = parser.parse_args()
    base = args.analysis_dir
    cells = json.loads((base / "formal-cell-summary.json").read_text(encoding="utf-8"))
    paired = json.loads((base / "formal-paired-bootstrap.json").read_text(encoding="utf-8"))
    states = json.loads((base / "formal-state-partition.json").read_text(encoding="utf-8"))
    out = base / "plots"
    grouped_goodput(cells, out / "joint-goodput-30-55.svg")
    paired_delta(paired, out / "paired-joint-goodput-delta.svg")
    state_audit(states, out / "adaptive-state-audit.svg")
    print(json.dumps({"status": "complete", "plots": [str(p) for p in sorted(out.glob("*.svg"))]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
