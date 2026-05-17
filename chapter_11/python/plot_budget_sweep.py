#!/usr/bin/env python3
"""
Dependency-free SVG plots for Chapter 11 budget sweeps.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


PLOTS = [
    ("ttft_mean_ms", "Mean TTFT", "ms"),
    ("ttft_p95_ms", "P95 TTFT", "ms"),
    ("total_mean_ms", "Mean Total Latency", "ms"),
    ("total_p95_ms", "P95 Total Latency", "ms"),
    ("throughput_req_s", "Throughput", "req/s"),
    ("gpu_power_mean_w", "Mean GPU Power", "W"),
    ("gpu_power_peak_w", "Peak GPU Power", "W"),
    ("energy_per_request_j", "Energy Per Request", "J/req"),
    ("error_rate", "Error Rate", "fraction"),
    ("vllm_queue_wait_mean_ms", "vLLM Queue Wait", "ms"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("summary_csv")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    summary_csv = Path(args.summary_csv)
    out_dir = Path(args.out_dir) if args.out_dir else summary_csv.parent / "plots"
    paths = plot_sweep(summary_csv, out_dir)
    for path in paths:
        print(path)


def plot_sweep(summary_csv: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(summary_csv)
    paths: list[Path] = []
    for key, title, unit in PLOTS:
        if any(is_finite(row.get(key)) for row in rows):
            path = out_dir / f"{key}.svg"
            write_single_plot(path, rows, key, title, unit)
            paths.append(path)
    dashboard = out_dir / "phase1_dashboard.svg"
    write_dashboard(dashboard, rows)
    paths.insert(0, dashboard)
    return paths


def read_rows(path: Path) -> list[dict[str, float | str | None]]:
    with path.open() as f:
        reader = csv.DictReader(f)
        rows = []
        for raw in reader:
            row: dict[str, float | str | None] = {}
            for key, value in raw.items():
                if value is None or value == "":
                    row[key] = None
                    continue
                try:
                    row[key] = float(value)
                except ValueError:
                    row[key] = value
            rows.append(row)
    rows.sort(key=lambda r: float(r.get("admission_fraction") or 0), reverse=True)
    return rows


def write_single_plot(path: Path, rows: list[dict[str, float | str | None]], key: str, title: str, unit: str) -> None:
    svg = render_plot(rows, key, title, unit, width=900, height=560)
    path.write_text(svg)


def write_dashboard(path: Path, rows: list[dict[str, float | str | None]]) -> None:
    width = 1400
    height = 1050
    cell_w = 700
    cell_h = 350
    selected = [
        ("ttft_mean_ms", "Mean TTFT", "ms"),
        ("total_mean_ms", "Mean Total Latency", "ms"),
        ("throughput_req_s", "Throughput", "req/s"),
        ("gpu_power_mean_w", "Mean GPU Power", "W"),
        ("energy_per_request_j", "Energy Per Request", "J/req"),
        ("error_rate", "Error Rate", "fraction"),
    ]
    parts = [
        svg_header(width, height),
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="32" y="42" font-size="28" font-family="Arial" font-weight="700">'
        "Chapter 11 Phase 1: Token-Budget Plant Curves</text>",
    ]
    for idx, (key, title, unit) in enumerate(selected):
        x = (idx % 2) * cell_w
        y = 70 + (idx // 2) * cell_h
        parts.append(f'<g transform="translate({x},{y})">')
        parts.append(render_plot_body(rows, key, title, unit, width=cell_w, height=cell_h, compact=True))
        parts.append("</g>")
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def render_plot(rows, key, title, unit, width=900, height=560) -> str:
    return "\n".join(
        [
            svg_header(width, height),
            '<rect width="100%" height="100%" fill="#ffffff"/>',
            render_plot_body(rows, key, title, unit, width, height, compact=False),
            "</svg>",
        ]
    )


def render_plot_body(rows, key, title, unit, width, height, compact=False) -> str:
    margin_l = 88 if not compact else 72
    margin_r = 35
    margin_t = 70 if not compact else 48
    margin_b = 70 if not compact else 58
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    points = [
        (float(row["admission_fraction"]), float(row[key]))
        for row in rows
        if is_finite(row.get("admission_fraction")) and is_finite(row.get(key))
    ]
    if not points:
        return ""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = nice_bounds(min(ys), max(ys))

    def sx(x):
        if x_max == x_min:
            return margin_l + plot_w / 2
        return margin_l + (x - x_min) / (x_max - x_min) * plot_w

    def sy(y):
        if y_max == y_min:
            return margin_t + plot_h / 2
        return margin_t + plot_h - (y - y_min) / (y_max - y_min) * plot_h

    grid = []
    for i in range(6):
        t = i / 5
        y_val = y_min + t * (y_max - y_min)
        y = sy(y_val)
        grid.append(f'<line x1="{margin_l}" y1="{y:.1f}" x2="{margin_l + plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        grid.append(f'<text x="{margin_l - 10}" y="{y + 4:.1f}" text-anchor="end" font-size="12" font-family="Arial" fill="#374151">{fmt(y_val)}</text>')
    for x_val in sorted(set(xs)):
        x = sx(x_val)
        grid.append(f'<line x1="{x:.1f}" y1="{margin_t}" x2="{x:.1f}" y2="{margin_t + plot_h}" stroke="#f3f4f6"/>')
        grid.append(f'<text x="{x:.1f}" y="{margin_t + plot_h + 24}" text-anchor="middle" font-size="12" font-family="Arial" fill="#374151">{x_val:g}</text>')

    polyline = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in points)
    dots = []
    for x, y in points:
        dots.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="5" fill="#2563eb"/>')
        dots.append(f'<text x="{sx(x):.1f}" y="{sy(y) - 10:.1f}" text-anchor="middle" font-size="11" font-family="Arial" fill="#1f2937">{fmt(y)}</text>')

    return "\n".join(
        [
            f'<text x="{margin_l}" y="{margin_t - 24}" font-size="{20 if compact else 24}" font-family="Arial" font-weight="700" fill="#111827">{title}</text>',
            f'<text x="{margin_l}" y="{height - 18}" font-size="13" font-family="Arial" fill="#374151">admission_fraction</text>',
            f'<text x="18" y="{margin_t - 12}" font-size="13" font-family="Arial" fill="#374151">{unit}</text>',
            *grid,
            f'<rect x="{margin_l}" y="{margin_t}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#9ca3af"/>',
            f'<polyline points="{polyline}" fill="none" stroke="#2563eb" stroke-width="3"/>',
            *dots,
        ]
    )


def nice_bounds(lo: float, hi: float) -> tuple[float, float]:
    if lo == hi:
        pad = abs(lo) * 0.1 if lo else 1.0
        return lo - pad, hi + pad
    pad = 0.08 * (hi - lo)
    return lo - pad, hi + pad


def is_finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def fmt(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def svg_header(width: int, height: int) -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'


if __name__ == "__main__":
    main()
