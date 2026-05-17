#!/usr/bin/env python3
"""
Dependency-free SVG time-series plots for Chapter 11 Phase 2 TTFT sweep.

Generates per-target:
  - TTFT over time (measured vs target)
  - Admission fraction over time
  - GPU power over time
  - 3-panel dashboard
"""

from __future__ import annotations

import math
from pathlib import Path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_result(ts: list[dict], summary: dict, plots_dir: Path) -> list[Path]:
    """Generate all SVG plots for one TTFT target run.

    ts: list of timeseries dicts from run_internal_ttft_sweep.
    summary: per-target summary dict (without the timeseries key).
    plots_dir: directory to write SVG files into.
    """
    plots_dir.mkdir(parents=True, exist_ok=True)
    target_ms = int(summary.get("target_ttft_ms", 0))
    prefix = f"target_{target_ms}ms"

    if not ts:
        return []

    t_vals = [row.get("t", 0.0) for row in ts]
    measured_ttft = [row.get("measured_ttft_ms") for row in ts]
    fraction = [row.get("admission_fraction") for row in ts]
    power = [row.get("gpu_power_w") for row in ts]
    target_line = [float(target_ms)] * len(ts)

    paths: list[Path] = []

    # 1. TTFT over time
    p = _svg_timeseries(
        t_vals=t_vals,
        series=[
            {"y": measured_ttft, "label": "Measured TTFT (ms)", "color": "#1565c0"},
            {"y": target_line, "label": f"Target {target_ms} ms", "color": "#c62828", "dashed": True},
        ],
        title=f"TTFT vs Time — target {target_ms} ms",
        xlabel="Time (s)",
        ylabel="TTFT (ms)",
    )
    path = plots_dir / f"{prefix}_ttft.svg"
    path.write_text(p)
    paths.append(path)

    # 2. Admission fraction over time
    p = _svg_timeseries(
        t_vals=t_vals,
        series=[
            {"y": fraction, "label": "Admission fraction", "color": "#2e7d32"},
        ],
        title=f"Admission Fraction vs Time — target {target_ms} ms",
        xlabel="Time (s)",
        ylabel="Fraction",
        y_min=0.0,
        y_max=1.05,
    )
    path = plots_dir / f"{prefix}_fraction.svg"
    path.write_text(p)
    paths.append(path)

    # 3. GPU power over time
    p = _svg_timeseries(
        t_vals=t_vals,
        series=[
            {"y": power, "label": "GPU Power (W)", "color": "#e65100"},
        ],
        title=f"GPU Power vs Time — target {target_ms} ms",
        xlabel="Time (s)",
        ylabel="Power (W)",
    )
    path = plots_dir / f"{prefix}_power.svg"
    path.write_text(p)
    paths.append(path)

    # 4. 3-panel dashboard
    p = _svg_dashboard(ts, target_ms, summary)
    path = plots_dir / f"{prefix}_dashboard.svg"
    path.write_text(p)
    paths.append(path)

    return paths


# ---------------------------------------------------------------------------
# SVG helpers
# ---------------------------------------------------------------------------

def _svg_timeseries(
    t_vals: list,
    series: list[dict],
    title: str,
    xlabel: str,
    ylabel: str,
    width: int = 800,
    height: int = 320,
    y_min: float | None = None,
    y_max: float | None = None,
) -> str:
    pad_l, pad_r, pad_t, pad_b = 72, 24, 38, 48
    w = width - pad_l - pad_r
    h = height - pad_t - pad_b

    all_y = [v for s in series for v in s["y"] if v is not None and math.isfinite(float(v))]
    if not all_y or not t_vals:
        return _empty_svg(width, height, "No data")

    t_min = min(t_vals)
    t_max = max(t_vals)
    data_y_min = min(all_y)
    data_y_max = max(all_y)
    pad_frac = 0.06

    if y_min is None:
        y_min = max(0.0, data_y_min - pad_frac * max(1.0, data_y_max - data_y_min))
    if y_max is None:
        y_max = data_y_max + pad_frac * max(1.0, data_y_max - data_y_min)
    if abs(y_max - y_min) < 1e-9:
        y_max = y_min + 1.0
    t_range = max(t_max - t_min, 1e-9)
    y_range = y_max - y_min

    def tx(t: float) -> float:
        return pad_l + (t - t_min) / t_range * w

    def ty(y: float) -> float:
        return pad_t + h - (y - y_min) / y_range * h

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        "<style>text{font-family:Arial,sans-serif;font-size:11px;fill:#222}</style>",
        f'<rect width="{width}" height="{height}" fill="#fff"/>',
        f'<rect x="{pad_l}" y="{pad_t}" width="{w}" height="{h}" fill="#fafafa" stroke="#ccc" stroke-width="1"/>',
        f'<text x="{width//2}" y="22" text-anchor="middle" font-size="14" font-weight="bold">{_esc(title)}</text>',
        f'<text x="{pad_l + w//2}" y="{height - 6}" text-anchor="middle">{_esc(xlabel)}</text>',
        f'<text x="13" y="{pad_t + h//2}" text-anchor="middle" transform="rotate(-90 13 {pad_t + h//2})">{_esc(ylabel)}</text>',
    ]

    for v in _nice_ticks(y_min, y_max, 5):
        if y_min - 1e-9 <= v <= y_max + 1e-9:
            yp = ty(v)
            lines.append(f'<line x1="{pad_l}" y1="{yp:.1f}" x2="{pad_l+w}" y2="{yp:.1f}" stroke="#e0e0e0" stroke-width="1"/>')
            lines.append(f'<text x="{pad_l-5}" y="{yp+4:.1f}" text-anchor="end">{_fmt(v)}</text>')

    for v in _nice_ticks(t_min, t_max, 6):
        if t_min - 1e-9 <= v <= t_max + 1e-9:
            xp = tx(v)
            lines.append(f'<line x1="{xp:.1f}" y1="{pad_t}" x2="{xp:.1f}" y2="{pad_t+h}" stroke="#e0e0e0" stroke-width="1"/>')
            lines.append(f'<text x="{xp:.1f}" y="{pad_t+h+16}" text-anchor="middle">{_fmt(v)}</text>')

    color_cycle = ["#1565c0", "#c62828", "#2e7d32", "#e65100", "#6a1b9a"]
    for idx, s in enumerate(series):
        color = s.get("color", color_cycle[idx % len(color_cycle)])
        dashed = s.get("dashed", False)
        pts = [
            (tx(float(t)), ty(float(y)))
            for t, y in zip(t_vals, s["y"])
            if y is not None and math.isfinite(float(y))
        ]
        if not pts:
            continue
        d = "M " + " L ".join(f"{x:.1f},{yy:.1f}" for x, yy in pts)
        dash = ' stroke-dasharray="7,4"' if dashed else ""
        lines.append(f'<path d="{d}" stroke="{color}" stroke-width="2" fill="none"{dash}/>')

    # Legend
    lx, ly = pad_l + 12, pad_t + 12
    for idx, s in enumerate(series):
        color = s.get("color", color_cycle[idx % len(color_cycle)])
        dashed = s.get("dashed", False)
        dash = ' stroke-dasharray="7,4"' if dashed else ""
        row_y = ly + idx * 18
        lines.append(f'<line x1="{lx}" y1="{row_y+6}" x2="{lx+24}" y2="{row_y+6}" stroke="{color}" stroke-width="2"{dash}/>')
        lines.append(f'<text x="{lx+28}" y="{row_y+10}">{_esc(s["label"])}</text>')

    lines.append("</svg>")
    return "\n".join(lines)


def _svg_dashboard(ts: list[dict], target_ms: int, summary: dict) -> str:
    """3-panel dashboard for one TTFT target run."""
    if not ts:
        return _empty_svg(900, 520, "No data")

    t_vals = [row.get("t", 0.0) for row in ts]
    measured_ttft = [row.get("measured_ttft_ms") for row in ts]
    fraction = [row.get("admission_fraction") for row in ts]
    power = [row.get("gpu_power_w") for row in ts]

    total_w = 900
    total_h = 520
    panel_w = 270
    panel_h = 200
    start_x = 16
    start_y = 52
    gap = 12

    kp_val = summary.get("kp", "?")
    ki_val = summary.get("ki", "?")
    ttft_mean = summary.get("ttft_mean_ms")
    ttft_p95 = summary.get("ttft_p95_ms")
    ttft_mean_str = f"{ttft_mean:.0f}" if ttft_mean is not None else "?"
    ttft_p95_str = f"{ttft_p95:.0f}" if ttft_p95 is not None else "?"
    fmin = summary.get("fraction_min", "?")
    fmax = summary.get("fraction_max", "?")

    title = (
        f"TTFT Controller — target {target_ms} ms  |  "
        f"TTFT mean {ttft_mean_str} ms  p95 {ttft_p95_str} ms  |  "
        f"kp={kp_val} ki={ki_val}  frac[{fmin},{fmax}]"
    )

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}" height="{total_h}">',
        "<style>text{font-family:Arial,sans-serif;font-size:11px;fill:#222}</style>",
        f'<rect width="{total_w}" height="{total_h}" fill="#fff"/>',
        f'<text x="{total_w//2}" y="28" text-anchor="middle" font-size="13" font-weight="bold">{_esc(title)}</text>',
    ]

    panels = [
        {
            "ys_list": [measured_ttft, [float(target_ms)] * len(ts)],
            "labels": [f"Measured TTFT", f"Target {target_ms} ms"],
            "colors": ["#1565c0", "#c62828"],
            "dashed": [False, True],
            "title": "TTFT (ms)",
            "ylabel": "ms",
            "y_min": None,
            "y_max": None,
        },
        {
            "ys_list": [fraction],
            "labels": ["Admission frac"],
            "colors": ["#2e7d32"],
            "dashed": [False],
            "title": "Admission Fraction",
            "ylabel": "frac",
            "y_min": 0.0,
            "y_max": 1.05,
        },
        {
            "ys_list": [power],
            "labels": ["GPU Power (W)"],
            "colors": ["#e65100"],
            "dashed": [False],
            "title": "GPU Power (W)",
            "ylabel": "W",
            "y_min": None,
            "y_max": None,
        },
    ]

    for i, panel in enumerate(panels):
        x0 = start_x + i * (panel_w + gap)
        y0 = start_y
        el = _mini_panel(
            x0, y0, panel_w, panel_h,
            t_vals, panel["ys_list"], panel["labels"], panel["colors"], panel["dashed"],
            panel["title"], panel["ylabel"],
            panel["y_min"], panel["y_max"],
        )
        lines.extend(el)

    # Summary stats below the panels
    sx = start_x
    sy = start_y + panel_h + 28
    lines.append(f'<text x="{sx}" y="{sy}" font-size="12" font-weight="bold">Summary</text>')

    stats = [
        ("Target TTFT", f"{target_ms} ms"),
        ("Measured mean", f"{ttft_mean_str} ms"),
        ("Measured p95", f"{ttft_p95_str} ms"),
        ("Throughput", _fmt_opt(summary.get("throughput_req_s"), ".2f") + " req/s"),
        ("GPU power mean", _fmt_opt(summary.get("gpu_power_mean_w"), ".1f") + " W"),
        ("Energy/req", _fmt_opt(summary.get("energy_per_request_j"), ".1f") + " J"),
        ("Error rate", _fmt_opt(summary.get("error_rate"), ".4f")),
        ("kp / ki", f"{kp_val} / {ki_val}"),
        ("Fraction range", f"[{fmin}, {fmax}]"),
        ("Reqs measured", str(summary.get("requests_measured", "?"))),
    ]
    col_w = 200
    for i, (label, value) in enumerate(stats):
        col = i % 4
        row = i // 4
        lines.append(
            f'<text x="{sx + col * col_w}" y="{sy + 18 + row * 16}">'
            f'{_esc(label)}: {_esc(value)}</text>'
        )

    lines.append("</svg>")
    return "\n".join(lines)


def _mini_panel(
    x0: int, y0: int, pw: int, ph: int,
    t_vals: list, ys_list: list, labels: list, colors: list, dashed_flags: list,
    title: str, ylabel: str,
    y_min: float | None, y_max: float | None,
) -> list[str]:
    pl, pr, pt, pb = 54, 8, 28, 36
    iw = pw - pl - pr
    ih = ph - pt - pb

    all_y = [v for ys in ys_list for v in ys if v is not None and math.isfinite(float(v))]
    if not all_y or not t_vals:
        return [f'<text x="{x0+10}" y="{y0+20}">No data</text>']

    t_min, t_max = min(t_vals), max(t_vals)
    data_y_min = min(all_y)
    data_y_max = max(all_y)
    pad_frac = 0.06
    if y_min is None:
        y_min = max(0.0, data_y_min - pad_frac * max(1.0, data_y_max - data_y_min))
    if y_max is None:
        y_max = data_y_max + pad_frac * max(1.0, data_y_max - data_y_min)
    if abs(y_max - y_min) < 1e-9:
        y_max = y_min + 1.0

    t_range = max(t_max - t_min, 1e-9)
    y_range = y_max - y_min

    def tx(t: float) -> float:
        return x0 + pl + (t - t_min) / t_range * iw

    def ty(y: float) -> float:
        return y0 + pt + ih - (y - y_min) / y_range * ih

    res = [
        f'<rect x="{x0+pl}" y="{y0+pt}" width="{iw}" height="{ih}" fill="#fafafa" stroke="#ccc" stroke-width="1"/>',
        f'<text x="{x0+pl+iw//2}" y="{y0+17}" text-anchor="middle" font-size="12" font-weight="bold">{_esc(title)}</text>',
        f'<text x="{x0+10}" y="{y0+pt+ih//2}" text-anchor="middle" font-size="9" transform="rotate(-90 {x0+10} {y0+pt+ih//2})">{_esc(ylabel)}</text>',
        f'<text x="{x0+pl+iw//2}" y="{y0+ph-4}" text-anchor="middle" font-size="9">Time (s)</text>',
    ]

    for v in _nice_ticks(y_min, y_max, 4):
        if y_min - 1e-9 <= v <= y_max + 1e-9:
            yp = ty(v)
            res.append(f'<line x1="{x0+pl}" y1="{yp:.1f}" x2="{x0+pl+iw}" y2="{yp:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
            res.append(f'<text x="{x0+pl-3}" y="{yp+3:.1f}" text-anchor="end" font-size="9">{_fmt(v)}</text>')

    for ys, label, color, is_dashed in zip(ys_list, labels, colors, dashed_flags):
        pts = [
            (tx(float(t)), ty(float(y)))
            for t, y in zip(t_vals, ys)
            if y is not None and math.isfinite(float(y))
        ]
        if not pts:
            continue
        d = "M " + " L ".join(f"{x:.1f},{yy:.1f}" for x, yy in pts)
        dash = ' stroke-dasharray="5,3"' if is_dashed else ""
        res.append(f'<path d="{d}" stroke="{color}" stroke-width="1.5" fill="none"{dash}/>')

    return res


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _nice_ticks(lo: float, hi: float, n: int) -> list[float]:
    span = hi - lo
    if span <= 0:
        return [lo]
    raw_step = span / n
    if raw_step <= 0:
        return [lo]
    mag = 10 ** math.floor(math.log10(raw_step))
    candidates = [mag * f for f in (1, 2, 2.5, 5, 10)]
    step = next((s for s in candidates if s >= raw_step), candidates[-1])
    first = math.ceil(lo / step) * step
    ticks: list[float] = []
    v = first
    while v <= hi + step * 0.01:
        ticks.append(round(v, 10))
        v += step
    return ticks


def _fmt(v: float) -> str:
    if abs(v) >= 1000:
        return f"{v:.0f}"
    if abs(v) >= 100:
        return f"{v:.0f}"
    if abs(v) >= 10:
        return f"{v:.1f}"
    return f"{v:.2f}"


def _fmt_opt(v: float | None, spec: str = ".1f") -> str:
    if v is None:
        return "?"
    return format(v, spec)


def _esc(s: str) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _empty_svg(width: int, height: int, msg: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        f'<text x="10" y="20">{_esc(msg)}</text></svg>'
    )


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Plot TTFT sweep results from a saved sweep_response.json")
    ap.add_argument("response_json")
    ap.add_argument("--out-dir", required=True)
    args_cli = ap.parse_args()

    with open(args_cli.response_json) as f:
        data = json.load(f)
    out = Path(args_cli.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = data.get("results", [])
    for res in results:
        ts = res.get("timeseries", [])
        summary = {k: v for k, v in res.items() if k != "timeseries"}
        paths = plot_result(ts, summary, out)
        for p in paths:
            print(p)
