#!/usr/bin/env python3
"""SVG plots for Chapter 11 load-step disturbance rejection experiment.

Four panels:
  1. Measured TTFT vs time (target line + step boundaries)
  2. Offered QPS vs time (step function)
  3. Actuator state (admission_fraction or dispatch_delay_ms) vs time
  4. GPU power vs time

Warmup phase is drawn in grey; each load step in alternating colours.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

STEP_COLORS = ["#4488cc", "#cc6622", "#22aa55", "#9933aa", "#aaaa22", "#22aaaa"]
WARMUP_COLOR = "#aaaaaa"


def _phase_color(phase: str) -> str:
    if phase == "warmup":
        return WARMUP_COLOR
    try:
        idx = int(phase.split("_")[1])
        return STEP_COLORS[idx % len(STEP_COLORS)]
    except Exception:
        return "#666666"


def _svg_open(w: int, h: int) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}">\n')


def _rect(x, y, w, h, fill="white", stroke="none", sw=1) -> str:
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>\n'


def _text(x, y, s, anchor="middle", size=10, color="#333", bold=False) -> str:
    fw = "bold" if bold else "normal"
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" '
            f'font-family="monospace" font-weight="{fw}" fill="{color}">{s}</text>\n')


def _polyline(pts, color, width=1.5) -> str:
    if len(pts) < 2:
        return ""
    coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    return f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="{width}"/>\n'


def _vline(x, y0, y1, color="#999", dash="4,3", width=1.2) -> str:
    return (f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y1}" '
            f'stroke="{color}" stroke-width="{width}" stroke-dasharray="{dash}"/>\n')


def _hline(y, x0, x1, color="#e44", dash="6,3", width=1.8) -> str:
    return (f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" '
            f'stroke="{color}" stroke-width="{width}" stroke-dasharray="{dash}"/>\n')


def _panel(ts: list[dict], key: str, label: str, unit: str,
           target_val: float | None = None,
           W: int = 860, H: int = 200,
           ML: int = 68, MR: int = 18, MT: int = 28, MB: int = 38) -> str:
    """Render one SVG panel for a given timeseries key."""
    times = [r.get("t", 0.0) for r in ts]
    vals = [r.get(key) for r in ts]
    phases = [r.get("phase", "warmup") for r in ts]

    finite = [v for v in vals if v is not None and math.isfinite(v)]
    if not finite:
        return ""

    t_min, t_max = min(times), max(times)
    v_min, v_max = min(finite), max(finite)
    if target_val is not None:
        v_min = min(v_min, target_val * 0.85)
        v_max = max(v_max, target_val * 1.20)
    pad = (v_max - v_min) * 0.08 or 5.0
    v_min -= pad
    v_max += pad

    t_rng = max(t_max - t_min, 1.0)
    v_rng = max(v_max - v_min, 1.0)
    pw = W - ML - MR
    ph = H - MT - MB

    def tx(t):
        return ML + pw * (t - t_min) / t_rng

    def ty(v):
        return MT + ph * (1.0 - (v - v_min) / v_rng)

    svg = _svg_open(W, H)
    svg += _rect(0, 0, W, H, "#fafafa")
    svg += _rect(ML, MT, pw, ph, "white", "#ccc")

    # Grid lines
    for i in range(6):
        v = v_min + i * v_rng / 5
        y = ty(v)
        svg += f'<line x1="{ML}" y1="{y:.1f}" x2="{ML + pw}" y2="{y:.1f}" stroke="#eee" stroke-width="1"/>\n'

    # Vertical step-boundary lines (where phase changes)
    prev_phase = None
    for t, p in zip(times, phases):
        if prev_phase is not None and p != prev_phase and prev_phase == "warmup":
            svg += _vline(tx(t), MT, MT + ph, "#555", "5,3", 1.5)
        elif prev_phase is not None and p != prev_phase:
            svg += _vline(tx(t), MT, MT + ph, "#aaa", "3,3")
        prev_phase = p

    # Target line
    if target_val is not None:
        svg += _hline(ty(target_val), ML, ML + pw)
        svg += _text(ML + pw - 2, ty(target_val) - 4, f"target={target_val:.0f}",
                     anchor="end", size=9, color="#e44")

    # Timeseries polyline, coloured by phase
    seg: list[tuple[float, float]] = []
    seg_phase = None
    for t, v, p in zip(times, vals, phases):
        if v is None or not math.isfinite(v):
            if seg:
                svg += _polyline(seg, _phase_color(seg_phase))
                seg = []
            continue
        if p != seg_phase and seg:
            svg += _polyline(seg, _phase_color(seg_phase))
            seg = [(tx(t), ty(v))]
            seg_phase = p
        else:
            seg.append((tx(t), ty(v)))
            if seg_phase is None:
                seg_phase = p
    if seg:
        svg += _polyline(seg, _phase_color(seg_phase))

    # Y-axis ticks and labels
    for i in range(6):
        v = v_min + i * v_rng / 5
        y = ty(v)
        svg += f'<line x1="{ML - 4}" y1="{y:.1f}" x2="{ML}" y2="{y:.1f}" stroke="#888" stroke-width="1"/>\n'
        svg += _text(ML - 6, y + 3, f"{v:.0f}", anchor="end", size=9, color="#555")

    # X-axis ticks and labels
    for i in range(9):
        t = t_min + i * t_rng / 8
        x = tx(t)
        svg += f'<line x1="{x:.1f}" y1="{MT + ph}" x2="{x:.1f}" y2="{MT + ph + 4}" stroke="#888" stroke-width="1"/>\n'
        svg += _text(x, MT + ph + 15, f"{t:.0f}s", size=9, color="#555")

    svg += _text(ML // 2, MT + ph // 2, f"{label} ({unit})", size=10, color="#333", bold=True)
    svg += _text(ML + pw // 2, H - 4, "elapsed time (s)", size=9, color="#777")
    svg += "</svg>\n"
    return svg


def plot_result(ts: list[dict], result: dict, plots_dir: Path) -> list[Path]:
    """Write per-panel and dashboard SVGs; return list of written paths."""
    plots_dir = Path(plots_dir)
    target = result.get("target_ttft_ms", 300.0)
    actuator = result.get("actuator", "token_budget")

    actuator_key = "dispatch_delay_ms" if actuator == "dispatch_delay" else "admission_fraction"
    actuator_label = "Dispatch Delay" if actuator == "dispatch_delay" else "Admission Fraction"
    actuator_unit = "ms" if actuator == "dispatch_delay" else "fraction"

    panels = [
        ("measured_ttft_ms", "Measured TTFT", "ms", target),
        ("offered_qps",       "Offered Load",  "req/s", None),
        (actuator_key,        actuator_label,  actuator_unit, None),
        ("gpu_power_w",       "GPU Power",     "W", None),
    ]

    paths: list[Path] = []
    svgs: list[str] = []
    for key, label, unit, tgt in panels:
        svg = _panel(ts, key, label, unit, tgt)
        if not svg:
            svgs.append("")
            continue
        p = plots_dir / f"{key}.svg"
        p.write_text(svg)
        paths.append(p)
        svgs.append(svg)

    # Dashboard: 2×2 grid
    PW, PH = 860, 200
    cols, rows = 2, 2
    dw, dh = PW * cols, PH * rows
    dash = _svg_open(dw, dh)
    dash += _rect(0, 0, dw, dh, "#f2f2f2")
    for idx, svg in enumerate(svgs[:4]):
        if not svg:
            continue
        col, row = idx % cols, idx // cols
        ox, oy = col * PW, row * PH
        inner = svg.split(f'viewBox="0 0 {PW} {PH}">\n', 1)[-1].rsplit("</svg>", 1)[0]
        dash += f'<g transform="translate({ox},{oy})">\n{inner}</g>\n'
    dash += "</svg>\n"
    dash_path = plots_dir / "dashboard.svg"
    dash_path.write_text(dash)
    paths.append(dash_path)

    return paths


if __name__ == "__main__":
    import sys
    ts_path = Path(sys.argv[1])
    result_path = Path(sys.argv[2])
    out_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else ts_path.parent / "plots"
    out_dir.mkdir(exist_ok=True)
    ts_data = json.loads(ts_path.read_text())
    result_data = json.loads(result_path.read_text())
    for p in plot_result(ts_data, result_data, out_dir):
        print(p)
