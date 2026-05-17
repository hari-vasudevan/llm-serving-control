#!/usr/bin/env python3
"""Chapter 11 load-step experiment plotter.

Generates a single SVG with three vertically-stacked subplots sharing the
same x-axis:
  top    — Offered load (req/s)
  middle — Measured TTFT (ms) with target line
  bottom — GPU power (W)

Vertical dashed lines mark QPS step boundaries across all three panels.
Each phase is coloured differently (warmup=grey, steps=blue/orange/green…).

Also generates a MATLAB .m script that re-creates the same figure with
linked axes so the user can zoom interactively and save as .fig.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

# ── colour palette ────────────────────────────────────────────────────────────
WARMUP_COLOR = "#aaaaaa"
STEP_COLORS = ["#3a76c4", "#d06020", "#229944", "#9933bb", "#bbbb22", "#22bbbb"]


def _phase_color(phase: str) -> str:
    if phase == "warmup":
        return WARMUP_COLOR
    try:
        idx = int(phase.split("_")[1])
        return STEP_COLORS[idx % len(STEP_COLORS)]
    except Exception:
        return "#666666"


# ── SVG primitives ─────────────────────────────────────────────────────────────

def _open(w: int, h: int) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}">\n')


def _rect(x, y, w, h, fill="white", stroke="none", sw=1) -> str:
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>\n')


def _txt(x, y, s, anchor="middle", size=11, color="#333", bold=False, rotate=0) -> str:
    fw = "bold" if bold else "normal"
    rot = f' transform="rotate({rotate},{x},{y})"' if rotate else ""
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" '
            f'font-family="monospace" font-weight="{fw}" fill="{color}"{rot}>{s}</text>\n')


def _poly(pts: list[tuple[float, float]], color: str, width: float = 1.6) -> str:
    if len(pts) < 2:
        return ""
    coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    return f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="{width}"/>\n'


def _vline(x: float, y0: float, y1: float,
           color: str = "#aaa", dash: str = "4,3", width: float = 1.2) -> str:
    return (f'<line x1="{x:.1f}" y1="{y0:.1f}" x2="{x:.1f}" y2="{y1:.1f}" '
            f'stroke="{color}" stroke-width="{width}" stroke-dasharray="{dash}"/>\n')


def _hline(y: float, x0: float, x1: float,
           color: str = "#dd3333", dash: str = "6,4", width: float = 1.8) -> str:
    return (f'<line x1="{x0:.1f}" y1="{y:.1f}" x2="{x1:.1f}" y2="{y:.1f}" '
            f'stroke="{color}" stroke-width="{width}" stroke-dasharray="{dash}"/>\n')


# ── subplot figure ─────────────────────────────────────────────────────────────

def _subplot_figure(ts: list[dict], result: dict) -> str:
    """Three vertically stacked panels sharing the same x-axis."""
    target = float(result.get("target_ttft_ms", 300.0))
    actuator = str(result.get("actuator", "token_budget"))
    act_key = "dispatch_delay_ms" if actuator == "dispatch_delay" else "admission_fraction"

    PANELS = [
        ("offered_qps",      "Load (req/s)",      None),
        ("measured_ttft_ms", "TTFT (ms)",          target),
        ("gpu_power_w",      "GPU Power (W)",      None),
    ]

    # Layout constants
    W, H = 940, 600
    ML, MR = 75, 25          # left / right margin (pixels)
    MT = 42                   # top margin (title space)
    MB = 52                   # bottom margin (x-axis labels)
    PG = 18                   # gap between panels (pixels)
    N = len(PANELS)
    PH = (H - MT - MB - (N - 1) * PG) // N   # plot-area height per panel

    PW = W - ML - MR          # plot area width

    times = [r.get("t", 0.0) for r in ts]
    phases = [r.get("phase", "warmup") for r in ts]
    t_min = min(times) if times else 0.0
    t_max = max(times) if times else 1.0
    t_rng = max(t_max - t_min, 1.0)

    def tx(t: float) -> float:
        return ML + PW * (t - t_min) / t_rng

    # Collect step-boundary x-positions (where phase changes, excluding warmup end)
    step_xs: list[float] = []
    for i in range(1, len(phases)):
        if phases[i] != phases[i - 1]:
            step_xs.append(tx(times[i]))

    svg = _open(W, H)
    svg += _rect(0, 0, W, H, "#f8f8f8")

    # Title
    svg += _txt(W // 2, 26,
                f"TTFT Load-Step  |  target={target:.0f} ms  |  actuator={actuator}",
                size=13, bold=True)

    for pi, (key, ylabel, target_val) in enumerate(PANELS):
        y_top = MT + pi * (PH + PG)      # top of this panel's plot area
        y_bot = y_top + PH               # bottom

        vals_raw = [r.get(key) for r in ts]
        finite = [v for v in vals_raw if v is not None and math.isfinite(v)]
        if not finite:
            svg += _rect(ML, y_top, PW, PH, "#eee", "#ccc")
            svg += _txt(ML + PW // 2, y_top + PH // 2, f"no data ({key})", size=10, color="#999")
            continue

        v_min, v_max = min(finite), max(finite)
        if target_val is not None:
            v_min = min(v_min, target_val * 0.8)
            v_max = max(v_max, target_val * 1.25)
        pad = (v_max - v_min) * 0.08 or 5.0
        v_min -= pad
        v_max += pad
        v_rng = max(v_max - v_min, 1.0)

        def ty(v: float, _vmin=v_min, _vrng=v_rng, _ytop=y_top, _ph=PH) -> float:
            return _ytop + _ph * (1.0 - (v - _vmin) / _vrng)

        # Panel background + border
        svg += _rect(ML, y_top, PW, PH, "white", "#cccccc")

        # Horizontal grid lines
        for gi in range(5):
            gy = ty(v_min + (gi + 1) * v_rng / 6)
            svg += (f'<line x1="{ML}" y1="{gy:.1f}" x2="{ML + PW}" y2="{gy:.1f}" '
                    f'stroke="#eeeeee" stroke-width="1"/>\n')

        # Step boundary vertical lines
        for sx in step_xs:
            svg += _vline(sx, y_top, y_bot, "#888", "5,4", 1.3)

        # Target horizontal line
        if target_val is not None:
            ty_target = ty(target_val)
            svg += _hline(ty_target, ML, ML + PW)
            svg += _txt(ML + PW - 3, ty_target - 5,
                        f"target={target_val:.0f}", anchor="end", size=9, color="#dd3333")

        # Data polyline, one segment per phase
        seg: list[tuple[float, float]] = []
        seg_phase: str | None = None
        for t, v, p in zip(times, vals_raw, phases):
            if v is None or not math.isfinite(v):
                if seg:
                    svg += _poly(seg, _phase_color(seg_phase))
                seg = []
                continue
            if p != seg_phase and seg:
                svg += _poly(seg, _phase_color(seg_phase))
                seg = [(tx(t), ty(v))]
                seg_phase = p
            else:
                seg.append((tx(t), ty(v)))
                if seg_phase is None:
                    seg_phase = p
        if seg:
            svg += _poly(seg, _phase_color(seg_phase))

        # Y-axis ticks + labels
        n_yticks = 5
        for gi in range(n_yticks + 1):
            v = v_min + gi * v_rng / n_yticks
            y = ty(v)
            svg += (f'<line x1="{ML - 4}" y1="{y:.1f}" x2="{ML}" y2="{y:.1f}" '
                    f'stroke="#888" stroke-width="1"/>\n')
            svg += _txt(ML - 6, y + 3.5, f"{v:.0f}", anchor="end", size=9, color="#555")

        # Y-axis label (rotated)
        svg += _txt(16, y_top + PH // 2, ylabel, size=10, color="#333",
                    bold=True, rotate=-90)

        # X-axis ticks + labels (only bottom panel)
        if pi == N - 1:
            n_xticks = 8
            for gi in range(n_xticks + 1):
                t_tick = t_min + gi * t_rng / n_xticks
                x = tx(t_tick)
                svg += (f'<line x1="{x:.1f}" y1="{y_bot}" x2="{x:.1f}" y2="{y_bot + 5}" '
                        f'stroke="#888" stroke-width="1"/>\n')
                svg += _txt(x, y_bot + 18, f"{t_tick:.0f}s", size=9, color="#555")

    # X-axis title
    svg += _txt(ML + PW // 2, H - 10, "Elapsed time (s)", size=11, color="#555")

    # Legend (warmup + steps)
    legend_x = ML + 8
    legend_y = MT + 8
    unique_phases: list[str] = []
    for p in phases:
        if p not in unique_phases:
            unique_phases.append(p)
    for i, p in enumerate(unique_phases):
        lx = legend_x + i * 110
        color = _phase_color(p)
        label = "warmup" if p == "warmup" else p.replace("_", " ")
        svg += f'<rect x="{lx}" y="{legend_y}" width="14" height="10" fill="{color}" rx="2"/>\n'
        svg += _txt(lx + 18, legend_y + 9, label, anchor="start", size=9, color="#444")

    svg += "</svg>\n"
    return svg


# ── MATLAB script generation ───────────────────────────────────────────────────

def _matlab_single(ts_path: str, target_ttft: float, out_dir: str) -> str:
    """Return MATLAB code for a single-target load-step figure."""
    return f"""%% Chapter 11 — Load Step Disturbance Rejection
%  target TTFT = {target_ttft:.0f} ms
%  Auto-generated by plot_load_step.py — do not edit by hand.

clear; close all;

ts_path   = fullfile('{out_dir}', 'timeseries.json');
fig_path  = fullfile('{out_dir}', 'results_{target_ttft:.0f}ms.fig');

%% Load timeseries
raw = jsondecode(fileread(ts_path));
n   = numel(raw);
t               = zeros(1, n);
offered_qps     = zeros(1, n);
measured_ttft   = NaN(1, n);
admission_frac  = NaN(1, n);
dispatch_delay  = NaN(1, n);
gpu_power       = NaN(1, n);
target_ttft_val = {target_ttft:.1f};
phase_cell      = cell(1, n);

for i = 1:n
    t(i) = raw(i).t;
    if isfield(raw(i), 'offered_qps') && ~isempty(raw(i).offered_qps)
        offered_qps(i) = raw(i).offered_qps;
    end
    if isfield(raw(i), 'measured_ttft_ms') && ~isempty(raw(i).measured_ttft_ms)
        measured_ttft(i) = raw(i).measured_ttft_ms;
    end
    if isfield(raw(i), 'admission_fraction') && ~isempty(raw(i).admission_fraction)
        admission_frac(i) = raw(i).admission_fraction;
    end
    if isfield(raw(i), 'dispatch_delay_ms') && ~isempty(raw(i).dispatch_delay_ms)
        dispatch_delay(i) = raw(i).dispatch_delay_ms;
    end
    if isfield(raw(i), 'gpu_power_w') && ~isempty(raw(i).gpu_power_w)
        gpu_power(i) = raw(i).gpu_power_w;
    end
    if isfield(raw(i), 'phase')
        phase_cell{{i}} = raw(i).phase;
    else
        phase_cell{{i}} = 'unknown';
    end
end

%% Step boundary times
phase_changes = [true, ~strcmp(phase_cell(1:end-1), phase_cell(2:end))];
step_times    = t(phase_changes);

%% Figure
fig = figure('Name', sprintf('Load Step — target=%dms', round(target_ttft_val)), ...
    'Position', [80 80 1050 720]);
tl = tiledlayout(3, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
title(tl, sprintf('Chapter 11  Load-Step Disturbance Rejection  (target TTFT = %d ms)', ...
    round(target_ttft_val)), 'FontSize', 13, 'FontWeight', 'bold');

%% Panel 1 — Offered load
ax1 = nexttile;
stairs(t, offered_qps, 'b-', 'LineWidth', 2);
ylabel('Load (req/s)', 'FontWeight', 'bold');
grid on; box on; xlim([min(t), max(t)]);
for k = 1:numel(step_times)
    xline(step_times(k), '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1);
end

%% Panel 2 — TTFT
ax2 = nexttile;
plot(t, measured_ttft, 'Color', [0.22 0.51 0.77], 'LineWidth', 1.4);
hold on;
yline(target_ttft_val, 'r--', 'LineWidth', 1.8, ...
    'Label', sprintf('target = %d ms', round(target_ttft_val)), ...
    'LabelHorizontalAlignment', 'right');
ylabel('TTFT (ms)', 'FontWeight', 'bold');
grid on; box on; xlim([min(t), max(t)]);
yl = ylim; ylim([0, max(yl(2), target_ttft_val * 1.3)]);
for k = 1:numel(step_times)
    xline(step_times(k), '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1);
end

%% Panel 3 — GPU power
ax3 = nexttile;
plot(t, gpu_power, 'Color', [0.8 0.4 0.1], 'LineWidth', 1.4);
ylabel('GPU Power (W)', 'FontWeight', 'bold');
xlabel('Elapsed time (s)', 'FontWeight', 'bold');
grid on; box on; xlim([min(t), max(t)]);
for k = 1:numel(step_times)
    xline(step_times(k), '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1);
end

%% Link x-axes so zoom/pan is synchronised
linkaxes([ax1, ax2, ax3], 'x');

%% Save
savefig(fig, fig_path);
fprintf('Saved: %s\\n', fig_path);
"""


def _matlab_multi(entries: list[dict], out_dir: str) -> str:
    """Return MATLAB code for a multi-target comparison figure (N columns, 3 rows)."""
    n = len(entries)
    load_lines = []
    for i, e in enumerate(entries):
        ts_rel = e["ts_rel"]
        tgt = e["target_ttft_ms"]
        load_lines.append(
            f"    load_entry(i).ts_path = fullfile(base_dir, '{ts_rel}');\n"
            f"    load_entry(i).target  = {tgt:.1f};\n"
        )
    load_block = f"n = {n};\nload_entry = struct('ts_path', {{''}},'target', {{0}});\n"
    for i, e in enumerate(entries):
        ts_rel = e["ts_rel"]
        tgt = e["target_ttft_ms"]
        load_block += (f"load_entry({i+1}).ts_path = fullfile(base_dir, '{ts_rel}');\n"
                       f"load_entry({i+1}).target  = {tgt:.1f};\n")

    return f"""%% Chapter 11 — Multi-Target Load-Step Comparison
%  Auto-generated by plot_load_step.py

clear; close all;

base_dir = '{out_dir}';
fig_path = fullfile(base_dir, 'comparison.fig');

{load_block}

%% Load all timeseries
all_t       = cell(1, n);
all_qps     = cell(1, n);
all_ttft    = cell(1, n);
all_power   = cell(1, n);
all_stimes  = cell(1, n);

for col = 1:n
    raw = jsondecode(fileread(load_entry(col).ts_path));
    ni  = numel(raw);
    tv  = zeros(1, ni);
    qv  = zeros(1, ni);
    fv  = NaN(1, ni);
    pv  = NaN(1, ni);
    phc = cell(1, ni);
    for i = 1:ni
        tv(i) = raw(i).t;
        if isfield(raw(i),'offered_qps') && ~isempty(raw(i).offered_qps), qv(i)=raw(i).offered_qps; end
        if isfield(raw(i),'measured_ttft_ms') && ~isempty(raw(i).measured_ttft_ms), fv(i)=raw(i).measured_ttft_ms; end
        if isfield(raw(i),'gpu_power_w') && ~isempty(raw(i).gpu_power_w), pv(i)=raw(i).gpu_power_w; end
        if isfield(raw(i),'phase'), phc{{i}}=raw(i).phase; else phc{{i}}=''; end
    end
    all_t{{col}}      = tv;
    all_qps{{col}}    = qv;
    all_ttft{{col}}   = fv;
    all_power{{col}}  = pv;
    chg = [true, ~strcmp(phc(1:end-1), phc(2:end))];
    all_stimes{{col}} = tv(chg);
end

%% Figure: 3 rows (Load / TTFT / Power), N columns (one per target)
fig = figure('Name', 'Load-Step Multi-Target', 'Position', [60 60 {min(400*n, 1600)} 720]);
tl  = tiledlayout(3, n, 'TileSpacing', 'compact', 'Padding', 'compact');
title(tl, 'Chapter 11  Load-Step Disturbance Rejection — Multi-Target', ...
    'FontSize', 13, 'FontWeight', 'bold');

ax_all = gobjects(3, n);
colors = lines(n);

for col = 1:n
    tgt  = load_entry(col).target;
    tv   = all_t{{col}};
    qv   = all_qps{{col}};
    fv   = all_ttft{{col}};
    pv   = all_power{{col}};
    stv  = all_stimes{{col}};

    %% Row 1: Load
    ax_all(1,col) = nexttile(col);
    stairs(tv, qv, 'b-', 'LineWidth',1.8); grid on; box on;
    title(sprintf('target = %d ms', round(tgt)));
    if col==1, ylabel('Load (req/s)','FontWeight','bold'); end
    for k=1:numel(stv), xline(stv(k),'--','Color',[.5 .5 .5],'LineWidth',1); end

    %% Row 2: TTFT
    ax_all(2,col) = nexttile(n+col);
    plot(tv, fv, 'Color', colors(col,:), 'LineWidth',1.5); hold on;
    yline(tgt, 'r--', 'LineWidth',1.8, 'Label', sprintf('%d ms',round(tgt)), ...
        'LabelHorizontalAlignment','right');
    grid on; box on;
    yl = ylim; ylim([0, max(yl(2), tgt*1.3)]);
    if col==1, ylabel('TTFT (ms)','FontWeight','bold'); end
    for k=1:numel(stv), xline(stv(k),'--','Color',[.5 .5 .5],'LineWidth',1); end

    %% Row 3: Power
    ax_all(3,col) = nexttile(2*n+col);
    plot(tv, pv, 'Color',[0.8 0.4 0.1],'LineWidth',1.5);
    grid on; box on;
    xlabel('Time (s)');
    if col==1, ylabel('GPU Power (W)','FontWeight','bold'); end
    for k=1:numel(stv), xline(stv(k),'--','Color',[.5 .5 .5],'LineWidth',1); end
end

%% Link axes: all panels in the same row share x-axis
for row = 1:3
    linkaxes(ax_all(row,:), 'x');
end

savefig(fig, fig_path);
fprintf('Saved comparison figure: %s\\n', fig_path);
"""


# ── public API ─────────────────────────────────────────────────────────────────

def plot_result(ts: list[dict], result: dict, plots_dir: Path) -> list[Path]:
    """Write the 3-panel subplot SVG for one target; return list of written paths."""
    plots_dir = Path(plots_dir)
    svg = _subplot_figure(ts, result)
    target = int(result.get("target_ttft_ms", 0))
    p = plots_dir / f"subplot_target_{target}ms.svg"
    p.write_text(svg)
    return [p]


def write_matlab_scripts(results_list: list[dict], out_dir: Path) -> list[Path]:
    """Write per-target and combined MATLAB .m scripts; return written paths."""
    out_dir = Path(out_dir)
    paths: list[Path] = []

    entries: list[dict] = []
    for res in results_list:
        target = float(res.get("target_ttft_ms", 0))
        # Each result was saved in a sub-directory named by target
        ts_rel = f"target_{int(target)}ms/timeseries.json"
        entries.append({"target_ttft_ms": target, "ts_rel": ts_rel})

        # Per-target script
        m_code = _matlab_single(
            ts_path=str(out_dir / ts_rel),
            target_ttft=target,
            out_dir=str(out_dir / f"target_{int(target)}ms"),
        )
        p = out_dir / f"target_{int(target)}ms" / "view_figure.m"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(m_code)
        paths.append(p)

    if len(results_list) > 1:
        m_code = _matlab_multi(entries, str(out_dir))
        p = out_dir / "view_comparison.m"
        p.write_text(m_code)
        paths.append(p)

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
