#!/usr/bin/env python3
"""Chapter 11 — QA scrolling video generator.

Takes qa_log.json + timeseries.json from a load-step run and produces an MP4
that replays the experiment at a configurable speed multiplier.

Layout (1280×720):
  Left 800 px  — scrolling chat pane (questions + answers with TTFT badge)
  Right 480 px — live TTFT trace (with step-function target) + QPS + dispatch delay,
                 with a bright cursor tracking experiment time

Usage:
  python make_video.py <run_dir> [--speed 5] [--fps 30] [--out video.mp4]

  run_dir: e.g. results/load_step_20260517_210757
           Must contain timeseries.json, qa_log.json, summary.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib import gridspec
import numpy as np


# ── colour scheme ────────────────────────────────────────────────────────────
BG         = "#0d0d0d"
CHAT_BG    = "#111111"
PANEL_BG   = "#141414"
Q_COLOR    = "#4a9eff"
A_COLOR    = "#e0e0e0"
A_PEND     = "#555555"
TTFT_LINE  = "#4a9eff"
QPS_LINE   = "#e0801a"
DELAY_LINE = "#44cc44"
TARGET_LINE= "#cc3333"
GRID_COL   = "#222222"
CURSOR_COL = "#ffffff"
BADGE_OK   = "#1a7a3a"
BADGE_SLOW = "#8a2020"


def _target_from_phase(phase: str, fallback: float) -> float:
    """Extract target TTFT from phase label like 'step_0_qps4_t200ms'."""
    m = re.search(r'_t(\d+)ms', phase)
    return float(m.group(1)) if m else fallback


def _ttft_badge_color(ttft_ms: float | None, target_ms: float) -> str:
    if ttft_ms is None:
        return "#444444"
    return BADGE_OK if ttft_ms <= target_ms * 1.15 else BADGE_SLOW


def load_data(run_dir: Path):
    qa_raw  = json.loads((run_dir / "qa_log.json").read_text())
    ts_raw  = json.loads((run_dir / "timeseries.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    return qa_raw, ts_raw, summary


def build_events(qa_log: list[dict]) -> list[dict]:
    events = []
    for entry in qa_log:
        # Use sent_at_s as a unique request ID so duplicate questions don't collide
        uid = entry["sent_at_s"]
        events.append({"kind": "question", "t": entry["sent_at_s"], "uid": uid, "entry": entry})
        events.append({"kind": "answer",   "t": entry["recv_at_s"], "uid": uid, "entry": entry})
    events.sort(key=lambda e: e["t"])
    return events


def make_video(run_dir: Path, speed: float, fps: int, out_path: Path) -> None:
    qa_log, ts_raw, summary = load_data(run_dir)

    targets_raw = summary.get("target_ttft_ms", 300.0)
    all_targets = targets_raw if isinstance(targets_raw, list) else [float(targets_raw)]
    first_target = all_targets[0]
    actuator = str(summary.get("actuator", "dispatch_delay"))

    if not qa_log:
        raise ValueError(f"qa_log.json is empty in {run_dir}")

    # Timeseries arrays
    ts_t      = np.array([r["t"]                          for r in ts_raw])
    ts_ttft   = np.array([r.get("measured_ttft_ms") or float("nan") for r in ts_raw])
    ts_qps    = np.array([r.get("offered_qps", 0)         for r in ts_raw])
    ts_delay  = np.array([r.get("dispatch_delay_ms") or float("nan") for r in ts_raw])
    ts_frac   = np.array([r.get("admission_fraction") or float("nan") for r in ts_raw])
    ts_target = np.array([r.get("target_ttft_ms") or float("nan")   for r in ts_raw])

    ctrl_arr  = ts_delay if actuator == "dispatch_delay" else ts_frac

    t_total = float(max(
        ts_t[-1] if len(ts_t) else 0.0,
        max(e["recv_at_s"] for e in qa_log),
    ))

    events   = build_events(qa_log)
    n_frames = max(1, int(math.ceil(t_total * fps / speed)))

    # ── figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(1280/96, 720/96), dpi=96, facecolor=BG)
    gs  = gridspec.GridSpec(1, 2, width_ratios=[800, 480],
                            left=0, right=1, top=1, bottom=0, wspace=0)

    ax_chat = fig.add_subplot(gs[0])
    ax_chat.set_facecolor(CHAT_BG)
    ax_chat.set_xlim(0, 1); ax_chat.set_ylim(0, 1)
    ax_chat.axis("off")

    gs_right = gridspec.GridSpecFromSubplotSpec(
        3, 1, subplot_spec=gs[1], hspace=0.40,
        height_ratios=[2.2, 1.1, 1.1])
    ax_ttft = fig.add_subplot(gs_right[0])
    ax_qps  = fig.add_subplot(gs_right[1])
    ax_ctrl = fig.add_subplot(gs_right[2])

    for ax in (ax_ttft, ax_qps, ax_ctrl):
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors="#888", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#333")
        ax.xaxis.label.set_color("#aaa")
        ax.yaxis.label.set_color("#aaa")
        ax.grid(True, color=GRID_COL, linewidth=0.5)

    ax_ttft.set_ylabel("TTFT (ms)", fontsize=8, color="#aaa")
    ax_qps.set_ylabel("QPS",         fontsize=8, color="#aaa")
    ctrl_label = "Delay (ms)" if actuator == "dispatch_delay" else "Adm. Fraction"
    ax_ctrl.set_ylabel(ctrl_label,   fontsize=8, color="#aaa")
    ax_ctrl.set_xlabel("Experiment time (s)", fontsize=8, color="#aaa")

    # Fixed axis ranges
    for ax in (ax_ttft, ax_qps, ax_ctrl):
        ax.set_xlim(0, t_total)

    ttft_max = float(np.nanmax(ts_ttft)) if np.any(~np.isnan(ts_ttft)) else first_target * 1.5
    tgt_max  = float(np.nanmax(all_targets))
    ax_ttft.set_ylim(0, max(ttft_max * 1.1, tgt_max * 1.4))
    ax_qps.set_ylim(0, float(np.nanmax(ts_qps)) * 1.3 if len(ts_qps) else 12)
    ctrl_max = float(np.nanmax(ctrl_arr)) if np.any(~np.isnan(ctrl_arr)) else 1.0
    ax_ctrl.set_ylim(0, max(ctrl_max * 1.2, 1.0))

    # Static step-function target reference (pre-drawn once)
    valid = ~np.isnan(ts_target)
    if np.any(valid):
        ax_ttft.plot(ts_t[valid], ts_target[valid],
                     color=TARGET_LINE, linewidth=1.5, linestyle="--",
                     alpha=0.8, drawstyle="steps-post", label="target")

    # Live traces
    line_ttft, = ax_ttft.plot([], [], color=TTFT_LINE,  linewidth=1.0)
    line_qps,  = ax_qps.plot([],  [], color=QPS_LINE,   linewidth=1.4,
                              drawstyle="steps-post")
    line_ctrl, = ax_ctrl.plot([], [], color=DELAY_LINE, linewidth=1.0)

    # Cursor: bright vertical line across all three panels
    cur_kw = dict(color=CURSOR_COL, linewidth=1.2, linestyle=":", alpha=0.7)
    vl_ttft = ax_ttft.axvline(0, **cur_kw)
    vl_qps  = ax_qps.axvline(0,  **cur_kw)
    vl_ctrl = ax_ctrl.axvline(0, **cur_kw)

    # Title
    tgt_str = "/".join(f"{t:.0f}" for t in all_targets)
    fig.text(0.0, 0.99,
             f"Chapter 11  |  targets={tgt_str} ms  |  {actuator}  |  {speed:.0f}× speed",
             color="#cccccc", fontsize=8.5, va="top", ha="left", fontfamily="monospace")

    # Chat state
    MAX_VISIBLE   = 11
    chat_entries: list[dict] = []
    pending: dict[str, dict] = {}
    ev_ptr = [0]

    def _redraw_chat(sim_t: float):
        ax_chat.cla()
        ax_chat.set_facecolor(CHAT_BG)
        ax_chat.set_xlim(0, 1); ax_chat.set_ylim(0, 1)
        ax_chat.axis("off")

        ax_chat.text(0.5, 0.979, "LLM Chat Replay",
                     color="#666", fontsize=8.5, ha="center", va="top",
                     fontfamily="monospace", transform=ax_chat.transAxes)

        visible = chat_entries[-MAX_VISIBLE:]
        n = len(visible)
        if n == 0:
            return

        row_h  = 0.90 / MAX_VISIBLE
        y_base = 0.952 - row_h * 0.5

        for idx, entry in enumerate(reversed(visible)):
            y        = y_base - idx * row_h
            answered = entry.get("answer") is not None
            tgt      = entry.get("target_ms", first_target)

            q_short = textwrap.shorten(entry["question"], width=62, placeholder="…")
            ax_chat.text(0.025, y + row_h * 0.27, f"Q  {q_short}",
                         color=Q_COLOR, fontsize=7.2, va="center",
                         fontfamily="monospace", transform=ax_chat.transAxes,
                         clip_on=True)

            if answered:
                a_short    = textwrap.shorten(entry["answer"], width=74, placeholder="…")
                badge_col  = _ttft_badge_color(entry.get("ttft_ms"), tgt)
                ttft_str   = f"{entry['ttft_ms']:.0f} ms" if entry.get("ttft_ms") else "?"
                ax_chat.text(
                    0.975, y - row_h * 0.06,
                    f"TTFT {ttft_str}",
                    color="white", fontsize=6, va="center", ha="right",
                    fontfamily="monospace", transform=ax_chat.transAxes,
                    bbox=dict(boxstyle="round,pad=0.18", fc=badge_col, ec="none"),
                    clip_on=True)
                ax_chat.text(0.025, y - row_h * 0.06, f"A  {a_short}",
                             color=A_COLOR, fontsize=7, va="center",
                             fontfamily="monospace", transform=ax_chat.transAxes,
                             clip_on=True)
            else:
                ax_chat.text(0.025, y - row_h * 0.06, "A  …",
                             color=A_PEND, fontsize=7, va="center",
                             fontfamily="monospace", transform=ax_chat.transAxes,
                             clip_on=True)

            if idx < n - 1:
                sep_y = y - row_h * 0.46
                ax_chat.axhline(sep_y, color="#1d1d1d", linewidth=0.6)

        # Sim-time + current target annotation
        cur_tgt = _target_from_phase(
            next((e["phase"] for e in reversed(chat_entries)
                  if e.get("phase")), ""),
            first_target)
        ax_chat.text(0.975, 0.012,
                     f"t={sim_t:.1f}s  target={cur_tgt:.0f}ms",
                     color="#555", fontsize=7, ha="right", va="bottom",
                     fontfamily="monospace", transform=ax_chat.transAxes)

    def update(frame: int):
        sim_t = frame * speed / fps

        while ev_ptr[0] < len(events):
            ev = events[ev_ptr[0]]
            if ev["t"] > sim_t:
                break
            entry = ev["entry"]
            uid   = ev["uid"]
            phase = entry.get("phase", "")
            tgt   = _target_from_phase(phase, first_target)

            if ev["kind"] == "question":
                chat_entry = {
                    "question":  entry["question"],
                    "answer":    None,
                    "ttft_ms":   None,
                    "target_ms": tgt,
                    "phase":     phase,
                }
                chat_entries.append(chat_entry)
                pending[uid] = chat_entry
            elif ev["kind"] == "answer" and uid in pending:
                pending[uid]["answer"]  = entry.get("answer", "")
                pending[uid]["ttft_ms"] = entry.get("ttft_ms")
                del pending[uid]
            ev_ptr[0] += 1

        _redraw_chat(sim_t)

        mask = ts_t <= sim_t
        line_ttft.set_data(ts_t[mask], ts_ttft[mask])
        line_qps.set_data(ts_t[mask],  ts_qps[mask])
        line_ctrl.set_data(ts_t[mask], ctrl_arr[mask])
        for vl in (vl_ttft, vl_qps, vl_ctrl):
            vl.set_xdata([sim_t])

        return []

    anim = FuncAnimation(fig, update, frames=n_frames,
                         interval=1000/fps, blit=False)
    writer = FFMpegWriter(fps=fps, bitrate=2500,
                          extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])
    print(f"Rendering {n_frames} frames @ {fps} fps ({speed}× speed) → {out_path}")
    anim.save(str(out_path), writer=writer, dpi=96)
    print(f"Saved: {out_path}")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate QA scrolling video from load-step run")
    ap.add_argument("run_dir", help="Load-step run directory (contains timeseries.json, qa_log.json)")
    ap.add_argument("--speed", type=float, default=5.0, help="Playback speed multiplier (default 5)")
    ap.add_argument("--fps",   type=int,   default=30)
    ap.add_argument("--out",   default=None, help="Output MP4 path (default: <run_dir>/qa_video.mp4)")
    args = ap.parse_args()

    run_dir  = Path(args.run_dir).resolve()
    out_path = Path(args.out) if args.out else run_dir / "qa_video.mp4"
    make_video(run_dir, speed=args.speed, fps=args.fps, out_path=out_path)


if __name__ == "__main__":
    main()
