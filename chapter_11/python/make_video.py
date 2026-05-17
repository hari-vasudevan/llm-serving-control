#!/usr/bin/env python3
"""Chapter 11 — QA scrolling video generator.

Takes qa_log.json + timeseries.json from a load-step run and produces an MP4
that replays the experiment at a configurable speed multiplier.

Layout (1280×720):
  Left 840 px  — scrolling chat pane (questions + answers with TTFT badge)
  Right 440 px — live TTFT trace + QPS indicator + dispatch-delay trace

Usage:
  python make_video.py <target_dir> [--speed 5] [--fps 30] [--out video.mp4]

  target_dir: e.g. results/load_step_20260517_170041/target_200ms
"""
from __future__ import annotations

import argparse
import json
import math
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib import gridspec
import numpy as np


# ── colour scheme ────────────────────────────────────────────────────────────
BG        = "#0d0d0d"
CHAT_BG   = "#111111"
PANEL_BG  = "#141414"
Q_COLOR   = "#4a9eff"
A_COLOR   = "#e0e0e0"
A_PEND    = "#555555"
TTFT_LINE = "#4a9eff"
QPS_LINE  = "#e0801a"
DELAY_LINE= "#44cc44"
GRID_COL  = "#222222"
TARGET_COL= "#cc3333"
BADGE_OK  = "#1a7a3a"
BADGE_SLOW= "#8a2020"


def _ttft_badge_color(ttft_ms: float | None, target_ms: float) -> str:
    if ttft_ms is None:
        return "#444444"
    return BADGE_OK if ttft_ms <= target_ms * 1.15 else BADGE_SLOW


def load_data(target_dir: Path):
    qa_raw  = json.loads((target_dir / "qa_log.json").read_text())
    ts_raw  = json.loads((target_dir / "timeseries.json").read_text())
    summary = json.loads((target_dir / "summary.json").read_text())
    return qa_raw, ts_raw, summary


def build_events(qa_log: list[dict]) -> list[dict]:
    """Return list of events sorted by time: question-sent and answer-received."""
    events = []
    for entry in qa_log:
        events.append({"kind": "question", "t": entry["sent_at_s"],  "entry": entry})
        events.append({"kind": "answer",   "t": entry["recv_at_s"],  "entry": entry})
    events.sort(key=lambda e: e["t"])
    return events


def make_video(target_dir: Path, speed: float, fps: int, out_path: Path) -> None:
    qa_log, ts_raw, summary = load_data(target_dir)
    target_ttft = float(summary.get("target_ttft_ms", 300.0))
    actuator    = str(summary.get("actuator", "dispatch_delay"))

    if not qa_log:
        raise ValueError(f"qa_log.json is empty in {target_dir}")

    # Timeseries arrays for the right panel
    ts_t     = np.array([r["t"]                 for r in ts_raw])
    ts_ttft  = np.array([r.get("measured_ttft_ms") or float("nan") for r in ts_raw])
    ts_qps   = np.array([r.get("offered_qps", 0)   for r in ts_raw])
    ts_delay = np.array([r.get("dispatch_delay_ms") or float("nan") for r in ts_raw])
    ts_frac  = np.array([r.get("admission_fraction") or float("nan") for r in ts_raw])

    t_total = max(float(ts_t[-1]) if len(ts_t) else 0.0,
                  max(e["recv_at_s"] for e in qa_log))

    events = build_events(qa_log)
    n_frames = max(1, int(math.ceil(t_total * fps / speed)))

    # ── figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(1280/96, 720/96), dpi=96, facecolor=BG)
    gs  = gridspec.GridSpec(1, 2, width_ratios=[840, 440],
                            left=0, right=1, top=1, bottom=0, wspace=0)

    # Chat pane (left)
    ax_chat = fig.add_subplot(gs[0])
    ax_chat.set_facecolor(CHAT_BG)
    ax_chat.set_xlim(0, 1)
    ax_chat.set_ylim(0, 1)
    ax_chat.axis("off")

    # Right column: 3 stacked panels
    gs_right = gridspec.GridSpecFromSubplotSpec(
        3, 1, subplot_spec=gs[1], hspace=0.35,
        height_ratios=[2, 1.2, 1.2])
    ax_ttft  = fig.add_subplot(gs_right[0])
    ax_qps   = fig.add_subplot(gs_right[1])
    ax_ctrl  = fig.add_subplot(gs_right[2])

    for ax in (ax_ttft, ax_qps, ax_ctrl):
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors="#888", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#333")
        ax.xaxis.label.set_color("#aaa")
        ax.yaxis.label.set_color("#aaa")
        ax.grid(True, color=GRID_COL, linewidth=0.5)

    ax_ttft.set_ylabel("TTFT (ms)", fontsize=8, color="#aaa")
    ax_qps.set_ylabel("QPS", fontsize=8, color="#aaa")
    ctrl_label = "Delay (ms)" if actuator == "dispatch_delay" else "Adm. Fraction"
    ax_ctrl.set_ylabel(ctrl_label, fontsize=8, color="#aaa")
    ax_ctrl.set_xlabel("Experiment time (s)", fontsize=8, color="#aaa")

    # Static reference lines
    ax_ttft.axhline(target_ttft, color=TARGET_COL, linewidth=1.2, linestyle="--", alpha=0.8)

    # Set fixed x range
    for ax in (ax_ttft, ax_qps, ax_ctrl):
        ax.set_xlim(0, t_total)

    # Set y ranges
    ttft_max = np.nanmax(ts_ttft) if np.any(~np.isnan(ts_ttft)) else target_ttft * 1.5
    ax_ttft.set_ylim(0, max(ttft_max * 1.15, target_ttft * 1.5))
    ax_qps.set_ylim(0, float(np.nanmax(ts_qps)) * 1.2 if len(ts_qps) else 10)
    ctrl_arr = ts_delay if actuator == "dispatch_delay" else ts_frac
    ctrl_max = float(np.nanmax(ctrl_arr)) if np.any(~np.isnan(ctrl_arr)) else 1.0
    ax_ctrl.set_ylim(0, max(ctrl_max * 1.2, 1.0))

    # Live traces (updated each frame)
    line_ttft,  = ax_ttft.plot([], [], color=TTFT_LINE, linewidth=1.0)
    line_qps,   = ax_qps.plot([],  [], color=QPS_LINE,  linewidth=1.4,
                               drawstyle="steps-post")
    line_ctrl,  = ax_ctrl.plot([], [], color=DELAY_LINE, linewidth=1.0)

    # Current-time vlines
    vl_ttft = ax_ttft.axvline(0, color="#555", linewidth=0.8, linestyle=":")
    vl_qps  = ax_qps.axvline(0,  color="#555", linewidth=0.8, linestyle=":")
    vl_ctrl = ax_ctrl.axvline(0, color="#555", linewidth=0.8, linestyle=":")

    # Title strip
    title_txt = fig.text(
        0.0, 0.98,
        f"Chapter 11 — TTFT Load-Step  |  target={target_ttft:.0f} ms  |  "
        f"actuator={actuator}  |  speed={speed:.0f}×",
        color="#cccccc", fontsize=9, va="top", ha="left",
        fontfamily="monospace",
    )

    # Chat state
    MAX_VISIBLE = 12
    chat_entries: list[dict] = []   # {question, answer or None, ttft_ms, t_q, t_a}
    pending_answers: dict[str, dict] = {}  # keyed by question text → entry dict

    chat_texts: list = []

    def _redraw_chat(sim_t: float):
        ax_chat.cla()
        ax_chat.set_facecolor(CHAT_BG)
        ax_chat.set_xlim(0, 1)
        ax_chat.set_ylim(0, 1)
        ax_chat.axis("off")

        # Header
        ax_chat.text(0.5, 0.975, "LLM Chat Replay",
                     color="#888", fontsize=9, ha="center", va="top",
                     fontfamily="monospace", transform=ax_chat.transAxes)

        visible = chat_entries[-MAX_VISIBLE:]
        n = len(visible)
        if n == 0:
            return

        row_h = 0.88 / MAX_VISIBLE  # height per chat slot
        y_base = 0.945 - row_h * 0.5

        for idx, entry in enumerate(reversed(visible)):
            y = y_base - idx * row_h
            answered = entry.get("answer") is not None

            # Question line
            q_short = textwrap.shorten(entry["question"], width=60, placeholder="…")
            ax_chat.text(0.03, y + row_h * 0.28, f"Q: {q_short}",
                         color=Q_COLOR, fontsize=7.5, va="center",
                         fontfamily="monospace", transform=ax_chat.transAxes,
                         clip_on=True)

            if answered:
                a_short = textwrap.shorten(entry["answer"], width=72, placeholder="…")
                badge_col = _ttft_badge_color(entry.get("ttft_ms"), target_ttft)
                ttft_str = f"{entry['ttft_ms']:.0f}ms" if entry.get("ttft_ms") else "?"
                # TTFT badge
                ax_chat.text(
                    0.97, y - row_h * 0.04,
                    f"TTFT {ttft_str}",
                    color="white", fontsize=6.5, va="center", ha="right",
                    fontfamily="monospace", transform=ax_chat.transAxes,
                    bbox=dict(boxstyle="round,pad=0.15", fc=badge_col, ec="none"),
                    clip_on=True,
                )
                ax_chat.text(0.03, y - row_h * 0.04, f"A: {a_short}",
                             color=A_COLOR, fontsize=7, va="center",
                             fontfamily="monospace", transform=ax_chat.transAxes,
                             clip_on=True)
            else:
                ax_chat.text(0.03, y - row_h * 0.04, "A: …",
                             color=A_PEND, fontsize=7, va="center",
                             fontfamily="monospace", transform=ax_chat.transAxes,
                             clip_on=True)

            # Separator line
            if idx < n - 1:
                ax_chat.axhline(y - row_h * 0.45,
                                color="#1e1e1e", linewidth=0.8,
                                transform=ax_chat.transAxes)

        # Sim-time stamp bottom
        ax_chat.text(0.97, 0.01, f"t = {sim_t:.1f}s",
                     color="#444", fontsize=7, ha="right", va="bottom",
                     fontfamily="monospace", transform=ax_chat.transAxes)

    # Event pointer
    ev_ptr = [0]

    def update(frame: int):
        sim_t = frame * speed / fps

        # Process events up to sim_t
        while ev_ptr[0] < len(events):
            ev = events[ev_ptr[0]]
            if ev["t"] > sim_t:
                break
            entry = ev["entry"]
            key = entry["question"]
            if ev["kind"] == "question":
                chat_entry = {
                    "question": key,
                    "answer": None,
                    "ttft_ms": None,
                }
                chat_entries.append(chat_entry)
                pending_answers[key] = chat_entry
            elif ev["kind"] == "answer":
                if key in pending_answers:
                    pending_answers[key]["answer"]  = entry.get("answer", "")
                    pending_answers[key]["ttft_ms"] = entry.get("ttft_ms")
                    del pending_answers[key]
            ev_ptr[0] += 1

        _redraw_chat(sim_t)

        # Update right-panel traces
        mask = ts_t <= sim_t
        t_vis = ts_t[mask]
        line_ttft.set_data(t_vis, ts_ttft[mask])
        line_qps.set_data(t_vis, ts_qps[mask])
        line_ctrl.set_data(t_vis, ctrl_arr[mask])
        vl_ttft.set_xdata([sim_t])
        vl_qps.set_xdata([sim_t])
        vl_ctrl.set_xdata([sim_t])

        return []

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000/fps, blit=False)

    writer = FFMpegWriter(fps=fps, bitrate=2000,
                          extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])
    print(f"Rendering {n_frames} frames at {fps} fps ({speed}× speed) → {out_path}")
    anim.save(str(out_path), writer=writer, dpi=96)
    print(f"Saved: {out_path}")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate QA scrolling video from load-step run")
    ap.add_argument("target_dir", help="Path to target_Nms directory (contains qa_log.json)")
    ap.add_argument("--speed", type=float, default=5.0, help="Playback speed multiplier (default 5)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", default=None,
                    help="Output MP4 path (default: <target_dir>/qa_video.mp4)")
    args = ap.parse_args()

    target_dir = Path(args.target_dir).resolve()
    out_path = Path(args.out) if args.out else target_dir / "qa_video.mp4"
    make_video(target_dir, speed=args.speed, fps=args.fps, out_path=out_path)


if __name__ == "__main__":
    main()
