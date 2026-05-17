#!/usr/bin/env python3
"""Chapter 11 — QA scrolling video generator.

Takes qa_log.json + timeseries.json from a load-step run and produces an MP4
that replays the experiment at a configurable speed multiplier.

Layout (1280×720):
  Left 800 px  — continuously scrolling chat pane (questions + answers)
  Right 480 px — live TTFT / QPS / dispatch delay / effective power traces

Fixes vs original:
  - uid uses enumeration index, not sent_at_s (avoids ms-precision collisions)
  - answer event fires at sent_at_s + ttft_ms/1000 (first token), not recv_at_s
    (full completion) — so answers appear while the question is still visible
  - continuous smooth scroll: y-positions are computed from entry age × current QPS
    so scroll speed tracks load doubling visibly
  - 4th right panel: effective GPU power (raw × duty_cycle)

Usage:
  python make_video.py <run_dir> [--speed 5] [--fps 30] [--out video.mp4]
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
BG          = "#0d0d0d"
CHAT_BG     = "#111111"
PANEL_BG    = "#141414"
Q_COLOR     = "#4a9eff"
A_COLOR     = "#e0e0e0"
A_PEND      = "#555555"
TTFT_LINE   = "#4a9eff"
QPS_LINE    = "#e0801a"
DELAY_LINE  = "#44cc44"
POWER_LINE  = "#cc7700"
TARGET_LINE = "#cc3333"
GRID_COL    = "#222222"
CURSOR_COL  = "#ffffff"
BADGE_OK    = "#1a7a3a"
BADGE_SLOW  = "#8a2020"

# Scroll window: how many sim-seconds of chat history to keep visible.
# At qps=4 a question stays visible for SCROLL_WINDOW * 4 = 48 rows.
# At qps=8 the same window contains 96 rows but we only render MAX_VISIBLE of them.
SCROLL_WINDOW = 12.0   # sim-seconds of history shown
MAX_VISIBLE   = 14     # max rows rendered (caps density at high QPS)


def _target_from_phase(phase: str, fallback: float) -> float:
    m = re.search(r'_t(\d+)ms', phase)
    return float(m.group(1)) if m else fallback


def _clean(text: str) -> str:
    """Escape $ so matplotlib doesn't try to render LLM math as LaTeX."""
    return text.replace("$", r"\$")


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
    """Build sorted event list.

    uid is the enumeration index — guaranteed unique, unlike sent_at_s which
    has only ms precision and collides at high QPS.

    Answer event fires at sent_at_s + ttft_ms/1000 (first-token arrival) so
    the answered text appears while the Q is still in the visible scroll window,
    rather than at recv_at_s (full completion, ~1400ms later at qps=8).
    """
    events = []
    for i, entry in enumerate(qa_log):
        uid = i
        t_question = entry["sent_at_s"]
        ttft_s     = (entry.get("ttft_ms") or 0.0) / 1000.0
        t_answer   = t_question + ttft_s
        events.append({"kind": "question", "t": t_question, "uid": uid, "entry": entry})
        events.append({"kind": "answer",   "t": t_answer,   "uid": uid, "entry": entry})
    events.sort(key=lambda e: e["t"])
    return events


def make_video(run_dir: Path, speed: float, fps: int, out_path: Path) -> None:
    qa_log, ts_raw, summary = load_data(run_dir)

    targets_raw  = summary.get("target_ttft_ms", 300.0)
    all_targets  = targets_raw if isinstance(targets_raw, list) else [float(targets_raw)]
    first_target = all_targets[0]
    actuator     = str(summary.get("actuator", "dispatch_delay"))

    if not qa_log:
        raise ValueError(f"qa_log.json is empty in {run_dir}")

    # ── timeseries arrays ────────────────────────────────────────────────────
    ts_t      = np.array([r["t"]                                          for r in ts_raw])
    ts_ttft   = np.array([r.get("measured_ttft_ms")  or float("nan")     for r in ts_raw])
    ts_qps    = np.array([r.get("offered_qps",   0)                      for r in ts_raw])
    ts_delay  = np.array([r.get("dispatch_delay_ms") or float("nan")     for r in ts_raw])
    ts_frac   = np.array([r.get("admission_fraction") or float("nan")    for r in ts_raw])
    ts_power  = np.array([r.get("gpu_power_w")    or float("nan")        for r in ts_raw])
    ts_target = np.array([r.get("target_ttft_ms") or float("nan")        for r in ts_raw])

    ctrl_arr = ts_delay if actuator == "dispatch_delay" else ts_frac

    # Effective power: raw × duty_cycle
    ts_active  = np.maximum(ts_ttft - ts_delay, 0.0)
    ts_duty    = np.minimum(ts_qps * ts_active / 1000.0, 1.0)
    ts_effpow  = ts_power * ts_duty

    t_total  = float(max(
        ts_t[-1] if len(ts_t) else 0.0,
        max(e["sent_at_s"] + (e.get("ttft_ms") or 0) / 1000 for e in qa_log),
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
        4, 1, subplot_spec=gs[1], hspace=0.45,
        height_ratios=[2.0, 1.0, 1.0, 1.0])
    ax_ttft  = fig.add_subplot(gs_right[0])
    ax_qps   = fig.add_subplot(gs_right[1])
    ax_ctrl  = fig.add_subplot(gs_right[2])
    ax_epow  = fig.add_subplot(gs_right[3])

    right_axes = (ax_ttft, ax_qps, ax_ctrl, ax_epow)
    for ax in right_axes:
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors="#888", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#333")
        ax.xaxis.label.set_color("#aaa")
        ax.yaxis.label.set_color("#aaa")
        ax.grid(True, color=GRID_COL, linewidth=0.5)

    ax_ttft.set_ylabel("TTFT (ms)",   fontsize=8, color="#aaa")
    ax_qps.set_ylabel("QPS",          fontsize=8, color="#aaa")
    ctrl_label = "Delay (ms)" if actuator == "dispatch_delay" else "Adm. Fraction"
    ax_ctrl.set_ylabel(ctrl_label,    fontsize=8, color="#aaa")
    ax_epow.set_ylabel("Eff.Pwr (W)", fontsize=8, color="#aaa")
    ax_epow.set_xlabel("Experiment time (s)", fontsize=8, color="#aaa")

    for ax in right_axes:
        ax.set_xlim(0, t_total)

    ttft_max = float(np.nanmax(ts_ttft)) if np.any(~np.isnan(ts_ttft)) else first_target * 1.5
    tgt_max  = float(np.nanmax(all_targets))
    ax_ttft.set_ylim(0, max(ttft_max * 1.1, tgt_max * 1.4))
    ax_qps.set_ylim(0, float(np.nanmax(ts_qps)) * 1.3 if len(ts_qps) else 12)
    ctrl_max = float(np.nanmax(ctrl_arr)) if np.any(~np.isnan(ctrl_arr)) else 1.0
    ax_ctrl.set_ylim(0, max(ctrl_max * 1.2, 1.0))
    epow_max = float(np.nanmax(ts_effpow)) if np.any(~np.isnan(ts_effpow)) else 70.0
    ax_epow.set_ylim(0, max(epow_max * 1.2, 5.0))

    # Static step-function target reference
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
    line_epow, = ax_epow.plot([], [], color=POWER_LINE, linewidth=1.0)

    # Cursor lines
    cur_kw = dict(color=CURSOR_COL, linewidth=1.2, linestyle=":", alpha=0.7)
    vlines = [ax.axvline(0, **cur_kw) for ax in right_axes]

    # Title
    tgt_str = "/".join(f"{t:.0f}" for t in all_targets)
    fig.text(0.0, 0.99,
             f"Chapter 11  |  targets={tgt_str} ms  |  {actuator}  |  {speed:.0f}× speed",
             color="#cccccc", fontsize=8.5, va="top", ha="left", fontfamily="monospace")

    # ── chat state ────────────────────────────────────────────────────────────
    chat_entries:  list[dict]  = []
    entry_times:   list[float] = []   # sim_t when each entry was ADDED (for smooth scroll)
    pending:       dict[int, dict] = {}
    ev_ptr = [0]

    def _current_qps(sim_t: float) -> float:
        """QPS from the most recent timeseries sample at or before sim_t."""
        idx = np.searchsorted(ts_t, sim_t, side="right") - 1
        if idx < 0:
            return float(ts_qps[0]) if len(ts_qps) else 4.0
        return float(ts_qps[min(idx, len(ts_qps) - 1)])

    def _redraw_chat(sim_t: float):
        ax_chat.cla()
        ax_chat.set_facecolor(CHAT_BG)
        ax_chat.set_xlim(0, 1); ax_chat.set_ylim(0, 1)
        ax_chat.axis("off")

        ax_chat.text(0.5, 0.979, "LLM Chat Replay",
                     color="#666", fontsize=8.5, ha="center", va="top",
                     fontfamily="monospace", transform=ax_chat.transAxes)

        cur_qps = _current_qps(sim_t)

        # Continuous smooth scroll: y-position is based on how long ago the entry
        # was added, scaled by current QPS. This makes scroll speed double visibly
        # when QPS doubles — each second of history moves through 1 row-height
        # per (1/qps) sim-seconds.
        row_h  = 0.90 / MAX_VISIBLE
        y_base = 0.952 - row_h * 0.5

        visible_items: list[tuple[float, dict]] = []  # (y, entry)
        for entry, t_added in zip(chat_entries, entry_times):
            age         = sim_t - t_added               # seconds since added
            rows_up     = age * cur_qps if cur_qps > 0 else 0.0
            y           = y_base - rows_up * row_h
            # Keep entries from y_base (bottom) up to top of pane
            if y > (y_base - MAX_VISIBLE * row_h) and y <= y_base + row_h:
                visible_items.append((y, entry))

        # Render bottom-most first (lowest y = most recent = bottom)
        visible_items.sort(key=lambda x: -x[0])   # top of list = highest y = oldest

        for rank, (y, entry) in enumerate(visible_items):
            answered = entry.get("answer") is not None
            tgt      = entry.get("target_ms", first_target)

            q_short = _clean(textwrap.shorten(entry["question"], width=60, placeholder="…"))
            ax_chat.text(0.025, y + row_h * 0.27, f"Q  {q_short}",
                         color=Q_COLOR, fontsize=7.0, va="center",
                         fontfamily="monospace", transform=ax_chat.transAxes,
                         clip_on=True)

            if answered:
                a_short   = _clean(textwrap.shorten(entry["answer"], width=72, placeholder="…"))
                badge_col = _ttft_badge_color(entry.get("ttft_ms"), tgt)
                ttft_str  = f"{entry['ttft_ms']:.0f}ms" if entry.get("ttft_ms") else "?"
                ax_chat.text(
                    0.975, y - row_h * 0.06,
                    f"TTFT {ttft_str}",
                    color="white", fontsize=5.8, va="center", ha="right",
                    fontfamily="monospace", transform=ax_chat.transAxes,
                    bbox=dict(boxstyle="round,pad=0.16", fc=badge_col, ec="none"),
                    clip_on=True)
                ax_chat.text(0.025, y - row_h * 0.06, f"A  {a_short}",
                             color=A_COLOR, fontsize=6.8, va="center",
                             fontfamily="monospace", transform=ax_chat.transAxes,
                             clip_on=True)
            else:
                ax_chat.text(0.025, y - row_h * 0.06, "A  …",
                             color=A_PEND, fontsize=6.8, va="center",
                             fontfamily="monospace", transform=ax_chat.transAxes,
                             clip_on=True)

            # Separator line between entries
            sep_y = y - row_h * 0.47
            if 0.02 < sep_y < 0.96:
                ax_chat.axhline(sep_y, color="#1d1d1d", linewidth=0.5)

        # Bottom status bar
        cur_tgt = _target_from_phase(
            next((e["phase"] for e in reversed(chat_entries) if e.get("phase")), ""),
            first_target)
        ax_chat.text(0.025, 0.012,
                     f"QPS {cur_qps:.0f}",
                     color="#888", fontsize=7, ha="left", va="bottom",
                     fontfamily="monospace", transform=ax_chat.transAxes)
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
                entry_times.append(ev["t"])
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
        line_epow.set_data(ts_t[mask], ts_effpow[mask])
        for vl in vlines:
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
