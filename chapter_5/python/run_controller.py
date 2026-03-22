#!/usr/bin/env python3
"""
run_controller.py  --  Chapter 5: Closed-loop integral controller on live vLLM

Controller law (correctly signed for GPU-contention plant):

    e_l[k]      = L_target - l_meas[k]      # positive when below target
    xi_l[k+1]   = xi_l[k] + e_l[k]          # integral of error
    B[k]        = clamp(B0 + K_il*xi_l[k], B_min, B_max)

    When l > target:  e_l < 0 → xi_l ↓ → K_il*xi_l ↓ → B ↓ → less contention → l ↓  ✓
    When l < target:  e_l > 0 → xi_l ↑ → B ↑ → more throughput                        ✓

Latency measurement: rolling mean of the last N raw TTFT samples from
the vLLM Prometheus histogram (per-tick deltas). Rolling mean smooths
the noisy per-tick measurements without adding lag beyond N/2 ticks.

Usage:
    python3 run_controller.py [--params controller_params.json] [options]

    # Steady-state test (60 ticks):
    python3 run_controller.py --n_ticks 60

    # Step-load spike:
    python3 run_controller.py --n_ticks 90 --spike_on 30 --spike_off 60 --spike_mult 3
"""

import argparse
import json
import math
import re
import threading
import time
import statistics
import sys
from collections import deque
from pathlib import Path
from datetime import datetime
from typing import Optional

import requests
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_PARAMS = Path(__file__).parent / "controller_params.json"
DEFAULT_URL    = "http://localhost:8001"
DEFAULT_MODEL  = "mlx-community/Qwen3-0.6B-4bit"

PROMPTS = [
    "What is 2+2?",
    "Name a colour.",
    "What is the capital of France?",
    "How many days in a week?",
    "Name a planet.",
    "What is the speed of light?",
    "Name a mammal.",
    "What is 10 times 10?",
]


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
def ttft_ms(url: str, model: str, prompt: str, timeout: int = 30) -> float:
    body = json.dumps({"model": model, "prompt": prompt,
                       "max_tokens": 1, "stream": True})
    t0 = time.perf_counter()
    with requests.post(f"{url}/v1/completions", data=body,
                       headers={"Content-Type": "application/json"},
                       stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        for chunk in resp.iter_lines():
            if chunk and chunk != b"data: [DONE]":
                return (time.perf_counter() - t0) * 1000
    return (time.perf_counter() - t0) * 1000


def get_metrics(url: str) -> dict:
    raw = requests.get(f"{url}/metrics", timeout=5).text
    out = {}
    for line in raw.splitlines():
        if line.startswith("#"):
            continue
        clean = re.sub(r"\{[^}]*\}", "", line).strip()
        parts = clean.split()
        if len(parts) < 2:
            continue
        try:
            v = float(parts[1])
        except ValueError:
            continue
        out[parts[0]] = out.get(parts[0], 0.0) + v
    return out


def fire_b_concurrent(url: str, model: str, b: int,
                      results: list, timeout: int = 30):
    """Fire B TTFT requests concurrently, filling results list in-place."""
    def worker(i):
        try:
            results[i] = ttft_ms(url, model, PROMPTS[i % len(PROMPTS)], timeout)
        except Exception:
            results[i] = float("nan")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(b)]
    for t in threads: t.start()
    for t in threads: t.join()


# ---------------------------------------------------------------------------
# Latency smoother
# ---------------------------------------------------------------------------
class RollingMean:
    """Rolling mean over the last N samples. Returns NaN until primed."""
    def __init__(self, n: int, init_val: float = None):
        self._buf  = deque(maxlen=n)
        self._n    = n
        if init_val is not None:
            for _ in range(n):
                self._buf.append(init_val)

    def update(self, val: float) -> float:
        if not math.isnan(val):
            self._buf.append(val)
        if len(self._buf) == 0:
            return float("nan")
        valid = [x for x in self._buf if not math.isnan(x)]
        return statistics.mean(valid) if valid else float("nan")


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------
class IntegralController:
    """
    Single integral controller: B[k] = B0 + K_il * xi_l[k]
    K_il > 0 (positive) for GPU-contention plant where more B → more latency.
    """
    def __init__(self, p: dict):
        c = p["controller"]
        self.K_il     = c["K_il"]         # positive
        self.B0       = c["B0"]
        self.B_min    = c["B_min"]
        self.B_max    = c["B_max"]
        self.xi_min   = c["xi_min"]
        self.xi_max   = c["xi_max"]
        self.L_target = p["L_target"]
        self.xi_l     = 0.0

    def step(self, l_meas: float) -> tuple:
        e_l      = self.L_target - l_meas
        xi_sat   = max(self.xi_min, min(self.xi_max, self.xi_l))
        B_raw    = self.B0 + self.K_il * xi_sat
        B_cmd    = int(np.clip(round(B_raw), self.B_min, self.B_max))

        # Conditional anti-windup
        at_min = B_cmd <= self.B_min and e_l < 0   # latency above target, B already min
        at_max = B_cmd >= self.B_max and e_l > 0   # latency below target, B already max
        if not (at_min or at_max):
            self.xi_l = max(self.xi_min, min(self.xi_max, xi_sat + e_l))
        else:
            self.xi_l = xi_sat

        return B_cmd, float(B_raw), e_l, xi_sat


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------
def run_experiment(url, model, ctrl, params, n_ticks, lambda_mean,
                   spike_on, spike_off, spike_mult, dt, n_smooth):

    L_target = ctrl.L_target
    smoother = RollingMean(n_smooth, init_val=L_target)

    # Prime histogram baseline
    metrics0 = get_metrics(url)
    prev_s   = metrics0.get("vllm:time_to_first_token_seconds_sum",   0.0)
    prev_c   = metrics0.get("vllm:time_to_first_token_seconds_count", 0.0)

    log = {k: [] for k in ["tick","lam","B","l_raw","l_smooth",
                            "l_target","e_l","xi_l","B_raw"]}

    print(f"\n{'═'*68}")
    print(f"CLOSED-LOOP RUN  --  {n_ticks} ticks  dt={dt}s  λ={lambda_mean}")
    if spike_on:
        print(f"  Spike ticks {spike_on}–{spike_off}: λ × {spike_mult:.0f}")
    print(f"  L_target={L_target:.0f} ms  K_il={ctrl.K_il:.6f}  B0={ctrl.B0}")
    print(f"{'═'*68}\n")
    print(f"{'k':>4} {'λ':>4} {'B':>3} {'l_raw':>7} {'l_sm':>7} {'e_l':>7} {'xi_l':>8}  note")
    print("-" * 68)

    for tick in range(n_ticks):
        lam = lambda_mean * spike_mult if (spike_on and spike_off
              and spike_on <= tick < spike_off) else lambda_mean
        a_k = int(np.random.poisson(lam))

        t0 = time.perf_counter()

        # --- Measure latency from histogram delta ---
        m  = get_metrics(url)
        cs = m.get("vllm:time_to_first_token_seconds_sum",   0.0)
        cc = m.get("vllm:time_to_first_token_seconds_count", 0.0)
        dc = cc - prev_c
        ds = cs - prev_s
        l_raw = (ds / dc) * 1000.0 if dc > 0 else float("nan")
        prev_s, prev_c = cs, cc

        l_smooth = smoother.update(l_raw)

        # Use smoothed value for control; fall back to target on NaN
        l_ctrl = l_smooth if not math.isnan(l_smooth) else L_target

        # --- Control ---
        B_cmd, B_raw, e_l, xi_l = ctrl.step(l_ctrl)

        # --- Actuate: fire exactly B_cmd concurrent TTFT requests ---
        results = [float("nan")] * B_cmd
        fire_b_concurrent(url, model, B_cmd, results)

        # Also update smoother with direct measurements if histogram was stale
        valid = [x for x in results if not math.isnan(x)]
        if valid and math.isnan(l_raw):
            l_raw    = statistics.mean(valid)
            l_smooth = smoother.update(l_raw)

        note = "SPIKE" if (spike_on and spike_off and spike_on <= tick < spike_off) else ""
        l_raw_str    = f"{l_raw:.0f}"    if not math.isnan(l_raw)    else "  NaN"
        l_smooth_str = f"{l_smooth:.0f}" if not math.isnan(l_smooth) else "  NaN"

        print(f"{tick+1:4d} {lam:4.0f} {B_cmd:3d} {l_raw_str:>7} "
              f"{l_smooth_str:>7} {e_l:>7.1f} {xi_l:>8.1f}  {note}")

        log["tick"].append(tick + 1)
        log["lam"].append(lam)
        log["B"].append(B_cmd)
        log["l_raw"].append(l_raw if not math.isnan(l_raw) else None)
        log["l_smooth"].append(l_smooth if not math.isnan(l_smooth) else None)
        log["l_target"].append(L_target)
        log["e_l"].append(e_l)
        log["xi_l"].append(xi_l)
        log["B_raw"].append(B_raw)

        elapsed = time.perf_counter() - t0
        if elapsed < dt:
            time.sleep(dt - elapsed)

    return log


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_results(log, params, out_dir, label=""):
    ticks     = log["tick"]
    L_tgt     = params["L_target"]
    spike_on  = params.get("spike_on")
    spike_off = params.get("spike_off")

    # Replace None with NaN for plotting
    l_raw    = [x if x is not None else float("nan") for x in log["l_raw"]]
    l_smooth = [x if x is not None else float("nan") for x in log["l_smooth"]]

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)

    # Panel 1: Latency
    ax = axes[0]
    ax.plot(ticks, l_raw,    color="lightblue", linewidth=0.8,
            alpha=0.6, label="l_raw (per-tick)")
    ax.plot(ticks, l_smooth, color="blue", linewidth=1.8,
            label="l_smooth (rolling mean)")
    ax.axhline(L_tgt, color="black", linestyle="--", linewidth=1.5,
               label=f"L_target = {L_tgt:.0f} ms")
    if spike_on and spike_off:
        ax.axvspan(spike_on, spike_off, alpha=0.12, color="orange",
                   label="Load spike")
    ax.set_ylabel("TTFT [ms]")
    ax.set_title(f"Chapter 5 — Closed-loop integral controller  {label}")
    ax.legend(loc="upper right"); ax.grid(True, alpha=0.4)

    # Panel 2: Batch size B
    ax = axes[1]
    ax.step(ticks, log["B"], color="purple", linewidth=1.5,
            where="post", label="B (batch size commanded)")
    ax.axhline(params["B0"], color="gray", linestyle=":",
               label=f"B0 = {params['B0']}")
    if spike_on and spike_off:
        ax.axvspan(spike_on, spike_off, alpha=0.12, color="orange")
    ax.set_ylabel("Batch size B [req]")
    ax.set_ylim(max(0, params["B_min"] - 0.5), params["B_max"] + 0.5)
    ax.legend(loc="upper right"); ax.grid(True, alpha=0.4)

    # Panel 3: Arrival rate and integrator state
    ax  = axes[2]
    ax2 = ax.twinx()
    ax.step(ticks, log["lam"], "k--", linewidth=1.2,
            where="post", label="λ (arrivals/tick)")
    ax2.plot(ticks, log["xi_l"], color="green", linewidth=1.2,
             alpha=0.7, label="ξ_l (integrator)")
    ax2.axhline(0, color="green", linestyle=":", alpha=0.4)
    if spike_on and spike_off:
        ax.axvspan(spike_on, spike_off, alpha=0.12, color="orange")
    ax.set_xlabel("Tick [k]")
    ax.set_ylabel("Arrivals / tick")
    ax2.set_ylabel("ξ_l  (integrator state)", color="green")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2,
              loc="upper right"); ax.grid(True, alpha=0.4)

    fig.tight_layout()
    ts   = datetime.now().strftime("%H%M%S")
    path = out_dir / f"ch5_closed_loop_{ts}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {path}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params",       default=str(DEFAULT_PARAMS))
    ap.add_argument("--url",          default=DEFAULT_URL)
    ap.add_argument("--model",        default=DEFAULT_MODEL)
    ap.add_argument("--n_ticks",      type=int,   default=60)
    ap.add_argument("--dt",           type=float, default=1.0)
    ap.add_argument("--lambda_mean",  type=float, default=3.0)
    ap.add_argument("--spike_on",     type=int,   default=None)
    ap.add_argument("--spike_off",    type=int,   default=None)
    ap.add_argument("--spike_mult",   type=float, default=3.0)
    ap.add_argument("--n_smooth",     type=int,   default=6,
                    help="Rolling window for latency smoothing")
    ap.add_argument("--out_dir",      default=str(Path(__file__).parent))
    ap.add_argument("--L_target",     type=float, default=None)
    args = ap.parse_args()

    if not Path(args.params).exists():
        sys.exit(f"Not found: {args.params}\nRun: python3 design_controller.py first.")

    with open(args.params) as f:
        params = json.load(f)
    print(f"Loaded: {args.params}")

    if args.L_target is not None:
        params["L_target"] = args.L_target
        params["controller"]["xi_min"] = (params["B_min"] - params["B0"]) / params["controller"]["K_il"]
        params["controller"]["xi_max"] = (params["B_max"] - params["B0"]) / params["controller"]["K_il"]
        print(f"L_target overridden to {args.L_target:.0f} ms")

    try:
        requests.get(f"{args.url}/health", timeout=5).raise_for_status()
        print(f"vLLM healthy at {args.url}")
    except Exception as e:
        sys.exit(f"vLLM not reachable: {e}")

    ctrl = IntegralController(params)
    print(f"Controller: K_il={ctrl.K_il:.6f}  B0={ctrl.B0}  "
          f"B=[{ctrl.B_min},{ctrl.B_max}]  L_target={ctrl.L_target:.0f} ms")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    params["spike_on"]  = args.spike_on
    params["spike_off"] = args.spike_off

    log = run_experiment(
        url         = args.url,
        model       = args.model,
        ctrl        = ctrl,
        params      = params,
        n_ticks     = args.n_ticks,
        lambda_mean = args.lambda_mean,
        spike_on    = args.spike_on,
        spike_off   = args.spike_off,
        spike_mult  = args.spike_mult,
        dt          = args.dt,
        n_smooth    = args.n_smooth,
    )

    # Summary
    valid_raw    = [x for x in log["l_raw"]    if x is not None]
    valid_smooth = [x for x in log["l_smooth"] if x is not None]
    print("\n=== Run summary ===")
    if valid_smooth:
        p95 = sorted(valid_smooth)[int(0.95 * len(valid_smooth))]
        print(f"  l_smooth: mean={statistics.mean(valid_smooth):.1f} ms  "
              f"p95={p95:.1f} ms  target={params['L_target']:.0f} ms")
    if valid_raw:
        print(f"  l_raw:    mean={statistics.mean(valid_raw):.1f} ms  "
              f"max={max(valid_raw):.1f} ms")
    b_vals = log["B"]
    print(f"  B:        mean={statistics.mean(b_vals):.2f}  "
          f"min={min(b_vals)}  max={max(b_vals)}")

    ts  = datetime.now().strftime("%H%M%S")
    lp  = out_dir / f"ch5_run_log_{ts}.json"
    with open(lp, "w") as f:
        json.dump({"params": params, "log": log}, f, indent=2)
    print(f"  Log: {lp}")

    plot_results(log, params, out_dir,
                 label=f"λ={args.lambda_mean}  dt={args.dt}s")


if __name__ == "__main__":
    main()
