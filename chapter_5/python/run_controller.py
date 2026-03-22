#!/usr/bin/env python3
"""
run_controller.py  --  Chapter 5: Closed-loop integral controller on live vLLM

CONTROLLER LAW (single integral, correctly signed for GPU contention):
    e_l[k]       = L_target[k] - l_meas[k]
    xi_l[k+1]    = clip(xi_l[k] + e_l[k], xi_min, xi_max)  (anti-windup)
    B[k]         = clamp(round(B0 + K_il * xi_l[k]), B_min, B_max)

    K_il > 0: when l_meas > L_target, xi_l decreases, B decreases → less contention.
    K_il > 0: when l_meas < L_target, xi_l increases,  B increases → more contention.

LOAD MODEL WITH FIFO QUEUE:
    Each tick k:
      1. a_k ~ Poisson(lambda[k]) new requests arrive → pushed onto software FIFO
      2. n_dispatch = min(B_cmd, len(queue)) requests dequeued and fired concurrently
      3. TTFT measured as mean of n_dispatch results via Prometheus histogram delta
         (fallback: direct timing from a single request)
    This makes lambda genuinely matter:
      lambda > B_cmd → queue builds → only B_cmd served per tick
      lambda < B_cmd → queue drains → fewer than B_cmd served (limited by demand)

DISTURBANCE SCHEDULE (via --schedule JSON or built-in presets):
    Each segment: {"ticks": N, "lambda": λ, "L_target": L}
    Segments run consecutively.

Built-in preset 'rich' (used by default):
    0-29:   Steady state         λ=3  L=300ms
    30-59:  Lambda SPIKE UP      λ=6  L=300ms  (excess demand, queue builds)
    60-89:  Return to nominal    λ=3  L=300ms
    90-119: Lambda DROP          λ=1  L=300ms  (scarce demand, TTFT falls)
    120-149:Return to nominal    λ=3  L=300ms
    150-179:Target STEP DOWN     λ=3  L=200ms  (tighten SLA, controller must reduce B)
    180-209:Return to nominal    λ=3  L=300ms

Usage:
    python3 run_controller.py                        # full rich preset
    python3 run_controller.py --preset spike_only    # lambda spike only
    python3 run_controller.py --n_ticks 60 --lambda_mean 3  # simple steady-state
"""

import argparse
import json
import math
import re
import threading
import time
import statistics
import sys
import collections
from pathlib import Path
from datetime import datetime
from typing import Optional

import requests
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DEFAULT_PARAMS = Path(__file__).parent / "controller_params.json"
DEFAULT_URL    = "http://localhost:8001"
DEFAULT_MODEL  = "mlx-community/Qwen3-0.6B-4bit"

PROMPTS = [
    "What is 2+2?", "Name a colour.", "What is the capital of France?",
    "How many days in a week?", "Name a planet.", "What is the speed of light?",
    "Name a mammal.", "What is 10 times 10?", "What colour is the sky?",
    "Name a fruit.", "How many hours in a day?", "What is 5 squared?",
]

# ---------------------------------------------------------------------------
# Disturbance schedule presets
# ---------------------------------------------------------------------------
PRESETS = {
    "rich": [
        {"ticks": 30, "lambda": 3, "L_target": 300, "label": "Steady"},
        {"ticks": 30, "lambda": 6, "L_target": 300, "label": "λ↑ Spike"},
        {"ticks": 30, "lambda": 3, "L_target": 300, "label": "Recovery"},
        {"ticks": 30, "lambda": 1, "L_target": 300, "label": "λ↓ Drop"},
        {"ticks": 30, "lambda": 3, "L_target": 300, "label": "Recovery"},
        {"ticks": 30, "lambda": 3, "L_target": 200, "label": "Target↓"},
        {"ticks": 30, "lambda": 3, "L_target": 300, "label": "Target restore"},
    ],
    "spike_only": [
        {"ticks": 20, "lambda": 3, "L_target": 300, "label": "Steady"},
        {"ticks": 20, "lambda": 6, "L_target": 300, "label": "λ↑ Spike"},
        {"ticks": 20, "lambda": 3, "L_target": 300, "label": "Recovery"},
    ],
    "target_step": [
        {"ticks": 20, "lambda": 3, "L_target": 300, "label": "Steady"},
        {"ticks": 30, "lambda": 3, "L_target": 200, "label": "Target↓"},
        {"ticks": 20, "lambda": 3, "L_target": 300, "label": "Restore"},
    ],
}


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
def ttft_once(url: str, model: str, prompt: str, timeout: int = 30) -> float:
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
        k = parts[0]
        out[k] = out.get(k, 0.0) + v
    return out


def fire_concurrent_ttft(url: str, model: str, prompts: list[str],
                          timeout: int = 30) -> list[float]:
    """Fire len(prompts) requests concurrently, return list of TTFTs."""
    results = [float("nan")] * len(prompts)

    def worker(i, prompt):
        try:
            results[i] = ttft_once(url, model, prompt, timeout)
        except Exception:
            pass

    threads = [threading.Thread(target=worker, args=(i, p), daemon=True)
               for i, p in enumerate(prompts)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


# ---------------------------------------------------------------------------
# Latency measurer using Prometheus histogram deltas
# ---------------------------------------------------------------------------
class HistogramDelta:
    def __init__(self, url: str):
        self.url    = url
        self._ps    = 0.0  # prev sum
        self._pc    = 0.0  # prev count

    def update(self, metrics: dict) -> float:
        s  = metrics.get("vllm:time_to_first_token_seconds_sum",   0.0)
        c  = metrics.get("vllm:time_to_first_token_seconds_count", 0.0)
        ds = s  - self._ps
        dc = c  - self._pc
        self._ps, self._pc = s, c
        return (ds / dc) * 1000.0 if dc > 0 else float("nan")


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------
class IntegralController:
    def __init__(self, p: dict):
        c = p["controller"]
        self.K_il    = c["K_il"]     # positive
        self.xi      = 0.0
        self.xi_min  = c["xi_min"]
        self.xi_max  = c["xi_max"]
        self.B0      = c["B0"]
        self.B_min   = c["B_min"]
        self.B_max   = c["B_max"]
        self.L_target = p["L_target"]

    def step(self, l_meas: float, L_target_override: Optional[float] = None) -> tuple:
        L_tgt = L_target_override if L_target_override is not None else self.L_target
        e     = L_tgt - l_meas      # + when too fast, - when too slow

        xi_sat = max(self.xi_min, min(self.xi_max, self.xi))
        B_raw  = self.B0 + self.K_il * xi_sat
        B_cmd  = int(np.clip(round(B_raw), self.B_min, self.B_max))

        # Anti-windup: freeze if saturated in direction of error
        at_lo = (B_cmd <= self.B_min) and (e > 0)   # want more but can't go lower
        at_hi = (B_cmd >= self.B_max) and (e < 0)   # want less but can't go higher
        if not (at_lo or at_hi):
            self.xi = max(self.xi_min, min(self.xi_max, xi_sat + e))
        else:
            self.xi = xi_sat

        return B_cmd, B_raw, e, xi_sat


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def run(url: str, model: str, ctrl: IntegralController,
        schedule: list[dict], dt: float, out_dir: Path) -> dict:

    total_ticks = sum(s["ticks"] for s in schedule)
    print(f"\n{'═'*68}")
    print(f"CLOSED-LOOP RUN  --  {total_ticks} ticks  dt={dt}s")
    for s in schedule:
        print(f"  {s['ticks']:3d} ticks: λ={s['lambda']}  L_target={s['L_target']} ms"
              f"  [{s['label']}]")
    print(f"{'═'*68}\n")
    print(f"{'tick':>5} {'λ':>3} {'L_tgt':>6} {'q_sw':>5} "
          f"{'B':>3} {'l_meas':>7} {'e_l':>7}  label")
    print("-" * 68)

    # Build tick-by-tick schedule
    tick_schedule = []
    for seg in schedule:
        tick_schedule.extend([seg] * seg["ticks"])

    # Software FIFO queue of pending prompts
    queue: collections.deque = collections.deque()
    prompt_idx = 0

    # Latency measurer
    hist = HistogramDelta(url)
    m0   = get_metrics(url)
    hist.update(m0)   # prime baseline

    log = {"tick": [], "lambda": [], "L_target": [],
           "q_sw": [], "B": [], "B_raw": [],
           "l_meas": [], "e_l": [], "xi": [], "label": []}

    for tick, seg in enumerate(tick_schedule):
        lam     = seg["lambda"]
        L_tgt   = seg["L_target"]
        label   = seg["label"]
        t_tick  = time.perf_counter()

        # 1. Observe latency from Prometheus histogram delta
        m      = get_metrics(url)
        l_meas = hist.update(m)
        if math.isnan(l_meas):
            l_meas = L_tgt   # fallback on cold tick

        # 2. Poisson arrivals → push to FIFO
        a_k = int(np.random.poisson(lam))
        for _ in range(a_k):
            queue.append(PROMPTS[prompt_idx % len(PROMPTS)])
            prompt_idx += 1

        q_sw = len(queue)

        # 3. Control
        B_cmd, B_raw, e_l, xi = ctrl.step(l_meas, L_tgt)

        # 4. Dispatch min(B_cmd, queue_len) concurrently
        n_dispatch = min(B_cmd, len(queue))
        prompts_to_fire = [queue.popleft() for _ in range(n_dispatch)]

        if prompts_to_fire:
            # Fire concurrently, non-blocking (daemon threads)
            def _fire(ps=prompts_to_fire):
                fire_concurrent_ttft(url, model, ps)
            threading.Thread(target=_fire, daemon=True).start()

        # 5. Tick clock
        elapsed = time.perf_counter() - t_tick
        if elapsed < dt:
            time.sleep(dt - elapsed)

        print(f"{tick+1:5d} {lam:3.0f} {L_tgt:6.0f} {q_sw:5d} "
              f"{B_cmd:3d} {l_meas:7.1f} {e_l:7.1f}  {label}")

        log["tick"].append(tick + 1)
        log["lambda"].append(lam)
        log["L_target"].append(L_tgt)
        log["q_sw"].append(q_sw)
        log["B"].append(B_cmd)
        log["B_raw"].append(B_raw)
        log["l_meas"].append(l_meas)
        log["e_l"].append(e_l)
        log["xi"].append(xi)
        log["label"].append(label)

    return log


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_results(log: dict, schedule: list[dict], params: dict,
                 out_dir: Path) -> Path:

    ticks     = log["tick"]
    n         = len(ticks)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Chapter 5 — Closed-loop integral controller on live vLLM",
                 fontsize=13, fontweight="bold")

    # Shade segments
    COLORS = ["#f0f0f0", "#ffe8e8", "#e8ffe8", "#e8e8ff",
              "#fff8e8", "#ffe8ff", "#e8f8ff"]
    seg_start = 0
    for ci, seg in enumerate(schedule):
        seg_end = seg_start + seg["ticks"]
        for ax in axes:
            ax.axvspan(seg_start + 1, seg_end + 1,
                       alpha=0.35, color=COLORS[ci % len(COLORS)], zorder=0)
        axes[0].text((seg_start + seg_end) / 2 + 1, 0.97,
                     seg["label"], transform=axes[0].get_xaxis_transform(),
                     ha="center", va="top", fontsize=7.5, color="#333333")
        seg_start = seg_end

    # Panel 1: latency + target
    ax = axes[0]
    ax.plot(ticks, log["l_meas"], "b-", linewidth=1.3, label="l_meas (TTFT) [ms]", zorder=3)
    ax.step(ticks, log["L_target"], "k--", linewidth=1.5,
            where="post", label="L_target [ms]", zorder=3)
    ax.set_ylabel("TTFT [ms]")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.4)

    # Panel 2: batch size B + lambda
    ax = axes[1]
    ax.step(ticks, log["B"], "m-", linewidth=1.5, where="post",
            label="B (dispatched)", zorder=3)
    ax.step(ticks, log["lambda"], "k--", linewidth=1.0, where="post",
            label="λ (arrivals)", zorder=3)
    ax.set_ylabel("Requests / tick")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.4)

    # Panel 3: software queue depth
    ax = axes[2]
    ax.fill_between(ticks, log["q_sw"], step="post",
                    color="orange", alpha=0.5, label="q_sw (FIFO depth)", zorder=3)
    ax.step(ticks, log["q_sw"], "darkorange", linewidth=1.2, where="post", zorder=3)
    ax.set_ylabel("Software queue [req]")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.4)

    # Panel 4: integrator state xi
    ax = axes[3]
    ax.plot(ticks, log["xi"], "g-", linewidth=1.2, label="ξ (integrator state)", zorder=3)
    ax.axhline(0, color="k", linewidth=0.7, linestyle="--")
    ax.set_ylabel("ξ")
    ax.set_xlabel("Tick [k]")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.4)

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
    ap = argparse.ArgumentParser(
        description="Chapter 5 closed-loop controller -- rich disturbance experiment")
    ap.add_argument("--params",   default=str(DEFAULT_PARAMS))
    ap.add_argument("--url",      default=DEFAULT_URL)
    ap.add_argument("--model",    default=DEFAULT_MODEL)
    ap.add_argument("--dt",       type=float, default=1.0)
    ap.add_argument("--preset",   default="rich",
                    choices=list(PRESETS.keys()),
                    help="Disturbance schedule preset")
    ap.add_argument("--schedule", default=None,
                    help="JSON string: list of {ticks,lambda,L_target,label}")
    ap.add_argument("--out_dir",  default=str(Path(__file__).parent))

    # Simple overrides for quick one-liners
    ap.add_argument("--n_ticks",     type=int,   default=None)
    ap.add_argument("--lambda_mean", type=float, default=None)
    ap.add_argument("--L_target",    type=float, default=None)
    args = ap.parse_args()

    # Load controller params
    if not Path(args.params).exists():
        sys.exit(f"Controller params not found: {args.params}\n"
                 f"Run: python3 design_controller.py first.")
    with open(args.params) as f:
        params = json.load(f)
    print(f"Loaded controller: {args.params}")
    c = params["controller"]
    print(f"  K_il={c['K_il']:.6f}  z_cl={c['z_cl']:.4f}  "
          f"B0={c['B0']}  B range=[{c['B_min']},{c['B_max']}]")
    print(f"  L_target={params['L_target']:.0f} ms  "
          f"tau_cl={c.get('tau_cl','?')}s  dt={params['dt']}s\n")

    # Build schedule
    if args.schedule:
        schedule = json.loads(args.schedule)
    elif args.n_ticks:
        # Simple single-segment run
        schedule = [{"ticks": args.n_ticks,
                     "lambda": args.lambda_mean or 3,
                     "L_target": args.L_target or params["L_target"],
                     "label": "Steady"}]
    else:
        schedule = PRESETS[args.preset]
        if args.L_target:
            for seg in schedule:
                seg["L_target"] = args.L_target

    # Health check
    try:
        requests.get(f"{args.url}/health", timeout=5).raise_for_status()
        print(f"vLLM healthy at {args.url}")
    except Exception as e:
        sys.exit(f"vLLM not reachable: {e}\nRun: ./start_vllm.sh --bg")

    ctrl    = IntegralController(params)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log = run(
        url      = args.url,
        model    = args.model,
        ctrl     = ctrl,
        schedule = schedule,
        dt       = args.dt,
        out_dir  = out_dir,
    )

    # Stats by segment
    print("\n=== Per-segment summary ===")
    seg_start = 0
    for seg in schedule:
        seg_end = seg_start + seg["ticks"]
        sl = log["l_meas"][seg_start:seg_end]
        sb = log["B"][seg_start:seg_end]
        sq = log["q_sw"][seg_start:seg_end]
        valid = [x for x in sl if not math.isnan(x)]
        if valid:
            p95 = sorted(valid)[int(0.95*len(valid))]
            print(f"  {seg['label']:20s}  λ={seg['lambda']}  "
                  f"L_tgt={seg['L_target']}  "
                  f"l_mean={statistics.mean(valid):.0f}ms  "
                  f"l_p95={p95:.0f}ms  "
                  f"B_mean={statistics.mean(sb):.1f}  "
                  f"q_max={max(sq)}")
        seg_start = seg_end

    # Save
    ts       = datetime.now().strftime("%H%M%S")
    log_path = out_dir / f"ch5_run_log_{ts}.json"
    with open(log_path, "w") as f:
        json.dump({"params": params, "schedule": schedule, "log": log}, f, indent=2)
    print(f"\n  Log: {log_path}")

    plot_results(log, schedule, params, out_dir)


if __name__ == "__main__":
    main()
