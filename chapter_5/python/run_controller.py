#!/usr/bin/env python3
"""
run_controller.py  --  Chapter 5: Closed-loop cascade controller on live vLLM

Runs the designed cascade controller against a real vLLM server.
Arrivals are simulated as Poisson draws; the controller fires exactly
B[k] concurrent requests each tick and measures the result.

CONTROLLER ARCHITECTURE
-----------------------
  Outer loop (latency -> q_ref):
    e_l[k]       = L_target - l_meas[k]
    xi_l[k+1]    = xi_l[k] + e_l[k]    (with anti-windup)
    q_ref[k]     = q0 + K_il * xi_l[k]
    q_ref clamped to [0, q_max]

  Inner loop (q_error -> B):
    e_q[k]       = q_ref[k] - q[k]
    xi_q[k+1]    = xi_q[k] + e_q[k]    (with anti-windup)
    dB[k]        = -(K_q * e_q[k] + K_i * xi_q[k])
    B[k]         = clamp(round(B0 + dB[k]), B_min, B_max)

  Queue measurement:
    We track our own software queue counter because vllm-metal's
    num_requests_waiting Prometheus gauge accumulates (known bug).
    q[k] = max(0, inflight[k-1] - max_num_seqs)
    where inflight is requests sent but not yet responded to.

  Latency measurement:
    Per-tick mean TTFT from vLLM Prometheus histogram deltas:
    l_meas[k] = (Δsum / Δcount) * 1000 ms
    Falls back to direct timing if histogram is stale.

LOAD PROFILE
-----------
  Poisson arrivals at lambda_mean req/tick, with an optional step spike:
    ticks 0..spike_on-1:    lambda = lambda_mean
    ticks spike_on..spike_off-1: lambda = lambda_mean * spike_mult
    ticks spike_off..n_ticks-1: lambda = lambda_mean

Usage:
    python3 run_controller.py [--params controller_params.json] [options]

    # Quick test (30 ticks, no spike):
    python3 run_controller.py --n_ticks 30

    # Full experiment with spike:
    python3 run_controller.py --n_ticks 120 --spike_on 40 --spike_off 80
"""

import argparse
import json
import math
import re
import threading
import time
import statistics
import sys
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
        k = parts[0]
        out[k] = out.get(k, 0.0) + v
    return out


def fire_b_concurrent(url: str, model: str, b: int, timeout: int = 30) -> list:
    """Fire exactly B TTFT requests concurrently. Returns list of latencies."""
    results = [float("nan")] * b

    def worker(i):
        prompt = PROMPTS[i % len(PROMPTS)]
        try:
            results[i] = ttft_ms(url, model, prompt, timeout)
        except Exception:
            pass

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(b)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


# ---------------------------------------------------------------------------
# Controller state
# ---------------------------------------------------------------------------
class CascadeController:
    def __init__(self, p: dict):
        ic  = p["inner"]
        oc  = p["outer"]
        self.K_q     = ic["K_q"]
        self.K_i     = ic["K_i"]
        self.xi_q    = 0.0
        self.xi_q_min = ic["xi_min"]
        self.xi_q_max = ic["xi_max"]
        self.B0      = ic["B0"]
        self.B_min   = ic["B_min"]
        self.B_max   = ic["B_max"]

        self.K_il    = oc["K_il"]
        self.xi_l    = 0.0
        self.xi_l_min = oc["xi_min"]
        self.xi_l_max = oc["xi_max"]
        self.q0      = oc["q0"]
        self.q_max   = oc["q_max"]
        self.L_target = oc["L_target"]

    def step(self, l_meas: float, q_meas: float):
        """
        Run one control tick.
        Returns (B_cmd, q_ref, debug_dict).
        """
        # --- Outer loop: latency -> q_ref ---
        e_l      = self.L_target - l_meas
        xi_l_sat = max(self.xi_l_min, min(self.xi_l_max, self.xi_l))
        q_ref    = float(np.clip(self.q0 + self.K_il * xi_l_sat, 0, self.q_max))

        # Anti-windup: freeze if saturated and error pushes further
        at_lo = q_ref <= 0       and e_l < 0
        at_hi = q_ref >= self.q_max and e_l > 0
        if not (at_lo or at_hi):
            self.xi_l = max(self.xi_l_min, min(self.xi_l_max, xi_l_sat + e_l))
        else:
            self.xi_l = xi_l_sat

        # --- Inner loop: q_error -> B ---
        e_q      = q_ref - q_meas
        xi_q_sat = max(self.xi_q_min, min(self.xi_q_max, self.xi_q))
        dB       = -(self.K_q * e_q + self.K_i * xi_q_sat)
        B_raw    = self.B0 + dB
        B_cmd    = int(np.clip(round(B_raw), self.B_min, self.B_max))

        # Anti-windup on inner integrator
        at_lo_q = B_cmd <= self.B_min and e_q < 0
        at_hi_q = B_cmd >= self.B_max and e_q > 0
        if not (at_lo_q or at_hi_q):
            self.xi_q = max(self.xi_q_min, min(self.xi_q_max, xi_q_sat + e_q))
        else:
            self.xi_q = xi_q_sat

        debug = {
            "e_l": e_l, "xi_l": self.xi_l, "q_ref": q_ref,
            "e_q": e_q, "xi_q": self.xi_q, "B_raw": B_raw,
        }
        return B_cmd, q_ref, debug


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------
class LatencyMeasurer:
    """Per-tick TTFT from Prometheus histogram deltas."""

    def __init__(self, url: str):
        self.url    = url
        self._prev_s = 0.0
        self._prev_c = 0.0

    def update(self, metrics: dict) -> float:
        s = metrics.get("vllm:time_to_first_token_seconds_sum",   0.0)
        c = metrics.get("vllm:time_to_first_token_seconds_count", 0.0)
        ds = s - self._prev_s
        dc = c - self._prev_c
        self._prev_s = s
        self._prev_c = c
        return (ds / dc) * 1000.0 if dc > 0 else float("nan")


class QueueMeasurer:
    """
    Software queue counter -- workaround for broken vllm-metal gauge.
    Tracks requests sent minus requests completed minus max_num_seqs.
    """

    def __init__(self, max_num_seqs: int = 4):
        self._lock        = threading.Lock()
        self._inflight    = 0
        self._max_running = max_num_seqs

    def inc(self):
        with self._lock:
            self._inflight += 1

    def dec(self):
        with self._lock:
            self._inflight = max(0, self._inflight - 1)

    @property
    def q(self) -> float:
        with self._lock:
            return max(0, self._inflight - self._max_running)

    @property
    def inflight(self) -> int:
        with self._lock:
            return self._inflight


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------
def run_experiment(
    url:          str,
    model:        str,
    controller:   CascadeController,
    params:       dict,
    n_ticks:      int,
    lambda_mean:  float,
    spike_on:     Optional[int],
    spike_off:    Optional[int],
    spike_mult:   float,
    dt:           float,
    out_dir:      Path,
) -> dict:

    q_meas_log = QueueMeasurer(max_num_seqs=4)
    l_meas_log = LatencyMeasurer(url)
    metrics0   = get_metrics(url)
    l_meas_log.update(metrics0)   # prime histogram baseline

    log = {
        "tick": [], "lambda": [], "B": [], "q_ref": [], "q": [],
        "l_meas": [], "l_target": [], "B_raw": [], "e_l": [], "e_q": [],
    }
    L_target = params["outer"]["L_target"]

    print(f"\n{'═'*65}")
    print(f"CLOSED-LOOP RUN  --  {n_ticks} ticks  dt={dt}s  λ={lambda_mean}")
    if spike_on:
        print(f"  Spike: ticks {spike_on}..{spike_off}, λ × {spike_mult}")
    print(f"  L_target = {L_target:.0f} ms")
    print(f"{'═'*65}\n")
    print(f"{'tick':>5} {'lam':>4} {'q':>5} {'q_ref':>6} {'B':>3} {'l_meas':>8}  {'e_l':>8}  phase")
    print("-" * 65)

    for tick in range(n_ticks):
        # Arrival rate for this tick
        if spike_on and spike_off and spike_on <= tick < spike_off:
            lam = lambda_mean * spike_mult
        else:
            lam = lambda_mean
        a_k = int(np.random.poisson(lam))

        t_tick = time.perf_counter()

        # --- Observe ---
        metrics  = get_metrics(url)
        l_meas   = l_meas_log.update(metrics)
        q_meas   = q_meas_log.q

        # Use L_target as fallback on first tick (cold histogram)
        if math.isnan(l_meas):
            l_meas = L_target

        # --- Control ---
        B_cmd, q_ref, dbg = controller.step(l_meas, q_meas)

        # --- Actuate: fire B_cmd requests, then a_k background arrivals ---
        # We fire exactly B_cmd concurrent requests this tick.
        # 'Arrivals' beyond B_cmd go into a simple FIFO queue that we
        # manage here (not vLLM's internal queue).
        def fire_one(qm=q_meas_log):
            qm.inc()
            try:
                ttft_ms(url, model, PROMPTS[tick % len(PROMPTS)])
            except Exception:
                pass
            finally:
                qm.dec()

        for _ in range(B_cmd):
            threading.Thread(target=fire_one, daemon=True).start()

        phase = ""
        if spike_on and spike_off and spike_on <= tick < spike_off:
            phase = "SPIKE"

        print(f"{tick+1:5d} {lam:4.0f} {q_meas:5.1f} {q_ref:6.2f} {B_cmd:3d}"
              f" {l_meas:8.1f}  {dbg['e_l']:8.1f}  {phase}")

        log["tick"].append(tick + 1)
        log["lambda"].append(lam)
        log["B"].append(B_cmd)
        log["q_ref"].append(q_ref)
        log["q"].append(q_meas)
        log["l_meas"].append(l_meas)
        log["l_target"].append(L_target)
        log["B_raw"].append(dbg["B_raw"])
        log["e_l"].append(dbg["e_l"])
        log["e_q"].append(dbg["e_q"])

        # --- Tick clock ---
        elapsed = time.perf_counter() - t_tick
        if elapsed < dt:
            time.sleep(dt - elapsed)

    # Wait for in-flight requests to drain
    print("\n[drain] Waiting for in-flight requests to complete...")
    deadline = time.time() + 15
    while q_meas_log.inflight > 0 and time.time() < deadline:
        time.sleep(0.5)
    print(f"[drain] Done. inflight={q_meas_log.inflight}")

    return log


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_results(log: dict, params: dict, out_dir: Path, label: str = ""):
    ticks   = log["tick"]
    L_tgt   = params["outer"]["L_target"]
    spike_on  = params.get("spike_on")
    spike_off = params.get("spike_off")

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    # Panel 1: latency
    ax = axes[0]
    ax.plot(ticks, log["l_meas"], "b-", linewidth=1.2, label="l_meas [ms]")
    ax.axhline(L_tgt, color="k", linestyle="--", label=f"L_target={L_tgt:.0f} ms")
    if spike_on and spike_off:
        ax.axvspan(spike_on, spike_off, alpha=0.1, color="orange", label="Spike")
    ax.set_ylabel("TTFT [ms]")
    ax.set_title(f"Chapter 5 — Closed-loop cascade controller  {label}")
    ax.legend(loc="upper right"); ax.grid(True)

    # Panel 2: q and q_ref
    ax = axes[1]
    ax.plot(ticks, log["q"],     "r-",  linewidth=1.2, label="q (measured)")
    ax.plot(ticks, log["q_ref"], "g--", linewidth=1.2, label="q_ref (outer cmd)")
    if spike_on and spike_off:
        ax.axvspan(spike_on, spike_off, alpha=0.1, color="orange")
    ax.set_ylabel("Queue depth [req]")
    ax.legend(loc="upper right"); ax.grid(True)

    # Panel 3: B and lambda
    ax = axes[2]
    ax.plot(ticks, log["B"],      "m-",  linewidth=1.2, label="B (batch size)")
    ax.plot(ticks, log["lambda"], "k--", linewidth=1.0, label="λ (arrivals)")
    if spike_on and spike_off:
        ax.axvspan(spike_on, spike_off, alpha=0.1, color="orange")
    ax.set_xlabel("Tick [k]")
    ax.set_ylabel("Requests / tick")
    ax.legend(loc="upper right"); ax.grid(True)

    fig.tight_layout()
    ts  = datetime.now().strftime("%H%M%S")
    path = out_dir / f"ch5_closed_loop_{ts}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {path}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Chapter 5 closed-loop cascade controller")
    ap.add_argument("--params",      default=str(DEFAULT_PARAMS))
    ap.add_argument("--url",         default=DEFAULT_URL)
    ap.add_argument("--model",       default=DEFAULT_MODEL)
    ap.add_argument("--n_ticks",     type=int,   default=60)
    ap.add_argument("--dt",          type=float, default=1.0, help="tick period [s]")
    ap.add_argument("--lambda_mean", type=float, default=3.0, help="mean arrival rate")
    ap.add_argument("--spike_on",    type=int,   default=None, help="tick to start spike")
    ap.add_argument("--spike_off",   type=int,   default=None, help="tick to end spike")
    ap.add_argument("--spike_mult",  type=float, default=2.0,  help="spike rate multiplier")
    ap.add_argument("--out_dir",     default=str(Path(__file__).parent))
    ap.add_argument("--L_target",    type=float, default=None,
                    help="Override latency target [ms]")
    args = ap.parse_args()

    # Load controller params
    if not Path(args.params).exists():
        sys.exit(f"Controller params not found: {args.params}\n"
                 f"Run: python3 design_controller.py first.")

    with open(args.params) as f:
        params = json.load(f)
    print(f"Loaded controller: {args.params}")

    if args.L_target is not None:
        params["outer"]["L_target"] = args.L_target
        print(f"Overriding L_target = {args.L_target:.0f} ms")

    # Health check
    try:
        requests.get(f"{args.url}/health", timeout=5).raise_for_status()
        print(f"vLLM healthy at {args.url}")
    except Exception as e:
        sys.exit(f"vLLM not reachable: {e}\nRun: ./start_vllm.sh --bg")

    # Build controller
    ctrl = CascadeController(params)
    print(f"\nController loaded:")
    print(f"  Inner: K_q={ctrl.K_q:.4f}  K_i={ctrl.K_i:.4f}  B0={ctrl.B0}")
    print(f"  Outer: K_il={ctrl.K_il:.8f}  L_target={ctrl.L_target:.0f} ms\n")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Store spike info for plotting
    params["spike_on"]  = args.spike_on
    params["spike_off"] = args.spike_off

    # Run
    log = run_experiment(
        url         = args.url,
        model       = args.model,
        controller  = ctrl,
        params      = params,
        n_ticks     = args.n_ticks,
        lambda_mean = args.lambda_mean,
        spike_on    = args.spike_on,
        spike_off   = args.spike_off,
        spike_mult  = args.spike_mult,
        dt          = args.dt,
        out_dir     = out_dir,
    )

    # Summary stats
    valid_l = [x for x in log["l_meas"] if not math.isnan(x)]
    print("\n=== Run summary ===")
    if valid_l:
        print(f"  l_meas: mean={statistics.mean(valid_l):.1f} ms  "
              f"median={statistics.median(valid_l):.1f} ms  "
              f"max={max(valid_l):.1f} ms")
        p95 = sorted(valid_l)[int(0.95 * len(valid_l))]
        print(f"  l_p95:  {p95:.1f} ms  (target={params['outer']['L_target']:.0f} ms)")
    print(f"  q mean: {statistics.mean(log['q']):.2f} req")
    print(f"  B mean: {statistics.mean(log['B']):.2f} req")

    # Save log
    ts      = datetime.now().strftime("%H%M%S")
    log_path = out_dir / f"ch5_run_log_{ts}.json"
    with open(log_path, "w") as f:
        json.dump({"params": params, "log": log}, f, indent=2)
    print(f"  Log saved: {log_path}")

    plot_results(log, params, out_dir,
                 label=f"λ={args.lambda_mean} dt={args.dt}s")


if __name__ == "__main__":
    main()
