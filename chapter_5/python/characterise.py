#!/usr/bin/env python3
"""
characterise.py  --  Chapter 5: vLLM plant identification

LATENCY DEFINITION
------------------
Latency is measured from the moment a request enters the Python FIFO queue
to the moment the first token arrives.  This is the correct SLO metric from
the user's perspective:

    l_total = t_first_token - t_enqueue
            = queue_wait_time + TTFT_on_dispatch

PLANT MODEL
-----------
For a FIFO queue with q requests waiting and B dispatched per tick (dt=1s):

    queue_wait(q, B) = (q / B) * dt * 1000    [ms]
    TTFT(B)          = alpha*B + gamma*B^2      [ms]  (no intercept)

    l_total(B, q)    = (q / B) * dt * 1000  +  alpha*B + gamma*B^2

Linearised at operating point (B0, q0=0):

    l_total ≈ alpha*B0 + gamma*B0^2
            + (alpha + 2*gamma*B0) * dB        [TTFT slope]
            + (dt*1000 / B0)       * dq        [queue slope]

So:
    beta_q   = d(l_total)/d(q)|_{B0}  = dt*1000 / B0   [ms/req, analytical]
    beta_eff = d(l_total)/d(B)|_{B0}  = alpha + 2*gamma*B0  [ms/req, from fit]

WHY THE CASCADE MAKES SENSE
---------------------------
With this latency definition the cascade architecture is correct:

    Outer loop:  l_total -> q_ref   (integrator, uses beta_q for gain)
    Inner loop:  q_ref   -> B       (pole placement, controls dispatch rate)

When l_total > L_target:
    -> outer decreases q_ref
    -> inner increases B (drain queue faster)
    -> queue_wait decreases
    -> l_total decreases  ✓

STAGES
------
Stage 1: Smoke test
Stage 2: B sweep at q=0, measure l_total (≈ TTFT since queue is empty)
         Fits alpha, gamma.  Computes beta_q analytically.

Usage:
    python3 characterise.py [--url URL] [--B0 B0] [--n_reps N]
"""

import argparse
import json
import time
import threading
import statistics
import re
import sys
from datetime import datetime
from pathlib import Path

import requests
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_URL   = "http://localhost:8001"
DEFAULT_MODEL = "mlx-community/Qwen3-0.6B-4bit"
DEFAULT_OUT   = Path(__file__).parent / "identified_params.json"

PROMPTS = [
    "What is 2+2?", "Name a colour.", "What is the capital of France?",
    "How many days in a week?", "Name a planet.", "What is the speed of light?",
]


def ttft_with_enqueue_time(url, model, prompt, t_enqueue, timeout=30):
    """
    Fire one request.  Returns l_total = (t_first_token - t_enqueue) in ms.
    t_enqueue is the timestamp when this request entered the queue.
    """
    body = json.dumps({"model": model, "prompt": prompt,
                       "max_tokens": 1, "stream": True})
    with requests.post(f"{url}/v1/completions", data=body,
                       headers={"Content-Type": "application/json"},
                       stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        for chunk in resp.iter_lines():
            if chunk and chunk != b"data: [DONE]":
                t_first_token = time.perf_counter()
                return (t_first_token - t_enqueue) * 1000
    return (time.perf_counter() - t_enqueue) * 1000


def get_metrics(url):
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
            out[parts[0]] = out.get(parts[0], 0.0) + float(parts[1])
        except ValueError:
            continue
    return out


def fire_concurrent_with_timestamps(url, model, b, timeout=30):
    """
    Enqueue B requests simultaneously (t_enqueue is the same for all),
    then fire them concurrently.  Returns list of l_total values.

    At q=0 (empty queue before dispatch), t_enqueue ≈ t_dispatch so
    l_total ≈ TTFT.  This is the correct Stage 2 measurement.
    """
    t_enqueue = time.perf_counter()   # all B enter the queue at the same instant
    results   = [float("nan")] * b

    def worker(i):
        prompt = PROMPTS[i % len(PROMPTS)]
        try:
            results[i] = ttft_with_enqueue_time(url, model, prompt, t_enqueue, timeout)
        except Exception:
            pass

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(b)]
    for t in threads: t.start()
    for t in threads: t.join()
    return results


def stage2_b_sweep(url, model, b_sweep, n_reps, B0, dt, out_dir):
    """
    B sweep at q_sw=0.

    We fire B requests concurrently with a shared enqueue timestamp.
    Since q=0 before dispatch, l_total = queue_wait(0,B) + TTFT(B) = TTFT(B).
    This fits alpha and gamma.

    beta_q = dt*1000/B0 is then computed analytically.
    """
    print("\n" + "═"*65)
    print("STAGE 2: B sweep at q_sw=0  [l_total = TTFT since q=0]")
    print(f"  b_sweep={b_sweep},  n_reps={n_reps},  B0={B0},  dt={dt}s")
    print("="*65 + "\n")

    print("[warmup] 3 serial requests...")
    for w in range(3):
        t0  = time.perf_counter()
        lat = ttft_with_enqueue_time(url, model, "Hello", t0)
        print(f"  warmup {w+1}: {lat:.0f} ms")
    print()

    l_mean, l_std = [], []

    for b in b_sweep:
        print(f"[stage2] B={b}  ({n_reps} reps)...")
        rep_means = []
        for r in range(n_reps):
            lats  = fire_concurrent_with_timestamps(url, model, b)
            valid = [x for x in lats if not np.isnan(x)]
            m     = statistics.mean(valid) if valid else float("nan")
            rep_means.append(m)
            print(f"  rep {r+1}: {m:.1f} ms  {[round(x) for x in lats]}")
        mu = statistics.mean(rep_means)
        sd = statistics.stdev(rep_means) if len(rep_means) > 1 else 0.0
        l_mean.append(mu); l_std.append(sd)
        print(f"  --> B={b}: {mu:.1f} ± {sd:.1f} ms\n")

    # Fit TTFT(B) = alpha*B + gamma*B^2  (no intercept, valid since queue is empty)
    B   = np.array(b_sweep, dtype=float)
    L   = np.array(l_mean,  dtype=float)
    A   = np.column_stack([B, B**2])
    p, _, _, _ = np.linalg.lstsq(A, L, rcond=None)
    alpha, gamma = float(p[0]), float(p[1])
    L_fit = A @ p
    r2    = float(1.0 - np.sum((L - L_fit)**2) / np.sum((L - np.mean(L))**2))

    ttft_at_B0 = alpha*B0 + gamma*B0**2

    # Analytical queue delay slope at operating point
    beta_q   = (dt * 1000.0) / B0   # ms per request in queue
    # TTFT slope at operating point
    beta_eff = alpha + 2*gamma*B0    # ms per unit increase in B

    print("RESULTS:")
    print(f"  TTFT(B) = {alpha:.4f}*B + ({gamma:.4f})*B^2   R^2={r2:.4f}")
    print(f"  TTFT(B0={B0})   = {ttft_at_B0:.2f} ms")
    print()
    print(f"  beta_q   = dt*1000/B0 = {dt*1000:.0f}/{B0} = {beta_q:.2f} ms/req")
    print(f"             [d(l_total)/d(q) at B0 -- ANALYTICAL, no stage 3 needed]")
    print()
    print(f"  beta_eff = alpha + 2*gamma*B0")
    print(f"           = {alpha:.4f} + 2*({gamma:.4f})*{B0} = {beta_eff:.4f} ms/req")
    print(f"             [d(TTFT)/dB at B0 -- from fit]")
    print()
    print(f"  Full linearised model at (B0={B0}, q0=0):")
    print(f"    l_total ≈ {ttft_at_B0:.2f} + {beta_eff:.2f}*(B-{B0}) + {beta_q:.2f}*(q-0)")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.errorbar(b_sweep, l_mean, l_std, fmt="bo", capsize=4,
                label="Measured l_total (≈TTFT at q=0)", lw=1.2)
    bf = np.linspace(0.5, max(b_sweep)+0.5, 300)
    ax.plot(bf, alpha*bf + gamma*bf**2, "r-", lw=2,
            label=f"αB+γB²  (R²={r2:.3f})")
    ax.plot(B0, ttft_at_B0, "gs", ms=10, label=f"B0={B0}")
    x_tan = np.array([B0-1.5, B0+1.5])
    ax.plot(x_tan, ttft_at_B0 + beta_eff*(x_tan-B0), "g--", lw=1.5,
            label=f"Slope β_eff={beta_eff:.1f} ms/req")
    ax.set_xlabel("B (concurrent requests)"); ax.set_ylabel("l_total [ms]")
    ax.set_title("TTFT(B) at q=0")
    ax.legend(fontsize=8); ax.grid(True)

    # Second panel: model surface l_total(B, q) -- the full cascade plant
    ax = axes[1]
    q_range = np.linspace(0, 15, 200)
    for b_plot in [1, 2, 3, 4, 6, 8]:
        ttft_b = alpha*b_plot + gamma*b_plot**2
        l_total_q = ttft_b + (q_range / b_plot) * dt * 1000
        ax.plot(q_range, l_total_q, label=f"B={b_plot}")
    ax.axhline(150, color="k", ls="--", lw=1, label="L_target=150ms")
    ax.set_xlabel("q_sw (software FIFO depth)")
    ax.set_ylabel("l_total [ms]")
    ax.set_title("Full plant: l_total(B, q) = TTFT(B) + q/B × dt × 1000")
    ax.legend(fontsize=8); ax.grid(True)

    fig.suptitle(f"Chapter 5 plant identification\n"
                 f"α={alpha:.3f}  γ={gamma:.4f}  β_q={beta_q:.2f}ms/req(analytical)"
                 f"  β_eff={beta_eff:.2f}ms/req",
                 fontsize=11)
    fig.tight_layout()
    path = out_dir / "ch5_stage2_b_sweep.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[plot] {path}\n")

    return {"alpha": alpha, "gamma": gamma, "r2": r2,
            "beta_q": beta_q, "beta_eff": beta_eff,
            "ttft_at_B0": ttft_at_B0, "B0": B0, "dt": dt,
            "b_sweep": b_sweep, "l_mean": l_mean, "l_std": l_std}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url",     default=DEFAULT_URL)
    ap.add_argument("--model",   default=DEFAULT_MODEL)
    ap.add_argument("--out",     default=str(DEFAULT_OUT))
    ap.add_argument("--B0",      type=int,   default=3)
    ap.add_argument("--b_sweep", type=int, nargs="+", default=[1,2,3,4,5,6,8])
    ap.add_argument("--n_reps",  type=int,   default=5)
    ap.add_argument("--dt",      type=float, default=1.0)
    args = ap.parse_args()

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: smoke test
    print("═"*65); print("STAGE 1: Smoke test"); print("═"*65 + "\n")
    try:
        requests.get(f"{args.url}/health", timeout=5).raise_for_status()
        print("[stage1] Health: OK")
    except Exception as e:
        sys.exit(f"[stage1] FAIL: {e}\n  Run: ./start_vllm.sh --bg")

    m = get_metrics(args.url)
    waiting = m.get("vllm:num_requests_waiting", 0)
    print(f"[stage1] num_requests_running = {m.get('vllm:num_requests_running', 0)}")
    print(f"[stage1] num_requests_waiting = {waiting}")
    if isinstance(waiting, float) and waiting > 0:
        sys.exit(f"\n[stage1] ERROR: waiting={waiting} -- stale queue.\n"
                 "  Fix: pkill -9 -f 'vllm serve' && ./start_vllm.sh --bg")
    print("[stage1] Clean. Proceeding.\n")

    s2 = stage2_b_sweep(args.url, args.model, args.b_sweep,
                        args.n_reps, args.B0, args.dt, out_dir)

    print("╔" + "═"*65 + "╗")
    print(f"║  IDENTIFIED PARAMETERS  ({args.model})")
    print("╠" + "═"*65 + "╣")
    print(f"║  Plant:  l_total(B,q) = alpha*B + gamma*B^2 + (q/B)*dt*1000")
    print(f"║")
    print(f"║  alpha   = {s2['alpha']:9.4f}  ms/req      R^2 = {s2['r2']:.4f}")
    print(f"║  gamma   = {s2['gamma']:9.4f}  ms/req^2")
    print(f"║")
    print(f"║  At B0={s2['B0']}, dt={s2['dt']}s:")
    print(f"║    TTFT(B0)  = {s2['ttft_at_B0']:.2f} ms")
    print(f"║    beta_q   = {s2['beta_q']:.2f} ms/req  [d(l_total)/d(q) -- ANALYTICAL]")
    print(f"║    beta_eff = {s2['beta_eff']:.2f} ms/req  [d(l_total)/d(B) -- from fit]")
    print("╚" + "═"*65 + "╝\n")

    result = {
        "model": args.model, "timestamp": datetime.now().isoformat(),
        "alpha": s2["alpha"], "gamma": s2["gamma"],
        "beta_q":   s2["beta_q"],    # analytical: dt*1000/B0
        "beta_eff": s2["beta_eff"],  # from fit: alpha + 2*gamma*B0
        "r2_stage2": s2["r2"],
        "B0": s2["B0"], "dt": s2["dt"],
        "ttft_at_B0": s2["ttft_at_B0"],
        "latency_definition": (
            "l_total = t_first_token - t_enqueue  "
            "(queue wait time + TTFT, from user perspective)"
        ),
        "plant_model": (
            "l_total(B, q) = alpha*B + gamma*B^2 + (q/B)*dt*1000"
        ),
        "stage2": s2,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
