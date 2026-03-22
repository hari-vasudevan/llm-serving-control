#!/usr/bin/env python3
"""
identify_beta.py  --  Chapter 5: Targeted beta identification

The model is:
    TTFT(B, q) = alpha*B + gamma*B^2 + beta*q

Alpha and gamma are already known from characterise.py Stage 2.
Beta is the queuing delay per waiting request.

METHOD: Single-probe approach
------------------------------
We vary the number of background requests to create different queue depths,
then fire a single probe request (B=1) and measure its TTFT.

For B_probe=1:
    TTFT_probe = alpha + gamma + beta*q

This is a straight line in q. We know alpha and gamma from Stage 2,
so we can isolate beta:
    beta = (TTFT_probe - (alpha + gamma)) / q

We repeat at multiple q levels and fit a line through the origin.

Why this works:
    - We directly control q via the number of background requests in flight
    - vllm-metal has max_num_seqs=4 slots: if we have N_bg background requests
      in flight and N_bg > 4, then q = N_bg - 4 requests are waiting
    - The probe goes to the back of the queue and experiences exactly q
      waiting slots ahead of it
    - Single probe (B=1) eliminates the B-dependent term, isolating beta*q

Usage:
    python3 identify_beta.py [--params identified_params.json]
"""

import argparse
import json
import threading
import time
import statistics
import math
from pathlib import Path
from datetime import datetime

import requests
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_PARAMS = Path(__file__).parent / "identified_params.json"
DEFAULT_URL    = "http://localhost:8001"
DEFAULT_MODEL  = "mlx-community/Qwen3-0.6B-4bit"

BG_PROMPT    = "Write a haiku about the ocean."   # longer -> slower -> queue persists longer
PROBE_PROMPT = "What is 2+2?"


def ttft_blocking(url: str, model: str, prompt: str,
                  n_tokens: int = 1, timeout: int = 60) -> float:
    """TTFT via streaming. Blocks until first token arrives."""
    body = json.dumps({
        "model": model, "prompt": prompt,
        "max_tokens": n_tokens, "stream": True,
    })
    t0 = time.perf_counter()
    with requests.post(f"{url}/v1/completions",
                       data=body,
                       headers={"Content-Type": "application/json"},
                       stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        for chunk in resp.iter_lines():
            if chunk and chunk != b"data: [DONE]":
                return (time.perf_counter() - t0) * 1000
    return (time.perf_counter() - t0) * 1000


def fire_background(url: str, model: str, n: int,
                    n_tokens: int, sem: threading.Semaphore,
                    done_event: threading.Event):
    """Fire n background requests; release sem when all are in flight."""
    lock      = threading.Lock()
    in_flight = [0]

    def worker():
        try:
            requests.post(f"{url}/v1/completions",
                          json={"model": model, "prompt": BG_PROMPT,
                                "max_tokens": n_tokens, "stream": False},
                          timeout=60)
        except Exception:
            pass
        finally:
            with lock:
                in_flight[0] += 1
                if in_flight[0] == n:
                    done_event.set()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(n)]
    for t in threads:
        t.start()
    # Signal that all requests have been submitted (not completed)
    # Give a moment for them to reach vLLM
    time.sleep(0.3)
    sem.release()


def probe_at_queue_depth(url: str, model: str, target_q: int,
                         max_num_seqs: int, n_tokens_bg: int,
                         n_reps: int) -> tuple[float, float, list]:
    """
    Fire target_q + max_num_seqs background requests to create queue depth
    of target_q, then measure probe TTFT.

    Returns (mean_ttft, std_ttft, all_ttfts)
    """
    n_bg = target_q + max_num_seqs   # fills all slots + puts target_q in queue
    ttfts = []

    for rep in range(n_reps):
        sem   = threading.Semaphore(0)
        done  = threading.Event()

        # Start background flood
        bg_thread = threading.Thread(
            target=fire_background,
            args=(url, model, n_bg, n_tokens_bg, sem, done),
            daemon=True
        )
        bg_thread.start()

        # Wait until all background requests are in flight (in vLLM's queue)
        sem.acquire()

        # Now fire the probe -- it goes to position q+1 in the queue
        t_probe = ttft_blocking(url, model, PROBE_PROMPT, 1)
        ttfts.append(t_probe)

        # Wait for background requests to finish before next rep
        done.wait(timeout=30)
        time.sleep(0.5)  # small gap between reps

        print(f"    rep {rep+1}/{n_reps}: q={target_q}  TTFT={t_probe:.0f} ms")

    mean = statistics.mean(ttfts)
    std  = statistics.stdev(ttfts) if len(ttfts) > 1 else 0.0
    return mean, std, ttfts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params",       default=str(DEFAULT_PARAMS))
    ap.add_argument("--url",          default=DEFAULT_URL)
    ap.add_argument("--model",        default=DEFAULT_MODEL)
    ap.add_argument("--max_num_seqs", type=int,   default=4)
    ap.add_argument("--n_tokens_bg",  type=int,   default=50,
                    help="tokens for background requests (longer = queue persists longer)")
    ap.add_argument("--q_sweep",      type=str,   default="0,1,2,3,4,6,8",
                    help="comma-separated queue depths to sweep")
    ap.add_argument("--n_reps",       type=int,   default=5,
                    help="probe repetitions per queue depth")
    ap.add_argument("--out",          default=str(DEFAULT_PARAMS))
    args = ap.parse_args()

    q_sweep = [int(x) for x in args.q_sweep.split(",")]

    # Load alpha, gamma from Stage 2
    if Path(args.params).exists():
        with open(args.params) as f:
            identified = json.load(f)
        alpha = identified["alpha"]
        gamma = identified["gamma"]
        print(f"Loaded: alpha={alpha:.4f}  gamma={gamma:.4f}")
    else:
        alpha, gamma = 143.89, -5.25
        print(f"[warn] params not found, using defaults: alpha={alpha}  gamma={gamma}")
        identified = {}

    # Health check
    try:
        requests.get(f"{args.url}/health", timeout=5).raise_for_status()
        print(f"vLLM healthy at {args.url}\n")
    except Exception as e:
        raise SystemExit(f"vLLM not reachable: {e}")

    print("=" * 65)
    print("BETA IDENTIFICATION  --  Single-probe method")
    print("=" * 65)
    print(f"  max_num_seqs = {args.max_num_seqs}")
    print(f"  n_tokens_bg  = {args.n_tokens_bg}  (longer = more stable queue)")
    print(f"  q_sweep      = {q_sweep}")
    print(f"  n_reps       = {args.n_reps}")
    print()
    print("Method:")
    print(f"  For each target q: fire {args.max_num_seqs}+q background requests,")
    print(f"  then fire 1 probe and measure TTFT.")
    print(f"  TTFT_probe = alpha + gamma + beta*q")
    print(f"  Known: alpha+gamma = {alpha + gamma:.2f} ms")
    print()

    # Warm up
    print("[warmup] 3 requests...")
    for w in range(3):
        lat = ttft_blocking(args.url, args.model, "Hello", 1)
        print(f"  {w+1}: {lat:.0f} ms")
    print()

    # Sweep
    q_vals   = []
    l_means  = []
    l_stds   = []
    residuals = []   # TTFT - (alpha + gamma)

    for q in q_sweep:
        n_bg = q + args.max_num_seqs
        print(f"[q={q}] Firing {n_bg} background requests ({q} waiting + {args.max_num_seqs} running)...")
        mean, std, ttfts = probe_at_queue_depth(
            url          = args.url,
            model        = args.model,
            target_q     = q,
            max_num_seqs = args.max_num_seqs,
            n_tokens_bg  = args.n_tokens_bg,
            n_reps       = args.n_reps,
        )
        resid = mean - (alpha + gamma)
        q_vals.append(q)
        l_means.append(mean)
        l_stds.append(std)
        residuals.append(resid)
        print(f"  --> mean={mean:.1f} ms  std={std:.1f} ms  residual={resid:.1f} ms\n")

    # Fit beta: residual = beta * q  (regression through origin)
    q_arr = np.array(q_vals,    dtype=float)
    r_arr = np.array(residuals, dtype=float)

    # Only use q > 0 for the regression (q=0 is just a baseline check)
    mask = q_arr > 0
    if mask.sum() >= 2:
        beta = float(np.dot(q_arr[mask], r_arr[mask]) / np.dot(q_arr[mask], q_arr[mask]))
        r_fit = beta * q_arr[mask]
        ss_res = np.sum((r_arr[mask] - r_fit)**2)
        ss_tot = np.sum((r_arr[mask] - np.mean(r_arr[mask]))**2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    else:
        beta = float("nan")
        r2   = float("nan")

    print("=" * 65)
    print(f"  alpha + gamma (at B=1) = {alpha + gamma:.2f} ms  (service, no queue)")
    print(f"  beta  = {beta:.4f} ms/req  (queuing delay per waiting request)")
    print(f"  R^2   = {r2:.4f}")
    print()
    print("  Full model at B=1:")
    print(f"    TTFT(B=1, q) = {alpha:.2f}*1 + {gamma:.4f}*1 + {beta:.2f}*q")
    print(f"                 = {alpha+gamma:.2f} + {beta:.2f}*q")
    print("=" * 65)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.errorbar(q_vals, l_means, l_stds, fmt="bo", capsize=4,
                linewidth=1.2, label="Probe TTFT [ms]")
    if not math.isnan(beta):
        q_fine = np.linspace(0, max(q_vals) * 1.05, 200)
        ax.plot(q_fine, (alpha + gamma) + beta * q_fine, "r-", linewidth=2,
                label=f"(α+γ) + β·q  (β={beta:.1f} ms/req)")
    ax.axhline(alpha + gamma, color="gray", linestyle="--",
               label=f"α+γ={alpha+gamma:.1f} ms (q=0 baseline)")
    ax.set_xlabel("Queue depth q [waiting requests]")
    ax.set_ylabel("Probe TTFT [ms]")
    ax.set_title("Probe TTFT vs queue depth")
    ax.legend(); ax.grid(True)

    ax = axes[1]
    ax.scatter(q_vals, residuals, color="b", s=60, zorder=5, label="Residual")
    if not math.isnan(beta):
        ax.plot(q_fine, beta * q_fine, "r-", linewidth=2,
                label=f"β·q  (β={beta:.1f}, R²={r2:.3f})")
    ax.axhline(0, color="gray", linestyle="--")
    ax.set_xlabel("Queue depth q [waiting requests]")
    ax.set_ylabel("TTFT - (α+γ)  [ms]")
    ax.set_title("Queuing contribution (residual)")
    ax.legend(); ax.grid(True)

    fig.suptitle(f"Beta identification — {args.model}", fontsize=12)
    fig.tight_layout()
    out_dir = Path(args.out).parent
    plot_path = out_dir / "ch5_beta_identification.png"
    fig.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[plot] {plot_path}")

    # Save updated params
    identified.update({
        "beta":           beta,
        "r2_beta":        r2,
        "beta_timestamp": datetime.now().isoformat(),
        "beta_method":    "single_probe",
        "beta_raw": {
            "q_vals":    q_vals,
            "l_means":   l_means,
            "l_stds":    l_stds,
            "residuals": residuals,
        },
    })
    with open(args.out, "w") as f:
        json.dump(identified, f, indent=2)
    print(f"[save] {args.out}")
    print(f"\nRun design_controller.py next to update controller_params.json")


if __name__ == "__main__":
    main()
