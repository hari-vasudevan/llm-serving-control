#!/usr/bin/env python3
"""
collect_ttft.py  --  Chapter 3 plant identification data collection
Standalone, no external dependencies (uses only stdlib).

Collects TTFT measurements for two experiments:
  Stage 1: B sweep at q=0  (single-shot bursts, no sustained arrivals)
  Stage 2: Sustained load at multiple lambda values

Results saved to: ttft_data.json
Run time: ~2-3 minutes total.
"""

import asyncio
import json
import time
import random
import math
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────
URL          = "http://localhost:11434/api/generate"
MODEL        = "qwen2.5:3b"
NUM_PREDICT  = 1
TIMEOUT      = 15       # s per request

# Stage 1: B sweep
B_SWEEP      = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12]
N_REPS       = 8        # burst repetitions per B value

# Stage 2: sustained load
LAMBDA_SWEEP = [2, 3, 4, 5, 7, 9]   # req/tick
SETTLE_TICKS = 20       # ticks before measuring (let queue reach steady state)
MEAS_TICKS   = 15       # ticks to average over
B_MAX        = 12
B_MIN        = 1
Q_MAX        = 20
LAT_WIN      = 20       # rolling window size

PROMPTS = [
    "What is 2+2?", "Name a planet.", "What colour is grass?",
    "What is water?", "How many days in a week?", "What is the moon?",
    "Name a primary colour.", "What is ice made of?",
    "How many hours in a day?", "What is the sun?"
]

OUT_PATH = Path(__file__).parent / "ttft_data.json"

# ── Single TTFT request (streaming, read first chunk) ─────────────────────
def ttft_request(prompt: str) -> float:
    """Fire one streaming request, return TTFT in ms. Blocking."""
    body = json.dumps({
        "model":   MODEL,
        "prompt":  prompt,
        "stream":  True,
        "options": {"num_predict": NUM_PREDICT}
    }).encode()

    req = urllib.request.Request(
        URL, data=body,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            resp.readline()          # blocks until first token
            return (time.perf_counter() - t0) * 1000
    except Exception as e:
        return float("nan")


# ── Fire B concurrent requests, return list of TTFTs ─────────────────────
def burst(b: int, executor: ThreadPoolExecutor) -> list:
    prompts  = [random.choice(PROMPTS) for _ in range(b)]
    futures  = [executor.submit(ttft_request, p) for p in prompts]
    results  = [f.result() for f in futures]
    return results


# ── Warm-up ───────────────────────────────────────────────────────────────
def warmup(executor, n=4):
    print(f"Warming up {MODEL} ({n} requests)...")
    futs = [executor.submit(ttft_request, "Hello") for _ in range(n)]
    for f in futs:
        f.result()
    print("Warm.\n")


# ── Stage 1: B sweep ─────────────────────────────────────────────────────
def stage1(executor):
    print("=" * 55)
    print("STAGE 1: B sweep at q=0  (alpha, gamma identification)")
    print("=" * 55)
    results = {}
    for b in B_SWEEP:
        reps = []
        for rep in range(N_REPS):
            lats = burst(b, executor)
            valid = [x for x in lats if not math.isnan(x)]
            if valid:
                reps.append(sum(valid) / len(valid))

        mean_lat = sum(reps) / len(reps) if reps else float("nan")
        std_lat  = (sum((x - mean_lat)**2 for x in reps) / max(len(reps)-1, 1))**0.5 if len(reps) > 1 else 0
        results[b] = {"mean": mean_lat, "std": std_lat, "reps": reps}
        print(f"  B={b:2d}:  mean TTFT = {mean_lat:6.1f} ms  (std = {std_lat:5.1f} ms,  n={len(reps)})")

    print()
    return results


# ── Stage 2: sustained load ───────────────────────────────────────────────
def stage2(executor):
    print("=" * 55)
    print("STAGE 2: Sustained load sweep  (beta identification)")
    print("=" * 55)
    results = {}

    for lam in LAMBDA_SWEEP:
        print(f"  lambda={lam} req/tick ...", flush=True)

        q_k     = float(lam)
        lat_buf = [200.0] * LAT_WIN   # warm-start buffer
        buf_idx = 0

        q_hist  = []
        b_hist  = []
        l_hist  = []

        total_ticks = SETTLE_TICKS + MEAS_TICKS

        for tick in range(total_ticks):
            # Poisson arrivals (approximate with rounded Gaussian for speed)
            import random
            a_k = max(0, round(random.gauss(lam, lam**0.5)))

            # Drain rule: try to serve current queue + arrivals
            b_k = max(B_MIN, min(B_MAX, round(q_k + a_k)))

            # Fire b_k concurrent requests
            lats = burst(b_k, executor)
            valid_lats = [x for x in lats if not math.isnan(x)]

            # Update rolling buffer
            for lat in valid_lats:
                lat_buf[buf_idx % LAT_WIN] = lat
                buf_idx += 1

            l_meas = sum(lat_buf) / len(lat_buf)

            # Queue update
            q_k = max(0, min(Q_MAX, q_k + a_k - b_k))

            q_hist.append(q_k)
            b_hist.append(b_k)
            l_hist.append(l_meas)

        # Average over measurement window
        meas_q = q_hist[SETTLE_TICKS:]
        meas_b = b_hist[SETTLE_TICKS:]
        meas_l = l_hist[SETTLE_TICKS:]

        q_ss = sum(meas_q) / len(meas_q)
        b_ss = sum(meas_b) / len(meas_b)
        l_ss = sum(meas_l) / len(meas_l)

        results[lam] = {
            "q_ss": q_ss, "b_ss": b_ss, "l_ss": l_ss,
            "q_hist": q_hist, "b_hist": b_hist, "l_hist": l_hist
        }
        print(f"    q_ss={q_ss:.2f}  b_ss={b_ss:.2f}  l_ss={l_ss:.1f} ms")

    print()
    return results


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    t_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=24) as executor:
        warmup(executor)
        s1 = stage1(executor)
        s2 = stage2(executor)

    elapsed = time.perf_counter() - t_start
    print(f"Total collection time: {elapsed:.1f} s\n")

    data = {
        "model":   MODEL,
        "stage1":  {str(k): v for k, v in s1.items()},
        "stage2":  {str(k): v for k, v in s2.items()},
        "config": {
            "b_sweep":      B_SWEEP,
            "lambda_sweep": LAMBDA_SWEEP,
            "n_reps":       N_REPS,
            "settle_ticks": SETTLE_TICKS,
            "meas_ticks":   MEAS_TICKS
        }
    }

    with open(OUT_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Data saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
