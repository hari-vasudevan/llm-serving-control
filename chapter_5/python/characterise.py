#!/usr/bin/env python3
"""
characterise.py  --  Chapter 5: vLLM plant identification

Identifies the static plant model:
    TTFT(B, q) = alpha*B + gamma*B^2 + beta*q

Stage 1: Smoke test -- confirm /metrics and /v1/completions are reachable
Stage 2: B sweep at q=0 -- fits alpha and gamma from concurrent TTFT measurements
Stage 3: Queue-load sweep -- fits beta by observing TTFT at different q levels
Stage 4: Operating envelope -- maps (lambda, q_ss, l_ss) for controller design

Key design choices vs. MATLAB version:
  - Python threads fire concurrent requests with near-zero dispatch overhead
    (MATLAB parfeval had ~75ms per-worker IPC overhead that prevented queue buildup)
  - Queue tracking uses a software counter (inflight - running) because
    vllm-metal's num_requests_waiting Prometheus gauge accumulates monotonically
    due to a known multiprocessing metric bug
  - All stages are self-contained; results are saved to identified_params.json

Usage:
    python3 characterise.py [--url URL] [--model MODEL] [--out OUT]
"""

import argparse
import json
import time
import threading
import statistics
import re
import sys
import signal
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------
DEFAULT_URL   = "http://localhost:8001"
DEFAULT_MODEL = "mlx-community/Qwen3-0.6B-4bit"
DEFAULT_OUT   = Path(__file__).parent / "identified_params.json"

PROMPTS = [
    "What is 2+2?",
    "Name a colour.",
    "What is the capital of France?",
    "How many days in a week?",
    "Name a planet.",
    "What is the speed of light?",
]


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
def ttft(url: str, model: str, prompt: str, n_tokens: int = 1,
         timeout: int = 30) -> float:
    """Return Time To First Token in ms using streaming."""
    body = json.dumps({
        "model": model, "prompt": prompt,
        "max_tokens": n_tokens, "stream": True,
    })
    t0 = time.perf_counter()
    with requests.post(f"{url}/v1/completions",
                       data=body,
                       headers={"Content-Type": "application/json"},
                       stream=True,
                       timeout=timeout) as resp:
        resp.raise_for_status()
        for chunk in resp.iter_lines():
            if chunk and chunk != b"data: [DONE]":
                return (time.perf_counter() - t0) * 1000
    return (time.perf_counter() - t0) * 1000


def e2e(url: str, model: str, prompt: str, n_tokens: int = 20,
        timeout: int = 30) -> float:
    """Return end-to-end latency in ms (non-streaming)."""
    t0 = time.perf_counter()
    requests.post(f"{url}/v1/completions",
                  json={"model": model, "prompt": prompt,
                        "max_tokens": n_tokens, "stream": False},
                  timeout=timeout).raise_for_status()
    return (time.perf_counter() - t0) * 1000


def get_metrics(url: str) -> dict:
    """Parse /metrics and return a flat dict of gauge/counter values."""
    raw  = requests.get(f"{url}/metrics", timeout=5).text
    out  = {}
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


def read_queue(url: str) -> float:
    """Read num_requests_waiting (treat with caution -- see module docstring)."""
    m = get_metrics(url)
    return m.get("vllm:num_requests_waiting", 0.0)


def fire_concurrent(url: str, model: str, b: int,
                    fn=None, n_tokens: int = 1,
                    timeout: int = 30) -> list[float]:
    """
    Fire B requests concurrently using threads.
    fn is ttft or e2e.  Returns list of latencies in ms.
    """
    if fn is None:
        fn = ttft
    results = [None] * b
    errors  = []

    def worker(i):
        prompt = PROMPTS[i % len(PROMPTS)]
        try:
            results[i] = fn(url, model, prompt, n_tokens, timeout)
        except Exception as ex:
            errors.append(str(ex))
            results[i] = float("nan")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(b)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


# ---------------------------------------------------------------------------
# Stage 2: B sweep at q=0
# ---------------------------------------------------------------------------
def stage2_b_sweep(url: str, model: str, b_sweep: list[int],
                   n_reps: int, out_dir: Path) -> dict:
    print("\n" + "═" * 63)
    print("STAGE 2: B sweep at q=0  [concurrent TTFT, q=0 guaranteed]")
    print("═" * 63 + "\n")

    # Warm up
    print("[stage2] Warming up (3 requests)...")
    for w in range(3):
        lat = ttft(url, model, "Hello", 1)
        print(f"  warmup {w+1}: {lat:.0f} ms")
    print()

    l_mean = []
    l_std  = []

    for b in b_sweep:
        print(f"[stage2] B={b}  ({n_reps} reps)...")
        rep_means = []
        for r in range(n_reps):
            lats = fire_concurrent(url, model, b, ttft, 1)
            valid = [x for x in lats if not np.isnan(x)]
            m     = statistics.mean(valid) if valid else float("nan")
            rep_means.append(m)
            print(f"  rep {r+1}: {m:.1f} ms  {[round(x) for x in lats]}")
        mu = statistics.mean(rep_means)
        sd = statistics.stdev(rep_means) if len(rep_means) > 1 else 0.0
        l_mean.append(mu)
        l_std.append(sd)
        print(f"  --> B={b}: {mu:.1f} ± {sd:.1f} ms\n")

    # Least-squares fit: l = alpha*B + gamma*B^2  (no intercept)
    B   = np.array(b_sweep, dtype=float)
    L   = np.array(l_mean,  dtype=float)
    A   = np.column_stack([B, B**2])
    p, _, _, _ = np.linalg.lstsq(A, L, rcond=None)
    alpha, gamma = float(p[0]), float(p[1])
    L_fit = A @ p
    ss_res = np.sum((L - L_fit)**2)
    ss_tot = np.sum((L - np.mean(L))**2)
    r2    = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    print(f"[stage2] alpha = {alpha:.4f} ms/req")
    print(f"[stage2] gamma = {gamma:.4f} ms/req^2")
    print(f"[stage2] R^2   = {r2:.4f}\n")

    # Plot
    fig, ax = plt.subplots()
    ax.errorbar(b_sweep, l_mean, l_std, fmt="bo", capsize=4,
                label="Measured", linewidth=1.2)
    b_fine = np.linspace(0.5, max(b_sweep) + 0.5, 300)
    ax.plot(b_fine, alpha*b_fine + gamma*b_fine**2, "r-", linewidth=2,
            label=f"αB + γB²  (R²={r2:.3f})")
    ax.set_xlabel("Concurrent batch B [req]")
    ax.set_ylabel("TTFT [ms]")
    ax.set_title(f"Stage 2 — B sweep (q=0)\nα={alpha:.3f}  γ={gamma:.4f}")
    ax.legend(); ax.grid(True)
    path = out_dir / "ch5_stage2_b_sweep.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[stage2] Plot saved: {path}\n")

    return {"alpha": alpha, "gamma": gamma, "r2": r2,
            "b_sweep": b_sweep, "l_mean": l_mean, "l_std": l_std}


# ---------------------------------------------------------------------------
# Stage 3: beta estimation via sustained load
# ---------------------------------------------------------------------------
def stage3_beta(url: str, model: str, alpha: float, gamma: float,
                rate: int, max_inflight: int, n_ticks: int,
                n_meas: int, dt: float, out_dir: Path) -> dict:
    """
    Fire requests at rate > max_num_seqs using a semaphore-backed thread pool.
    Track inflight count ourselves (gauge workaround).
    Measure per-tick mean TTFT using histogram deltas from /metrics.
    """
    print("\n" + "═" * 63)
    print(f"STAGE 3: Beta identification  [rate={rate} req/s, dt={dt}s]")
    print("═" * 63 + "\n")
    print("Note: vllm-metal num_requests_waiting gauge accumulates (known bug).")
    print("      Using software inflight counter as queue proxy.\n")

    sem      = threading.Semaphore(max_inflight)
    lock     = threading.Lock()
    inflight = [0]
    done     = [0]

    def fire_one():
        sem.acquire()
        with lock:
            inflight[0] += 1
        try:
            e2e(url, model, PROMPTS[done[0] % len(PROMPTS)], 20)
        except Exception:
            pass
        finally:
            with lock:
                inflight[0] -= 1
                done[0]     += 1
            sem.release()

    # Get baseline histogram counters
    def get_ttft_hist(metrics: dict):
        s = metrics.get("vllm:time_to_first_token_seconds_sum",   0.0)
        c = metrics.get("vllm:time_to_first_token_seconds_count", 0.0)
        return s, c

    stop = threading.Event()

    def load_thread():
        interval = 1.0 / rate
        while not stop.is_set():
            threading.Thread(target=fire_one, daemon=True).start()
            time.sleep(interval)

    loader = threading.Thread(target=load_thread, daemon=True)
    loader.start()
    print(f"[stage3] Load thread started at {rate} req/s...")
    time.sleep(2)   # ramp up

    q_log = []
    l_log = []
    prev_s, prev_c = get_ttft_hist(get_metrics(url))

    for tick in range(n_ticks + n_meas):
        phase = "build" if tick < n_ticks else "MEAS "
        t_tick = time.perf_counter()

        # Software queue estimate
        with lock:
            q_k = max(0, inflight[0] - 4)   # 4 = max_num_seqs

        # Per-tick TTFT from histogram delta
        m         = get_metrics(url)
        curr_s, curr_c = get_ttft_hist(m)
        d_s = curr_s - prev_s
        d_c = curr_c - prev_c
        l_k = (d_s / d_c) * 1000.0 if d_c > 0 else float("nan")
        prev_s, prev_c = curr_s, curr_c

        q_log.append(q_k)
        l_log.append(l_k)
        print(f"  tick {tick+1:3d} [{phase}]  q={q_k:5.1f}  l={l_k:6.0f} ms")

        elapsed = time.perf_counter() - t_tick
        if elapsed < dt:
            time.sleep(dt - elapsed)

    stop.set()
    loader.join(timeout=5)

    # Fit beta from measurement window
    q_meas = [q_log[n_ticks + i] for i in range(n_meas)]
    l_meas = [l_log[n_ticks + i] for i in range(n_meas) if not np.isnan(l_log[n_ticks + i])]

    q_ss = statistics.mean(q_meas) if q_meas else 0.0
    l_ss = statistics.mean(l_meas) if l_meas else float("nan")
    b_ss = min(rate, 4)   # max_num_seqs caps running concurrency
    svc  = alpha * b_ss + gamma * b_ss**2

    print(f"\n[stage3] q_ss={q_ss:.2f}  l_ss={l_ss:.0f} ms  b_ss={b_ss}  svc={svc:.0f} ms")

    if q_ss > 0.5 and not np.isnan(l_ss):
        beta = (l_ss - svc) / q_ss
        print(f"[stage3] beta = {beta:.4f} ms/req\n")
    else:
        beta = float("nan")
        print("[stage3] WARNING: q_ss too small -- beta not identifiable.\n"
              "         Will use analytical estimate in design_controller.py\n")

    # Plot
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ticks = range(1, n_ticks + n_meas + 1)
    ax1.plot(ticks, q_log, "b-o", markersize=4, label="q (software counter)")
    ax2.plot(ticks, [l if not np.isnan(l) else None for l in l_log],
             "r-s", markersize=4, label="TTFT [ms]")
    ax1.axvline(n_ticks + 0.5, color="k", linestyle="--", label="Measurement window")
    ax1.set_xlabel("Tick"); ax1.set_ylabel("q [req]", color="b")
    ax2.set_ylabel("TTFT [ms]", color="r")
    ax1.set_title("Stage 3 — Queue buildup")
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
    ax1.grid(True)
    path = out_dir / "ch5_stage3_queue.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[stage3] Plot saved: {path}\n")

    return {"beta": beta, "q_ss": q_ss, "l_ss": l_ss}


# ---------------------------------------------------------------------------
# Stage 4: Operating envelope
# ---------------------------------------------------------------------------
def stage4_envelope(url: str, model: str, lambda_sweep: list[int],
                    settle: int, meas: int, dt: float, out_dir: Path) -> dict:
    print("\n" + "═" * 63)
    print("STAGE 4: Operating envelope")
    print("═" * 63 + "\n")

    env_lambda, env_q, env_l, env_wall = [], [], [], []

    for lam in lambda_sweep:
        b_k = min(lam, 8)
        print(f"[stage4] lambda={lam}  b={b_k}  ({settle} settle + {meas} meas ticks)...")

        sem  = threading.Semaphore(b_k + 2)
        lock = threading.Lock()
        inf  = [0]; done_cnt = [0]
        stop = threading.Event()

        def fire_one_s4(sem=sem, lock=lock, inf=inf, done_cnt=done_cnt):
            sem.acquire()
            with lock: inf[0] += 1
            try:
                e2e(url, model, PROMPTS[done_cnt[0] % len(PROMPTS)], 20)
            except Exception:
                pass
            finally:
                with lock: inf[0] -= 1; done_cnt[0] += 1
                sem.release()

        def load_s4(stop=stop, lam=lam):
            interval = 1.0 / lam
            while not stop.is_set():
                threading.Thread(target=fire_one_s4, daemon=True).start()
                time.sleep(interval)

        loader = threading.Thread(target=load_s4, daemon=True)
        loader.start()
        time.sleep(1.5)   # ramp up

        q_all, l_all, w_all = [], [], []
        prev_s, prev_c = 0.0, 0.0
        m0 = get_metrics(url)
        prev_s = m0.get("vllm:time_to_first_token_seconds_sum", 0.0)
        prev_c = m0.get("vllm:time_to_first_token_seconds_count", 0.0)

        for tick in range(settle + meas):
            phase = "settle" if tick < settle else "MEAS  "
            t0    = time.perf_counter()

            with lock:
                q_k = max(0, inf[0] - 4)
            m_now = get_metrics(url)
            cs = m_now.get("vllm:time_to_first_token_seconds_sum", 0.0)
            cc = m_now.get("vllm:time_to_first_token_seconds_count", 0.0)
            l_k = ((cs - prev_s) / (cc - prev_c)) * 1000 if (cc - prev_c) > 0 else float("nan")
            prev_s, prev_c = cs, cc

            elapsed = time.perf_counter() - t0
            wall    = elapsed
            if elapsed < dt:
                time.sleep(dt - elapsed)
            wall = time.perf_counter() - t0

            q_all.append(q_k); l_all.append(l_k); w_all.append(wall * 1000)
            print(f"  tick {tick+1:2d} [{phase}]  q={q_k:5.1f}  l={l_k:6.0f} ms  wall={wall*1000:.0f} ms")

        stop.set()

        meas_q = [q_all[settle + i] for i in range(meas)]
        meas_l = [l_all[settle + i] for i in range(meas) if not np.isnan(l_all[settle + i])]
        q_ss   = statistics.mean(meas_q) if meas_q else 0.0
        l_ss   = statistics.mean(meas_l) if meas_l else float("nan")

        env_lambda.append(lam); env_q.append(q_ss); env_l.append(l_ss)
        print(f"  --> q_ss={q_ss:.2f}  l_ss={l_ss:.0f} ms\n")

        # Drain
        print("  Draining queue (5s)...")
        time.sleep(5)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(env_lambda, env_l, "b-o", linewidth=1.5, markersize=6)
    ax1.set_xlabel("λ [req/s]"); ax1.set_ylabel("l_ss [ms]")
    ax1.set_title("Latency vs arrival rate"); ax1.grid(True)
    ax2.plot(env_lambda, env_q, "r-o", linewidth=1.5, markersize=6)
    ax2.set_xlabel("λ [req/s]"); ax2.set_ylabel("q_ss [req]")
    ax2.set_title("Queue depth vs arrival rate"); ax2.grid(True)
    fig.tight_layout()
    path = out_dir / "ch5_stage4_envelope.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[stage4] Plot saved: {path}\n")

    return {"lambda": env_lambda, "q_ss": env_q, "l_ss": env_l}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Chapter 5 vLLM plant identification")
    p.add_argument("--url",   default=DEFAULT_URL,   help="vLLM base URL")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Model name")
    p.add_argument("--out",   default=str(DEFAULT_OUT), help="Output JSON path")
    p.add_argument("--skip_stage3", action="store_true", help="Skip beta identification")
    p.add_argument("--skip_stage4", action="store_true", help="Skip envelope sweep")
    args = p.parse_args()

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: smoke test
    print("═" * 63)
    print("STAGE 1: Smoke test")
    print("═" * 63 + "\n")
    try:
        r = requests.get(f"{args.url}/health", timeout=5)
        r.raise_for_status()
        print(f"[stage1] Health: OK")
    except Exception as e:
        sys.exit(f"[stage1] FAIL: {e}\nRun: ./start_vllm.sh --bg")

    m = get_metrics(args.url)
    print(f"[stage1] num_requests_running = {m.get('vllm:num_requests_running', 'N/A')}")
    print(f"[stage1] num_requests_waiting = {m.get('vllm:num_requests_waiting', 'N/A')}")
    print(f"[stage1] model in use: {args.model}\n")

    # Stage 2: B sweep
    s2 = stage2_b_sweep(
        url       = args.url,
        model     = args.model,
        b_sweep   = [1, 2, 3, 4, 5, 6, 8],
        n_reps    = 5,
        out_dir   = out_dir,
    )
    alpha = s2["alpha"]; gamma = s2["gamma"]

    # Stage 3: beta
    if not args.skip_stage3:
        s3 = stage3_beta(
            url         = args.url,
            model       = args.model,
            alpha       = alpha,
            gamma       = gamma,
            rate        = 12,         # req/s, > max_num_seqs
            max_inflight= 16,
            n_ticks     = 12,
            n_meas      = 8,
            dt          = 1.0,
            out_dir     = out_dir,
        )
        beta = s3["beta"]
    else:
        s3 = {}; beta = float("nan")

    # Stage 4: envelope
    if not args.skip_stage4:
        s4 = stage4_envelope(
            url           = args.url,
            model         = args.model,
            lambda_sweep  = [1, 2, 3, 4, 6],
            settle        = 10,
            meas          = 6,
            dt            = 1.0,
            out_dir       = out_dir,
        )
    else:
        s4 = {}

    # Summary
    print("╔" + "═"*62 + "╗")
    print(f"║  IDENTIFIED PARAMETERS  ({args.model})")
    print("╠" + "═"*62 + "╣")
    print(f"║  alpha = {alpha:8.4f}  ms/req      R² = {s2['r2']:.4f}")
    print(f"║  gamma = {gamma:8.4f}  ms/req²")
    beta_str = f"{beta:.4f}" if not np.isnan(beta) else "NaN  (use analytical estimate)"
    print(f"║  beta  = {beta_str}")
    print("╚" + "═"*62 + "╝\n")

    # Save
    result = {
        "model":     args.model,
        "timestamp": datetime.now().isoformat(),
        "alpha":     alpha,
        "gamma":     gamma,
        "beta":      beta if not np.isnan(beta) else None,
        "r2_stage2": s2["r2"],
        "stage2":    s2,
        "stage3":    s3,
        "stage4":    s4,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
