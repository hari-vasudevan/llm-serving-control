#!/usr/bin/env python3
"""
characterise_remote.py  --  Chapter 7 native-vLLM characterization

This chapter now uses vLLM's own scheduler queue rather than an extra software
wrapper queue. The control input is client-side concurrency C, and the measured
output is user-facing first-token latency:

    l_total = t_first_token - t_enqueue

The script fires bursts of size C directly at the deployed vLLM endpoint and
fits a local latency-vs-concurrency model around an operating point C0.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests

from vllm_native import (
    fire_burst,
    get_metrics,
    jitter_prompt_offset,
    metric_delta_mean_ms,
    percentile,
    wait_for_health,
)


DEFAULT_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8001")
DEFAULT_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
DEFAULT_OUT = Path(__file__).parent / "identified_params.json"


def robust_slope(c_values, l_values, c0):
    pairs = [(c, l) for c, l in zip(c_values, l_values) if math.isfinite(l)]
    pairs = sorted(pairs)
    if len(pairs) < 2:
        return 1.0, "fallback"

    lower = max((p for p in pairs if p[0] <= c0), key=lambda x: x[0], default=None)
    upper = min((p for p in pairs if p[0] >= c0), key=lambda x: x[0], default=None)
    if lower and upper and lower[0] != upper[0]:
        slope = (upper[1] - lower[1]) / (upper[0] - lower[0])
        if slope > 0:
            return float(slope), "local_secant"

    low, high = pairs[0], pairs[-1]
    if high[0] != low[0]:
        slope = (high[1] - low[1]) / (high[0] - low[0])
        if slope > 0:
            return float(slope), "global_secant"

    return 1.0, "fallback"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", ""))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--c-sweep", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    ap.add_argument("--C0", type=int, default=3)
    ap.add_argument("--n-reps", type=int, default=6)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--prompt-repeat", type=int, default=64)
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("═" * 72)
    print("STAGE 1: Smoke test")
    print("═" * 72)
    if not wait_for_health(args.url, timeout=180):
        sys.exit(f"Health check failed for {args.url}")

    try:
        resp = requests.get(f"{args.url}/health", timeout=10)
        print(f"[stage1] Health: {resp.status_code} {resp.text.strip()}")
    except Exception as exc:
        sys.exit(f"[stage1] Health request failed: {exc}")

    metrics = get_metrics(args.url)
    print(f"[stage1] waiting={metrics.get('vllm:num_requests_waiting', 0.0)}")
    print(f"[stage1] running={metrics.get('vllm:num_requests_running', 0.0)}")
    print(f"[stage1] queue_time_hist_count={metrics.get('vllm:request_queue_time_seconds_count', 0.0)}")
    print(f"[stage1] ttft_hist_count={metrics.get('vllm:time_to_first_token_seconds_count', 0.0)}")

    print("\n" + "═" * 72)
    print("STAGE 2: Concurrency sweep on native vLLM queue")
    print("═" * 72)
    print(f"c_sweep={args.c_sweep}  n_reps={args.n_reps}  C0={args.C0}")
    print(f"prompt_repeat={args.prompt_repeat}  max_tokens={args.max_tokens}\n")

    print("[warmup] 4 serial requests...")
    for i in range(4):
        lats = fire_burst(
            args.url,
            args.model,
            1,
            prompt_repeat=args.prompt_repeat,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            api_key=args.api_key,
            seed_offset=1000 + i,
        )
        print(f"  warmup {i+1}: {lats[0]:.1f} ms")
    print()

    l_mean = []
    l_std = []
    l_p95 = []
    ttft_server_mean = []
    queue_server_mean = []
    e2e_server_mean = []
    raw_reps = {}

    for c in args.c_sweep:
        rep_client = []
        rep_ttft = []
        rep_queue = []
        rep_e2e = []
        print(f"[C={c}] {args.n_reps} reps (rep 1 discarded for fitting)...")
        for rep in range(args.n_reps):
            before = get_metrics(args.url)
            lats = fire_burst(
                args.url,
                args.model,
                c,
                prompt_repeat=args.prompt_repeat,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                api_key=args.api_key,
                seed_offset=jitter_prompt_offset(),
            )
            after = get_metrics(args.url)
            valid = [x for x in lats if math.isfinite(x)]
            mean_client = statistics.mean(valid) if valid else float("nan")
            mean_ttft = metric_delta_mean_ms(before, after, "vllm:time_to_first_token_seconds")
            mean_queue = metric_delta_mean_ms(before, after, "vllm:request_queue_time_seconds")
            mean_e2e = metric_delta_mean_ms(before, after, "vllm:e2e_request_latency_seconds")
            rep_client.append(mean_client)
            rep_ttft.append(mean_ttft if mean_ttft is not None else float("nan"))
            rep_queue.append(mean_queue if mean_queue is not None else float("nan"))
            rep_e2e.append(mean_e2e if mean_e2e is not None else float("nan"))
            tag = "[DISCARD]" if rep == 0 else ""
            print(
                f"  rep {rep+1}: client={mean_client:7.1f}ms  "
                f"server_ttft={mean_ttft if mean_ttft is not None else float('nan'):7.1f}ms  "
                f"server_queue={mean_queue if mean_queue is not None else float('nan'):7.1f}ms  {tag}"
            )

        raw_reps[str(c)] = {
            "client_mean_ms": rep_client,
            "server_ttft_ms": rep_ttft,
            "server_queue_ms": rep_queue,
            "server_e2e_ms": rep_e2e,
        }

        used_client = [x for x in rep_client[1:] if math.isfinite(x)]
        used_ttft = [x for x in rep_ttft[1:] if math.isfinite(x)]
        used_queue = [x for x in rep_queue[1:] if math.isfinite(x)]
        used_e2e = [x for x in rep_e2e[1:] if math.isfinite(x)]

        l_mean.append(statistics.mean(used_client) if used_client else float("nan"))
        l_std.append(statistics.stdev(used_client) if len(used_client) > 1 else 0.0)
        l_p95.append(percentile(used_client, 0.95) if used_client else float("nan"))
        ttft_server_mean.append(statistics.mean(used_ttft) if used_ttft else float("nan"))
        queue_server_mean.append(statistics.mean(used_queue) if used_queue else float("nan"))
        e2e_server_mean.append(statistics.mean(used_e2e) if used_e2e else float("nan"))
        print(
            f"  -> C={c}: client_mean={l_mean[-1]:.1f}ms  "
            f"p95={l_p95[-1]:.1f}ms  queue_mean={queue_server_mean[-1]:.1f}ms\n"
        )

    c_arr = np.array(args.c_sweep, dtype=float)
    l_arr = np.array(l_mean, dtype=float)
    valid = np.isfinite(l_arr)
    coeffs = np.polyfit(c_arr[valid], l_arr[valid], deg=2) if np.count_nonzero(valid) >= 3 else np.array([0.0, 1.0, l_arr[valid][0]])
    fit_vals = np.polyval(coeffs, c_arr[valid])
    denom = np.sum((l_arr[valid] - np.mean(l_arr[valid])) ** 2)
    r2 = float(1.0 - np.sum((l_arr[valid] - fit_vals) ** 2) / denom) if denom > 0 else 1.0

    poly_slope = float(coeffs[1] + 2 * coeffs[0] * args.C0)
    slope_secant, slope_method = robust_slope(args.c_sweep, l_mean, args.C0)
    beta_c = poly_slope if poly_slope > 0 else slope_secant
    if beta_c <= 0:
        beta_c = 1.0
        slope_method = "fallback"
    l0 = float(np.polyval(coeffs, args.C0))
    if not math.isfinite(l0):
        nearest_idx = int(np.argmin(np.abs(c_arr - args.C0)))
        l0 = float(l_arr[nearest_idx])

    print("RESULTS:")
    print(f"  fit: L(C) = {coeffs[0]:.4f}*C^2 + {coeffs[1]:.4f}*C + {coeffs[2]:.4f}")
    print(f"  R^2 = {r2:.4f}")
    print(f"  operating point: C0={args.C0}  L(C0)≈{l0:.2f} ms")
    print(f"  beta_c = dL/dC|C0 ≈ {beta_c:.4f} ms per concurrency  [{slope_method}]")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.errorbar(args.c_sweep, l_mean, l_std, fmt="bo", capsize=4, label="Client l_total mean ± std")
    c_fine = np.linspace(min(args.c_sweep), max(args.c_sweep), 200)
    ax.plot(c_fine, np.polyval(coeffs, c_fine), "r-", lw=2, label=f"quadratic fit  R²={r2:.3f}")
    ax.axvline(args.C0, color="g", ls="--", lw=1.2, label=f"C0={args.C0}")
    ax.set_xlabel("Client concurrency C")
    ax.set_ylabel("First-token latency [ms]")
    ax.set_title("Client-observed latency vs concurrency")
    ax.grid(True)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(args.c_sweep, ttft_server_mean, "m-o", label="vLLM TTFT mean [ms]")
    ax.plot(args.c_sweep, queue_server_mean, "orange", marker="s", label="vLLM queue mean [ms]")
    ax.plot(args.c_sweep, e2e_server_mean, "k--^", label="vLLM e2e mean [ms]")
    ax.set_xlabel("Client concurrency C")
    ax.set_ylabel("Server-side metric [ms]")
    ax.set_title("vLLM metrics during concurrency sweep")
    ax.grid(True)
    ax.legend(fontsize=8)

    fig.suptitle("Chapter 7 — Native vLLM characterization", fontsize=12, fontweight="bold")
    fig.tight_layout()
    plot_path = out_path.parent / "ch7_native_concurrency_sweep.png"
    fig.savefig(plot_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {plot_path}")

    result = {
        "timestamp": datetime.now().isoformat(),
        "url": args.url,
        "model": args.model,
        "plant_model": "L(C) = client first-token latency vs client concurrency on native vLLM",
        "latency_definition": "l_total = t_first_token - t_enqueue",
        "c_sweep": args.c_sweep,
        "C0": args.C0,
        "dt": args.dt,
        "max_tokens": args.max_tokens,
        "prompt_repeat": args.prompt_repeat,
        "l_client_mean_ms": l_mean,
        "l_client_std_ms": l_std,
        "l_client_p95_ms": l_p95,
        "ttft_server_mean_ms": ttft_server_mean,
        "queue_server_mean_ms": queue_server_mean,
        "e2e_server_mean_ms": e2e_server_mean,
        "poly_coeffs": coeffs.tolist(),
        "r2": r2,
        "L0_ms": l0,
        "beta_c": beta_c,
        "beta_c_method": slope_method if poly_slope <= 0 else "poly_derivative",
        "raw_reps": raw_reps,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[save] {out_path}")


if __name__ == "__main__":
    main()
