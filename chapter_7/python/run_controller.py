#!/usr/bin/env python3
"""run_controller.py  --  Chapter 7 native-vLLM single-loop controller."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import threading
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests

from vllm_native import (
    BackgroundLoad,
    ResultsSink,
    fire_burst,
    get_metrics,
    make_prompt,
    percentile,
    wait_for_health,
)


DEFAULT_PARAMS = Path(__file__).parent / "controller_params.json"
DEFAULT_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8001")
DEFAULT_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")


PROMPT_OFFSET = 20_000
SCHEDULE = [
    {"ticks": 8, "bg_rps": 0.0, "L_target": None, "label": "Steady"},
    {"ticks": 10, "bg_rps": 0.8, "L_target": None, "label": "Background load"},
    {"ticks": 8, "bg_rps": 1.6, "L_target": None, "label": "Load spike"},
    {"ticks": 8, "bg_rps": 0.4, "L_target": None, "label": "Recovery"},
    {"ticks": 6, "bg_rps": 0.4, "L_target": "tight", "label": "Target tighten"},
    {"ticks": 8, "bg_rps": 0.0, "L_target": None, "label": "Cool-down"},
]


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def launch_controlled_burst(url, model, concurrency, sink, *, prompt_repeat, max_tokens, api_key, timeout, seed_base):
    def worker(i):
        try:
            lats = fire_burst(
                url,
                model,
                1,
                prompt_repeat=prompt_repeat,
                max_tokens=max_tokens,
                timeout=timeout,
                api_key=api_key,
                seed_offset=seed_base + i,
            )
            if lats and math.isfinite(lats[0]):
                sink.append(lats[0])
        except Exception:
            pass

    for i in range(concurrency):
        threading.Thread(target=worker, args=(i,), daemon=True).start()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default=str(DEFAULT_PARAMS))
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", ""))
    ap.add_argument("--dt", type=float, default=None)
    ap.add_argument("--out-dir", default=str(Path(__file__).parent))
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--background-scale", type=float, default=1.0)
    args = ap.parse_args()

    with open(args.params) as f:
        ctrl = json.load(f)

    if not wait_for_health(args.url, timeout=180):
        raise SystemExit(f"health check failed for {args.url}")
    requests.get(f"{args.url}/health", timeout=10).raise_for_status()

    dt = args.dt if args.dt is not None else float(ctrl["dt"])
    L_nom = float(ctrl["L_target_ms"])
    L_tight = round(0.85 * L_nom, 1)
    schedule = []
    for segment in SCHEDULE:
        copied = dict(segment)
        copied["bg_rps"] = copied["bg_rps"] * args.background_scale
        copied["L_target"] = L_tight if copied["L_target"] == "tight" else L_nom
        schedule.append(copied)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    background = BackgroundLoad(
        args.url,
        args.model,
        api_key=args.api_key,
        prompt_repeat=ctrl["prompt_repeat"],
        max_tokens=ctrl["max_tokens"],
    )
    background.start()

    sink = ResultsSink()
    xi = 0.0
    tick = 0
    seed = PROMPT_OFFSET

    log = {k: [] for k in [
        "tick", "bg_rps", "L_target", "L_meas", "C_cmd", "q_waiting",
        "q_running", "queue_mean_ms", "ttft_mean_ms", "e", "xi", "label"
    ]}

    print(f"{'tick':>4} {'bg':>4} {'L_tgt':>7} {'L_meas':>8} {'C':>3} {'q_w':>5} {'q_r':>5} {'queue_ms':>9} {'ttft_ms':>8}  label", flush=True)
    print("-" * 92, flush=True)

    try:
        for segment in schedule:
            background.set_rate(segment["bg_rps"])
            for _ in range(segment["ticks"]):
                tick += 1
                t0 = time.perf_counter()

                completed = [x for x in sink.drain() if math.isfinite(x)]
                L_meas = statistics.mean(completed) if completed else float(segment["L_target"])

                metrics = get_metrics(args.url)
                q_waiting = metrics.get("vllm:num_requests_waiting", 0.0)
                q_running = metrics.get("vllm:num_requests_running", 0.0)

                queue_count = metrics.get("vllm:request_queue_time_seconds_count", 0.0)
                queue_sum = metrics.get("vllm:request_queue_time_seconds_sum", 0.0)
                ttft_count = metrics.get("vllm:time_to_first_token_seconds_count", 0.0)
                ttft_sum = metrics.get("vllm:time_to_first_token_seconds_sum", 0.0)
                queue_mean_ms = 1000.0 * queue_sum / queue_count if queue_count > 0 else float("nan")
                ttft_mean_ms = 1000.0 * ttft_sum / ttft_count if ttft_count > 0 else float("nan")

                e = float(segment["L_target"]) - L_meas
                xi_sat = clamp(xi, ctrl["xi_min"], ctrl["xi_max"])
                C_cmd = int(round(clamp(ctrl["C0"] + ctrl["K_i"] * xi_sat, ctrl["C_min"], ctrl["C_max"])))

                at_min = (C_cmd <= ctrl["C_min"]) and (e < 0)
                at_max = (C_cmd >= ctrl["C_max"]) and (e > 0)
                if not (at_min or at_max):
                    xi = clamp(xi_sat + e, ctrl["xi_min"], ctrl["xi_max"])
                else:
                    xi = xi_sat

                burst_thread = threading.Thread(
                    target=launch_controlled_burst,
                    args=(args.url, args.model, C_cmd, sink),
                    kwargs={
                        "prompt_repeat": ctrl["prompt_repeat"],
                        "max_tokens": ctrl["max_tokens"],
                        "api_key": args.api_key,
                        "timeout": args.timeout,
                        "seed_base": seed,
                    },
                    daemon=True,
                )
                seed += C_cmd
                burst_thread.start()

                elapsed = time.perf_counter() - t0
                if elapsed < dt:
                    time.sleep(dt - elapsed)

                print(
                    f"{tick:4d} {segment['bg_rps']:4.1f} {segment['L_target']:7.1f} {L_meas:8.1f} "
                    f"{C_cmd:3d} {q_waiting:5.1f} {q_running:5.1f} {queue_mean_ms:9.1f} {ttft_mean_ms:8.1f}  {segment['label']}"
                , flush=True)

                for key, value in [
                    ("tick", tick),
                    ("bg_rps", segment["bg_rps"]),
                    ("L_target", segment["L_target"]),
                    ("L_meas", L_meas),
                    ("C_cmd", C_cmd),
                    ("q_waiting", q_waiting),
                    ("q_running", q_running),
                    ("queue_mean_ms", queue_mean_ms),
                    ("ttft_mean_ms", ttft_mean_ms),
                    ("e", e),
                    ("xi", xi),
                    ("label", segment["label"]),
                ]:
                    log[key].append(value)
    finally:
        background.stop()

    valid = [x for x in log["L_meas"] if math.isfinite(x)]
    print("\n=== Summary ===", flush=True)
    print(f"  L_mean = {statistics.mean(valid):.1f} ms", flush=True)
    print(f"  L_p95  = {percentile(valid, 0.95):.1f} ms", flush=True)
    print(f"  C_mean = {statistics.mean(log['C_cmd']):.2f}", flush=True)
    print(f"  q_wait_mean = {statistics.mean(log['q_waiting']):.2f}", flush=True)
    print(f"  background sent={background.sent} done={background.done} errors={background.errors}", flush=True)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    ticks = log["tick"]
    axes[0].plot(ticks, log["L_meas"], "b-", lw=1.4, label="client latency")
    axes[0].plot(ticks, log["L_target"], "k--", lw=1.2, label="target")
    axes[0].set_ylabel("Latency [ms]")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.4)

    axes[1].step(ticks, log["C_cmd"], "m-", where="post", label="C_cmd")
    axes[1].step(ticks, log["bg_rps"], "g--", where="post", label="background rps")
    axes[1].set_ylabel("Load / concurrency")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.4)

    axes[2].plot(ticks, log["q_waiting"], color="darkorange", label="vllm waiting")
    axes[2].plot(ticks, log["q_running"], color="teal", label="vllm running")
    axes[2].set_ylabel("vLLM queue")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.4)

    axes[3].plot(ticks, log["xi"], "r-", lw=1.3, label="integrator")
    axes[3].axhline(0, color="k", lw=0.8, ls="--")
    axes[3].set_ylabel("xi")
    axes[3].set_xlabel("Tick")
    axes[3].legend(fontsize=8)
    axes[3].grid(True, alpha=0.4)

    fig.suptitle("Chapter 7 — Single-loop latency control on native vLLM", fontsize=12, fontweight="bold")
    fig.tight_layout()
    ts = datetime.now().strftime("%H%M%S")
    plot_path = out_dir / f"ch7_single_loop_{ts}.png"
    log_path = out_dir / f"ch7_single_loop_log_{ts}.json"
    fig.savefig(plot_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    with open(log_path, "w") as f:
        json.dump({"controller": ctrl, "schedule": schedule, "log": log}, f, indent=2)
    print(f"[plot] {plot_path}", flush=True)
    print(f"[log]  {log_path}", flush=True)


if __name__ == "__main__":
    main()
