#!/usr/bin/env python3
"""
Chapter 11 Phase 2 closed-loop TTFT sweep runner.

Posts to /run_internal_ttft_sweep on the deployed Modal wrapper, which runs
the TTFT PI controller inside the GPU container and returns per-target
summaries plus time-series controller state.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_ttft_sweep import plot_result


DEFAULT_TARGETS = [150.0, 200.0, 300.0]


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data: str) -> None:
        for f in self.files:
            f.write(data)
            f.flush()

    def flush(self) -> None:
        for f in self.files:
            f.flush()


def main() -> None:
    ap = argparse.ArgumentParser(description="Chapter 11 Phase 2 TTFT sweep runner")
    ap.add_argument("--url", required=True, help="Base URL for the Chapter 11 Modal wrapper")
    ap.add_argument("--target-ttft-ms", type=float, nargs="+", default=DEFAULT_TARGETS)
    ap.add_argument("--offered-rate-qps", type=float, default=4.0)
    ap.add_argument("--duration-s", type=float, default=60.0)
    ap.add_argument("--warmup-s", type=float, default=10.0)
    ap.add_argument("--settle-s", type=float, default=3.0)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--prompt-repeat", type=int, default=64)
    ap.add_argument("--feedback-period-s", type=float, default=0.5)
    ap.add_argument("--ttft-window", type=int, default=20)
    ap.add_argument("--kp", type=float, default=0.15)
    ap.add_argument("--ki", type=float, default=0.02)
    ap.add_argument("--fraction-min", type=float, default=0.25)
    ap.add_argument("--fraction-max", type=float, default=1.0)
    ap.add_argument("--max-outstanding", type=int, default=256)
    ap.add_argument("--metric-period-s", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--timeout-s", type=float, default=7200.0)
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "results"))
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / f"ttft_sweep_{stamp}"
    logs_dir = out_dir / "logs"
    plots_dir = out_dir / "plots"
    logs_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    with (logs_dir / "run_ttft_sweep.log").open("w") as log_f:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = Tee(old_stdout, log_f)
        sys.stderr = Tee(old_stderr, log_f)
        try:
            run(args, out_dir, plots_dir)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def run(args: argparse.Namespace, out_dir: Path, plots_dir: Path) -> None:
    print(f"[start] Chapter 11 Phase 2 TTFT sweep {datetime.now().isoformat()}", flush=True)
    print(f"[out] {out_dir}", flush=True)
    wait_for_health(args.url)

    payload = {
        "target_ttft_ms": args.target_ttft_ms,
        "offered_rate_qps": args.offered_rate_qps,
        "duration_s": args.duration_s,
        "warmup_s": args.warmup_s,
        "settle_s": args.settle_s,
        "max_tokens": args.max_tokens,
        "prompt_repeat": args.prompt_repeat,
        "feedback_period_s": args.feedback_period_s,
        "ttft_window": args.ttft_window,
        "kp": args.kp,
        "ki": args.ki,
        "fraction_min": args.fraction_min,
        "fraction_max": args.fraction_max,
        "max_outstanding": args.max_outstanding,
        "metric_period_s": args.metric_period_s,
        "seed": args.seed,
    }
    write_json(out_dir / "sweep_request.json", payload)
    print("[request]", json.dumps(payload, indent=2), flush=True)

    endpoint = f"{args.url.rstrip('/')}/run_internal_ttft_sweep"
    t0 = time.perf_counter()
    print(f"[post] {endpoint}", flush=True)
    status, text = post_json(endpoint, payload, timeout_s=args.timeout_s)
    elapsed = time.perf_counter() - t0
    print(f"[response] status={status} elapsed_s={elapsed:.1f}", flush=True)
    if status < 200 or status >= 300:
        raise RuntimeError(f"POST {endpoint} failed with HTTP {status}: {text[:2000]}")

    result = json.loads(text)
    result["client_elapsed_s"] = elapsed
    result["client_timestamp"] = datetime.now().isoformat()
    write_json(out_dir / "sweep_response.json", result)

    results = result.get("results", [])

    # Plot and save per-target data while timeseries is still in memory.
    all_plot_paths: list[Path] = []
    summaries = []
    for res in results:
        target_ms = int(res.get("target_ttft_ms", 0))
        ts = res.get("timeseries", [])
        sample_errors = res.get("sample_errors", [])
        if sample_errors:
            print(f"[errors target={int(res.get('target_ttft_ms', 0))}ms] {sample_errors}", flush=True)
        summary = {k: v for k, v in res.items() if k != "timeseries"}

        paths = plot_result(ts, summary, plots_dir)
        all_plot_paths.extend(paths)

        ts_dir = out_dir / f"target_{target_ms}ms"
        ts_dir.mkdir(exist_ok=True)
        write_json(ts_dir / "timeseries.json", ts)
        summaries.append(summary)

    write_json(out_dir / "sweep_summary.json", summaries)
    write_summary_csv(out_dir / "sweep_summary.csv", summaries)
    print("[summary]", json.dumps(summaries, indent=2), flush=True)

    write_json(out_dir / "plot_manifest.json", [str(p) for p in all_plot_paths])
    for path in all_plot_paths:
        print(f"[plot] {path}", flush=True)
    print(f"[done] wrote {out_dir}", flush=True)


def wait_for_health(url: str, timeout_s: float = 900.0) -> None:
    """Wait for the Modal endpoint to become healthy.

    Modal holds HTTP connections open during container cold-start and proxies
    them once the container is ready (~4 min for Qwen2.5-3B).  A per-request
    timeout shorter than the cold-start window causes the client to drop the
    connection before Modal can proxy it; the container never appears healthy.
    We use per-request timeout up to 300 s so one queued request survives
    the full cold-start, then wait 10 s between retries.
    """
    deadline = time.time() + timeout_s
    last_error = ""
    while time.time() < deadline:
        remaining = deadline - time.time()
        req_timeout = min(remaining, 300.0)
        try:
            status, text = get_text(f"{url.rstrip('/')}/health", timeout_s=req_timeout)
            if 200 <= status < 300:
                print(f"[health] ok {text[:300]}", flush=True)
                return
            last_error = f"{status}: {text[:200]}"
        except Exception as exc:
            last_error = repr(exc)
        print(f"[health] waiting (retry in 10 s): {last_error}", flush=True)
        time.sleep(10)
    raise RuntimeError(f"health check failed for {url}: {last_error}")


def get_text(url: str, timeout_s: float) -> tuple[int, str]:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")


def post_json(url: str, payload: Any, timeout_s: float) -> tuple[int, str]:
    raw = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=raw,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")


def write_json(path: Path, payload: Any) -> None:
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def write_summary_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    keys = [
        "target_ttft_ms",
        "offered_rate_qps",
        "kp",
        "ki",
        "fraction_min",
        "fraction_max",
        "requests_measured",
        "requests_ok",
        "error_rate",
        "throughput_req_s",
        "ttft_mean_ms",
        "ttft_p95_ms",
        "total_mean_ms",
        "total_p95_ms",
        "vllm_queue_wait_mean_ms",
        "vllm_ttft_mean_ms",
        "vllm_e2e_mean_ms",
        "gpu_power_mean_w",
        "gpu_power_peak_w",
        "energy_j",
        "energy_per_request_j",
    ]
    extra = sorted({
        key for summary in summaries for key in summary
        if key not in keys
    })
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys + extra, extrasaction="ignore")
        writer.writeheader()
        for summary in summaries:
            writer.writerow({key: summary.get(key) for key in keys + extra})


if __name__ == "__main__":
    main()
