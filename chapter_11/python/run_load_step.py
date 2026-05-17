#!/usr/bin/env python3
"""Chapter 11 load-step disturbance rejection runner.

Posts to /run_internal_load_step on the deployed Modal wrapper.
Supports multiple targets in one call — the server runs them sequentially
so the container stays warm between targets.

Default experiment (dispatch_delay, 3 targets, 3 load levels):
  warmup  30s  qps=4  (dispatch_delay, settle to target TTFT)
  step 0  90s  qps=4  (baseline)
  step 1  90s  qps=8  (load doubles  → natural TTFT rises → delay reduced)
  step 2  90s  qps=4  (load halves   → delay increased back)

After the run:
  - Per-target subplot SVG (Load / TTFT / Power, shared x-axis)
  - Per-target MATLAB view_figure.m  (linked axes, saves .fig)
  - Multi-target view_comparison.m   (3 × N grid, linked per row)
"""
from __future__ import annotations

import argparse
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
from plot_load_step import plot_result, write_matlab_scripts


DEFAULT_LOAD_STEPS = [
    {"qps": 4.0, "duration_s": 90.0},
    {"qps": 8.0, "duration_s": 90.0},
    {"qps": 4.0, "duration_s": 90.0},
]
DEFAULT_TARGETS = [200.0, 350.0, 500.0]


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
    ap = argparse.ArgumentParser(description="Chapter 11 load-step disturbance rejection runner")
    ap.add_argument("--url", required=True, help="Base URL of Modal wrapper")
    ap.add_argument("--target-ttft-ms", type=float, nargs="+", default=DEFAULT_TARGETS)
    ap.add_argument("--actuator", default="dispatch_delay",
                    choices=["dispatch_delay", "token_budget"])
    # Warmup
    ap.add_argument("--warmup-qps", type=float, default=4.0)
    ap.add_argument("--warmup-s", type=float, default=30.0)
    ap.add_argument("--warmup-fraction", type=float, default=0.08,
                    help="Open-loop fraction during warmup (token_budget only)")
    # Load steps
    ap.add_argument("--load-steps-json", default=None,
                    help='JSON array e.g. \'[{"qps":4,"duration_s":90},{"qps":8,"duration_s":90}]\'')
    # Controller
    ap.add_argument("--kp", type=float, default=0.05)
    ap.add_argument("--ki", type=float, default=0.005)
    ap.add_argument("--fraction-min", type=float, default=0.05)
    ap.add_argument("--fraction-max", type=float, default=1.0)
    ap.add_argument("--max-delay-ms", type=float, default=2000.0)
    ap.add_argument("--ttft-window", type=int, default=10)
    ap.add_argument("--feedback-period-s", type=float, default=0.5)
    # Request shape
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--prompt-repeat", type=int, default=64)
    ap.add_argument("--max-outstanding", type=int, default=256)
    ap.add_argument("--metric-period-s", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--timeout-s", type=float, default=7200.0)
    ap.add_argument("--out-dir",
                    default=str(Path(__file__).resolve().parent / "results"))
    args = ap.parse_args()

    load_steps = json.loads(args.load_steps_json) if args.load_steps_json else DEFAULT_LOAD_STEPS

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / f"load_step_{stamp}"
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    with (logs_dir / "run_load_step.log").open("w") as log_f:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = Tee(old_stdout, log_f)
        sys.stderr = Tee(old_stderr, log_f)
        try:
            _run(args, load_steps, out_dir)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def _run(args: argparse.Namespace, load_steps: list[dict], out_dir: Path) -> None:
    print(f"[start] Chapter 11 load-step sweep {datetime.now().isoformat()}", flush=True)
    print(f"[out]   {out_dir}", flush=True)

    targets = args.target_ttft_ms
    print(f"[targets] {targets}", flush=True)
    print(f"[load_steps] {load_steps}", flush=True)

    wait_for_health(args.url)

    endpoint = f"{args.url.rstrip('/')}/run_internal_load_step"

    # Loop targets on the client side — one HTTP request per target (~5 min each).
    # Results are written to disk immediately so a failure on a later target
    # doesn't lose data from already-completed targets.
    # A health wait before each target handles container cold-starts gracefully.
    results_list: list[dict] = []
    all_plot_paths: list[Path] = []
    for i, target in enumerate(targets):
        # Re-confirm container is up before each target (may cold-start between runs)
        if i > 0:
            print(f"\n[health] waiting for container before target {i+1}...", flush=True)
            wait_for_health(args.url)

        payload = {
            "target_ttft_ms": float(target),
            "actuator": args.actuator,
            "warmup_qps": args.warmup_qps,
            "warmup_s": args.warmup_s,
            "warmup_fraction": args.warmup_fraction,
            "load_steps": load_steps,
            "kp": args.kp,
            "ki": args.ki,
            "fraction_min": args.fraction_min,
            "fraction_max": args.fraction_max,
            "max_delay_ms": args.max_delay_ms,
            "ttft_window": args.ttft_window,
            "feedback_period_s": args.feedback_period_s,
            "max_tokens": args.max_tokens,
            "prompt_repeat": args.prompt_repeat,
            "max_outstanding": args.max_outstanding,
            "metric_period_s": args.metric_period_s,
            "seed": args.seed,
        }
        _write_json(out_dir / f"request_target{int(target)}ms.json", payload)
        print(f"\n[target {i+1}/{len(targets)}] {target}ms  POST {endpoint}", flush=True)

        t0 = time.perf_counter()
        status, text = _post_json(endpoint, payload, timeout_s=args.timeout_s)
        elapsed = time.perf_counter() - t0
        print(f"[response] status={status} elapsed_s={elapsed:.1f}", flush=True)

        if status < 200 or status >= 300:
            raise RuntimeError(f"POST {endpoint} failed HTTP {status}: {text[:2000]}")

        raw = json.loads(text)
        raw["client_elapsed_s"] = elapsed
        raw["client_timestamp"] = datetime.now().isoformat()
        results_list.append(raw)

        # Write per-target data to disk immediately
        tgt_key = int(raw.get("target_ttft_ms", target))
        tdir = out_dir / f"target_{tgt_key}ms"
        tdir.mkdir(exist_ok=True)

        ts = raw.get("timeseries", [])
        _write_json(tdir / "timeseries.json", ts)
        _write_json(tdir / "step_summaries.json", raw.get("step_summaries", []))
        _write_json(tdir / "qa_log.json", raw.get("qa_log", []))
        _write_json(tdir / "summary.json", {k: v for k, v in raw.items()
                                             if k not in ("timeseries", "qa_log")})

        plots_dir = tdir / "plots"
        plots_dir.mkdir(exist_ok=True)
        paths = plot_result(ts, raw, plots_dir)
        all_plot_paths.extend(paths)

        print(f"\n[target={tgt_key}ms] step_summaries:", flush=True)
        for ss in raw.get("step_summaries", []):
            print(f"  qps={ss['qps']}  mean={ss.get('ttft_mean_ms', 'N/A')}ms  "
                  f"p95={ss.get('ttft_p95_ms', 'N/A')}ms  "
                  f"std={ss.get('ttft_stdev_ms', 'N/A')}ms", flush=True)

    # Generate MATLAB scripts (single + combined)
    matlab_paths = write_matlab_scripts(results_list, out_dir)
    all_plot_paths.extend(matlab_paths)

    _write_json(out_dir / "plot_manifest.json", [str(p) for p in all_plot_paths])

    for p in all_plot_paths:
        print(f"[output] {p}", flush=True)
    print(f"\n[done] {out_dir}", flush=True)

    return out_dir, results_list


# ── network helpers ────────────────────────────────────────────────────────────

def wait_for_health(url: str, timeout_s: float = 900.0) -> None:
    deadline = time.time() + timeout_s
    last_error = ""
    while time.time() < deadline:
        remaining = deadline - time.time()
        req_timeout = min(remaining, 300.0)
        try:
            status, text = _get(f"{url.rstrip('/')}/health", timeout_s=req_timeout)
            if 200 <= status < 300:
                print(f"[health] ok {text[:200]}", flush=True)
                return
            last_error = f"HTTP {status}: {text[:200]}"
        except Exception as exc:
            last_error = repr(exc)
        print(f"[health] waiting (retry in 10s): {last_error}", flush=True)
        time.sleep(10)
    raise RuntimeError(f"health check timed out for {url}: {last_error}")


def _get(url: str, timeout_s: float) -> tuple[int, str]:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")


def _post_json(url: str, payload: Any, timeout_s: float) -> tuple[int, str]:
    raw = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=raw, method="POST",
                                  headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")


def _write_json(path: Path, payload: Any) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
