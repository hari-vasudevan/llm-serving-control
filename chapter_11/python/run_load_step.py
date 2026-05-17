#!/usr/bin/env python3
"""Chapter 11 load-step disturbance rejection runner.

Posts to /run_internal_load_step on the deployed Modal wrapper.

Default experiment (token_budget actuator):
  - Warmup 90s at qps=2, fraction=0.08  → TTFT ≈ 300ms (queue built)
  - Step 1: qps=2, 90s   (closed-loop baseline)
  - Step 2: qps=4, 90s   (load doubled  → TTFT spikes → fraction rises → reconverges)
  - Step 3: qps=2, 90s   (load halved   → TTFT drops  → fraction falls → reconverges)

Total: ~6 min experiment + ~4 min cold-start if container is down.
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
from plot_load_step import plot_result


DEFAULT_LOAD_STEPS = [
    {"qps": 2.0, "duration_s": 90.0},
    {"qps": 4.0, "duration_s": 90.0},
    {"qps": 2.0, "duration_s": 90.0},
]


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
    ap.add_argument("--target-ttft-ms", type=float, default=300.0)
    ap.add_argument("--actuator", default="token_budget",
                    choices=["token_budget", "dispatch_delay"])
    # Warmup
    ap.add_argument("--warmup-qps", type=float, default=2.0)
    ap.add_argument("--warmup-s", type=float, default=90.0)
    ap.add_argument("--warmup-fraction", type=float, default=0.08,
                    help="Open-loop fraction during warmup (token_budget only)")
    # Load steps — JSON or defaults
    ap.add_argument("--load-steps-json", default=None,
                    help='JSON array, e.g. \'[{"qps":2,"duration_s":90},{"qps":4,"duration_s":90}]\'')
    # Controller
    ap.add_argument("--kp", type=float, default=0.05)
    ap.add_argument("--ki", type=float, default=0.01)
    ap.add_argument("--fraction-min", type=float, default=0.05)
    ap.add_argument("--fraction-max", type=float, default=1.0)
    ap.add_argument("--max-delay-ms", type=float, default=2000.0)
    ap.add_argument("--ttft-window", type=int, default=20)
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
    plots_dir = out_dir / "plots"
    logs_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    with (logs_dir / "run_load_step.log").open("w") as log_f:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = Tee(old_stdout, log_f)
        sys.stderr = Tee(old_stderr, log_f)
        try:
            _run(args, load_steps, out_dir, plots_dir)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def _run(args: argparse.Namespace, load_steps: list[dict],
         out_dir: Path, plots_dir: Path) -> None:
    print(f"[start] Chapter 11 load-step sweep {datetime.now().isoformat()}", flush=True)
    print(f"[out]   {out_dir}", flush=True)

    wait_for_health(args.url)

    payload = {
        "target_ttft_ms": args.target_ttft_ms,
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
    write_json(out_dir / "request.json", payload)
    print("[request]", json.dumps(payload, indent=2), flush=True)

    endpoint = f"{args.url.rstrip('/')}/run_internal_load_step"
    t0 = time.perf_counter()
    print(f"[post] {endpoint}", flush=True)
    status, text = post_json(endpoint, payload, timeout_s=args.timeout_s)
    elapsed = time.perf_counter() - t0
    print(f"[response] status={status} elapsed_s={elapsed:.1f}", flush=True)

    if status < 200 or status >= 300:
        raise RuntimeError(f"POST {endpoint} failed HTTP {status}: {text[:2000]}")

    result = json.loads(text)
    result["client_elapsed_s"] = elapsed
    result["client_timestamp"] = datetime.now().isoformat()
    write_json(out_dir / "response.json", result)

    ts = result.get("timeseries", [])
    step_summaries = result.get("step_summaries", [])
    write_json(out_dir / "timeseries.json", ts)
    write_json(out_dir / "step_summaries.json", step_summaries)

    paths = plot_result(ts, result, plots_dir)
    write_json(out_dir / "plot_manifest.json", [str(p) for p in paths])

    print("[step_summaries]", json.dumps(step_summaries, indent=2), flush=True)
    print(f"[summary] target={result.get('target_ttft_ms')}ms  "
          f"overall_mean={result.get('ttft_mean_ms'):.1f}ms  "
          f"error_rate={result.get('error_rate')}", flush=True)
    for p in paths:
        print(f"[plot] {p}", flush=True)
    print(f"[done] {out_dir}", flush=True)


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
    raise RuntimeError(f"health check failed for {url}: {last_error}")


def _get(url: str, timeout_s: float) -> tuple[int, str]:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")


def post_json(url: str, payload: Any, timeout_s: float) -> tuple[int, str]:
    raw = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=raw, method="POST",
                                  headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")


def write_json(path: Path, payload: Any) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
