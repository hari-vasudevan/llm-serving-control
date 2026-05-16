#!/usr/bin/env python3
"""
Chapter 10 top-level query benchmark.

This runner sends ordinary OpenAI-compatible `/v1/completions` requests to a
vLLM endpoint while scraping metrics and optional power data. It is intentionally
outside the controller implementation: the client offers load, vLLM admits work,
and this script measures the outcome.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests


PROMPT_SEEDS = [
    "Explain how a queueing controller changes latency in an LLM serving system.",
    "Summarize the difference between admission control and GPU scheduling.",
    "Describe why measuring power matters when evaluating inference latency.",
    "Explain why a nonzero queue wait target might smooth bursty load.",
    "Compare throughput, latency, and energy per request in one paragraph.",
]


@dataclass
class RequestRecord:
    request_id: int
    target_wait_ms: float
    offered_rate_qps: float
    t_send: float
    t_first_token: float | None
    t_done: float
    status: str
    http_status: int | None
    ttft_ms: float | None
    total_latency_ms: float
    prompt_chars: int
    max_tokens: int
    error: str


@dataclass
class MetricRecord:
    t: float
    target_wait_ms: float
    offered_rate_qps: float
    metrics: dict[str, float]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="Base URL for vLLM or Chapter 10 wrapper")
    ap.add_argument("--model", default=os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"))
    ap.add_argument("--api-key", default=os.getenv("VLLM_API_KEY", ""))
    ap.add_argument("--target-wait-ms", type=float, nargs="+", default=[0, 50, 100, 200, 400])
    ap.add_argument("--offered-rate-qps", type=float, default=4.0)
    ap.add_argument("--duration-s", type=float, default=90.0)
    ap.add_argument("--warmup-s", type=float, default=15.0)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--prompt-repeat", type=int, default=96)
    ap.add_argument("--max-outstanding", type=int, default=128)
    ap.add_argument("--metric-period-s", type=float, default=1.0)
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "results"))
    ap.add_argument("--seed", type=int, default=10)
    args = ap.parse_args()

    random.seed(args.seed)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    for target in args.target_wait_ms:
        run_name = f"target_{int(target):04d}ms_rate_{args.offered_rate_qps:g}qps"
        out_dir = out_root / run_name
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {run_name} ===", flush=True)
        summary = run_one_target(args, target, out_dir)
        all_summaries.append(summary)
        print(json.dumps(summary, indent=2), flush=True)

    with (out_root / "sweep_summary.json").open("w") as f:
        json.dump(all_summaries, f, indent=2)
    write_summary_csv(out_root / "sweep_summary.csv", all_summaries)
    print(f"\n[done] wrote {out_root}", flush=True)


def run_one_target(args: argparse.Namespace, target_wait_ms: float, out_dir: Path) -> dict[str, Any]:
    configure_target(args.url, target_wait_ms)
    wait_for_health(args.url)

    records: list[RequestRecord] = []
    metrics: list[MetricRecord] = []
    stop = threading.Event()
    sem = threading.Semaphore(args.max_outstanding)
    lock = threading.Lock()
    request_counter = 0

    metric_thread = threading.Thread(
        target=scrape_metrics_loop,
        args=(args.url, target_wait_ms, args.offered_rate_qps, args.metric_period_s, stop, metrics),
        daemon=True,
    )
    metric_thread.start()

    print(f"[warmup] {args.warmup_s:.1f}s at {args.offered_rate_qps:.2f} qps", flush=True)
    t_start = time.perf_counter()
    t_measure_start = t_start + args.warmup_s
    t_end = t_measure_start + args.duration_s
    next_arrival = t_start
    threads: list[threading.Thread] = []

    while time.perf_counter() < t_end:
        now = time.perf_counter()
        if now < next_arrival:
            time.sleep(min(0.02, next_arrival - now))
            continue

        if sem.acquire(timeout=0.2):
            request_counter += 1
            req_id = request_counter
            measure = now >= t_measure_start
            thread = threading.Thread(
                target=fire_request,
                args=(args, target_wait_ms, req_id, measure, sem, lock, records),
                daemon=True,
            )
            thread.start()
            threads.append(thread)

        interval = random.expovariate(args.offered_rate_qps) if args.offered_rate_qps > 0 else 1.0
        next_arrival += interval

    for thread in threads:
        thread.join(timeout=180)
    stop.set()
    metric_thread.join(timeout=5)

    write_request_csv(out_dir / "requests.csv", records)
    write_metrics_jsonl(out_dir / "metrics.jsonl", metrics)

    summary = summarize(records, metrics)
    summary.update(
        {
            "target_wait_ms": target_wait_ms,
            "offered_rate_qps": args.offered_rate_qps,
            "duration_s": args.duration_s,
            "warmup_s": args.warmup_s,
            "url": args.url,
            "model": args.model,
        }
    )
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    return summary


def configure_target(url: str, target_wait_ms: float) -> None:
    # The Chapter 10 wrapper can expose this endpoint. Direct vLLM will return
    # 404, which is fine when the target is configured by environment/deploy.
    try:
        requests.post(
            f"{url.rstrip('/')}/control/queue_wait_target",
            json={"target_wait_ms": target_wait_ms},
            timeout=5,
        )
    except Exception:
        pass


def wait_for_health(url: str, timeout_s: float = 180.0) -> None:
    deadline = time.time() + timeout_s
    last_error = ""
    while time.time() < deadline:
        try:
            resp = requests.get(f"{url.rstrip('/')}/health", timeout=5)
            if resp.ok:
                return
            last_error = f"{resp.status_code}: {resp.text[:200]}"
        except Exception as exc:
            last_error = repr(exc)
        time.sleep(2)
    raise RuntimeError(f"health check failed for {url}: {last_error}")


def fire_request(
    args: argparse.Namespace,
    target_wait_ms: float,
    request_id: int,
    measure: bool,
    sem: threading.Semaphore,
    lock: threading.Lock,
    records: list[RequestRecord],
) -> None:
    prompt = make_prompt(request_id, args.prompt_repeat)
    t_send = time.perf_counter()
    t_first = None
    status = "ok"
    http_status = None
    error = ""
    try:
        body = {
            "model": args.model,
            "prompt": prompt,
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
            "stream": True,
        }
        headers = {"Content-Type": "application/json"}
        if args.api_key:
            headers["Authorization"] = f"Bearer {args.api_key}"
        with requests.post(
            f"{args.url.rstrip('/')}/v1/completions",
            json=body,
            headers=headers,
            stream=True,
            timeout=180,
        ) as resp:
            http_status = resp.status_code
            resp.raise_for_status()
            for chunk in resp.iter_lines():
                if chunk and chunk != b"data: [DONE]":
                    if t_first is None:
                        t_first = time.perf_counter()
    except Exception as exc:
        status = "error"
        error = repr(exc)
    finally:
        t_done = time.perf_counter()
        if measure:
            record = RequestRecord(
                request_id=request_id,
                target_wait_ms=target_wait_ms,
                offered_rate_qps=args.offered_rate_qps,
                t_send=t_send,
                t_first_token=t_first,
                t_done=t_done,
                status=status,
                http_status=http_status,
                ttft_ms=(1000.0 * (t_first - t_send)) if t_first is not None else None,
                total_latency_ms=1000.0 * (t_done - t_send),
                prompt_chars=len(prompt),
                max_tokens=args.max_tokens,
                error=error,
            )
            with lock:
                records.append(record)
        sem.release()


def scrape_metrics_loop(
    url: str,
    target_wait_ms: float,
    offered_rate_qps: float,
    period_s: float,
    stop: threading.Event,
    sink: list[MetricRecord],
) -> None:
    while not stop.is_set():
        t0 = time.perf_counter()
        metrics = {}
        try:
            raw = requests.get(f"{url.rstrip('/')}/metrics", timeout=5).text
            if raw.lstrip().startswith("{"):
                metrics = flatten_json_metrics(json.loads(raw))
            else:
                metrics = parse_prometheus_metrics(raw)
        except Exception as exc:
            metrics = {"scrape_error": 1.0, "scrape_error_hash": float(abs(hash(repr(exc))) % 10_000)}
        sink.append(MetricRecord(time.perf_counter(), target_wait_ms, offered_rate_qps, metrics))
        elapsed = time.perf_counter() - t0
        time.sleep(max(0.0, period_s - elapsed))


def make_prompt(index: int, repeat: int) -> str:
    seed = PROMPT_SEEDS[index % len(PROMPT_SEEDS)]
    return " ".join([seed] * max(1, repeat))


def parse_prometheus_metrics(raw: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in raw.splitlines():
        if not line or line.startswith("#"):
            continue
        name_value = line.split()
        if len(name_value) < 2:
            continue
        name = name_value[0].split("{", 1)[0]
        try:
            out[name] = out.get(name, 0.0) + float(name_value[-1])
        except ValueError:
            continue
    return out


def flatten_json_metrics(payload: dict[str, Any]) -> dict[str, float]:
    out = {}
    for key, value in payload.items():
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            out[key] = float(value)
    return out


def summarize(records: list[RequestRecord], metrics: list[MetricRecord]) -> dict[str, Any]:
    ok = [r for r in records if r.status == "ok"]
    ttft = [r.ttft_ms for r in ok if r.ttft_ms is not None and math.isfinite(r.ttft_ms)]
    total = [r.total_latency_ms for r in ok if math.isfinite(r.total_latency_ms)]
    duration = max((r.t_done for r in ok), default=0.0) - min((r.t_send for r in ok), default=0.0)
    power = latest_series(metrics, "gpu_power_w")
    energy_j = integrate_series(power)
    completed = len(ok)
    return {
        "requests_measured": len(records),
        "requests_ok": completed,
        "error_rate": 1.0 - completed / max(len(records), 1),
        "throughput_req_s": completed / max(duration, 1e-9),
        "ttft_mean_ms": safe_mean(ttft),
        "ttft_p50_ms": percentile(ttft, 50),
        "ttft_p95_ms": percentile(ttft, 95),
        "total_mean_ms": safe_mean(total),
        "total_p50_ms": percentile(total, 50),
        "total_p95_ms": percentile(total, 95),
        "gpu_power_mean_w": safe_mean([v for _, v in power]),
        "gpu_power_peak_w": max([v for _, v in power], default=None),
        "energy_j": energy_j,
        "energy_per_request_j": energy_j / completed if completed > 0 and energy_j is not None else None,
        "vllm_queue_wait_mean_ms": (
            metric_hist_mean_ms(metrics, "vllm:request_queue_time_seconds")
            or metric_average(metrics, "vllm_queue_mean_ms")
        ),
        "vllm_ttft_mean_ms": (
            metric_hist_mean_ms(metrics, "vllm:time_to_first_token_seconds")
            or metric_average(metrics, "vllm_ttft_mean_ms")
        ),
        "vllm_e2e_mean_ms": (
            metric_hist_mean_ms(metrics, "vllm:e2e_request_latency_seconds")
            or metric_average(metrics, "vllm_e2e_mean_ms")
        ),
    }


def metric_hist_mean_ms(records: list[MetricRecord], stem: str) -> float | None:
    if not records:
        return None
    first = records[0].metrics
    last = records[-1].metrics
    d_sum = last.get(f"{stem}_sum", 0.0) - first.get(f"{stem}_sum", 0.0)
    d_count = last.get(f"{stem}_count", 0.0) - first.get(f"{stem}_count", 0.0)
    if d_count <= 0:
        return None
    return 1000.0 * d_sum / d_count


def metric_average(records: list[MetricRecord], key: str) -> float | None:
    values = [r.metrics[key] for r in records if key in r.metrics and math.isfinite(r.metrics[key])]
    return statistics.mean(values) if values else None


def latest_series(records: list[MetricRecord], key: str) -> list[tuple[float, float]]:
    out = []
    for r in records:
        value = r.metrics.get(key)
        if value is not None and math.isfinite(value):
            out.append((r.t, value))
    return out


def integrate_series(series: list[tuple[float, float]]) -> float | None:
    if len(series) < 2:
        return None
    total = 0.0
    for (t0, v0), (t1, v1) in zip(series, series[1:]):
        total += 0.5 * (v0 + v1) * max(0.0, t1 - t0)
    return total


def safe_mean(values: list[float]) -> float | None:
    finite = [v for v in values if v is not None and math.isfinite(v)]
    return statistics.mean(finite) if finite else None


def percentile(values: list[float], pct: float) -> float | None:
    finite = sorted(v for v in values if v is not None and math.isfinite(v))
    if not finite:
        return None
    idx = min(len(finite) - 1, max(0, round((pct / 100.0) * (len(finite) - 1))))
    return finite[idx]


def write_request_csv(path: Path, records: list[RequestRecord]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(RequestRecord.__dataclass_fields__.keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def write_metrics_jsonl(path: Path, records: list[MetricRecord]) -> None:
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(asdict(record)) + "\n")


def write_summary_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    keys = sorted({key for summary in summaries for key in summary})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(summary)


if __name__ == "__main__":
    main()
