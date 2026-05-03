#!/usr/bin/env python3
"""
gpu_batch_server.py -- Chapter 9 lower-level GPU batching plant.

Plant vocabulary follows Chapter 2:
  inner loop: B -> q
  outer loop: q_ref -> L_mean / L_p95

The server owns the physical plant:
  - FIFO queue q[k]
  - exact batch-size actuator B[k]
  - fixed GPU batch workload
  - measured batch service time

MATLAB owns identification and the cascade controller.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import signal
import threading
import time
import uuid
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from statistics import mean
from typing import Any

from workloads import FixedMatmulWorkload, WorkloadConfig


@dataclass
class RequestItem:
    request_id: str
    enqueue_time: float
    source: str


class Plant:
    def __init__(self, workload: FixedMatmulWorkload, log_dir: Path, initial_b: int, tick_s: float):
        self.workload = workload
        self.log_dir = log_dir
        self.tick_s = tick_s
        self.lock = threading.Lock()
        self.fifo: deque[RequestItem] = deque()
        self.B_current = initial_b
        self.running = True
        self.batch_id = 0
        self.enqueued = 0
        self.completed = 0
        self.arrivals_tick = 0
        self.completions_tick = 0
        self.service_ms_recent: deque[float] = deque(maxlen=100)
        self.l_total_recent: deque[float] = deque(maxlen=1000)
        self.qwait_recent: deque[float] = deque(maxlen=1000)
        self.q_samples_tick: list[int] = []
        self.last_batch: dict[str, Any] = {}
        self.controller_config: dict[str, Any] = {}

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.batch_csv = (self.log_dir / "batch_log.csv").open("w", newline="")
        self.request_csv = (self.log_dir / "request_log.csv").open("w", newline="")
        self.tick_csv = (self.log_dir / "tick_log.csv").open("w", newline="")
        self.batch_writer = csv.DictWriter(self.batch_csv, fieldnames=[
            "batch_id", "t_dispatch", "t_finish", "B_cmd", "B_actual",
            "q_before", "q_after", "service_time_ms", "throughput_jobs_s",
        ])
        self.request_writer = csv.DictWriter(self.request_csv, fieldnames=[
            "request_id", "batch_id", "enqueue_time", "dispatch_time", "finish_time",
            "queue_wait_ms", "service_time_ms", "total_latency_ms", "source",
        ])
        self.tick_writer = csv.DictWriter(self.tick_csv, fieldnames=[
            "t", "B_current", "q_sw", "q_mean_tick", "q_max_tick",
            "arrivals_tick", "completions_tick", "service_mean_ms",
            "l_mean_ms", "l_p95_ms", "queue_wait_mean_ms",
        ])
        self.batch_writer.writeheader()
        self.request_writer.writeheader()
        self.tick_writer.writeheader()

    def enqueue(self, count: int, source: str) -> dict[str, Any]:
        now = time.perf_counter()
        items = [RequestItem(str(uuid.uuid4()), now, source) for _ in range(max(count, 0))]
        with self.lock:
            self.fifo.extend(items)
            self.enqueued += len(items)
            self.arrivals_tick += len(items)
            q = len(self.fifo)
        return {"status": "queued_batch", "count": len(items), "q_sw": q}

    def set_B(self, B: int) -> dict[str, Any]:
        with self.lock:
            self.B_current = max(1, int(B))
            return {"status": "ok", "B_current": self.B_current}

    def reset(self) -> dict[str, Any]:
        with self.lock:
            self.fifo.clear()
            self.batch_id = 0
            self.enqueued = 0
            self.completed = 0
            self.arrivals_tick = 0
            self.completions_tick = 0
            self.service_ms_recent.clear()
            self.l_total_recent.clear()
            self.qwait_recent.clear()
            self.q_samples_tick.clear()
            self.last_batch = {}
        return {"status": "reset"}

    def metrics(self) -> dict[str, Any]:
        with self.lock:
            q_sw = len(self.fifo)
            q_samples = self.q_samples_tick[:] or [q_sw]
            service = list(self.service_ms_recent)
            l_total = list(self.l_total_recent)
            qwait = list(self.qwait_recent)
            out = {
                "status": "ok",
                "B_current": self.B_current,
                "q_sw": q_sw,
                "q_mean_tick": mean(q_samples),
                "q_max_tick": max(q_samples),
                "arrivals_tick": self.arrivals_tick,
                "completions_tick": self.completions_tick,
                "service_mean_ms": _safe_mean(service),
                "l_mean_ms": _safe_mean(l_total),
                "l_p95_ms": _percentile(l_total, 95.0),
                "queue_wait_mean_ms": _safe_mean(qwait),
                "enqueued": self.enqueued,
                "completed": self.completed,
                "last_batch": self.last_batch,
                "controller_config": self.controller_config,
            }
            self.arrivals_tick = 0
            self.completions_tick = 0
            self.q_samples_tick = []
            self.tick_writer.writerow({k: out[k] for k in self.tick_writer.fieldnames if k in out} | {"t": time.time()})
            self.tick_csv.flush()
            return out

    def worker_loop(self) -> None:
        next_tick = time.perf_counter()
        while self.running:
            now = time.perf_counter()
            if now < next_tick:
                time.sleep(next_tick - now)
            tick_start = time.perf_counter()
            next_tick = tick_start + self.tick_s

            with self.lock:
                self.q_samples_tick.append(len(self.fifo))
                b_now = self.B_current
                batch = []
                while self.fifo and len(batch) < b_now:
                    batch.append(self.fifo.popleft())
                q_after = len(self.fifo)
                q_before = q_after + len(batch)

            if not batch:
                continue

            dispatch_t = time.perf_counter()
            self.workload.synchronize()
            t0 = time.perf_counter()
            self.workload.run(len(batch))
            self.workload.synchronize()
            finish_t = time.perf_counter()
            service_ms = 1000.0 * (finish_t - t0)
            throughput = 1000.0 * len(batch) / max(service_ms, 1e-9)

            request_rows = []
            qwaits = []
            totals = []
            for item in batch:
                q_wait_ms = 1000.0 * (dispatch_t - item.enqueue_time)
                total_ms = 1000.0 * (finish_t - item.enqueue_time)
                qwaits.append(q_wait_ms)
                totals.append(total_ms)
                request_rows.append({
                    "request_id": item.request_id,
                    "batch_id": self.batch_id,
                    "enqueue_time": item.enqueue_time,
                    "dispatch_time": dispatch_t,
                    "finish_time": finish_t,
                    "queue_wait_ms": q_wait_ms,
                    "service_time_ms": service_ms,
                    "total_latency_ms": total_ms,
                    "source": item.source,
                })

            batch_row = {
                "batch_id": self.batch_id,
                "t_dispatch": dispatch_t,
                "t_finish": finish_t,
                "B_cmd": b_now,
                "B_actual": len(batch),
                "q_before": q_before,
                "q_after": q_after,
                "service_time_ms": service_ms,
                "throughput_jobs_s": throughput,
            }

            with self.lock:
                self.completed += len(batch)
                self.completions_tick += len(batch)
                self.service_ms_recent.append(service_ms)
                self.qwait_recent.extend(qwaits)
                self.l_total_recent.extend(totals)
                self.last_batch = batch_row.copy()
                self.batch_writer.writerow(batch_row)
                self.batch_csv.flush()
                for row in request_rows:
                    self.request_writer.writerow(row)
                self.request_csv.flush()
                self.batch_id += 1

    def close(self) -> None:
        self.running = False
        self.batch_csv.close()
        self.request_csv.close()
        self.tick_csv.close()

    def set_controller_config_xml(self, xml_text: str) -> dict[str, Any]:
        cfg = _parse_controller_xml(xml_text)
        with self.lock:
            self.controller_config = cfg
        return {"status": "ok", "controller_config": cfg}

    def run_characterisation(self, body: dict[str, Any]) -> dict[str, Any]:
        dt = float(body.get("dt", self.tick_s))
        b_sweep = [int(x) for x in body.get("B_sweep", [4, 8, 12, 16, 24, 32])]
        lambda_sweep = [float(x) for x in body.get("lambda_sweep", [8, 12, 16, 20, 24, 28, 32])]
        lambda_char = float(body.get("lambda_char", 24))
        b0_probe = int(body.get("B0_probe", 16))
        ticks_per_point = int(body.get("ticks_per_point", 30))
        settle_ticks = int(body.get("settle_ticks", 8))
        source = str(body.get("source", "modal_characterise"))

        b_results = []
        for b_cmd in b_sweep:
            logs = self._run_block(b_cmd, lambda_char, dt, ticks_per_point, f"{source}_B_{b_cmd}")
            b_results.append({"B": b_cmd, **_summarise_logs(logs, settle_ticks)})

        lambda_results = []
        for lam in lambda_sweep:
            logs = self._run_block(b0_probe, lam, dt, ticks_per_point, f"{source}_lam_{lam:g}")
            lambda_results.append({"lambda": lam, **_summarise_logs(logs, settle_ticks)})

        return {
            "status": "ok",
            "dt": dt,
            "lambda_char": lambda_char,
            "B0_probe": b0_probe,
            "ticks_per_point": ticks_per_point,
            "settle_ticks": settle_ticks,
            "B_results": b_results,
            "lambda_results": lambda_results,
        }

    def run_characterisation_block(self, body: dict[str, Any]) -> dict[str, Any]:
        dt = float(body.get("dt", self.tick_s))
        b_cmd = int(body.get("B", body.get("B_cmd", self.B_current)))
        lambda_tick = float(body.get("lambda", body.get("lambda_tick", 1)))
        ticks_per_point = int(body.get("ticks_per_point", 24))
        settle_ticks = int(body.get("settle_ticks", 6))
        source = str(body.get("source", "modal_characterise_block"))

        logs = self._run_block(b_cmd, lambda_tick, dt, ticks_per_point, source)
        return {
            "status": "ok",
            "B": b_cmd,
            "lambda": lambda_tick,
            "dt": dt,
            "ticks_per_point": ticks_per_point,
            "settle_ticks": settle_ticks,
            **_summarise_logs(logs, settle_ticks),
        }

    def _run_block(self, b_cmd: int, lambda_tick: float, dt: float, n_ticks: int, source: str) -> list[dict[str, Any]]:
        self.reset()
        self.set_B(b_cmd)
        time.sleep(2.0 * self.tick_s)
        logs = []
        for k in range(n_ticks):
            arrivals = max(0, int(round(lambda_tick)))
            self.enqueue(arrivals, f"{source}_tick_{k:03d}")
            time.sleep(dt)
            m = self.metrics()
            m["arrivals_injected"] = arrivals
            logs.append(m)
        return logs


def make_handler(plant: Plant):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/health":
                self._send({"status": "ok"})
            elif self.path == "/metrics":
                self._send(plant.metrics())
            elif self.path == "/logs":
                self._send({
                    "batch_log": str(plant.log_dir / "batch_log.csv"),
                    "request_log": str(plant.log_dir / "request_log.csv"),
                    "tick_log": str(plant.log_dir / "tick_log.csv"),
                })
            else:
                self._send({"error": "not found"}, status=404)

        def do_POST(self) -> None:
            body = self._read_json()
            if self.path == "/control":
                self._send(plant.set_B(int(body.get("B", plant.B_current))))
            elif self.path == "/enqueue_batch":
                count = int(body.get("count", body.get("arrivals", 1)))
                self._send(plant.enqueue(count, str(body.get("source", "matlab"))))
            elif self.path == "/reset":
                self._send(plant.reset())
            elif self.path == "/controller_config":
                xml_text = str(body.get("xml", ""))
                if not xml_text:
                    self._send({"error": "missing xml field"}, status=400)
                else:
                    self._send(plant.set_controller_config_xml(xml_text))
            elif self.path == "/characterise":
                self._send(plant.run_characterisation(body))
            elif self.path == "/characterise_block":
                self._send(plant.run_characterisation_block(body))
            else:
                self._send({"error": "not found"}, status=404)

        def _read_json(self) -> dict[str, Any]:
            n = int(self.headers.get("Content-Length", "0"))
            if n <= 0:
                return {}
            return json.loads(self.rfile.read(n).decode("utf-8"))

        def _send(self, payload: dict[str, Any], status: int = 200) -> None:
            raw = json.dumps(_json_safe(payload), allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    return Handler


def _safe_mean(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return mean(finite) if finite else float("nan")


def _summarise_logs(logs: list[dict[str, Any]], settle_ticks: int) -> dict[str, Any]:
    use = logs[min(len(logs), settle_ticks):]
    return {
        "q_mean_tick": _safe_mean([float(x.get("q_mean_tick", float("nan"))) for x in use]),
        "q_max_tick": max([float(x.get("q_max_tick", 0.0)) for x in use], default=0.0),
        "l_mean_ms": _safe_mean([float(x.get("l_mean_ms", float("nan"))) for x in use if x.get("l_mean_ms") is not None]),
        "l_p95_ms": _safe_mean([float(x.get("l_p95_ms", float("nan"))) for x in use if x.get("l_p95_ms") is not None]),
        "service_mean_ms": _safe_mean([float(x.get("service_mean_ms", float("nan"))) for x in use if x.get("service_mean_ms") is not None]),
        "queue_wait_mean_ms": _safe_mean([float(x.get("queue_wait_mean_ms", float("nan"))) for x in use if x.get("queue_wait_mean_ms") is not None]),
        "completions_tick": _safe_mean([float(x.get("completions_tick", float("nan"))) for x in use]),
    }


def _percentile(values: list[float], pct: float) -> float:
    finite = sorted(v for v in values if math.isfinite(v))
    if not finite:
        return float("nan")
    idx = min(len(finite) - 1, max(0, round((pct / 100.0) * (len(finite) - 1))))
    return finite[idx]


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _parse_controller_xml(xml_text: str) -> dict[str, Any]:
    root = ET.fromstring(xml_text)
    out: dict[str, Any] = {}
    for child in root:
        if list(child):
            out[child.tag] = {grand.tag: _coerce_xml_value(grand.text) for grand in child}
        else:
            out[child.tag] = _coerce_xml_value(child.text)
    return out


def _coerce_xml_value(value: str | None) -> Any:
    if value is None:
        return ""
    text = value.strip()
    if text == "":
        return ""
    try:
        as_float = float(text)
    except ValueError:
        return text
    if as_float.is_integer():
        return int(as_float)
    return as_float


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8019)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cuda")
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--initial-B", type=int, default=8)
    parser.add_argument("--tick-s", type=float, default=0.1)
    parser.add_argument("--log-dir", default="logs")
    args = parser.parse_args()

    random.seed(9)
    cfg = WorkloadConfig(device=args.device, dim=args.dim, layers=args.layers, dtype=args.dtype)
    workload = FixedMatmulWorkload(cfg)
    plant = Plant(workload, Path(args.log_dir), args.initial_B, args.tick_s)
    worker = threading.Thread(target=plant.worker_loop, daemon=True)
    worker.start()

    server = ThreadingHTTPServer((args.host, args.port), make_handler(plant))
    print(f"[ch9] serving on http://{args.host}:{args.port}", flush=True)
    print(f"[ch9] workload={asdict(cfg)}", flush=True)
    print(f"[ch9] logs={Path(args.log_dir).resolve()}", flush=True)

    def stop(_sig: int, _frame: Any) -> None:
        plant.close()
        server.shutdown()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    server.serve_forever()


if __name__ == "__main__":
    main()
