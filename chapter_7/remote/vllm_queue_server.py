#!/usr/bin/env python3
"""
vllm_queue_server.py  --  Chapter 7 remote queue server for GPU-backed vLLM

This keeps the Chapter 6 control surface:
  - POST /enqueue
  - POST /control   {"B": N}
  - GET  /metrics

but replaces Ollama with an OpenAI-compatible vLLM server on Linux/NVIDIA.
That lets the local controller keep the same "remote queue server" shape while
moving the actual inference work onto a real GPU host.
"""

import argparse
import collections
import json
import math
import os
import re
import statistics
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

import requests


_lock = threading.Lock()
_fifo = collections.deque()
_B = 2
_dt = 1.0
_l_buf = collections.deque(maxlen=200)
_ttft_buf = collections.deque(maxlen=200)
_ttft_recent = collections.deque(maxlen=20)
_tick = 0
_dispatched = 0
_completed = 0
_errors = 0
_backend_url = "http://localhost:8001"
_metrics_url = "http://localhost:8001/metrics"
_health_url = "http://localhost:8001/health"
_model = "Qwen/Qwen2.5-0.5B-Instruct"
_api_key = ""
_B_min = 1
_B_max = 8
_timeout = 90.0


def _headers():
    h = {"Content-Type": "application/json"}
    if _api_key:
        h["Authorization"] = f"Bearer {_api_key}"
    return h


def _parse_prometheus_metrics(raw_text):
    metrics = {}
    for line in raw_text.splitlines():
        if not line or line.startswith("#"):
            continue
        clean = re.sub(r"\{[^}]*\}", "", line).strip()
        parts = clean.split()
        if len(parts) < 2:
            continue
        try:
            metrics[parts[0]] = metrics.get(parts[0], 0.0) + float(parts[1])
        except ValueError:
            continue
    return metrics


def _fetch_backend_metrics():
    try:
        raw = requests.get(_metrics_url, timeout=5).text
        return _parse_prometheus_metrics(raw)
    except Exception:
        return {}


def _hist_mean_ms(metrics, stem):
    total = metrics.get(f"{stem}_sum")
    count = metrics.get(f"{stem}_count")
    if total is None or count in (None, 0):
        return None
    return round((total / count) * 1000.0, 2)


def dispatch_one(item, idx, result_lock, results):
    global _completed, _errors

    prompt = item[0]
    t_enq = item[1]
    cb = item[2] if len(item) > 2 else None
    short = prompt[:40].replace("\n", " ")
    t_disp = time.perf_counter()
    q_wait_ms = (t_disp - t_enq) * 1000.0

    body = {
        "model": _model,
        "prompt": prompt,
        "max_tokens": 1,
        "stream": True,
        "temperature": 0.0,
    }

    try:
        with requests.post(
            f"{_backend_url}/v1/completions",
            data=json.dumps(body),
            headers=_headers(),
            stream=True,
            timeout=_timeout,
        ) as resp:
            resp.raise_for_status()
            got_token = False
            for chunk in resp.iter_lines():
                if chunk and chunk != b"data: [DONE]":
                    t_first = time.perf_counter()
                    ttft_ms = (t_first - t_disp) * 1000.0
                    l_ms = (t_first - t_enq) * 1000.0
                    with _lock:
                        _ttft_buf.append(ttft_ms)
                        _ttft_recent.append(ttft_ms)
                        _l_buf.append(l_ms)
                        _completed += 1
                    with result_lock:
                        results.append((ttft_ms, l_ms))
                    got_token = True
                    if cb:
                        cb(l_ms)
                    print(
                        f"  [{idx+1}] ✓ '{short}'  TTFT={ttft_ms:.0f}ms  "
                        f"q_wait={q_wait_ms:.0f}ms  l_total={l_ms:.0f}ms",
                        flush=True,
                    )
                    break
            if not got_token:
                raise RuntimeError("stream ended before first token")
    except Exception as ex:
        with _lock:
            _errors += 1
        if cb:
            cb(float("nan"))
        print(f"  [{idx+1}] ✗ '{short}'  ERR: {ex}", flush=True)


def dispatcher():
    global _tick, _dispatched

    print(
        f"\n{'━'*68}\n"
        f"  DISPATCHER  model={_model}  dt={_dt}s  B=[{_B_min},{_B_max}]\n"
        f"  backend={_backend_url}\n"
        f"{'━'*68}\n",
        flush=True,
    )

    while True:
        t_tick = time.perf_counter()
        with _lock:
            b_now = _B
            batch = []
            while _fifo and len(batch) < b_now:
                batch.append(_fifo.popleft())
            _tick += 1
            _dispatched += len(batch)
            q_now = len(_fifo)

        ts = datetime.now().strftime("%H:%M:%S")
        if batch:
            print(f"\n[{ts}] Tick {_tick:5d}  B={b_now}  dispatch={len(batch)}  q={q_now}", flush=True)
            result_lock = threading.Lock()
            results = []
            threads = [
                threading.Thread(
                    target=dispatch_one,
                    args=(item, i, result_lock, results),
                    daemon=True,
                )
                for i, item in enumerate(batch)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            if results:
                ttfts = [r[0] for r in results]
                with _lock:
                    recent = [x for x in list(_ttft_recent) if not math.isnan(x)]
                print(
                    f"  ── TTFT mean={statistics.mean(ttfts):.0f}ms  "
                    f"ttft_recent(n={len(recent)})="
                    f"{statistics.mean(recent):.0f}ms  q={q_now}",
                    flush=True,
                )
        elif _tick % 5 == 0:
            print(f"[{ts}] Tick {_tick:5d}  q=0  B={b_now}  (idle)", flush=True)

        elapsed = time.perf_counter() - t_tick
        if elapsed < _dt:
            time.sleep(_dt - elapsed)


def get_metrics():
    backend = _fetch_backend_metrics()
    with _lock:
        q = len(_fifo)
        b = _B
        lbuf = list(_l_buf)
        tbuf = list(_ttft_buf)
        trec = [x for x in list(_ttft_recent) if not math.isnan(x)]
        tick = _tick
        dispatched = _dispatched
        completed = _completed
        errors = _errors

    def safe_mean(values):
        return round(statistics.mean(values), 2) if values else None

    def safe_p95(values):
        if not values:
            return None
        ordered = sorted(values)
        return round(ordered[int(0.95 * (len(ordered) - 1))], 2)

    return {
        "q_sw": q,
        "B_current": b,
        "ttft_recent_mean": safe_mean(trec),
        "ttft_recent_n": len(trec),
        "ttft_mean": safe_mean(tbuf),
        "ttft_p95": safe_p95(tbuf),
        "l_total_mean": safe_mean(lbuf),
        "l_total_p95": safe_p95(lbuf),
        "vllm_num_requests_waiting": backend.get("vllm:num_requests_waiting"),
        "vllm_num_requests_running": backend.get("vllm:num_requests_running"),
        "vllm_ttft_mean_ms": _hist_mean_ms(backend, "vllm:time_to_first_token_seconds"),
        "vllm_e2e_mean_ms": _hist_mean_ms(backend, "vllm:e2e_request_latency_seconds"),
        "vllm_queue_mean_ms": _hist_mean_ms(backend, "vllm:request_queue_time_seconds"),
        "ticks": tick,
        "dispatched": dispatched,
        "completed": completed,
        "errors": errors,
        "model": _model,
        "dt": _dt,
        "B_min": _B_min,
        "B_max": _B_max,
        "backend_url": _backend_url,
        "timestamp": datetime.now().isoformat(),
    }


def prom_metrics():
    m = get_metrics()

    def gauge(name, help_text, value):
        val = 0 if value is None else value
        return f"# HELP {name} {help_text}\n# TYPE {name} gauge\n{name} {val}\n"

    return (
        gauge("llm_queue_depth", "Wrapper FIFO depth", m["q_sw"]) +
        gauge("llm_batch_size", "Current batch size", m["B_current"]) +
        gauge("llm_ttft_recent_mean", "Recent wrapper TTFT in ms", m["ttft_recent_mean"]) +
        gauge("llm_vllm_waiting", "vLLM waiting requests", m["vllm_num_requests_waiting"]) +
        gauge("llm_vllm_running", "vLLM running requests", m["vllm_num_requests_running"]) +
        f"# TYPE llm_completed_total counter\nllm_completed_total {m['completed']}\n"
    )


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        return

    def _json(self, code, obj):
        payload = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _text(self, code, text):
        payload = text.encode()
        self.send_response(code)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"status": "ok", "model": _model, "q_sw": len(_fifo), "B": _B})
        elif self.path == "/metrics":
            self._json(200, get_metrics())
        elif self.path == "/prom_metrics":
            self._text(200, prom_metrics())
        elif self.path == "/status":
            m = get_metrics()
            m["queue_sample"] = [item[0][:50] for item in list(_fifo)[:5]]
            self._json(200, m)
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        global _B, _l_buf, _ttft_buf, _ttft_recent, _dispatched, _completed, _errors, _tick

        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n)
        if self.path == "/enqueue":
            try:
                data = json.loads(body) if body else {}
                prompt = data.get("prompt", "Hello")
                t_enq = time.perf_counter()
                with _lock:
                    _fifo.append((prompt, t_enq))
                self._json(202, {"status": "enqueued", "q_sw": len(_fifo)})
            except Exception as ex:
                self._json(400, {"error": str(ex)})
        elif self.path == "/enqueue_sync":
            try:
                data = json.loads(body) if body else {}
                prompt = data.get("prompt", "Hello")
                timeout = float(data.get("timeout", 120))
                event = threading.Event()
                result = {}

                def cb(latency):
                    result["l_total"] = latency
                    event.set()

                with _lock:
                    _fifo.append((prompt, time.perf_counter(), cb))
                if event.wait(timeout=timeout):
                    self._json(200, {"status": "ok", "l_total": round(result["l_total"], 2)})
                else:
                    self._json(504, {"error": "timeout"})
            except Exception as ex:
                self._json(400, {"error": str(ex)})
        elif self.path == "/control":
            try:
                data = json.loads(body) if body else {}
                new_b = max(_B_min, min(_B_max, int(data.get("B", _B))))
                with _lock:
                    old_b = _B
                    _B = new_b
                print(f"  [control] B: {old_b} -> {new_b}", flush=True)
                self._json(200, {"ok": True, "B_old": old_b, "B_new": new_b})
            except Exception as ex:
                self._json(400, {"error": str(ex)})
        elif self.path == "/reset":
            with _lock:
                _fifo.clear()
                _l_buf = collections.deque(maxlen=200)
                _ttft_buf = collections.deque(maxlen=200)
                _ttft_recent = collections.deque(maxlen=20)
                _dispatched = 0
                _completed = 0
                _errors = 0
                _tick = 0
            print("  [reset] Cleared.", flush=True)
            self._json(200, {"ok": True})
        else:
            self._json(404, {"error": "not found"})


def main():
    global _backend_url, _metrics_url, _health_url, _model, _B, _dt, _B_min, _B_max, _api_key, _timeout

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--backend-url", type=str, default="http://localhost:8001")
    ap.add_argument("--metrics-url", type=str, default="")
    ap.add_argument("--health-url", type=str, default="")
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--api-key", type=str, default="")
    ap.add_argument("--B-init", type=int, default=2)
    ap.add_argument("--B-min", type=int, default=1)
    ap.add_argument("--B-max", type=int, default=8)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--timeout", type=float, default=90.0)
    args = ap.parse_args()

    _backend_url = args.backend_url.rstrip("/")
    _metrics_url = args.metrics_url or f"{_backend_url}/metrics"
    _health_url = args.health_url or f"{_backend_url}/health"
    _model = args.model
    _api_key = args.api_key
    _B = args.B_init
    _B_min = args.B_min
    _B_max = args.B_max
    _dt = args.dt
    _timeout = args.timeout

    with open("/tmp/ch7_vllm_queue_server.pid", "w") as f:
        f.write(str(os.getpid()))

    try:
        requests.get(_health_url, timeout=5).raise_for_status()
        print(f"[startup] vLLM OK at {_backend_url}", flush=True)
    except Exception as ex:
        print(f"[startup] WARNING: backend health check failed: {ex}", flush=True)

    threading.Thread(target=dispatcher, daemon=True).start()
    server = HTTPServer(("0.0.0.0", args.port), Handler)
    print(f"[startup] Listening on 0.0.0.0:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[shutdown] Stopped.", flush=True)


if __name__ == "__main__":
    main()
