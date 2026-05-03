#!/usr/bin/env python3
"""
Chapter 8 Modal wrapper queue server.

This server restores a Chapter 2 style control surface:
  - MATLAB sets B through /control
  - MATLAB injects arrivals through /enqueue
  - the wrapper dispatches exactly B requests per tick into vLLM

The implementation is intentionally verbose so that `tail` on the Modal logs
shows what the Mac sent, what the server received, queue evolution, dispatch
activity, latency samples, and native vLLM metrics.
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
import urllib.parse
from dataclasses import dataclass, asdict
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

import requests


@dataclass
class QueueItem:
    request_id: str
    prompt: str
    prompt_chars: int
    prompt_repeat: int
    max_tokens: int
    temperature: float
    source: str
    client_ts: str
    enqueued_wall: str
    enqueued_perf: float


TRACE_PREFIX = "CH8"
LOCK = threading.Lock()
FIFO = collections.deque()
RECENT_EVENTS = collections.deque(maxlen=500)
RECENT_TICKS = collections.deque(maxlen=120)
L_MEAN_BUF = collections.deque(maxlen=300)
TTFT_BUF = collections.deque(maxlen=300)
QWAIT_BUF = collections.deque(maxlen=300)
ARRIVAL_TS = collections.deque(maxlen=400)
REQ_COUNTER = 0
TICK = 0
DISPATCHED = 0
COMPLETED = 0
ERRORS = 0
B = 4
DT = 1.0
B_MIN = 1
B_MAX = 50
MAX_TOKENS_DEFAULT = 32
PROMPT_REPEAT_DEFAULT = 192
TIMEOUT = 180.0
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
BACKEND_URL = "http://127.0.0.1:8001"
METRICS_URL = "http://127.0.0.1:8001/metrics"
HEALTH_URL = "http://127.0.0.1:8001/health"
API_KEY = ""
LAST_CONTROL_SOURCE = "startup"
LAST_CONTROL_TS = ""
QUEUE_AREA = 0.0
QUEUE_LAST_TS = time.perf_counter()
TICK_ARRIVALS = 0
TICK_COMPLETIONS = 0
TICK_Q_MAX = 0
LAST_TICK_SUMMARY = {
    "tick": 0,
    "q_mean_tick": 0.0,
    "q_max_tick": 0,
    "arrivals_tick": 0,
    "completions_tick": 0,
    "service_rate_tick": 0.0,
    "lambda_tick": 0.0,
}


def log(message):
    print(f"[{TRACE_PREFIX} {datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def short_prompt(text, limit=72):
    clean = text.replace("\n", " ").strip()
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3] + "..."


def parse_metrics_text(raw_text):
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


def hist_mean_ms(metrics, stem):
    total = metrics.get(f"{stem}_sum")
    count = metrics.get(f"{stem}_count")
    if total is None or count in (None, 0):
        return None
    return round((total / count) * 1000.0, 2)


def fetch_backend_metrics():
    try:
        raw = requests.get(METRICS_URL, timeout=5).text
        return parse_metrics_text(raw)
    except Exception as exc:
        log(f"metrics fetch failed: {exc}")
        return {}


def recent_arrival_rate():
    now = time.perf_counter()
    recent = [t for t in ARRIVAL_TS if now - t <= 10.0]
    if not recent:
        return 0.0
    return round(len(recent) / 10.0, 2)


def update_queue_area_locked(now=None):
    global QUEUE_AREA, QUEUE_LAST_TS
    if now is None:
        now = time.perf_counter()
    dt = now - QUEUE_LAST_TS
    if dt > 0:
        QUEUE_AREA += len(FIFO) * dt
        QUEUE_LAST_TS = now
    return now


def new_request_id():
    global REQ_COUNTER
    with LOCK:
        REQ_COUNTER += 1
        return f"r{REQ_COUNTER:06d}"


def build_queue_item(prompt, prompt_repeat, max_tokens, temperature, source, client_ts):
    expanded_prompt = prompt if prompt_repeat <= 1 else (prompt + " ") * prompt_repeat
    request_id = new_request_id()
    return QueueItem(
        request_id=request_id,
        prompt=expanded_prompt,
        prompt_chars=len(expanded_prompt),
        prompt_repeat=prompt_repeat,
        max_tokens=max_tokens,
        temperature=temperature,
        source=source,
        client_ts=client_ts,
        enqueued_wall=datetime.now().isoformat(),
        enqueued_perf=time.perf_counter(),
    )


def enqueue_item(item):
    global TICK_ARRIVALS, TICK_Q_MAX
    with LOCK:
        now = update_queue_area_locked()
        FIFO.append(item)
        ARRIVAL_TS.append(item.enqueued_perf)
        TICK_ARRIVALS += 1
        q_now = len(FIFO)
        TICK_Q_MAX = max(TICK_Q_MAX, q_now)
        RECENT_EVENTS.append(
            {
                "request_id": item.request_id,
                "event": "enqueue",
                "q_sw": q_now,
                "source": item.source,
                "prompt_chars": item.prompt_chars,
            }
        )
    return q_now


def safe_mean(values):
    return round(statistics.mean(values), 2) if values else None


def safe_p95(values):
    if not values:
        return None
    ordered = sorted(values)
    idx = int(0.95 * (len(ordered) - 1))
    return round(ordered[idx], 2)


def build_metrics():
    backend = fetch_backend_metrics()
    with LOCK:
        q = len(FIFO)
        b = B
        dispatched = DISPATCHED
        completed = COMPLETED
        errors = ERRORS
        tick = TICK
        last_control_source = LAST_CONTROL_SOURCE
        last_control_ts = LAST_CONTROL_TS
        last_tick = dict(LAST_TICK_SUMMARY)
        latencies = list(L_MEAN_BUF)
        ttfts = list(TTFT_BUF)
        qwaits = list(QWAIT_BUF)
        recent_events = list(RECENT_EVENTS)[-10:]
        recent_ticks = list(RECENT_TICKS)[-5:]

    metrics = {
        "status": "ok",
        "model": MODEL,
        "backend_url": BACKEND_URL,
        "q_sw": q,
        "B_current": b,
        "B_min": B_MIN,
        "B_max": B_MAX,
        "dt": DT,
        "ticks": tick,
        "dispatched": dispatched,
        "completed": completed,
        "errors": errors,
        "lambda_10s_est": recent_arrival_rate(),
        "l_mean_ms": safe_mean(latencies),
        "l_p95_ms": safe_p95(latencies),
        "ttft_mean_ms": safe_mean(ttfts),
        "ttft_p95_ms": safe_p95(ttfts),
        "queue_wait_mean_ms": safe_mean(qwaits),
        "queue_wait_p95_ms": safe_p95(qwaits),
        "q_mean_tick": last_tick["q_mean_tick"],
        "q_max_tick": last_tick["q_max_tick"],
        "arrivals_tick": last_tick["arrivals_tick"],
        "completions_tick": last_tick["completions_tick"],
        "service_rate_tick": last_tick["service_rate_tick"],
        "lambda_tick": last_tick["lambda_tick"],
        "vllm_num_requests_waiting": backend.get("vllm:num_requests_waiting"),
        "vllm_num_requests_running": backend.get("vllm:num_requests_running"),
        "vllm_ttft_mean_ms": hist_mean_ms(backend, "vllm:time_to_first_token_seconds"),
        "vllm_e2e_mean_ms": hist_mean_ms(backend, "vllm:e2e_request_latency_seconds"),
        "vllm_queue_mean_ms": hist_mean_ms(backend, "vllm:request_queue_time_seconds"),
        "last_control_source": last_control_source,
        "last_control_ts": last_control_ts,
        "recent_events": recent_events,
        "recent_ticks": recent_ticks,
        "timestamp": datetime.now().isoformat(),
    }
    return metrics


def prom_metrics_text():
    m = build_metrics()
    lines = []
    gauges = {
        "ch8_q_sw": m["q_sw"],
        "ch8_q_mean_tick": m["q_mean_tick"],
        "ch8_q_max_tick": m["q_max_tick"],
        "ch8_B_current": m["B_current"],
        "ch8_lambda_10s_est": m["lambda_10s_est"],
        "ch8_lambda_tick": m["lambda_tick"],
        "ch8_arrivals_tick": m["arrivals_tick"],
        "ch8_completions_tick": m["completions_tick"],
        "ch8_service_rate_tick": m["service_rate_tick"],
        "ch8_l_mean_ms": m["l_mean_ms"] or 0,
        "ch8_ttft_mean_ms": m["ttft_mean_ms"] or 0,
        "ch8_queue_wait_mean_ms": m["queue_wait_mean_ms"] or 0,
        "ch8_vllm_num_requests_waiting": m["vllm_num_requests_waiting"] or 0,
        "ch8_vllm_num_requests_running": m["vllm_num_requests_running"] or 0,
    }
    for name, value in gauges.items():
        lines.append(f"# TYPE {name} gauge")
        lines.append(f"{name} {value}")
    return "\n".join(lines) + "\n"


def dispatch_one(item, batch_index, result_lock, results):
    global COMPLETED, ERRORS, TICK_COMPLETIONS

    body = {
        "model": MODEL,
        "prompt": item.prompt,
        "max_tokens": item.max_tokens,
        "stream": True,
        "temperature": item.temperature,
    }
    t_dispatch = time.perf_counter()
    q_wait_ms = (t_dispatch - item.enqueued_perf) * 1000.0
    log(
        "dispatch request_id=%s batch_idx=%d q_wait=%.0fms prompt_chars=%d max_tokens=%d"
        % (item.request_id, batch_index, q_wait_ms, item.prompt_chars, item.max_tokens)
    )

    try:
        with requests.post(
            f"{BACKEND_URL}/v1/completions",
            data=json.dumps(body),
            headers=headers(),
            stream=True,
            timeout=TIMEOUT,
        ) as resp:
            resp.raise_for_status()
            first_token = False
            for chunk in resp.iter_lines():
                if chunk and chunk != b"data: [DONE]":
                    t_first = time.perf_counter()
                    ttft_ms = (t_first - t_dispatch) * 1000.0
                    l_total_ms = (t_first - item.enqueued_perf) * 1000.0
                    with LOCK:
                        TTFT_BUF.append(ttft_ms)
                        L_MEAN_BUF.append(l_total_ms)
                        QWAIT_BUF.append(q_wait_ms)
                        COMPLETED += 1
                        TICK_COMPLETIONS += 1
                        RECENT_EVENTS.append(
                            {
                                "request_id": item.request_id,
                                "event": "complete",
                                "ttft_ms": round(ttft_ms, 2),
                                "l_total_ms": round(l_total_ms, 2),
                                "q_wait_ms": round(q_wait_ms, 2),
                            }
                        )
                    with result_lock:
                        results.append((ttft_ms, l_total_ms, q_wait_ms))
                    log(
                        "complete request_id=%s ttft=%.0fms q_wait=%.0fms l_total=%.0fms prompt='%s'"
                        % (
                            item.request_id,
                            ttft_ms,
                            q_wait_ms,
                            l_total_ms,
                            short_prompt(item.prompt),
                        )
                    )
                    first_token = True
                    break
            if not first_token:
                raise RuntimeError("stream ended before first token")
    except Exception as exc:
        with LOCK:
            ERRORS += 1
            RECENT_EVENTS.append(
                {
                    "request_id": item.request_id,
                    "event": "error",
                    "message": str(exc),
                }
            )
        log(f"error request_id={item.request_id} err={exc}")


def dispatcher():
    global TICK, DISPATCHED, TICK_ARRIVALS, TICK_COMPLETIONS, TICK_Q_MAX, LAST_TICK_SUMMARY
    log(
        "dispatcher start backend=%s model=%s dt=%.2fs B=[%d,%d]"
        % (BACKEND_URL, MODEL, DT, B_MIN, B_MAX)
    )
    tick_index = 0
    tick_start = time.perf_counter()
    with LOCK:
        update_queue_area_locked(tick_start)
        area_start = QUEUE_AREA
        TICK_ARRIVALS = 0
        TICK_COMPLETIONS = 0
        TICK_Q_MAX = len(FIFO)
    while True:
        t0 = time.perf_counter()
        tick_index += 1
        with LOCK:
            update_queue_area_locked(t0)
            b_now = B
            batch = []
            while FIFO and len(batch) < b_now:
                batch.append(FIFO.popleft())
            update_queue_area_locked()
            q_after_pop = len(FIFO)
            TICK = tick_index
            tick_now = tick_index
            DISPATCHED += len(batch)

        if batch:
            log(
                "tick=%d dispatch=%d q_after_pop=%d B=%d lambda_10s=%.2f"
                % (tick_now, len(batch), q_after_pop, b_now, recent_arrival_rate())
            )
            result_lock = threading.Lock()
            results = []
            threads = [
                threading.Thread(
                    target=dispatch_one,
                    args=(item, i + 1, result_lock, results),
                    daemon=True,
                )
                for i, item in enumerate(batch)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            if results:
                ttfts = [r[0] for r in results]
                lats = [r[1] for r in results]
                waits = [r[2] for r in results]
                backend = fetch_backend_metrics()
                log(
                    "tick=%d summary ttft_mean=%.0fms l_mean=%.0fms q_wait_mean=%.0fms "
                    "vllm_waiting=%s vllm_running=%s"
                    % (
                        tick_now,
                        statistics.mean(ttfts),
                        statistics.mean(lats),
                        statistics.mean(waits),
                        backend.get("vllm:num_requests_waiting"),
                        backend.get("vllm:num_requests_running"),
                    )
                )
        elif tick_now % 5 == 0:
            backend = fetch_backend_metrics()
            log(
                "tick=%d idle q=0 B=%d lambda_10s=%.2f vllm_waiting=%s vllm_running=%s"
                % (
                    tick_now,
                    b_now,
                    recent_arrival_rate(),
                    backend.get("vllm:num_requests_waiting"),
                    backend.get("vllm:num_requests_running"),
                )
            )

        elapsed = time.perf_counter() - t0
        if elapsed < DT:
            time.sleep(DT - elapsed)
        tick_end = time.perf_counter()
        with LOCK:
            update_queue_area_locked(tick_end)
            tick_area = QUEUE_AREA - area_start
            tick_elapsed = max(tick_end - tick_start, 1e-6)
            q_mean_tick = tick_area / tick_elapsed
            tick_summary = {
                "tick": tick_now,
                "q_mean_tick": round(q_mean_tick, 2),
                "q_max_tick": int(TICK_Q_MAX),
                "arrivals_tick": int(TICK_ARRIVALS),
                "completions_tick": int(TICK_COMPLETIONS),
                "service_rate_tick": round(TICK_COMPLETIONS / tick_elapsed, 2),
                "lambda_tick": round(TICK_ARRIVALS / tick_elapsed, 2),
            }
            LAST_TICK_SUMMARY = tick_summary
            RECENT_TICKS.append(tick_summary)
            area_start = QUEUE_AREA
            tick_start = tick_end
            TICK_ARRIVALS = 0
            TICK_COMPLETIONS = 0
            TICK_Q_MAX = len(FIFO)
        if batch or tick_summary["arrivals_tick"] > 0 or tick_summary["q_max_tick"] > 0:
            log(
                "tick=%d plant q_mean=%.2f q_max=%d arrivals=%d completions=%d service_rate=%.2f"
                % (
                    tick_now,
                    tick_summary["q_mean_tick"],
                    tick_summary["q_max_tick"],
                    tick_summary["arrivals_tick"],
                    tick_summary["completions_tick"],
                    tick_summary["service_rate_tick"],
                )
            )


def headers():
    h = {"Content-Type": "application/json"}
    if API_KEY:
        h["Authorization"] = f"Bearer {API_KEY}"
    return h


def parse_json(handler):
    length = int(handler.headers.get("Content-Length", "0") or "0")
    raw = handler.rfile.read(length).decode("utf-8") if length > 0 else "{}"
    if not raw.strip():
        return {}
    return json.loads(raw)


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class Handler(BaseHTTPRequestHandler):
    server_version = "Chapter8Wrapper/0.1"

    def log_message(self, fmt, *args):
        log("http " + fmt % args)

    def _send_json(self, status, payload):
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _send_text(self, status, payload):
        raw = payload.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/health":
            health = {"status": "ok", "model": MODEL, "q_sw": len(FIFO), "B": B}
            self._send_json(200, health)
            return
        if parsed.path == "/metrics":
            self._send_json(200, build_metrics())
            return
        if parsed.path == "/metrics/prom":
            self._send_text(200, prom_metrics_text())
            return
        self._send_json(404, {"status": "error", "message": "not found"})

    def do_POST(self):
        global B, LAST_CONTROL_SOURCE, LAST_CONTROL_TS

        parsed = urllib.parse.urlparse(self.path)
        try:
            body = parse_json(self)
        except Exception as exc:
            self._send_json(400, {"status": "error", "message": str(exc)})
            return

        if parsed.path == "/enqueue":
            prompt = body.get("prompt", "")
            prompt_repeat = int(body.get("prompt_repeat", PROMPT_REPEAT_DEFAULT))
            max_tokens = int(body.get("max_tokens", MAX_TOKENS_DEFAULT))
            temperature = float(body.get("temperature", 0.0))
            source = body.get("source", "matlab")
            client_ts = body.get("client_ts", "")
            item = build_queue_item(
                prompt=prompt,
                prompt_repeat=prompt_repeat,
                max_tokens=max_tokens,
                temperature=temperature,
                source=source,
                client_ts=client_ts,
            )
            q_now = enqueue_item(item)
            request_id = item.request_id
            enqueued_wall = item.enqueued_wall
            log(
                "recv enqueue request_id=%s client=%s q=%d prompt_chars=%d repeat=%d max_tokens=%d client_ts=%s prompt='%s'"
                % (
                    request_id,
                    self.client_address[0],
                    q_now,
                    item.prompt_chars,
                    prompt_repeat,
                    max_tokens,
                    client_ts,
                    short_prompt(prompt),
                )
            )
            self._send_json(
                200,
                {
                    "status": "queued",
                    "request_id": request_id,
                    "q_sw": q_now,
                    "timestamp": enqueued_wall,
                },
            )
            return

        if parsed.path == "/enqueue_batch":
            prompt = body.get("prompt", "")
            count = int(body.get("count", 1))
            prompt_repeat = int(body.get("prompt_repeat", PROMPT_REPEAT_DEFAULT))
            max_tokens = int(body.get("max_tokens", MAX_TOKENS_DEFAULT))
            temperature = float(body.get("temperature", 0.0))
            source = body.get("source", "matlab_batch")
            client_ts = body.get("client_ts", "")
            count = max(1, min(count, 1000))

            request_ids = []
            for _ in range(count):
                item = build_queue_item(
                    prompt=prompt,
                    prompt_repeat=prompt_repeat,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    source=source,
                    client_ts=client_ts,
                )
                q_now = enqueue_item(item)
                request_ids.append(item.request_id)

            log(
                "recv enqueue_batch client=%s count=%d q=%d prompt_chars=%d repeat=%d max_tokens=%d source=%s prompt='%s'"
                % (
                    self.client_address[0],
                    count,
                    q_now,
                    len(prompt) * max(prompt_repeat, 1),
                    prompt_repeat,
                    max_tokens,
                    source,
                    short_prompt(prompt),
                )
            )
            self._send_json(
                200,
                {
                    "status": "queued_batch",
                    "count": count,
                    "first_request_id": request_ids[0],
                    "last_request_id": request_ids[-1],
                    "q_sw": q_now,
                    "timestamp": datetime.now().isoformat(),
                },
            )
            return

        if parsed.path == "/control":
            b_new = int(body.get("B", B))
            b_new = max(B_MIN, min(B_MAX, b_new))
            source = body.get("source", "matlab")
            note = body.get("note", "")
            with LOCK:
                B = b_new
                LAST_CONTROL_SOURCE = source
                LAST_CONTROL_TS = datetime.now().isoformat()
                q_now = len(FIFO)
                RECENT_EVENTS.append(
                    {
                        "event": "control",
                        "B": B,
                        "source": source,
                        "q_sw": q_now,
                    }
                )
            log(
                "recv control client=%s set_B=%d q=%d source=%s note='%s'"
                % (self.client_address[0], b_new, q_now, source, note)
            )
            self._send_json(200, {"status": "ok", "B": b_new, "q_sw": q_now})
            return

        if parsed.path == "/reset":
            global DISPATCHED, COMPLETED, ERRORS, TICK
            global QUEUE_AREA, QUEUE_LAST_TS, TICK_ARRIVALS, TICK_COMPLETIONS, TICK_Q_MAX, LAST_TICK_SUMMARY
            with LOCK:
                update_queue_area_locked()
                FIFO.clear()
                update_queue_area_locked()
                L_MEAN_BUF.clear()
                TTFT_BUF.clear()
                QWAIT_BUF.clear()
                ARRIVAL_TS.clear()
                RECENT_EVENTS.clear()
                RECENT_TICKS.clear()
                DISPATCHED = 0
                COMPLETED = 0
                ERRORS = 0
                TICK = 0
                QUEUE_AREA = 0.0
                QUEUE_LAST_TS = time.perf_counter()
                TICK_ARRIVALS = 0
                TICK_COMPLETIONS = 0
                TICK_Q_MAX = 0
                LAST_TICK_SUMMARY = {
                    "tick": 0,
                    "q_mean_tick": 0.0,
                    "q_max_tick": 0,
                    "arrivals_tick": 0,
                    "completions_tick": 0,
                    "service_rate_tick": 0.0,
                    "lambda_tick": 0.0,
                }
            log(f"recv reset client={self.client_address[0]} queue, buffers, and counters cleared")
            self._send_json(200, {"status": "ok"})
            return

        self._send_json(404, {"status": "error", "message": "not found"})


def wait_for_backend():
    log(f"waiting for backend health at {HEALTH_URL}")
    for _ in range(240):
        try:
            resp = requests.get(HEALTH_URL, timeout=5)
            if resp.ok:
                log("backend healthy")
                return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("backend did not become healthy")


def main():
    global TRACE_PREFIX, MODEL, BACKEND_URL, METRICS_URL, HEALTH_URL
    global B, B_MIN, B_MAX, DT, API_KEY, MAX_TOKENS_DEFAULT, PROMPT_REPEAT_DEFAULT, TIMEOUT

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--backend-url", default="http://127.0.0.1:8001")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--B-init", type=int, default=4)
    parser.add_argument("--B-min", type=int, default=1)
    parser.add_argument("--B-max", type=int, default=50)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--prompt-repeat", type=int, default=192)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--trace-prefix", default="CH8")
    args = parser.parse_args()

    TRACE_PREFIX = args.trace_prefix
    MODEL = args.model
    BACKEND_URL = args.backend_url
    METRICS_URL = f"{BACKEND_URL}/metrics"
    HEALTH_URL = f"{BACKEND_URL}/health"
    B = args.B_init
    B_MIN = args.B_min
    B_MAX = args.B_max
    DT = args.dt
    API_KEY = args.api_key
    MAX_TOKENS_DEFAULT = args.max_tokens
    PROMPT_REPEAT_DEFAULT = args.prompt_repeat
    TIMEOUT = args.timeout

    wait_for_backend()
    threading.Thread(target=dispatcher, daemon=True).start()
    server = ThreadedHTTPServer((args.host, args.port), Handler)
    log(
        "http server start host=%s port=%d backend=%s B_init=%d B_max=%d dt=%.2f"
        % (args.host, args.port, BACKEND_URL, B, B_MAX, DT)
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
