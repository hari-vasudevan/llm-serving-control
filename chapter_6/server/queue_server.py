#!/usr/bin/env python3
"""
queue_server.py  --  Chapter 6: Real LLM Queue Server

Sits between the controller and Ollama. Implements a REAL scheduler queue
so that l_total = queue_wait + TTFT is genuinely observable.

WHY THIS EXISTS (Chapter 5 lesson)
------------------------------------
In Chapter 5, we maintained a Python software FIFO queue, but requests
were dispatched to vLLM immediately within the same tick they arrived.
The actual queuing time was ~0ms because requests didn't genuinely wait --
they were dispatched before any latency accumulated.

Here, the queue_server is the dispatcher. Requests sit in the queue
until the dispatcher thread picks them up. The dispatcher fires exactly
B requests per tick (set by the controller). This means:

    l_total = (tick_of_dispatch - tick_of_arrival) * dt * 1000  [queue_wait]
            + (t_first_token - t_dispatch)                       [TTFT]

This is the genuine cascade plant:
    l_total(B, q) = alpha*B + gamma*B^2 + (q/B)*dt*1000

The controller observes l_total and q_sw via /metrics, and sets B via /control.

ARCHITECTURE
------------
  Client (controller or load generator)
    POST /enqueue          -- submit a prompt, get back when dispatched+complete
    GET  /metrics          -- queue depth, l_total stats, B setting
    POST /control          -- set B (batch size per dispatch tick)
    GET  /health           -- simple health check
    GET  /status           -- full server state

  Dispatcher thread (runs every dt seconds)
    - Dequeues min(B, len(queue)) requests
    - Fires them concurrently to Ollama
    - Records l_total = t_first_token - t_enqueue for each

  Metrics are exposed in both JSON (/metrics) and Prometheus text (/prom_metrics)
  so either the controller or a Prometheus scraper can read them.

FLOW
----
  1. Request arrives at /enqueue -> stamped t_enqueue -> pushed to FIFO
  2. Dispatcher wakes every dt -> dequeues B requests -> fires to Ollama
  3. Each response: t_first_token - t_enqueue = l_total -> stored in ring buffer
  4. Controller reads /metrics: q_sw, l_total_mean, l_total_p95, B_current
  5. Controller sends POST /control {"B": 5} to adjust dispatch rate

Usage:
    python3 queue_server.py [--port 8002] [--ollama_port 11434] [--model qwen2.5:0.5b]
    python3 queue_server.py --port 8002 --dt 1.0 --B_init 3 --B_max 8
"""

import argparse
import collections
import json
import math
import statistics
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

import requests

# ---------------------------------------------------------------------------
# Global state (protected by lock)
# ---------------------------------------------------------------------------
_lock         = threading.Lock()
_fifo         = collections.deque()          # (prompt, t_enqueue) pairs
_B            = 3                            # current dispatch batch size
_dt           = 1.0                          # dispatcher tick period [s]
_l_total_buf  = collections.deque(maxlen=200) # ring buffer of recent l_total values
_tick_count   = 0
_dispatched   = 0
_completed    = 0
_errors       = 0

# Per-tick stats for Prometheus-style histogram delta
_tick_l_sum   = 0.0
_tick_l_count = 0

# Config (set at startup)
_ollama_url   = "http://localhost:11434"
_model        = "qwen2.5:0.5b"
_B_min        = 1
_B_max        = 8


# ---------------------------------------------------------------------------
# Dispatcher thread
# ---------------------------------------------------------------------------
def dispatcher_loop():
    """
    Runs forever. Every _dt seconds, dequeues up to _B requests and
    fires them concurrently to Ollama. Records l_total for each response.
    """
    global _tick_count, _dispatched, _completed, _errors
    global _tick_l_sum, _tick_l_count

    print(f"[dispatcher] Started. dt={_dt}s  B_init={_B}", flush=True)

    while True:
        t_tick = time.perf_counter()

        with _lock:
            b_now   = _B
            # Dequeue up to b_now requests
            batch   = []
            while _fifo and len(batch) < b_now:
                batch.append(_fifo.popleft())
            _tick_count  += 1
            _dispatched  += len(batch)
            q_before      = len(_fifo)

        tick_num = _tick_count
        if batch:
            # Reset per-tick accumulator
            with _lock:
                _tick_l_sum   = 0.0
                _tick_l_count = 0

            results = []
            result_lock = threading.Lock()

            def fire_one(prompt, t_enqueue, idx=0):
                global _completed, _errors, _tick_l_sum, _tick_l_count
                try:
                    resp = requests.post(
                        f"{_ollama_url}/api/generate",
                        json={"model": _model, "prompt": prompt,
                              "stream": False, "options": {"num_predict": 1}},
                        timeout=60)
                    resp.raise_for_status()
                    t_done   = time.perf_counter()
                    l_total  = (t_done - t_enqueue) * 1000.0
                    with result_lock:
                        results.append(l_total)
                    with _lock:
                        _l_total_buf.append(l_total)
                        _tick_l_sum   += l_total
                        _tick_l_count += 1
                        _completed    += 1
                except Exception as ex:
                    with _lock:
                        _errors += 1
                    print(f"[dispatcher] Request error: {ex}", flush=True)

            threads = [
                threading.Thread(target=fire_one, args=(p, t), daemon=True)
                for p, t in batch
            ]
            for t in threads: t.start()
            for t in threads: t.join()

            mean_l = statistics.mean(results) if results else float("nan")
            print(
                f"[tick {tick_num:5d}]  q_before={q_before:4d}  "
                f"dispatched={len(batch)}  B={b_now}  "
                f"l_mean={mean_l:.0f}ms",
                flush=True)
        else:
            print(
                f"[tick {tick_num:5d}]  q_before={q_before:4d}  "
                f"dispatched=0  B={b_now}  (idle)",
                flush=True)

        # Sleep for remainder of tick
        elapsed = time.perf_counter() - t_tick
        sleep_time = _dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------
def get_metrics_dict():
    with _lock:
        q_sw    = len(_fifo)
        b_cur   = _B
        buf     = list(_l_total_buf)
        disp    = _dispatched
        comp    = _completed
        err     = _errors
        ticks   = _tick_count

    l_mean = statistics.mean(buf) if buf else float("nan")
    l_p95  = sorted(buf)[int(0.95 * len(buf))] if buf else float("nan")
    l_p99  = sorted(buf)[int(0.99 * len(buf))] if buf else float("nan")
    l_min  = min(buf) if buf else float("nan")
    l_max  = max(buf) if buf else float("nan")

    return {
        "q_sw":          q_sw,
        "B_current":     b_cur,
        "l_total_mean":  round(l_mean, 2) if not math.isnan(l_mean) else None,
        "l_total_p95":   round(l_p95,  2) if not math.isnan(l_p95)  else None,
        "l_total_p99":   round(l_p99,  2) if not math.isnan(l_p99)  else None,
        "l_total_min":   round(l_min,  2) if not math.isnan(l_min)  else None,
        "l_total_max":   round(l_max,  2) if not math.isnan(l_max)  else None,
        "n_in_buf":      len(buf),
        "ticks":         ticks,
        "dispatched":    disp,
        "completed":     comp,
        "errors":        err,
        "model":         _model,
        "dt":            _dt,
        "B_min":         _B_min,
        "B_max":         _B_max,
        "timestamp":     datetime.now().isoformat(),
    }


def get_prom_metrics():
    """Prometheus text format for compatibility with controllers that scrape /prom_metrics."""
    m = get_metrics_dict()
    lines = [
        "# HELP llm_queue_depth Current software FIFO queue depth",
        "# TYPE llm_queue_depth gauge",
        f"llm_queue_depth {m['q_sw']}",
        "",
        "# HELP llm_batch_size Current dispatch batch size B",
        "# TYPE llm_batch_size gauge",
        f"llm_batch_size {m['B_current']}",
        "",
        "# HELP llm_l_total_mean_ms Mean l_total over last 200 requests [ms]",
        "# TYPE llm_l_total_mean_ms gauge",
        f"llm_l_total_mean_ms {m['l_total_mean'] or 0}",
        "",
        "# HELP llm_l_total_p95_ms p95 l_total over last 200 requests [ms]",
        "# TYPE llm_l_total_p95_ms gauge",
        f"llm_l_total_p95_ms {m['l_total_p95'] or 0}",
        "",
        "# HELP llm_completed_total Total completed requests",
        "# TYPE llm_completed_total counter",
        f"llm_completed_total {m['completed']}",
        "",
        "# HELP llm_errors_total Total failed requests",
        "# TYPE llm_errors_total counter",
        f"llm_errors_total {m['errors']}",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
class QueueHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        pass   # suppress default access log; dispatcher prints its own

    def send_json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_text(self, code, text):
        body = text.encode()
        self.send_response(code)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self.send_json(200, {"status": "ok", "model": _model,
                                  "q_sw": len(_fifo), "B": _B})

        elif self.path == "/metrics":
            self.send_json(200, get_metrics_dict())

        elif self.path == "/prom_metrics":
            self.send_text(200, get_prom_metrics())

        elif self.path == "/status":
            m = get_metrics_dict()
            m["ollama_url"]  = _ollama_url
            m["fifo_sample"] = [p[:30] for p, _ in list(_fifo)[:5]]
            self.send_json(200, m)

        else:
            self.send_json(404, {"error": "not found"})

    def do_POST(self):
        length  = int(self.headers.get("Content-Length", 0))
        body    = self.rfile.read(length)

        if self.path == "/enqueue":
            # Enqueue a prompt and wait for it to complete.
            # Returns l_total (queue_wait + TTFT) in the response.
            try:
                data    = json.loads(body)
                prompt  = data.get("prompt", "Hello")
                t_enq   = time.perf_counter()
                done_ev = threading.Event()
                result  = [None]

                # We use a callback mechanism:
                # push (prompt, t_enqueue, done_event, result_list) onto FIFO
                with _lock:
                    _fifo.append((prompt, t_enq))

                # For now, return immediately with enqueue acknowledgement.
                # For synchronous l_total measurement, client should use /enqueue_sync.
                q_pos = len(_fifo)
                self.send_json(202, {
                    "status":    "enqueued",
                    "q_position": q_pos,
                    "t_enqueue": t_enq,
                })

            except Exception as e:
                self.send_json(400, {"error": str(e)})

        elif self.path == "/enqueue_sync":
            # Enqueue and BLOCK until the request completes.
            # Returns l_total measured from t_enqueue to t_first_token.
            # This is the correct measurement for the cascade plant model.
            try:
                data    = json.loads(body)
                prompt  = data.get("prompt", "Hello")
                t_enq   = time.perf_counter()
                done_ev = threading.Event()
                result  = {"l_total": None, "error": None}

                def callback(l_total_ms):
                    result["l_total"] = l_total_ms
                    done_ev.set()

                # Push to FIFO with callback
                with _lock:
                    _fifo.append((prompt, t_enq, callback))

                # Wait for dispatcher to pick it up and complete it
                timeout = data.get("timeout", 120)
                if done_ev.wait(timeout=timeout):
                    self.send_json(200, {
                        "status":  "ok",
                        "l_total": round(result["l_total"], 2),
                        "prompt":  prompt[:50],
                    })
                else:
                    self.send_json(504, {"error": "timeout", "timeout_s": timeout})

            except Exception as e:
                self.send_json(400, {"error": str(e)})

        elif self.path == "/control":
            # Set batch size B.  Called by the cascade controller each tick.
            # Body: {"B": 5}
            global _B
            try:
                data = json.loads(body)
                new_B = int(data.get("B", _B))
                new_B = max(_B_min, min(_B_max, new_B))
                with _lock:
                    old_B = _B
                    _B    = new_B
                self.send_json(200, {"ok": True, "B_old": old_B, "B_new": new_B})
            except Exception as e:
                self.send_json(400, {"error": str(e)})

        elif self.path == "/reset":
            # Clear the queue and reset metrics.  Useful between experiments.
            global _l_total_buf, _dispatched, _completed, _errors, _tick_count
            with _lock:
                _fifo.clear()
                _l_total_buf  = collections.deque(maxlen=200)
                _dispatched   = 0
                _completed    = 0
                _errors       = 0
                _tick_count   = 0
            self.send_json(200, {"ok": True, "message": "queue and metrics reset"})

        else:
            self.send_json(404, {"error": "not found"})


# ---------------------------------------------------------------------------
# Dispatcher with callback support
# ---------------------------------------------------------------------------
def dispatcher_loop_v2():
    """
    Enhanced dispatcher that supports callbacks for /enqueue_sync.
    Each queue item is either (prompt, t_enqueue) or (prompt, t_enqueue, callback).
    """
    global _tick_count, _dispatched, _completed, _errors

    print(f"[dispatcher] Started. dt={_dt}s  B_init={_B}  model={_model}", flush=True)

    while True:
        t_tick = time.perf_counter()

        with _lock:
            b_now  = _B
            batch  = []
            while _fifo and len(batch) < b_now:
                batch.append(_fifo.popleft())
            _tick_count += 1
            _dispatched += len(batch)
            q_before     = len(_fifo)

        if batch:
            result_lock = threading.Lock()
            results     = []

            def fire_one(item):
                global _completed, _errors
                # Unpack -- item may have 2 or 3 elements
                prompt    = item[0]
                t_enqueue = item[1]
                callback  = item[2] if len(item) > 2 else None

                try:
                    resp = requests.post(
                        f"{_ollama_url}/api/generate",
                        json={"model": _model, "prompt": prompt,
                              "stream": False, "options": {"num_predict": 1}},
                        timeout=60)
                    resp.raise_for_status()
                    t_done  = time.perf_counter()
                    l_total = (t_done - t_enqueue) * 1000.0

                    with _lock:
                        _l_total_buf.append(l_total)
                        _completed += 1
                    with result_lock:
                        results.append(l_total)

                    if callback:
                        callback(l_total)

                except Exception as ex:
                    with _lock:
                        _errors += 1
                    if callback:
                        callback(float("nan"))
                    print(f"[dispatcher] Error: {ex}", flush=True)

            threads = [
                threading.Thread(target=fire_one, args=(item,), daemon=True)
                for item in batch
            ]
            for t in threads: t.start()
            for t in threads: t.join()

            mean_l = statistics.mean(r for r in results if not math.isnan(r)) \
                     if results else float("nan")
            print(
                f"[tick {_tick_count:5d}]  q={q_before:3d}  "
                f"B={b_now}  dispatched={len(batch)}  "
                f"l_mean={mean_l:.0f}ms",
                flush=True)
        else:
            if _tick_count % 10 == 0:   # only print every 10 idle ticks
                print(
                    f"[tick {_tick_count:5d}]  q={q_before:3d}  B={b_now}  idle",
                    flush=True)

        elapsed = time.perf_counter() - t_tick
        if elapsed < _dt:
            time.sleep(_dt - elapsed)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global _ollama_url, _model, _B, _dt, _B_min, _B_max

    ap = argparse.ArgumentParser(description="Chapter 6 LLM queue server")
    ap.add_argument("--port",        type=int,   default=8002)
    ap.add_argument("--ollama_port", type=int,   default=11434)
    ap.add_argument("--ollama_host", type=str,   default="localhost")
    ap.add_argument("--model",       type=str,   default="qwen2.5:0.5b")
    ap.add_argument("--B_init",      type=int,   default=3)
    ap.add_argument("--B_min",       type=int,   default=1)
    ap.add_argument("--B_max",       type=int,   default=8)
    ap.add_argument("--dt",          type=float, default=1.0,
                    help="Dispatcher tick period [s]")
    args = ap.parse_args()

    _ollama_url = f"http://{args.ollama_host}:{args.ollama_port}"
    _model      = args.model
    _B          = args.B_init
    _B_min      = args.B_min
    _B_max      = args.B_max
    _dt         = args.dt

    print("=" * 60, flush=True)
    print(f"  Chapter 6 Queue Server", flush=True)
    print(f"  Listening on  : 0.0.0.0:{args.port}", flush=True)
    print(f"  Ollama        : {_ollama_url}", flush=True)
    print(f"  Model         : {_model}", flush=True)
    print(f"  B_init={_B}  B=[{_B_min},{_B_max}]  dt={_dt}s", flush=True)
    print("=" * 60, flush=True)

    # Verify Ollama is reachable
    try:
        r = requests.get(f"{_ollama_url}/api/tags", timeout=5)
        r.raise_for_status()
        print(f"[startup] Ollama OK at {_ollama_url}", flush=True)
    except Exception as e:
        print(f"[startup] WARNING: Ollama not reachable at {_ollama_url}: {e}", flush=True)
        print(f"[startup] Server will start anyway; dispatcher will retry.", flush=True)

    # Save PID
    import os
    with open("/tmp/queue_server.pid", "w") as f:
        f.write(str(os.getpid()))

    # Start dispatcher
    disp_thread = threading.Thread(target=dispatcher_loop_v2, daemon=True)
    disp_thread.start()

    # Start HTTP server
    server = HTTPServer(("0.0.0.0", args.port), QueueHandler)
    print(f"[startup] Queue server ready on port {args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[shutdown] Queue server stopped.", flush=True)


if __name__ == "__main__":
    main()
