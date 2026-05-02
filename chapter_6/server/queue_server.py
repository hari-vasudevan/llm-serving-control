#!/usr/bin/env python3
"""
queue_server.py  --  Chapter 6: Real LLM Queue Server

FIX v2: Added l_total_recent_mean based on last 10 completions only.
This prevents cold-start samples from poisoning the controller's
latency measurement for the first 100+ ticks.
"""

import argparse
import collections
import json
import math
import os
import statistics
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

import requests

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_lock        = threading.Lock()
_fifo        = collections.deque()
_B           = 2
_dt          = 1.0
_l_buf       = collections.deque(maxlen=200)   # long history for diagnostics
_l_recent    = collections.deque(maxlen=10)    # short window for controller
_tick        = 0
_dispatched  = 0
_completed   = 0
_errors      = 0

_ollama_url  = "http://localhost:11434"
_model       = "qwen2.5:0.5b"
_B_min       = 1
_B_max       = 8


def dispatcher():
    global _tick, _dispatched, _completed, _errors

    print(f"\n{'━'*64}", flush=True)
    print(f"  DISPATCHER STARTED  model={_model}  dt={_dt}s  B_max={_B_max}", flush=True)
    print(f"{'━'*64}\n", flush=True)

    while True:
        t_tick = time.perf_counter()

        with _lock:
            b_now = _B
            batch = []
            while _fifo and len(batch) < b_now:
                batch.append(_fifo.popleft())
            _tick       += 1
            _dispatched += len(batch)
            q_now        = len(_fifo)

        tick = _tick
        ts   = datetime.now().strftime("%H:%M:%S")

        if batch:
            print(f"\n[{ts}] ── Tick {tick:5d}  B={b_now}  dispatch={len(batch)}  q_after={q_now}",
                  flush=True)
            result_lock = threading.Lock()
            results     = []

            def fire(item, idx):
                global _completed, _errors
                prompt    = item[0]
                t_enqueue = item[1]
                callback  = item[2] if len(item) > 2 else None
                short     = prompt[:30].replace('\n', ' ')
                t_dispatch = time.perf_counter()
                q_wait_ms  = (t_dispatch - t_enqueue) * 1000.0
                print(f"  [{idx+1}] → '{short}'  q_wait={q_wait_ms:.0f}ms", flush=True)
                try:
                    resp = requests.post(
                        f"{_ollama_url}/api/generate",
                        json={"model": _model, "prompt": prompt,
                              "stream": False, "options": {"num_predict": 1}},
                        timeout=90)
                    resp.raise_for_status()
                    t_done  = time.perf_counter()
                    ttft_ms = (t_done - t_dispatch) * 1000.0
                    l_total = (t_done - t_enqueue) * 1000.0
                    with _lock:
                        _l_buf.append(l_total)
                        _l_recent.append(l_total)
                        _completed += 1
                    with result_lock:
                        results.append(l_total)
                    print(f"  [{idx+1}] ✓ '{short}'  TTFT={ttft_ms:.0f}ms  "
                          f"q_wait={q_wait_ms:.0f}ms  l_total={l_total:.0f}ms", flush=True)
                    if callback:
                        callback(l_total)
                except Exception as ex:
                    with _lock:
                        _errors += 1
                    print(f"  [{idx+1}] ✗ '{short}'  ERROR: {ex}", flush=True)
                    if callback:
                        callback(float("nan"))

            threads = [threading.Thread(target=fire, args=(item, i), daemon=True)
                       for i, item in enumerate(batch)]
            for t in threads: t.start()
            for t in threads: t.join()

            valid = [r for r in results if not math.isnan(r)]
            if valid:
                with _lock:
                    rec = list(_l_recent)
                rec_valid = [r for r in rec if not math.isnan(r)]
                rec_mean  = round(statistics.mean(rec_valid), 0) if rec_valid else float("nan")
                print(f"  ── tick {tick}: mean={statistics.mean(valid):.0f}ms  "
                      f"recent(n={len(rec_valid)})={rec_mean:.0f}ms  q={q_now}",
                      flush=True)
        else:
            if tick % 5 == 0:
                print(f"[{ts}] tick {tick:5d}  q=0  B={b_now}  (idle)", flush=True)

        elapsed = time.perf_counter() - t_tick
        if elapsed < _dt:
            time.sleep(_dt - elapsed)


def get_metrics():
    with _lock:
        q      = len(_fifo)
        b      = _B
        buf    = list(_l_buf)
        recent = list(_l_recent)
        d, c, e, t = _dispatched, _completed, _errors, _tick

    def safe(fn, lst):
        try: return round(fn(lst), 2) if lst else None
        except: return None

    rec_valid = [x for x in recent if not math.isnan(x)]

    return {
        "q_sw":                q,
        "B_current":           b,
        # Short window (last 10) -- USE THIS in the controller
        "l_total_recent_mean": safe(statistics.mean, rec_valid),
        "l_total_recent_n":    len(rec_valid),
        # Long window (last 200) -- for diagnostics only
        "l_total_mean":        safe(statistics.mean, buf),
        "l_total_p95":         safe(lambda x: sorted(x)[int(0.95*len(x))], buf),
        "l_total_std":         safe(statistics.stdev, buf) if len(buf) > 1 else None,
        "l_total_min":         safe(min, buf),
        "l_total_max":         safe(max, buf),
        "n_in_buf":            len(buf),
        "ticks":               t,
        "dispatched":          d,
        "completed":           c,
        "errors":              e,
        "model":               _model,
        "dt":                  _dt,
        "B_min":               _B_min,
        "B_max":               _B_max,
        "timestamp":           datetime.now().isoformat(),
    }


def prom_metrics():
    m = get_metrics()
    def g(name, help_str, val):
        return f"# HELP {name} {help_str}\n# TYPE {name} gauge\n{name} {val or 0}\n"
    return (
        g("llm_queue_depth",         "FIFO queue depth",               m["q_sw"]) +
        g("llm_batch_size",          "Dispatch batch size B",          m["B_current"]) +
        g("llm_l_total_recent_mean", "Mean l_total last 10 reqs [ms]", m["l_total_recent_mean"]) +
        g("llm_l_total_mean",        "Mean l_total last 200 reqs [ms]",m["l_total_mean"]) +
        g("llm_l_total_p95",         "p95 l_total last 200 reqs [ms]", m["l_total_p95"]) +
        f"# TYPE llm_completed_total counter\nllm_completed_total {m['completed']}\n"
        f"# TYPE llm_errors_total counter\nllm_errors_total {m['errors']}\n"
    )


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def json(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(b))
        self.end_headers()
        self.wfile.write(b)

    def text(self, code, s):
        b = s.encode()
        self.send_response(code)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", len(b))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if   self.path == "/health":
            self.json(200, {"status":"ok","model":_model,"q_sw":len(_fifo),"B":_B})
        elif self.path == "/metrics":
            self.json(200, get_metrics())
        elif self.path == "/prom_metrics":
            self.text(200, prom_metrics())
        elif self.path == "/status":
            m = get_metrics()
            m["ollama_url"]   = _ollama_url
            m["queue_sample"] = [item[0][:40] for item in list(_fifo)[:5]]
            self.json(200, m)
        else:
            self.json(404, {"error":"not found"})

    def do_POST(self):
        n    = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n)

        if self.path == "/enqueue":
            try:
                d = json.loads(body)
                t_enq = time.perf_counter()
                with _lock:
                    _fifo.append((d.get("prompt","Hello"), t_enq))
                self.json(202, {"status":"enqueued","q_sw":len(_fifo)})
            except Exception as e:
                self.json(400, {"error":str(e)})

        elif self.path == "/enqueue_sync":
            try:
                d = json.loads(body)
                t_enq = time.perf_counter()
                ev, res = threading.Event(), {}
                def cb(l): res["l_total"]=l; ev.set()
                with _lock:
                    _fifo.append((d.get("prompt","Hello"), t_enq, cb))
                if ev.wait(timeout=d.get("timeout",120)):
                    self.json(200, {"status":"ok","l_total":round(res["l_total"],2)})
                else:
                    self.json(504, {"error":"timeout"})
            except Exception as e:
                self.json(400, {"error":str(e)})

        elif self.path == "/control":
            global _B
            try:
                d = json.loads(body)
                newB = max(_B_min, min(_B_max, int(d.get("B",_B))))
                with _lock:
                    old, _B = _B, newB
                print(f"  [control] B: {old} → {newB}", flush=True)
                self.json(200, {"ok":True,"B_old":old,"B_new":newB})
            except Exception as e:
                self.json(400, {"error":str(e)})

        elif self.path == "/reset":
            global _l_buf, _l_recent, _dispatched, _completed, _errors, _tick
            with _lock:
                _fifo.clear()
                _l_buf      = collections.deque(maxlen=200)
                _l_recent   = collections.deque(maxlen=10)
                _dispatched = 0
                _completed  = 0
                _errors     = 0
                _tick       = 0
            print("  [reset] Cleared.", flush=True)
            self.json(200, {"ok":True})

        else:
            self.json(404, {"error":"not found"})


def main():
    global _ollama_url, _model, _B, _dt, _B_min, _B_max
    ap = argparse.ArgumentParser()
    ap.add_argument("--port",        type=int,   default=8002)
    ap.add_argument("--ollama_host", type=str,   default="localhost")
    ap.add_argument("--ollama_port", type=int,   default=11434)
    ap.add_argument("--model",       type=str,   default="qwen2.5:0.5b")
    ap.add_argument("--B_init",      type=int,   default=2)
    ap.add_argument("--B_min",       type=int,   default=1)
    ap.add_argument("--B_max",       type=int,   default=8)
    ap.add_argument("--dt",          type=float, default=1.0)
    args = ap.parse_args()
    _ollama_url = f"http://{args.ollama_host}:{args.ollama_port}"
    _model = args.model
    _B = args.B_init
    _B_min = args.B_min
    _B_max = args.B_max
    _dt = args.dt

    with open("/tmp/queue_server.pid","w") as f:
        f.write(str(os.getpid()))
    try:
        requests.get(f"{_ollama_url}/api/tags", timeout=5).raise_for_status()
        print(f"[startup] Ollama OK at {_ollama_url}", flush=True)
    except Exception as e:
        print(f"[startup] WARNING: {e}", flush=True)

    threading.Thread(target=dispatcher, daemon=True).start()
    server = HTTPServer(("0.0.0.0", args.port), Handler)
    print(f"[startup] Listening on 0.0.0.0:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[shutdown] Stopped.", flush=True)


if __name__ == "__main__":
    main()
