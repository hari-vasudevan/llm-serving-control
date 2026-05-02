#!/usr/bin/env python3
"""
queue_server.py  --  Chapter 6: Real LLM Queue Server  (v3)

KEY FIX: Exposes ttft_recent_mean (dispatch-to-completion, last 10).
The controller MUST use this, NOT l_total_recent_mean.

Why: l_total = queue_wait + TTFT. When queue is large, l_total is dominated
by queue_wait which INCREASES when B decreases (fewer dispatched per tick).
This inverts the control sign and makes the controller destabilising.

TTFT = dispatch-to-completion only. This is what B actually controls.
B larger -> TTFT larger (more CPU sharing). This relationship is stable.
"""

import argparse, collections, json, math, os, statistics, threading, time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
import requests

_lock       = threading.Lock()
_fifo       = collections.deque()
_B          = 2
_dt         = 1.0
_l_buf      = collections.deque(maxlen=200)   # l_total history (diagnostic)
_ttft_buf   = collections.deque(maxlen=200)   # TTFT history (diagnostic)
_ttft_recent= collections.deque(maxlen=10)    # TTFT last 10 -- USE IN CONTROLLER
_tick = 0; _dispatched = 0; _completed = 0; _errors = 0
_ollama_url = "http://localhost:11434"
_model      = "qwen2.5:0.5b"
_B_min = 1; _B_max = 8


def dispatcher():
    global _tick, _dispatched, _completed, _errors
    print(f"\n{'━'*64}\n  DISPATCHER  model={_model}  dt={_dt}s  B_max={_B_max}\n{'━'*64}\n",
          flush=True)

    while True:
        t_tick = time.perf_counter()
        with _lock:
            b_now = _B
            batch = []
            while _fifo and len(batch) < b_now:
                batch.append(_fifo.popleft())
            _tick += 1; _dispatched += len(batch); q_now = len(_fifo)

        ts = datetime.now().strftime("%H:%M:%S")

        if batch:
            print(f"\n[{ts}] Tick {_tick:5d}  B={b_now}  dispatch={len(batch)}  q={q_now}",
                  flush=True)
            result_lock = threading.Lock(); results = []

            def fire(item, idx):
                global _completed, _errors
                prompt = item[0]; t_enq = item[1]
                cb     = item[2] if len(item) > 2 else None
                short  = prompt[:28].replace('\n',' ')
                t_disp = time.perf_counter()
                qw_ms  = (t_disp - t_enq) * 1000.0
                print(f"  [{idx+1}] → '{short}'  q_wait={qw_ms:.0f}ms", flush=True)
                try:
                    resp = requests.post(f"{_ollama_url}/api/generate",
                        json={"model":_model,"prompt":prompt,
                              "stream":False,"options":{"num_predict":1}},
                        timeout=90)
                    resp.raise_for_status()
                    t_done  = time.perf_counter()
                    ttft_ms = (t_done - t_disp) * 1000.0   # dispatch → done
                    l_ms    = (t_done - t_enq)  * 1000.0   # enqueue  → done
                    with _lock:
                        _l_buf.append(l_ms)
                        _ttft_buf.append(ttft_ms)
                        _ttft_recent.append(ttft_ms)
                        _completed += 1
                    with result_lock:
                        results.append((ttft_ms, l_ms))
                    print(f"  [{idx+1}] ✓ '{short}'  TTFT={ttft_ms:.0f}ms  "
                          f"q_wait={qw_ms:.0f}ms  l_total={l_ms:.0f}ms", flush=True)
                    if cb: cb(l_ms)
                except Exception as ex:
                    with _lock: _errors += 1
                    print(f"  [{idx+1}] ✗ '{short}'  ERR:{ex}", flush=True)
                    if cb: cb(float("nan"))

            threads = [threading.Thread(target=fire, args=(item,i), daemon=True)
                       for i,item in enumerate(batch)]
            for t in threads: t.start()
            for t in threads: t.join()
            if results:
                tts = [r[0] for r in results]; lls = [r[1] for r in results]
                with _lock: rec = [x for x in list(_ttft_recent) if not math.isnan(x)]
                print(f"  ── TTFT mean={statistics.mean(tts):.0f}ms  "
                      f"ttft_recent(n={len(rec)})="
                      f"{statistics.mean(rec):.0f}ms  q={q_now}", flush=True)
        else:
            if _tick % 5 == 0:
                print(f"[{ts}] Tick {_tick:5d}  q=0  B={b_now}  (idle)", flush=True)

        elapsed = time.perf_counter() - t_tick
        if elapsed < _dt: time.sleep(_dt - elapsed)


def get_metrics():
    with _lock:
        q=len(_fifo); b=_B; lbuf=list(_l_buf); tbuf=list(_ttft_buf)
        trec=list(_ttft_recent); d=_dispatched; c=_completed; e=_errors; t=_tick
    def safe(fn,lst):
        try: return round(fn(lst),2) if lst else None
        except: return None
    trec_v = [x for x in trec if not math.isnan(x)]
    return {
        "q_sw":              q,
        "B_current":         b,
        # ── USE THESE IN THE CONTROLLER ──
        "ttft_recent_mean":  safe(statistics.mean, trec_v),   # TTFT last 10 (for control)
        "ttft_recent_n":     len(trec_v),
        # ── DIAGNOSTICS ──
        "ttft_mean":         safe(statistics.mean, tbuf),
        "ttft_p95":          safe(lambda x: sorted(x)[int(0.95*len(x))], tbuf),
        "l_total_mean":      safe(statistics.mean, lbuf),
        "l_total_p95":       safe(lambda x: sorted(x)[int(0.95*len(x))], lbuf),
        "l_total_recent_mean": safe(statistics.mean,
                                    [x for x in list(_ttft_recent) if not math.isnan(x)]),
        "n_buf":             len(tbuf),
        "ticks":t,"dispatched":d,"completed":c,"errors":e,
        "model":_model,"dt":_dt,"B_min":_B_min,"B_max":_B_max,
        "timestamp":datetime.now().isoformat(),
    }


def prom_metrics():
    m = get_metrics()
    def g(n,h,v): return f"# HELP {n} {h}\n# TYPE {n} gauge\n{n} {v or 0}\n"
    return (g("llm_queue_depth","FIFO depth",m["q_sw"]) +
            g("llm_batch_size","B",m["B_current"]) +
            g("llm_ttft_recent_mean","TTFT last 10 [ms]",m["ttft_recent_mean"]) +
            g("llm_ttft_mean","TTFT last 200 [ms]",m["ttft_mean"]) +
            g("llm_l_total_mean","l_total last 200 [ms]",m["l_total_mean"]) +
            f"# TYPE llm_completed_total counter\nllm_completed_total {m['completed']}\n")


class Handler(BaseHTTPRequestHandler):
    def log_message(self,*a): pass
    def _json(self,code,obj):
        b=json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type","application/json")
        self.send_header("Content-Length",len(b))
        self.end_headers(); self.wfile.write(b)
    def _text(self,code,s):
        b=s.encode()
        self.send_response(code)
        self.send_header("Content-Type","text/plain")
        self.send_header("Content-Length",len(b))
        self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if   self.path=="/health":
            self._json(200,{"status":"ok","model":_model,"q_sw":len(_fifo),"B":_B})
        elif self.path=="/metrics":      self._json(200,get_metrics())
        elif self.path=="/prom_metrics": self._text(200,prom_metrics())
        elif self.path=="/status":
            m=get_metrics(); m["ollama_url"]=_ollama_url
            m["queue_sample"]=[item[0][:40] for item in list(_fifo)[:5]]
            self._json(200,m)
        else: self._json(404,{"error":"not found"})
    def do_POST(self):
        n=int(self.headers.get("Content-Length",0)); body=self.rfile.read(n)
        if self.path=="/enqueue":
            try:
                d=json.loads(body); t_enq=time.perf_counter()
                with _lock: _fifo.append((d.get("prompt","Hello"),t_enq))
                self._json(202,{"status":"enqueued","q_sw":len(_fifo)})
            except Exception as e: self._json(400,{"error":str(e)})
        elif self.path=="/enqueue_sync":
            try:
                d=json.loads(body); t_enq=time.perf_counter()
                ev,res=threading.Event(),{}
                def cb(l): res["l_total"]=l; ev.set()
                with _lock: _fifo.append((d.get("prompt","Hello"),t_enq,cb))
                if ev.wait(timeout=d.get("timeout",120)):
                    self._json(200,{"status":"ok","l_total":round(res["l_total"],2)})
                else: self._json(504,{"error":"timeout"})
            except Exception as e: self._json(400,{"error":str(e)})
        elif self.path=="/control":
            global _B
            try:
                d=json.loads(body); newB=max(_B_min,min(_B_max,int(d.get("B",_B))))
                with _lock: old,_B=_B,newB
                print(f"  [control] B: {old} → {newB}",flush=True)
                self._json(200,{"ok":True,"B_old":old,"B_new":newB})
            except Exception as e: self._json(400,{"error":str(e)})
        elif self.path=="/reset":
            global _l_buf,_ttft_buf,_ttft_recent,_dispatched,_completed,_errors,_tick
            with _lock:
                _fifo.clear()
                _l_buf=collections.deque(maxlen=200); _ttft_buf=collections.deque(maxlen=200)
                _ttft_recent=collections.deque(maxlen=10)
                _dispatched=0; _completed=0; _errors=0; _tick=0
            print("  [reset] Cleared.",flush=True); self._json(200,{"ok":True})
        else: self._json(404,{"error":"not found"})


def main():
    global _ollama_url,_model,_B,_dt,_B_min,_B_max
    ap=argparse.ArgumentParser()
    ap.add_argument("--port",       type=int,  default=8002)
    ap.add_argument("--ollama_host",type=str,  default="localhost")
    ap.add_argument("--ollama_port",type=int,  default=11434)
    ap.add_argument("--model",      type=str,  default="qwen2.5:0.5b")
    ap.add_argument("--B_init",     type=int,  default=2)
    ap.add_argument("--B_min",      type=int,  default=1)
    ap.add_argument("--B_max",      type=int,  default=4)
    ap.add_argument("--dt",         type=float,default=1.0)
    args=ap.parse_args()
    _ollama_url=f"http://{args.ollama_host}:{args.ollama_port}"
    _model=args.model; _B=args.B_init; _B_min=args.B_min; _B_max=args.B_max; _dt=args.dt
    with open("/tmp/queue_server.pid","w") as f: f.write(str(os.getpid()))
    try:
        requests.get(f"{_ollama_url}/api/tags",timeout=5).raise_for_status()
        print(f"[startup] Ollama OK at {_ollama_url}",flush=True)
    except Exception as e: print(f"[startup] WARNING: {e}",flush=True)
    threading.Thread(target=dispatcher,daemon=True).start()
    server=HTTPServer(("0.0.0.0",args.port),Handler)
    print(f"[startup] Listening on 0.0.0.0:{args.port}",flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: print("\n[shutdown] Stopped.",flush=True)

if __name__=="__main__": main()
