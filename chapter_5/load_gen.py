#!/usr/bin/env python3
"""
load_gen.py -- Background load generator for Chapter 5 characterisation.

Fires concurrent HTTP requests to vLLM at a fixed rate, continuously.
MATLAB reads /metrics each tick; this script produces the actual queue.

Usage:
  python3 load_gen.py --rate 10 --workers 12 --tokens 20 --port 8001 &
  echo $! > /tmp/load_gen.pid

  # Later, to stop:
  kill $(cat /tmp/load_gen.pid)

Arguments:
  --rate     target requests per second (default: 10)
  --workers  concurrent thread count    (default: 12)
  --tokens   max_tokens per request     (default: 20)
  --port     vLLM port                  (default: 8001)
  --duration seconds to run (0=forever) (default: 0)
  --pid_file write PID to file          (default: /tmp/load_gen.pid)
"""

import argparse
import threading
import time
import requests
import sys
import os
import signal
import random

URL   = None
MODEL = "mlx-community/Qwen3-0.6B-4bit"
PROMPTS = [
    "What is 2+2?",
    "Name a colour.",
    "What is the capital of France?",
    "How many days in a week?",
    "Name a planet.",
    "What is the speed of light?",
    "Name a mammal.",
    "What is 10 times 10?",
]

stop_event = threading.Event()
stats = {"sent": 0, "done": 0, "errors": 0}
stats_lock = threading.Lock()


def fire_one(tokens):
    prompt = random.choice(PROMPTS)
    try:
        resp = requests.post(URL, json={
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": tokens,
            "stream": False,
        }, timeout=30)
        resp.raise_for_status()
        with stats_lock:
            stats["done"] += 1
    except Exception as e:
        with stats_lock:
            stats["errors"] += 1


def worker_loop(tokens, rate_per_worker, interval):
    """Each worker fires one request, waits interval, repeats."""
    while not stop_event.is_set():
        with stats_lock:
            stats["sent"] += 1
        t = threading.Thread(target=fire_one, args=(tokens,), daemon=True)
        t.start()
        time.sleep(interval)


def stats_printer():
    while not stop_event.is_set():
        time.sleep(5)
        with stats_lock:
            print(f"[load_gen] sent={stats['sent']}  done={stats['done']}  errors={stats['errors']}", flush=True)


def main():
    global URL

    parser = argparse.ArgumentParser()
    parser.add_argument("--rate",     type=float, default=10.0)
    parser.add_argument("--workers",  type=int,   default=12)
    parser.add_argument("--tokens",   type=int,   default=20)
    parser.add_argument("--port",     type=int,   default=8001)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--pid_file", type=str,   default="/tmp/load_gen.pid")
    args = parser.parse_args()

    URL = f"http://localhost:{args.port}/v1/completions"

    # Write PID
    with open(args.pid_file, "w") as f:
        f.write(str(os.getpid()))
    print(f"[load_gen] PID={os.getpid()} written to {args.pid_file}", flush=True)
    print(f"[load_gen] rate={args.rate} req/s  workers={args.workers}  tokens={args.tokens}  port={args.port}", flush=True)

    # Handle Ctrl-C / kill
    def handle_signal(sig, frame):
        print("\n[load_gen] Stopping...", flush=True)
        stop_event.set()
        sys.exit(0)
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT,  handle_signal)

    # Each worker fires at rate/workers req/s
    interval = args.workers / args.rate   # seconds between fires per worker
    print(f"[load_gen] Each worker fires every {interval:.2f}s", flush=True)

    threads = []
    for _ in range(args.workers):
        t = threading.Thread(target=worker_loop, args=(args.tokens, args.rate/args.workers, interval), daemon=True)
        t.start()
        threads.append(t)
        time.sleep(interval / args.workers)  # stagger start slightly

    # Stats printer
    sp = threading.Thread(target=stats_printer, daemon=True)
    sp.start()

    print(f"[load_gen] Running. Stop with: kill $(cat {args.pid_file})", flush=True)

    if args.duration > 0:
        time.sleep(args.duration)
        stop_event.set()
    else:
        # Run until killed
        while not stop_event.is_set():
            time.sleep(1)

    print("[load_gen] Done.", flush=True)


if __name__ == "__main__":
    main()
