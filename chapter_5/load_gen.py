#!/usr/bin/env python3
"""
load_gen.py -- Background load generator for Chapter 5 characterisation.

Fires concurrent HTTP requests to vLLM at a controlled rate.
Uses a semaphore to cap outstanding inflight requests, preventing
unbounded queue buildup in vLLM when requests are slow to complete.

Usage:
  python3 load_gen.py --rate 10 --max_inflight 12 --tokens 20 --port 8001 &
  echo $! > /tmp/load_gen.pid
  kill $(cat /tmp/load_gen.pid)   # to stop
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
stats      = {"sent": 0, "done": 0, "errors": 0}
stats_lock = threading.Lock()


def fire_one(semaphore, tokens):
    """Fire one request; release semaphore when done."""
    prompt = random.choice(PROMPTS)
    try:
        requests.post(URL, json={
            "model":      MODEL,
            "prompt":     prompt,
            "max_tokens": tokens,
            "stream":     False,
        }, timeout=30)
        with stats_lock:
            stats["done"] += 1
    except Exception:
        with stats_lock:
            stats["errors"] += 1
    finally:
        semaphore.release()


def main():
    global URL

    parser = argparse.ArgumentParser()
    parser.add_argument("--rate",        type=float, default=8.0,
                        help="target requests per second")
    parser.add_argument("--max_inflight",type=int,   default=12,
                        help="max concurrent outstanding requests (backpressure cap)")
    parser.add_argument("--tokens",      type=int,   default=20)
    parser.add_argument("--port",        type=int,   default=8001)
    parser.add_argument("--duration",    type=float, default=0.0,
                        help="seconds to run (0 = run until killed)")
    parser.add_argument("--pid_file",    type=str,   default="/tmp/load_gen.pid")
    args = parser.parse_args()

    URL = f"http://localhost:{args.port}/v1/completions"
    interval = 1.0 / args.rate      # seconds between request launches

    # Semaphore caps how many requests can be in-flight simultaneously.
    # This is the key backpressure mechanism that prevents queue explosion.
    sem = threading.Semaphore(args.max_inflight)

    with open(args.pid_file, "w") as f:
        f.write(str(os.getpid()))

    print(f"[load_gen] PID={os.getpid()}  rate={args.rate} req/s  "
          f"max_inflight={args.max_inflight}  tokens={args.tokens}", flush=True)
    print(f"[load_gen] interval={interval*1000:.0f}ms between launches", flush=True)

    def handle_signal(sig, frame):
        stop_event.set()
        sys.exit(0)
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT,  handle_signal)

    t_start   = time.time()
    t_stat    = t_start
    n_sent    = 0

    while not stop_event.is_set():
        if args.duration > 0 and (time.time() - t_start) >= args.duration:
            break

        # Acquire semaphore -- blocks if max_inflight already outstanding.
        # This is non-blocking with a short timeout so we can check stop_event.
        if not sem.acquire(timeout=0.5):
            continue

        with stats_lock:
            stats["sent"] += 1
        n_sent += 1

        t = threading.Thread(target=fire_one, args=(sem, args.tokens), daemon=True)
        t.start()

        # Stats print every 5 seconds
        if time.time() - t_stat >= 5:
            with stats_lock:
                print(f"[load_gen] sent={stats['sent']}  done={stats['done']}  "
                      f"errors={stats['errors']}  inflight≈{stats['sent']-stats['done']}",
                      flush=True)
            t_stat = time.time()

        time.sleep(interval)

    stop_event.set()
    print("[load_gen] Stopped.", flush=True)


if __name__ == "__main__":
    main()
