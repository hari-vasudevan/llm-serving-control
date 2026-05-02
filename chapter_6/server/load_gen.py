#!/usr/bin/env python3
"""
load_gen.py  --  Chapter 6: Test load generator for the queue server

Fires requests at a controlled rate to verify the queue is building up
and l_total measurements are correct.

Usage:
    python3 load_gen.py --rate 5 --duration 30
    python3 load_gen.py --rate 10 --duration 60 --server http://192.168.x.x:8002
"""

import argparse
import json
import math
import statistics
import time
import threading
import requests

PROMPTS = [
    "What is 2+2?", "Name a colour.", "Capital of France?",
    "Days in a week?", "Name a planet.", "Speed of light?",
    "Name a mammal.", "10 times 10?",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server",   default="http://localhost:8002")
    ap.add_argument("--rate",     type=float, default=5.0, help="requests/sec")
    ap.add_argument("--duration", type=float, default=30.0, help="seconds")
    args = ap.parse_args()

    interval = 1.0 / args.rate
    stop     = threading.Event()
    sent     = [0]

    def fire(i):
        prompt = PROMPTS[i % len(PROMPTS)]
        try:
            requests.post(f"{args.server}/enqueue",
                          json={"prompt": prompt}, timeout=10)
        except Exception:
            pass

    print(f"Firing at {args.rate} req/s for {args.duration}s -> {args.server}")
    print(f"Watching metrics every 5s...\n")

    # Fire in background
    def loader():
        i = 0
        while not stop.is_set():
            threading.Thread(target=fire, args=(i,), daemon=True).start()
            sent[0] += 1
            i += 1
            time.sleep(interval)

    threading.Thread(target=loader, daemon=True).start()

    # Print metrics every 5s
    t0 = time.time()
    while time.time() - t0 < args.duration:
        time.sleep(5)
        try:
            m = requests.get(f"{args.server}/metrics", timeout=3).json()
            print(f"  t={time.time()-t0:.0f}s  "
                  f"q={m['q_sw']}  "
                  f"B={m['B_current']}  "
                  f"l_mean={m['l_total_mean']}ms  "
                  f"l_p95={m['l_total_p95']}ms  "
                  f"sent={sent[0]}")
        except Exception as e:
            print(f"  metrics error: {e}")

    stop.set()
    time.sleep(1)

    # Final metrics
    try:
        m = requests.get(f"{args.server}/metrics", timeout=3).json()
        print(f"\nFinal: q={m['q_sw']}  completed={m['completed']}  "
              f"l_mean={m['l_total_mean']}ms  l_p95={m['l_total_p95']}ms")
    except Exception:
        pass


if __name__ == "__main__":
    main()
