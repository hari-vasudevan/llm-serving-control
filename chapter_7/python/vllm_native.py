#!/usr/bin/env python3
"""Shared helpers for Chapter 7 native-vLLM experiments."""

from __future__ import annotations

import json
import random
import re
import threading
import time
from typing import Iterable

import requests


PROMPT_SEEDS = [
    "Explain why feedback control can stabilize a service under disturbance.",
    "Summarize the difference between queueing delay and model TTFT.",
    "Describe why GPUs batch inference differently from CPUs.",
    "Explain how concurrency affects latency in an LLM serving system.",
    "What is the role of a scheduler in an inference server?",
    "Summarize why a latency target can be regulated with integral action.",
]


def auth_headers(api_key: str = "") -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def make_prompt(index: int, repeat: int = 64) -> str:
    seed = PROMPT_SEEDS[index % len(PROMPT_SEEDS)]
    body = " ".join([seed] * repeat)
    return (
        "You are part of a control-systems experiment for LLM serving. "
        "Respond in one short paragraph.\n\n"
        f"{body}"
    )


def parse_prom_metrics(raw_text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
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


def get_metrics(url: str, timeout: float = 10.0) -> dict[str, float]:
    raw = requests.get(f"{url}/metrics", timeout=timeout).text
    return parse_prom_metrics(raw)


def metric_delta_mean_ms(before: dict[str, float], after: dict[str, float], stem: str) -> float | None:
    sum_key = f"{stem}_sum"
    count_key = f"{stem}_count"
    d_sum = after.get(sum_key, 0.0) - before.get(sum_key, 0.0)
    d_count = after.get(count_key, 0.0) - before.get(count_key, 0.0)
    if d_count <= 0:
        return None
    return 1000.0 * d_sum / d_count


def wait_for_health(url: str, timeout: float = 180.0, poll_s: float = 2.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            resp = requests.get(f"{url}/health", timeout=5)
            if resp.ok:
                return True
        except Exception:
            pass
        time.sleep(poll_s)
    return False


def first_token_latency_ms(
    url: str,
    model: str,
    prompt: str,
    t_enqueue: float,
    *,
    max_tokens: int = 32,
    timeout: float = 120.0,
    api_key: str = "",
) -> float:
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "stream": True,
            "temperature": 0.0,
        }
    )
    with requests.post(
        f"{url}/v1/completions",
        data=body,
        headers=auth_headers(api_key),
        stream=True,
        timeout=timeout,
    ) as resp:
        resp.raise_for_status()
        for chunk in resp.iter_lines():
            if chunk and chunk != b"data: [DONE]":
                return (time.perf_counter() - t_enqueue) * 1000.0
    return (time.perf_counter() - t_enqueue) * 1000.0


def fire_burst(
    url: str,
    model: str,
    concurrency: int,
    *,
    prompt_repeat: int = 64,
    max_tokens: int = 32,
    timeout: float = 120.0,
    api_key: str = "",
    seed_offset: int = 0,
) -> list[float]:
    t_enqueue = time.perf_counter()
    results = [float("nan")] * concurrency

    def worker(i: int):
        prompt = make_prompt(seed_offset + i, repeat=prompt_repeat)
        try:
            results[i] = first_token_latency_ms(
                url,
                model,
                prompt,
                t_enqueue,
                max_tokens=max_tokens,
                timeout=timeout,
                api_key=api_key,
            )
        except Exception:
            results[i] = float("nan")

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(concurrency)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return results


class ResultsSink:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._buf: list[float] = []

    def append(self, value: float) -> None:
        with self._lock:
            self._buf.append(value)

    def drain(self) -> list[float]:
        with self._lock:
            out, self._buf = self._buf, []
        return out


class BackgroundLoad:
    """Simple rate-based background load against the same vLLM endpoint."""

    def __init__(
        self,
        url: str,
        model: str,
        *,
        api_key: str = "",
        prompt_repeat: int = 64,
        max_tokens: int = 32,
        max_outstanding: int = 64,
    ) -> None:
        self.url = url
        self.model = model
        self.api_key = api_key
        self.prompt_repeat = prompt_repeat
        self.max_tokens = max_tokens
        self._rate_rps = 0.0
        self._stop = threading.Event()
        self._sem = threading.Semaphore(max_outstanding)
        self._seed = 10_000
        self.sent = 0
        self.done = 0
        self.errors = 0
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def set_rate(self, rate_rps: float) -> None:
        self._rate_rps = max(0.0, float(rate_rps))

    def _fire_one(self, seed_idx: int) -> None:
        try:
            _ = first_token_latency_ms(
                self.url,
                self.model,
                make_prompt(seed_idx, repeat=self.prompt_repeat),
                time.perf_counter(),
                max_tokens=self.max_tokens,
                timeout=120.0,
                api_key=self.api_key,
            )
            self.done += 1
        except Exception:
            self.errors += 1
        finally:
            self._sem.release()

    def _run(self) -> None:
        next_launch = time.perf_counter()
        while not self._stop.is_set():
            rate = self._rate_rps
            if rate <= 0:
                time.sleep(0.1)
                next_launch = time.perf_counter()
                continue

            interval = 1.0 / rate
            now = time.perf_counter()
            if now < next_launch:
                time.sleep(min(0.05, next_launch - now))
                continue

            if not self._sem.acquire(timeout=0.2):
                continue

            seed_idx = self._seed
            self._seed += 1
            self.sent += 1
            thread = threading.Thread(target=self._fire_one, args=(seed_idx,), daemon=True)
            thread.start()
            next_launch += interval


def percentile(values: Iterable[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    idx = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * q))))
    return ordered[idx]


def jitter_prompt_offset() -> int:
    return random.randint(0, 1_000_000)
