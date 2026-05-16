#!/usr/bin/env python3
"""
Experimental vLLM scheduler hook for Chapter 10.

This module is loaded by vLLM through `--scheduler-cls`.

Intent:
  - sit at the lowest practical admission layer inside vLLM,
  - observe the scheduler waiting queue,
  - adjust the scheduler's admission capacity from a queue-wait setpoint,
  - leave Qwen execution, KV-cache management, and token generation to vLLM.

Important:
  vLLM's scheduler class is a private/unstable extension point. This module is
  written defensively and logs what it can observe. If a future vLLM version
  changes internal names, the class should fail open rather than crash serving.
"""

from __future__ import annotations

import logging
import os
import json
import time
from collections.abc import Iterable
from typing import Any


LOG = logging.getLogger("ch10.controlled_scheduler")


def _import_default_scheduler():
    candidates = [
        "vllm.v1.core.sched.scheduler",
        "vllm.core.scheduler",
    ]
    errors = []
    for module_name in candidates:
        try:
            module = __import__(module_name, fromlist=["Scheduler"])
            return getattr(module, "Scheduler")
        except Exception as exc:  # pragma: no cover - depends on vLLM version
            errors.append(f"{module_name}: {exc!r}")
    raise ImportError("Could not import vLLM Scheduler. Tried: " + "; ".join(errors))


DefaultScheduler = _import_default_scheduler()


class ControlledScheduler(DefaultScheduler):
    """A thin queue-wait controller around vLLM's default scheduler."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.target_queue_ms = float(os.getenv("CH10_TARGET_QUEUE_MS", "0"))
        self.control_file = os.getenv("CH10_CONTROL_FILE", "/tmp/ch10_scheduler_control.json")
        self.kp = float(os.getenv("CH10_QUEUE_KP", "0.02"))
        self.ki = float(os.getenv("CH10_QUEUE_KI", "0.002"))
        self.control_period_s = float(os.getenv("CH10_CONTROL_PERIOD_S", "0.25"))
        self.min_num_seqs = int(os.getenv("CH10_MIN_NUM_SEQS", "1"))
        self.max_num_seqs_override = _env_int_or_none("CH10_MAX_NUM_SEQS")
        self.enabled = os.getenv("CH10_SCHEDULER_ENABLED", "1") not in {"0", "false", "False"}

        self._last_control_t = 0.0
        self._last_control_file_mtime = 0.0
        self._xi = 0.0
        self._request_first_seen: dict[str, float] = {}
        self._nominal_max_num_seqs = self._read_scheduler_attr("max_num_seqs")
        self._nominal_max_num_batched_tokens = self._read_scheduler_attr("max_num_batched_tokens")
        self._last_budget = self._nominal_max_num_seqs or self.max_num_seqs_override

        LOG.warning(
            "CH10 ControlledScheduler loaded: enabled=%s target_queue_ms=%.2f "
            "kp=%.5f ki=%.5f nominal_max_num_seqs=%s nominal_max_num_batched_tokens=%s",
            self.enabled,
            self.target_queue_ms,
            self.kp,
            self.ki,
            self._nominal_max_num_seqs,
            self._nominal_max_num_batched_tokens,
        )

    def schedule(self, *args: Any, **kwargs: Any) -> Any:
        if self.enabled:
            self._control_admission_budget()
        return super().schedule(*args, **kwargs)

    def _control_admission_budget(self) -> None:
        now = time.monotonic()
        if now - self._last_control_t < self.control_period_s:
            return
        self._last_control_t = now
        self._load_control_file()

        waiting = list(self._waiting_requests())
        waiting_ids = [_request_id(req) for req in waiting]
        waiting_ids = [rid for rid in waiting_ids if rid]
        waiting_set = set(waiting_ids)

        for rid in waiting_ids:
            self._request_first_seen.setdefault(rid, now)
        for rid in list(self._request_first_seen):
            if rid not in waiting_set:
                del self._request_first_seen[rid]

        waits_ms = [(now - self._request_first_seen[rid]) * 1000.0 for rid in waiting_ids]
        mean_wait_ms = sum(waits_ms) / len(waits_ms) if waits_ms else 0.0
        oldest_wait_ms = max(waits_ms) if waits_ms else 0.0

        nominal = self.max_num_seqs_override or self._nominal_max_num_seqs
        if not nominal:
            LOG.debug("CH10 no max_num_seqs-like field found; skipping control")
            return

        # Positive error means the queue is waiting less than requested, so we
        # admit more slowly. Negative error means waiting is too high, so we
        # open the scheduler back up.
        error_ms = self.target_queue_ms - mean_wait_ms
        self._xi = _clamp(self._xi + error_ms * self.control_period_s, -20_000.0, 20_000.0)
        reduction = self.kp * error_ms + self.ki * self._xi
        budget = int(round(_clamp(nominal - reduction, self.min_num_seqs, nominal)))

        self._write_scheduler_attr("max_num_seqs", budget)
        self._last_budget = budget

        LOG.info(
            "CH10 scheduler control waiting=%d mean_wait_ms=%.2f oldest_wait_ms=%.2f "
            "target_ms=%.2f budget=%d nominal=%s xi=%.2f",
            len(waiting_ids),
            mean_wait_ms,
            oldest_wait_ms,
            self.target_queue_ms,
            budget,
            nominal,
            self._xi,
        )

    def _load_control_file(self) -> None:
        try:
            stat = os.stat(self.control_file)
        except FileNotFoundError:
            return
        except Exception as exc:
            LOG.debug("CH10 could not stat control file %s: %r", self.control_file, exc)
            return
        if stat.st_mtime <= self._last_control_file_mtime:
            return
        self._last_control_file_mtime = stat.st_mtime
        try:
            with open(self.control_file) as f:
                payload = json.load(f)
            if "target_queue_ms" in payload:
                self.target_queue_ms = float(payload["target_queue_ms"])
            if "enabled" in payload:
                self.enabled = bool(payload["enabled"])
            if "kp" in payload:
                self.kp = float(payload["kp"])
            if "ki" in payload:
                self.ki = float(payload["ki"])
            LOG.warning(
                "CH10 scheduler control updated from file: enabled=%s target_queue_ms=%.2f kp=%.5f ki=%.5f",
                self.enabled,
                self.target_queue_ms,
                self.kp,
                self.ki,
            )
        except Exception as exc:
            LOG.warning("CH10 failed to load control file %s: %r", self.control_file, exc)

    def _waiting_requests(self) -> Iterable[Any]:
        for name in ("waiting", "waiting_queue", "waiting_requests", "waiting_req_queue"):
            value = getattr(self, name, None)
            if value is not None:
                return _iter_collection(value)

        scheduler = getattr(self, "scheduler", None)
        if scheduler is not None:
            for name in ("waiting", "waiting_queue", "waiting_requests", "waiting_req_queue"):
                value = getattr(scheduler, name, None)
                if value is not None:
                    return _iter_collection(value)

        return []

    def _read_scheduler_attr(self, name: str) -> int | None:
        for holder_name in ("scheduler_config", "config"):
            holder = getattr(self, holder_name, None)
            if holder is not None and hasattr(holder, name):
                try:
                    return int(getattr(holder, name))
                except Exception:
                    return None
        if hasattr(self, name):
            try:
                return int(getattr(self, name))
            except Exception:
                return None
        return None

    def _write_scheduler_attr(self, name: str, value: int) -> None:
        for holder_name in ("scheduler_config", "config"):
            holder = getattr(self, holder_name, None)
            if holder is not None and hasattr(holder, name):
                try:
                    setattr(holder, name, int(value))
                    return
                except Exception as exc:
                    LOG.debug("CH10 could not set %s.%s: %r", holder_name, name, exc)
        if hasattr(self, name):
            try:
                setattr(self, name, int(value))
            except Exception as exc:
                LOG.debug("CH10 could not set self.%s: %r", name, exc)


def _env_int_or_none(name: str) -> int | None:
    raw = os.getenv(name)
    if raw in (None, ""):
        return None
    return int(raw)


def _iter_collection(value: Any) -> Iterable[Any]:
    if isinstance(value, dict):
        return value.values()
    try:
        return list(value)
    except TypeError:
        return []


def _request_id(req: Any) -> str:
    for name in ("request_id", "req_id", "id"):
        value = getattr(req, name, None)
        if value:
            return str(value)
    if isinstance(req, tuple) and req:
        return _request_id(req[0])
    if isinstance(req, dict):
        for name in ("request_id", "req_id", "id"):
            value = req.get(name)
            if value:
                return str(value)
    return str(id(req)) if req is not None else ""


def _clamp(value: float, lo: float, hi: float) -> float:
    return min(max(value, lo), hi)
