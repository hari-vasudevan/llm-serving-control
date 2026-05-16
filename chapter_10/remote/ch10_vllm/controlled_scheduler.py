#!/usr/bin/env python3
"""
Experimental vLLM scheduler hook for Chapter 10.

This module is loaded by vLLM through `--scheduler-cls`.

Intent:
  - sit at the lowest practical admission layer inside vLLM,
  - observe the scheduler waiting queue,
  - adjust the scheduler's admission capacity from a queue-wait setpoint,
  - leave Qwen execution, KV-cache management, and token generation to vLLM.

Actuator strategy (session 2 fix):
  The v1 scheduler's schedule() reads self.max_num_scheduled_tokens (token
  budget) and self.max_num_running_reqs (sequence count ceiling) each call.
  These are the real admission gates. We temporarily cap both before calling
  super().schedule(), then restore. This makes the parent's own admission
  logic respect our budget without reaching into its internals.

  The old approach (mutating scheduler_config.max_num_seqs via
  _write_scheduler_attr) failed because: (a) the config value is copied to
  self.max_num_running_reqs at init and not re-read, and (b) the token
  budget, not the sequence count, is the binding constraint for small models.

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
    """A thin queue-wait controller around vLLM's default scheduler.

    The controller computes an admission_fraction in [0, 1] that scales the
    token budget and sequence ceiling before each schedule() call. When
    target_queue_ms=0 and the queue is empty, admission_fraction=1 (full
    capacity). When the controller wants to build queue wait toward a nonzero
    target, it reduces the fraction, causing requests to accumulate in the
    waiting queue.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.target_queue_ms = float(os.getenv("CH10_TARGET_QUEUE_MS", "0"))
        self.control_file = os.getenv("CH10_CONTROL_FILE", "/tmp/ch10_scheduler_control.json")
        self.kp = float(os.getenv("CH10_QUEUE_KP", "0.02"))
        self.ki = float(os.getenv("CH10_QUEUE_KI", "0.002"))
        self.control_period_s = float(os.getenv("CH10_CONTROL_PERIOD_S", "0.25"))
        self.enabled = os.getenv("CH10_SCHEDULER_ENABLED", "1") not in {"0", "false", "False"}

        # minimum tokens to always allow (keeps decode alive for running reqs)
        self.min_token_budget = int(os.getenv("CH10_MIN_TOKEN_BUDGET", "1"))
        self.min_running_reqs = int(os.getenv("CH10_MIN_RUNNING_REQS", "1"))

        self._last_control_t = 0.0
        self._last_control_file_mtime = 0.0
        self._xi = 0.0  # integrator state
        self._admission_fraction = 1.0  # 0..1, computed by PI controller
        self._request_first_seen: dict[str, float] = {}

        # capture nominal capacities set by vLLM at init
        self._nominal_max_tokens = getattr(self, "max_num_scheduled_tokens", None)
        self._nominal_max_running = getattr(self, "max_num_running_reqs", None)

        LOG.warning(
            "CH10 ControlledScheduler loaded: enabled=%s target_queue_ms=%.2f "
            "kp=%.5f ki=%.5f nominal_max_tokens=%s nominal_max_running=%s",
            self.enabled,
            self.target_queue_ms,
            self.kp,
            self.ki,
            self._nominal_max_tokens,
            self._nominal_max_running,
        )

    # ------------------------------------------------------------------
    # schedule() override: the core actuator
    # ------------------------------------------------------------------
    def schedule(self, *args: Any, **kwargs: Any) -> Any:
        if self.enabled:
            self._update_controller()

        # apply the admission fraction by temporarily capping both gates
        saved_tokens = None
        saved_running = None
        if self.enabled and self._admission_fraction < 1.0:
            if self._nominal_max_tokens is not None:
                saved_tokens = getattr(self, "max_num_scheduled_tokens", None)
                capped_tokens = max(
                    self.min_token_budget,
                    int(round(self._nominal_max_tokens * self._admission_fraction)),
                )
                try:
                    self.max_num_scheduled_tokens = capped_tokens
                except Exception:
                    saved_tokens = None  # attribute is read-only; skip

            if self._nominal_max_running is not None:
                saved_running = getattr(self, "max_num_running_reqs", None)
                capped_running = max(
                    self.min_running_reqs,
                    int(round(self._nominal_max_running * self._admission_fraction)),
                )
                try:
                    self.max_num_running_reqs = capped_running
                except Exception:
                    saved_running = None

        try:
            result = super().schedule(*args, **kwargs)
        finally:
            # always restore nominal capacities
            if saved_tokens is not None:
                self.max_num_scheduled_tokens = saved_tokens
            if saved_running is not None:
                self.max_num_running_reqs = saved_running

        return result

    # ------------------------------------------------------------------
    # PI controller: updates admission_fraction at control_period_s
    # ------------------------------------------------------------------
    def _update_controller(self) -> None:
        now = time.monotonic()
        if now - self._last_control_t < self.control_period_s:
            return
        self._last_control_t = now
        self._load_control_file()

        # if target is 0 ms, run at full capacity (no queue desired)
        if self.target_queue_ms <= 0.0:
            self._admission_fraction = 1.0
            self._xi = 0.0
            return

        # measure current waiting queue
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

        # pi control law
        # positive error = queue waiting less than target -> reduce fraction
        # negative error = queue waiting more than target -> increase fraction
        error_ms = self.target_queue_ms - mean_wait_ms
        dt = self.control_period_s
        self._xi = _clamp(self._xi + error_ms * dt, -50_000.0, 50_000.0)

        # controller output: how much to reduce from full capacity
        # when error is positive (under-waiting), we want to slow admission
        # so reduction > 0 -> fraction < 1
        reduction = self.kp * error_ms + self.ki * self._xi
        # reduction is in "units of target_ms" roughly; normalize to a fraction
        # scale factor: reduction of target_queue_ms corresponds to ~50% cut
        scale = 2.0 / max(self.target_queue_ms, 1.0)
        self._admission_fraction = _clamp(1.0 - reduction * scale, 0.01, 1.0)

        LOG.info(
            "CH10 control waiting=%d mean_wait=%.1fms oldest=%.1fms "
            "target=%.1fms err=%.1f xi=%.1f frac=%.3f "
            "tok_cap=%s run_cap=%s",
            len(waiting_ids),
            mean_wait_ms,
            oldest_wait_ms,
            self.target_queue_ms,
            error_ms,
            self._xi,
            self._admission_fraction,
            int(round(self._nominal_max_tokens * self._admission_fraction))
            if self._nominal_max_tokens
            else "?",
            int(round(self._nominal_max_running * self._admission_fraction))
            if self._nominal_max_running
            else "?",
        )

    # ------------------------------------------------------------------
    # control file hot-reload
    # ------------------------------------------------------------------
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
            # reset integrator on target change to avoid windup carryover
            self._xi = 0.0
            self._admission_fraction = 1.0
            LOG.warning(
                "CH10 scheduler control updated from file: enabled=%s "
                "target_queue_ms=%.2f kp=%.5f ki=%.5f",
                self.enabled,
                self.target_queue_ms,
                self.kp,
                self.ki,
            )
        except Exception as exc:
            LOG.warning("CH10 failed to load control file %s: %r", self.control_file, exc)

    # ------------------------------------------------------------------
    # defensive introspection of vLLM internals
    # ------------------------------------------------------------------
    def _waiting_requests(self) -> Iterable[Any]:
        # v1 scheduler: self.waiting is a RequestQueue
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
