#!/usr/bin/env python3
"""
Chapter 11 vLLM scheduler hook: token-budget actuator.

Phase 1 is deliberately open-loop. The scheduler reads an admission_fraction
from a JSON control file and applies it directly by temporarily capping the
vLLM scheduler's token budget and running-request ceiling for each schedule()
call.

The goal is plant characterization:

    admission_fraction -> TTFT / latency / throughput / power / energy

Closed-loop TTFT and power PI modes will build on this same actuator after the
open-loop curves tell us the plant gain and useful operating region.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any


LOG = logging.getLogger("ch11.controlled_scheduler")


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
    """Open-loop token-budget scheduler for Chapter 11 Phase 1."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.control_file = os.getenv("CH11_CONTROL_FILE", "/tmp/ch11_scheduler_control.json")
        self.control_period_s = float(os.getenv("CH11_CONTROL_PERIOD_S", "0.25"))
        self.enabled = os.getenv("CH11_SCHEDULER_ENABLED", "1") not in {"0", "false", "False"}
        self.mode = os.getenv("CH11_CONTROL_MODE", "open_loop")
        self.min_token_budget = int(os.getenv("CH11_MIN_TOKEN_BUDGET", "1"))
        self.min_running_reqs = int(os.getenv("CH11_MIN_RUNNING_REQS", "1"))
        self._admission_fraction = _clamp(float(os.getenv("CH11_ADMISSION_FRACTION", "1.0")), 0.01, 1.0)
        self._last_control_t = 0.0
        self._last_control_file_mtime = 0.0
        self._last_log_t = 0.0

        self._nominal_max_tokens = getattr(self, "max_num_scheduled_tokens", None)
        self._nominal_max_running = getattr(self, "max_num_running_reqs", None)

        LOG.warning(
            "CH11 ControlledScheduler loaded: enabled=%s mode=%s frac=%.3f "
            "nominal_max_tokens=%s nominal_max_running=%s control_file=%s",
            self.enabled,
            self.mode,
            self._admission_fraction,
            self._nominal_max_tokens,
            self._nominal_max_running,
            self.control_file,
        )

    def schedule(self, *args: Any, **kwargs: Any) -> Any:
        self._maybe_load_control_file()

        saved_tokens = None
        saved_running = None
        should_cap = self.enabled and self.mode == "open_loop" and self._admission_fraction < 1.0
        if should_cap:
            if self._nominal_max_tokens is not None:
                saved_tokens = getattr(self, "max_num_scheduled_tokens", None)
                try:
                    self.max_num_scheduled_tokens = self._token_cap()
                except Exception:
                    saved_tokens = None

            if self._nominal_max_running is not None:
                saved_running = getattr(self, "max_num_running_reqs", None)
                try:
                    self.max_num_running_reqs = self._running_cap()
                except Exception:
                    saved_running = None

        self._periodic_log()
        try:
            return super().schedule(*args, **kwargs)
        finally:
            if saved_tokens is not None:
                self.max_num_scheduled_tokens = saved_tokens
            if saved_running is not None:
                self.max_num_running_reqs = saved_running

    def _maybe_load_control_file(self) -> None:
        now = time.monotonic()
        if now - self._last_control_t < self.control_period_s:
            return
        self._last_control_t = now

        try:
            stat = os.stat(self.control_file)
        except FileNotFoundError:
            return
        except Exception as exc:
            LOG.debug("CH11 could not stat control file %s: %r", self.control_file, exc)
            return

        if stat.st_mtime <= self._last_control_file_mtime:
            return
        self._last_control_file_mtime = stat.st_mtime

        try:
            with open(self.control_file) as f:
                payload = json.load(f)
            if "enabled" in payload:
                self.enabled = bool(payload["enabled"])
            if "mode" in payload:
                self.mode = str(payload["mode"])
            if "admission_fraction" in payload:
                self._admission_fraction = _clamp(float(payload["admission_fraction"]), 0.01, 1.0)
            LOG.warning(
                "CH11 scheduler control updated: enabled=%s mode=%s frac=%.3f "
                "tok_cap=%s run_cap=%s",
                self.enabled,
                self.mode,
                self._admission_fraction,
                self._token_cap(),
                self._running_cap(),
            )
        except Exception as exc:
            LOG.warning("CH11 failed to load control file %s: %r", self.control_file, exc)

    def _token_cap(self) -> str | int:
        if self._nominal_max_tokens is None:
            return "?"
        return max(self.min_token_budget, int(round(self._nominal_max_tokens * self._admission_fraction)))

    def _running_cap(self) -> str | int:
        if self._nominal_max_running is None:
            return "?"
        return max(self.min_running_reqs, int(round(self._nominal_max_running * self._admission_fraction)))

    def _periodic_log(self) -> None:
        now = time.monotonic()
        if now - self._last_log_t < 5.0:
            return
        self._last_log_t = now
        LOG.info(
            "CH11 actuator status enabled=%s mode=%s frac=%.3f tok_cap=%s run_cap=%s",
            self.enabled,
            self.mode,
            self._admission_fraction,
            self._token_cap(),
            self._running_cap(),
        )


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
