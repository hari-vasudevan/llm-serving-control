#!/usr/bin/env python3
"""
Chapter 11 vLLM scheduler hook: token-budget actuator.

Phase 1 is open-loop. The scheduler reads an admission_fraction from a JSON
control file and applies it directly by temporarily capping the vLLM scheduler's
token budget and running-request ceiling for each schedule() call.

The goal is plant characterization:

    admission_fraction -> TTFT / latency / throughput / power / energy

Phase 2 adds a TTFT PI mode on the same actuator. The wrapper writes rolling
measured TTFT into the control file; the scheduler adjusts admission_fraction.
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
    """Token-budget scheduler for Chapter 11 open-loop and TTFT-control modes."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.control_file = os.getenv("CH11_CONTROL_FILE", "/tmp/ch11_scheduler_control.json")
        self.status_file = os.getenv("CH11_STATUS_FILE", "/tmp/ch11_scheduler_status.json")
        self.control_period_s = float(os.getenv("CH11_CONTROL_PERIOD_S", "0.25"))
        self.enabled = os.getenv("CH11_SCHEDULER_ENABLED", "1") not in {"0", "false", "False"}
        self.mode = os.getenv("CH11_CONTROL_MODE", "open_loop")
        self.min_token_budget = int(os.getenv("CH11_MIN_TOKEN_BUDGET", "1"))
        self.min_running_reqs = int(os.getenv("CH11_MIN_RUNNING_REQS", "1"))
        self.fraction_min = float(os.getenv("CH11_FRACTION_MIN", "0.25"))
        self.fraction_max = float(os.getenv("CH11_FRACTION_MAX", "1.0"))
        self.kp = float(os.getenv("CH11_TTFT_KP", "0.15"))
        self.ki = float(os.getenv("CH11_TTFT_KI", "0.02"))
        self.target_ttft_ms = float(os.getenv("CH11_TARGET_TTFT_MS", "150"))
        self.measured_ttft_ms: float | None = None
        self._xi = 0.0
        self._admission_fraction = _clamp(
            float(os.getenv("CH11_ADMISSION_FRACTION", "1.0")),
            self.fraction_min,
            self.fraction_max,
        )
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
        should_cap = self.enabled and self.mode in {"open_loop", "ttft"} and self._admission_fraction < 1.0
        # In ttft mode, only cap scheduled_tokens — never cap running_reqs.
        # Capping running_reqs forces preemption of mid-prefill requests which
        # crashes vLLM under load. Token-budget capping alone is sufficient
        # to regulate TTFT because it throttles batch size and prefill throughput.
        should_cap_running = should_cap and self.mode == "open_loop"
        if should_cap:
            if self._nominal_max_tokens is not None:
                saved_tokens = getattr(self, "max_num_scheduled_tokens", None)
                try:
                    self.max_num_scheduled_tokens = self._token_cap()
                except Exception:
                    saved_tokens = None

        if should_cap_running:
            if self._nominal_max_running is not None:
                saved_running = getattr(self, "max_num_running_reqs", None)
                try:
                    self.max_num_running_reqs = self._running_cap()
                except Exception:
                    saved_running = None

        self._periodic_log()
        try:
            return super().schedule(*args, **kwargs)
        except Exception as exc:
            LOG.error("CH11 super().schedule() raised %r — disabling cap for this call", exc)
            # Restore and retry without cap so vLLM stays alive.
            if saved_tokens is not None:
                try:
                    self.max_num_scheduled_tokens = saved_tokens
                except Exception:
                    pass
            if saved_running is not None:
                try:
                    self.max_num_running_reqs = saved_running
                except Exception:
                    pass
            return super().schedule(*args, **kwargs)
        finally:
            if saved_tokens is not None:
                try:
                    self.max_num_scheduled_tokens = saved_tokens
                except Exception:
                    pass
            if saved_running is not None:
                try:
                    self.max_num_running_reqs = saved_running
                except Exception:
                    pass

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
                new_mode = str(payload["mode"])
                if new_mode != self.mode:
                    self._xi = 0.0
                self.mode = new_mode
            if "fraction_min" in payload:
                self.fraction_min = _clamp(float(payload["fraction_min"]), 0.01, 1.0)
            if "fraction_max" in payload:
                self.fraction_max = _clamp(float(payload["fraction_max"]), self.fraction_min, 1.0)
            if "kp" in payload:
                self.kp = float(payload["kp"])
            if "ki" in payload:
                self.ki = float(payload["ki"])
            if "target_ttft_ms" in payload:
                target = float(payload["target_ttft_ms"])
                if abs(target - self.target_ttft_ms) > 1e-9:
                    self._xi = 0.0
                self.target_ttft_ms = target
            if "measured_ttft_ms" in payload and payload["measured_ttft_ms"] is not None:
                self.measured_ttft_ms = float(payload["measured_ttft_ms"])
            if "admission_fraction" in payload:
                self._admission_fraction = _clamp(float(payload["admission_fraction"]), self.fraction_min, self.fraction_max)
            if self.mode == "ttft":
                self._update_ttft_pi()
            LOG.warning(
                "CH11 scheduler control updated: enabled=%s mode=%s frac=%.3f "
                "target_ttft=%.1f measured_ttft=%s kp=%.4f ki=%.4f tok_cap=%s run_cap=%s",
                self.enabled,
                self.mode,
                self._admission_fraction,
                self.target_ttft_ms,
                "%.1f" % self.measured_ttft_ms if self.measured_ttft_ms is not None else "None",
                self.kp,
                self.ki,
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
            "CH11 actuator status enabled=%s mode=%s frac=%.3f target_ttft=%.1f measured_ttft=%s "
            "xi=%.3f tok_cap=%s run_cap=%s",
            self.enabled,
            self.mode,
            self._admission_fraction,
            self.target_ttft_ms,
            "%.1f" % self.measured_ttft_ms if self.measured_ttft_ms is not None else "None",
            self._xi,
            self._token_cap(),
            self._running_cap(),
        )
        self._write_status()

    def _update_ttft_pi(self) -> None:
        if self.target_ttft_ms <= 0.0 or self.measured_ttft_ms is None:
            return

        # Plant has negative gain: fraction↑ → TTFT↓ (more throughput → less queue wait).
        # Flip sign so the PI drives fraction DOWN when TTFT is below target, which
        # throttles prefill bandwidth and raises TTFT toward the setpoint.
        e_norm = (self.measured_ttft_ms - self.target_ttft_ms) / max(self.target_ttft_ms, 1.0)
        dt = self.control_period_s
        at_floor = self._admission_fraction <= self.fraction_min + 1e-9
        at_ceiling = self._admission_fraction >= self.fraction_max - 1e-9

        should_integrate = True
        if at_floor and e_norm < 0.0:
            should_integrate = False
        if at_ceiling and e_norm > 0.0:
            should_integrate = False
        if should_integrate:
            self._xi = _clamp(self._xi + e_norm * dt, -20.0, 20.0)

        delta = self.kp * e_norm + self.ki * self._xi
        old_fraction = self._admission_fraction
        self._admission_fraction = _clamp(
            self._admission_fraction + delta,
            self.fraction_min,
            self.fraction_max,
        )
        LOG.info(
            "CH11 ttft_pi target=%.1f measured=%.1f e_norm=%.3f xi=%.3f "
            "frac %.3f->%.3f delta=%.4f",
            self.target_ttft_ms,
            self.measured_ttft_ms,
            e_norm,
            self._xi,
            old_fraction,
            self._admission_fraction,
            delta,
        )

    def _write_status(self) -> None:
        payload = {
            "timestamp": time.time(),
            "enabled": self.enabled,
            "mode": self.mode,
            "admission_fraction": self._admission_fraction,
            "target_ttft_ms": self.target_ttft_ms,
            "measured_ttft_ms": self.measured_ttft_ms,
            "kp": self.kp,
            "ki": self.ki,
            "xi": self._xi,
            "token_cap": self._token_cap(),
            "running_cap": self._running_cap(),
            "fraction_min": self.fraction_min,
            "fraction_max": self.fraction_max,
        }
        try:
            tmp = self.status_file + ".tmp"
            with open(tmp, "w") as f:
                json.dump(payload, f)
            os.replace(tmp, self.status_file)
        except Exception as exc:
            LOG.debug("CH11 failed to write status file %s: %r", self.status_file, exc)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
