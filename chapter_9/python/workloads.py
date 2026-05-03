"""
workloads.py -- Chapter 9 fixed GPU batch workloads.

The Chapter 9 plant is deliberately lower-level than vLLM.  A "request" is a
fixed-size tensor job; the scheduler stacks exactly B queued jobs into one
GPU batch and measures the batch service time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


DeviceKind = Literal["cpu", "cuda", "mps"]


@dataclass(frozen=True)
class WorkloadConfig:
    device: DeviceKind = "cuda"
    dim: int = 1024
    layers: int = 6
    dtype: str = "float32"
    seed: int = 9


class FixedMatmulWorkload:
    """Fixed batched matrix/vector workload with deterministic dimensions."""

    def __init__(self, cfg: WorkloadConfig):
        self.cfg = cfg
        self.device = _resolve_device(cfg.device)
        self.dtype = _resolve_dtype(cfg.dtype)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(cfg.seed)

        weights = []
        scale = 1.0 / (cfg.dim ** 0.5)
        for _ in range(cfg.layers):
            w = torch.randn(cfg.dim, cfg.dim, generator=gen, dtype=self.dtype) * scale
            weights.append(w.to(self.device))
        self.weights = weights

        # Warm up allocation paths and kernels outside measured service time.
        self.run(1)
        self.synchronize()

    def run(self, batch_size: int) -> None:
        x = torch.ones((batch_size, self.cfg.dim), device=self.device, dtype=self.dtype)
        for w in self.weights:
            x = torch.relu(x @ w)
        # Force the result to stay live until the final op is enqueued.
        self._last = x.sum()

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            torch.mps.synchronize()


def _resolve_device(requested: DeviceKind) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    raise RuntimeError(
        f"Requested device {requested!r} is not available. "
        "Use --device cpu for a smoke test, or run on a CUDA/MPS machine."
    )


def _resolve_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype {name!r}")
