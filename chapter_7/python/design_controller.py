#!/usr/bin/env python3
"""
design_controller.py  --  Chapter 7 single-loop controller design

Controller structure:
    e[k]  = L_target - L_meas[k]
    xi[k] = clamp(xi[k-1] + e[k], xi_min, xi_max)
    C[k]  = clamp(round(C0 + K_i * xi[k]), C_min, C_max)

Where:
    C     = client-side concurrency command
    L_meas= client first-token latency

For a local linear model L ≈ L0 + beta_c * (C - C0), choose:
    z_cl = exp(-dt / tau_cl)
    K_i  = (1 - z_cl) / beta_c
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


DEFAULT_PARAMS = Path(__file__).parent / "identified_params.json"
DEFAULT_OUT = Path(__file__).parent / "controller_params.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default=str(DEFAULT_PARAMS))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--C0", type=int, default=None)
    ap.add_argument("--C-min", type=int, default=1)
    ap.add_argument("--C-max", type=int, default=8)
    ap.add_argument("--L-target", type=float, default=None)
    ap.add_argument("--dt", type=float, default=None)
    ap.add_argument("--tau-cl", type=float, default=20.0)
    args = ap.parse_args()

    with open(args.params) as f:
        identified = json.load(f)

    beta_c = float(identified["beta_c"])
    if beta_c <= 0:
        raise SystemExit(f"beta_c must be positive, got {beta_c}")

    C0 = args.C0 if args.C0 is not None else int(identified["C0"])
    dt = args.dt if args.dt is not None else float(identified.get("dt", 1.0))
    L0 = float(identified["L0_ms"])
    L_target = args.L_target if args.L_target is not None else round(1.15 * L0, 1)

    z_cl = math.exp(-dt / args.tau_cl)
    K_i = (1.0 - z_cl) / beta_c

    bounds = sorted(((args.C_min - C0) / K_i, (args.C_max - C0) / K_i))
    xi_min, xi_max = float(bounds[0]), float(bounds[1])

    print("=== Chapter 7 Single-Loop Design ===")
    print(f"  beta_c   = {beta_c:.4f} ms/concurrency")
    print(f"  C0       = {C0}")
    print(f"  L0       = {L0:.2f} ms")
    print(f"  L_target = {L_target:.2f} ms")
    print(f"  dt       = {dt:.2f} s")
    print(f"  tau_cl   = {args.tau_cl:.2f} s")
    print(f"  z_cl     = {z_cl:.6f}")
    print(f"  K_i      = {K_i:.8f}")
    print(f"  C range  = [{args.C_min}, {args.C_max}]")

    result = {
        "model": identified.get("model", "unknown"),
        "url": identified.get("url", ""),
        "latency_definition": identified.get("latency_definition"),
        "plant_model": identified.get("plant_model"),
        "beta_c": beta_c,
        "C0": C0,
        "C_min": args.C_min,
        "C_max": args.C_max,
        "L0_ms": L0,
        "L_target_ms": L_target,
        "dt": dt,
        "tau_cl": args.tau_cl,
        "z_cl": z_cl,
        "K_i": K_i,
        "xi_min": xi_min,
        "xi_max": xi_max,
        "prompt_repeat": identified.get("prompt_repeat", 64),
        "max_tokens": identified.get("max_tokens", 32),
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
