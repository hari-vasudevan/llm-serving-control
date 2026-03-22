#!/usr/bin/env python3
"""
design_controller.py  --  Chapter 5: Cascade controller design

For Chapter 5 with vLLM on Apple Silicon the effective plant is:

    TTFT(B) = alpha*B + gamma*B^2    (q≈0 always in vllm-metal)

This is an INVERTED sign relative to the classic queuing cascade:
  * In queuing theory:  more B → drains queue → lower latency
  * In our GPU system:  more B → more contention → HIGHER TTFT

Therefore the controller must DECREASE B when latency exceeds target.

We design a single integral controller (degenerate cascade, q-loop bypassed):

    e_l[k]      = L_target - l_meas[k]
    xi_l[k+1]   = xi_l[k] + e_l[k]
    B[k]        = clamp(round(B0 + K_il * xi_l[k]), B_min, B_max)

Gain from desired closed-loop pole:
    Linearised plant:  Δl = beta_eff * ΔB
    where beta_eff = d(TTFT)/dB|_{B0} = alpha + 2*gamma*B0  > 0

    CL characteristic:  z = 1 - K_il * beta_eff  (note: MINUS, not plus)
    Target pole:        z_cl = exp(-dt / tau_cl)
    Gain:               K_il = (1 - z_cl) / beta_eff   [POSITIVE]

Sign check:
    K_il > 0, beta_eff > 0.
    When l > target:  e_l < 0 → xi_l decreases → K_il*xi_l decreases → B decreases ✓
    When l < target:  e_l > 0 → xi_l increases → B increases ✓

Usage:
    python3 design_controller.py [--params identified_params.json]
"""

import argparse
import json
import math
import numpy as np
from pathlib import Path

DEFAULT_PARAMS = Path(__file__).parent / "identified_params.json"
DEFAULT_OUT    = Path(__file__).parent / "controller_params.json"


def design_integral(alpha: float, gamma: float, B0: int, dt: float,
                    tau_cl: float, B_min: int, B_max: int) -> dict:
    """
    Single integral controller correctly signed for GPU-contention plant.
    """
    beta_eff = alpha + 2 * gamma * B0   # d(TTFT)/dB at B0
    z_cl     = math.exp(-dt / tau_cl)
    K_il     = (1.0 - z_cl) / beta_eff  # POSITIVE

    # Verify stability
    z_actual = 1.0 - K_il * beta_eff
    stable   = abs(z_actual) < 1.0

    print(f"  [integral] beta_eff = {beta_eff:.4f} ms/req  (alpha + 2*gamma*B0)")
    print(f"  [integral] K_il     = {K_il:.8f}  (positive → reduces B when l > target)")
    print(f"  [integral] z_cl     = {z_actual:.6f}  stable={stable}")

    # Anti-windup: B = B0 + K_il * xi_l in [B_min, B_max]
    # K_il > 0 → xi_l range: [(B_min - B0)/K_il, (B_max - B0)/K_il]
    xi_min = (B_min - B0) / K_il
    xi_max = (B_max - B0) / K_il
    print(f"  [integral] xi_l range: [{xi_min:.1f}, {xi_max:.1f}]")

    return {
        "K_il":     K_il,
        "z_cl":     z_actual,
        "tau_cl":   tau_cl,
        "beta_eff": beta_eff,
        "B0":       B0,
        "B_min":    B_min,
        "B_max":    B_max,
        "xi_min":   xi_min,
        "xi_max":   xi_max,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params",  default=str(DEFAULT_PARAMS))
    ap.add_argument("--out",     default=str(DEFAULT_OUT))
    ap.add_argument("--alpha",   type=float, default=None)
    ap.add_argument("--gamma",   type=float, default=None)
    ap.add_argument("--B0",      type=int,   default=3)
    ap.add_argument("--B_min",   type=int,   default=1)
    ap.add_argument("--B_max",   type=int,   default=8)
    ap.add_argument("--L_target",type=float, default=250.0,
                    help="Latency target [ms]. Set near/below typical TTFT "
                         "so controller has to actively regulate.")
    ap.add_argument("--dt",      type=float, default=1.0)
    ap.add_argument("--tau_cl",  type=float, default=10.0,
                    help="Closed-loop time constant [s]")
    args = ap.parse_args()

    # Load identified params
    identified = {}
    if Path(args.params).exists():
        with open(args.params) as f:
            identified = json.load(f)
        print(f"Loaded: {args.params}")
    else:
        print(f"[warn] {args.params} not found, using defaults")

    alpha = args.alpha if args.alpha is not None else identified.get("alpha", 143.89)
    gamma = args.gamma if args.gamma is not None else identified.get("gamma", -5.25)

    print(f"\n=== Plant parameters ===")
    print(f"  alpha = {alpha:.4f} ms/req")
    print(f"  gamma = {gamma:.4f} ms/req²")
    print(f"  B0={args.B0}  B range=[{args.B_min}, {args.B_max}]")
    print(f"  L_target={args.L_target:.0f} ms  dt={args.dt}s  tau_cl={args.tau_cl}s\n")

    print("=== Integral controller design ===")
    ctrl = design_integral(
        alpha  = alpha,
        gamma  = gamma,
        B0     = args.B0,
        dt     = args.dt,
        tau_cl = args.tau_cl,
        B_min  = args.B_min,
        B_max  = args.B_max,
    )

    print(f"\n╔{'═'*62}╗")
    print(f"║  CONTROLLER DESIGN  --  Chapter 5")
    print(f"╠{'═'*62}╣")
    print(f"║  Plant:  alpha={alpha:.4f}  gamma={gamma:.4f}")
    print(f"║          beta_eff={ctrl['beta_eff']:.4f} ms/req @ B0={args.B0}")
    print(f"║  Gain:   K_il={ctrl['K_il']:.8f}  (positive)")
    print(f"║  Pole:   z_cl={ctrl['z_cl']:.4f}  tau_cl={ctrl['tau_cl']:.0f}s")
    print(f"║  Target: L_target={args.L_target:.0f} ms")
    print(f"║  Law:    B[k] = clamp(B0 + K_il*xi_l[k], {args.B_min}, {args.B_max})")
    print(f"║          xi_l[k+1] = xi_l[k] + (L_target - l_meas[k])")
    print(f"╚{'═'*62}╝\n")

    result = {
        "model":     identified.get("model", "unknown"),
        "alpha":     alpha,
        "gamma":     gamma,
        "B0":        args.B0,
        "B_min":     args.B_min,
        "B_max":     args.B_max,
        "L_target":  args.L_target,
        "dt":        args.dt,
        "controller": ctrl,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
