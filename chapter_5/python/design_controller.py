#!/usr/bin/env python3
"""
design_controller.py  --  Chapter 5: Cascade controller design

Reads identified_params.json (from characterise.py) and designs
the discrete-time cascade controller:

  Inner loop:  B[k] -> q[k]   (Franklin augmented LQR / pole placement)
  Outer loop:  q_ref[k] -> l[k]   (integral-only, analytical gain)

Plant model:
    q[k+1]  = q[k] + a[k] - B[k]          (queue state equation)
    l[k]    = alpha*B[k] + gamma*B[k]^2 + beta*q[k]   (latency model)

Inner loop design (Franklin augmented, dt=1s):
    Augmented state:  x = [q_error; xi_q]  where xi_q is integrator
    A_aug = [[1, 0], [-1, 1]]
    B_aug = [[-1], [0]]
    Pole placement to put CL poles at z1, z2 = exp(-dt/tau1), exp(-dt/tau2)

Outer loop design:
    Linearised plant:  Δl = beta * Δq_ref   (assuming inner loop tracking)
    Integrator:        xi_l[k+1] = xi_l[k] + e_l[k]
    CL pole:           z_cl = exp(-dt/tau_out)
    Gain:              K_il = (z_cl - 1) / beta

If beta is NaN in the JSON (vllm-metal gauge bug), it is estimated from:
    beta = l_e2e(B=max_num_seqs, q=0) / max_num_seqs
         = (alpha*4 + gamma*16 + l_gen) / 4

Usage:
    python3 design_controller.py [--params PATH] [--out PATH]
"""

import argparse
import json
import math
import numpy as np
from pathlib import Path

DEFAULT_PARAMS = Path(__file__).parent / "identified_params.json"
DEFAULT_OUT    = Path(__file__).parent / "controller_params.json"

# ---------------------------------------------------------------------------
# Inner loop: Franklin augmented pole placement
# ---------------------------------------------------------------------------
def design_inner(dt: float, B_min: int, B_max: int, B0: int, q0: int,
                 tau1: float, tau2: float) -> dict:
    """
    Augmented state: x = [dq, xi_q]
      A_aug = [[1, 0], [-1, 1]]
      B_aug = [[-1], [0]]
    Desired poles: z1 = exp(-dt/tau1),  z2 = exp(-dt/tau2)
    Gain K = [K_q, K_i] from pole placement.
    """
    A = np.array([[1.0, 0.0],
                  [-1.0, 1.0]])
    B = np.array([[-1.0],
                  [0.0]])

    z1 = math.exp(-dt / tau1)
    z2 = math.exp(-dt / tau2)

    # Characteristic polynomial of desired CL system:
    # (z - z1)(z - z2) = z^2 - (z1+z2)z + z1*z2
    # Ackermann's formula:
    #   K = e_n^T * C^{-1} * p(A)
    # where e_n = [0 1] (last basis vector), C = controllability matrix,
    # p(A) = desired characteristic polynomial evaluated at A

    C = np.hstack([B, A @ B])   # controllability matrix: [B, AB]
    e2 = np.array([[0.0, 1.0]])

    p_A = A @ A - (z1 + z2) * A + z1 * z2 * np.eye(2)   # p(A)
    K   = (e2 @ np.linalg.inv(C) @ p_A).flatten()        # [K_q, K_i]

    K_q = float(K[0])
    K_i = float(K[1])

    # Anti-windup bounds for xi_q
    # B = B0 - K_q * dq - K_i * xi_q   => xi_q in [(B0 - B_max)/K_i, (B0 - B_min)/K_i]
    if abs(K_i) > 1e-12:
        bounds = [(B0 - B_max) / K_i, (B0 - B_min) / K_i]
        xi_min = min(bounds)
        xi_max = max(bounds)
    else:
        xi_min, xi_max = -1e6, 1e6

    # Closed-loop poles (verify)
    K_row = K.reshape(1, 2)
    A_cl  = A - B @ K_row
    poles = np.linalg.eigvals(A_cl)

    print(f"  [inner] K_q={K_q:.4f}  K_i={K_i:.4f}")
    print(f"  [inner] Desired poles: z1={z1:.4f}  z2={z2:.4f}")
    print(f"  [inner] Actual CL poles: {poles.real}")
    print(f"  [inner] xi_q range: [{xi_min:.2f}, {xi_max:.2f}]")

    return {
        "K_q":     K_q,
        "K_i":     K_i,
        "xi_min":  xi_min,
        "xi_max":  xi_max,
        "poles_cl": poles.real.tolist(),
        "B0":      B0,
        "q0":      q0,
        "B_min":   B_min,
        "B_max":   B_max,
    }


# ---------------------------------------------------------------------------
# Outer loop: integral-only analytical gain
# ---------------------------------------------------------------------------
def design_outer(beta: float, dt: float, tau_out: float,
                 L_target: float, q0: int, q_max: int) -> dict:
    """
    Linearised plant (assuming perfect inner-loop tracking):
        Δl = beta * Δq_ref
    Integrator:
        xi_l[k+1] = xi_l[k] + e_l[k]    where e_l = l_target - l_meas
    CL pole:
        z_cl = 1 + beta * K_il = exp(-dt / tau_out)
    Gain:
        K_il = (exp(-dt/tau_out) - 1) / beta
    """
    z_cl = math.exp(-dt / tau_out)
    K_il = (z_cl - 1.0) / beta

    stable = abs(z_cl) < 1.0
    print(f"  [outer] K_il={K_il:.8f}  z_cl={z_cl:.6f}  stable={stable}")
    print(f"  [outer] beta={beta:.2f} ms/req  tau_out={tau_out:.0f}s")

    # Anti-windup for xi_l: q_ref = q0 + K_il * xi_l  in [0, q_max]
    if abs(K_il) > 1e-12:
        bounds = [(0 - q0) / K_il, (q_max - q0) / K_il]
        xi_min = min(bounds)
        xi_max = max(bounds)
    else:
        xi_min, xi_max = -1e6, 1e6

    print(f"  [outer] xi_l range: [{xi_min:.2f}, {xi_max:.2f}]")

    return {
        "K_il":     K_il,
        "z_cl":     z_cl,
        "tau_out":  tau_out,
        "L_target": L_target,
        "q0":       q0,
        "q_max":    q_max,
        "xi_min":   xi_min,
        "xi_max":   xi_max,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Chapter 5 cascade controller design")
    ap.add_argument("--params",  default=str(DEFAULT_PARAMS))
    ap.add_argument("--out",     default=str(DEFAULT_OUT))

    # Plant parameters (overrides JSON if provided)
    ap.add_argument("--alpha",    type=float, default=None)
    ap.add_argument("--gamma",    type=float, default=None)
    ap.add_argument("--beta",     type=float, default=None)

    # Operating conditions
    ap.add_argument("--B0",       type=int,   default=3)
    ap.add_argument("--q0",       type=int,   default=0)
    ap.add_argument("--B_min",    type=int,   default=1)
    ap.add_argument("--B_max",    type=int,   default=8)
    ap.add_argument("--q_max",    type=int,   default=8)
    ap.add_argument("--lambda_mean", type=int, default=3)
    ap.add_argument("--L_target", type=float, default=800.0)    # ms
    ap.add_argument("--dt",       type=float, default=1.0)      # s

    # Controller tuning
    ap.add_argument("--tau1",     type=float, default=2.0)      # inner CL tau1 [s]
    ap.add_argument("--tau2",     type=float, default=3.0)      # inner CL tau2 [s]
    ap.add_argument("--tau_out",  type=float, default=20.0)     # outer CL tau [s]
    ap.add_argument("--l_gen",    type=float, default=700.0)    # generation latency [ms]
    ap.add_argument("--max_num_seqs", type=int, default=4)

    args = ap.parse_args()

    # Load identified params
    if Path(args.params).exists():
        with open(args.params) as f:
            identified = json.load(f)
        print(f"Loaded: {args.params}")
    else:
        identified = {}
        print(f"[warn] {args.params} not found, using command-line values only")

    alpha = args.alpha if args.alpha is not None else identified.get("alpha", 33.34)
    gamma = args.gamma if args.gamma is not None else identified.get("gamma", -1.36)
    beta_raw = args.beta if args.beta is not None else identified.get("beta")

    # Beta estimation if not identified
    if beta_raw is None or (isinstance(beta_raw, float) and math.isnan(beta_raw)):
        B4   = args.max_num_seqs
        l_e2e_B4 = alpha * B4 + gamma * B4**2 + args.l_gen
        beta = l_e2e_B4 / args.max_num_seqs
        print(f"\n[beta] Not identified -- using analytical estimate:")
        print(f"  l_e2e(B={B4}, q=0) ≈ {alpha}*{B4} + {gamma}*{B4**2} + {args.l_gen:.0f} = {l_e2e_B4:.1f} ms")
        print(f"  beta = {l_e2e_B4:.1f} / {args.max_num_seqs} = {beta:.2f} ms/req")
    else:
        beta = float(beta_raw)
        print(f"\n[beta] Using identified value: {beta:.4f} ms/req")

    print(f"\n=== Plant parameters ===")
    print(f"  alpha = {alpha:.4f} ms/req")
    print(f"  gamma = {gamma:.4f} ms/req²")
    print(f"  beta  = {beta:.4f} ms/req")
    print(f"  B0={args.B0}  q0={args.q0}  lambda_mean={args.lambda_mean}")
    print(f"  B range=[{args.B_min}, {args.B_max}]  q_max={args.q_max}\n")

    print("=== Inner loop design ===")
    inner = design_inner(
        dt    = args.dt,
        B_min = args.B_min,
        B_max = args.B_max,
        B0    = args.B0,
        q0    = args.q0,
        tau1  = args.tau1,
        tau2  = args.tau2,
    )

    print("\n=== Outer loop design ===")
    outer = design_outer(
        beta     = beta,
        dt       = args.dt,
        tau_out  = args.tau_out,
        L_target = args.L_target,
        q0       = args.q0,
        q_max    = args.q_max,
    )

    # Summary
    print("\n╔" + "═"*62 + "╗")
    print("║  CONTROLLER DESIGN  --  Chapter 5 Cascade")
    print("╠" + "═"*62 + "╣")
    print(f"║  Plant:  alpha={alpha:.4f}  gamma={gamma:.4f}  beta={beta:.2f}")
    print(f"║  Inner:  K_q={inner['K_q']:.4f}  K_i={inner['K_i']:.4f}")
    print(f"║          CL poles ≈ {inner['poles_cl']}")
    print(f"║  Outer:  K_il={outer['K_il']:.8f}")
    print(f"║          z_cl={outer['z_cl']:.4f}  tau_out={outer['tau_out']:.0f}s")
    print(f"║  Target: {outer['L_target']:.0f} ms")
    print("╚" + "═"*62 + "╝\n")

    result = {
        "model":      identified.get("model", "unknown"),
        "alpha": alpha, "gamma": gamma, "beta": beta,
        "B0":    args.B0, "q0": args.q0,
        "B_min": args.B_min, "B_max": args.B_max,
        "q_max": args.q_max,
        "lambda_mean": args.lambda_mean,
        "L_target":    args.L_target,
        "dt":          args.dt,
        "l_gen":       args.l_gen,
        "inner": inner,
        "outer": outer,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
