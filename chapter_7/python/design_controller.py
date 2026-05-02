#!/usr/bin/env python3
"""
design_controller.py  --  Chapter 7: Cascade controller design

PLANT MODEL
-----------
    l_total(B, q) = alpha*B + gamma*B^2 + (q/B)*dt*1000

where q = q_sw is the software FIFO depth (requests enqueued but not yet
dispatched) and l_total is measured from enqueue to first token.

LINEARISATION at (B0, q0=0):
    dl/dq |_{B0,q0} = dt*1000 / B0       = beta_q    [analytical]
    dl/dB |_{B0,q0} = alpha + 2*gamma*B0 = beta_eff  [from B sweep fit]

CASCADE CONTROLLER SIGN ANALYSIS
---------------------------------
Outer loop  (l_total -> q_ref):

    e_l = L_target - l_total
    xi_l[k+1] = xi_l[k] + e_l[k]
    q_ref[k]  = q0 + K_il * xi_l[k]

    We need K_il > 0.  CL pole: z = 1 - beta_q * K_il
    l > target (e_l < 0) -> xi_l down -> K_il*xi_l down -> q_ref down
    -> inner: q_ref < q_sw -> B up -> queue drains -> queue_wait down  ✓

    Setting z_out = exp(-dt/tau_out):
      K_il = (1 - z_out) / beta_q   [POSITIVE: 1-z_out>0, beta_q>0]

Inner loop  (q_ref -> B via queue error):

    e_q  = q_ref - q_sw
    xi_q[k+1] = xi_q[k] + e_q[k]
    dB[k]     = -(K_q * e_q[k] + K_i * xi_q[k])
    B[k]      = clamp(round(B0 + dB[k]), B_min, B_max)

    We need K_q > 0 so that:
      q_sw > q_ref  (e_q < 0)  -> dB = -(K_q * neg) > 0
                                -> B increases -> faster drain  ✓
      q_sw < q_ref  (e_q > 0)  -> dB = -(K_q * pos) < 0
                                -> B decreases -> queue builds  ✓

Usage:
    python3 design_controller.py [--params identified_params.json] [options]
"""

import argparse
import json
import math
import numpy as np
from pathlib import Path

DEFAULT_PARAMS = Path(__file__).parent / "identified_params.json"
DEFAULT_OUT    = Path(__file__).parent / "controller_params.json"


def design_inner(dt, B_min, B_max, B0, tau1, tau2):
    """
    Franklin augmented pole placement for SOFTWARE FIFO plant.
    FIFO: q[k+1]=q[k]+a-B, e_q=q_ref-q_sw
    => A_aug=[[1,0],[1,1]], B_aug=[[1],[0]]  (opposite sign from Ch2b)
    Ackermann gives K_q>0, K_i>0 directly. No negation needed.
    Law: dB = -(K_q*e_q + K_i*xi_q)
    Check: q_sw>q_ref -> e_q<0 -> dB=-(K_q*neg)>0 -> B up -> drains FIFO
    """
    A = np.array([[1.0, 0.0], [1.0, 1.0]])   # FIFO plant (NOT Ch2b sign)
    B = np.array([[1.0], [0.0]])
    z1 = math.exp(-dt / tau1)
    z2 = math.exp(-dt / tau2)
    C   = np.hstack([B, A @ B])
    e2  = np.array([[0.0, 1.0]])
    p_A = A @ A - (z1+z2)*A + z1*z2*np.eye(2)
    K   = (e2 @ np.linalg.inv(C) @ p_A).flatten()   # no negation; K_q>0 naturally
    K_q, K_i = float(K[0]), float(K[1])

    if abs(K_i) > 1e-12:
        bounds   = [(B0-B_max)/K_i, (B0-B_min)/K_i]
        xi_q_min = min(bounds); xi_q_max = max(bounds)
    else:
        xi_q_min, xi_q_max = -1e6, 1e6

    A_cl  = A - B @ K.reshape(1, 2)
    poles = np.linalg.eigvals(A_cl).real.tolist()
    assert K_q > 0, f"K_q={K_q:.4f} should be positive"
    assert K_i > 0, f"K_i={K_i:.4f} should be positive"

    print(f"  [inner] K_q={K_q:.4f} (>0 ✓)  K_i={K_i:.4f} (>0 ✓)")
    print(f"  [inner] CL poles: z1={z1:.4f}  z2={z2:.4f}  actual={[f'{p:.4f}' for p in poles]}")
    return {"K_q": K_q, "K_i": K_i, "xi_q_min": xi_q_min, "xi_q_max": xi_q_max,
            "poles_cl": poles, "B0": B0, "B_min": B_min, "B_max": B_max}


def design_outer(beta_q, dt, tau_out, q0, q_max):
    """
    Integral outer loop using beta_q = dt*1000/B0  (analytical).
    K_il = (z_out - 1) / beta_q  [NEGATIVE]
    """
    z_out = math.exp(-dt / tau_out)
    K_il  = (1.0 - z_out) / beta_q   # POSITIVE
    assert K_il > 0, f"K_il={K_il:.6f} should be positive"

    if abs(K_il) > 1e-12:
        bounds   = [(0-q0)/K_il, (q_max-q0)/K_il]
        xi_l_min = min(bounds); xi_l_max = max(bounds)
    else:
        xi_l_min, xi_l_max = -1e6, 1e6

    print(f"  [outer] beta_q = {beta_q:.4f} ms/req = dt*1000/B0  (analytical)")
    print(f"  [outer] K_il   = {K_il:.8f}  (>0 ✓)")
    print(f"  [outer] z_out  = {z_out:.6f}  tau_out={tau_out:.0f}s")
    return {"K_il": K_il, "z_out": z_out, "tau_out": tau_out, "beta_q": beta_q,
            "q0": q0, "q_max": q_max, "xi_l_min": xi_l_min, "xi_l_max": xi_l_max}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params",   default=str(DEFAULT_PARAMS))
    ap.add_argument("--out",      default=str(DEFAULT_OUT))
    ap.add_argument("--alpha",    type=float, default=None)
    ap.add_argument("--gamma",    type=float, default=None)
    ap.add_argument("--B0",       type=int,   default=3)
    ap.add_argument("--B_min",    type=int,   default=1)
    ap.add_argument("--B_max",    type=int,   default=8)
    ap.add_argument("--q_max",    type=int,   default=30)
    ap.add_argument("--q0",       type=int,   default=0)
    ap.add_argument("--L_target", type=float, default=150.0)
    ap.add_argument("--dt",       type=float, default=1.0)
    ap.add_argument("--tau1",     type=float, default=2.0)
    ap.add_argument("--tau2",     type=float, default=3.0)
    ap.add_argument("--tau_out",  type=float, default=20.0)
    args = ap.parse_args()

    identified = {}
    if Path(args.params).exists():
        with open(args.params) as f:
            identified = json.load(f)
        print(f"Loaded: {args.params}")

    alpha = args.alpha if args.alpha is not None else identified.get("alpha")
    gamma = args.gamma if args.gamma is not None else identified.get("gamma")
    if alpha is None or gamma is None:
        ap.error("alpha/gamma not in identified_params.json -- run characterise.py first")

    # beta_q: analytical from dt and B0
    # This is d(l_total)/d(q) at B0 -- the queue contribution to latency slope
    beta_q = (args.dt * 1000.0) / args.B0

    # beta_eff: from fit, d(l_total)/d(B) at B0 (informational, not used for gain)
    beta_eff = alpha + 2*gamma*args.B0
    ttft_B0  = alpha*args.B0 + gamma*args.B0**2

    print(f"\n=== Plant ===")
    print(f"  l_total(B,q) = alpha*B + gamma*B^2 + (q/B)*dt*1000")
    print(f"  alpha   = {alpha:.4f} ms/req")
    print(f"  gamma   = {gamma:.4f} ms/req^2")
    print(f"  B0={args.B0}  TTFT(B0)={ttft_B0:.2f} ms")
    print(f"  beta_q  = dt*1000/B0 = {args.dt*1000:.0f}/{args.B0} = {beta_q:.2f} ms/req  [ANALYTICAL]")
    print(f"  beta_eff= alpha+2*gamma*B0 = {beta_eff:.4f} ms/req  [TTFT slope, informational]\n")

    print("=== Inner loop ===")
    inner = design_inner(args.dt, args.B_min, args.B_max, args.B0, args.tau1, args.tau2)

    print("\n=== Outer loop ===")
    outer = design_outer(beta_q, args.dt, args.tau_out, args.q0, args.q_max)

    print(f"\n╔{'═'*66}╗")
    print(f"║  CASCADE CONTROLLER  --  Chapter 7")
    print(f"╠{'═'*66}╣")
    print(f"║  Plant:  l_total(B,q) = {alpha:.3f}*B + ({gamma:.4f})*B^2 + (q/B)*{args.dt*1000:.0f}")
    print(f"║          beta_q = {beta_q:.2f} ms/req  [d(l_total)/d(q), analytical]")
    print(f"║")
    print(f"║  Outer:  K_il={outer['K_il']:.8f}  z_out={outer['z_out']:.4f}  tau_out={args.tau_out:.0f}s")
    print(f"║          e_l = L_target - l_total,  q_ref = q0 + K_il*xi_l")
    print(f"║")
    print(f"║  Inner:  K_q={inner['K_q']:.4f}  K_i={inner['K_i']:.4f}")
    print(f"║          CL poles ≈ {inner['poles_cl']}")
    print(f"║          dB = -(K_q*e_q + K_i*xi_q),  e_q = q_ref - q_sw")
    print(f"║")
    print(f"║  L_target={args.L_target:.0f} ms    B0={args.B0}    q0={args.q0}")
    print(f"╚{'═'*66}╝\n")

    result = {
        "model":    identified.get("model", "unknown"),
        "alpha": alpha, "gamma": gamma,
        "beta_q": beta_q, "beta_eff": beta_eff,
        "B0": args.B0, "q0": args.q0,
        "B_min": args.B_min, "B_max": args.B_max, "q_max": args.q_max,
        "L_target": args.L_target, "dt": args.dt,
        "inner": inner, "outer": outer,
        "latency_definition": "l_total = t_first_token - t_enqueue",
        "plant_model": "l_total(B,q) = alpha*B + gamma*B^2 + (q/B)*dt*1000",
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
