#!/usr/bin/env python3
"""
identify_plant.py  --  Chapter 3: Plant identification for qwen2.5:3b

Standalone -- no dependency on MATLAB, setup_plant.m, or any workspace.
Run directly:   python3 -u identify_plant.py

Identifies the three parameters of:
    l_mean = alpha * B  +  gamma * B^2  +  beta * q

Stage 1: B sweep at q=0  ->  fits alpha, gamma
Stage 2: sustained load sweep  ->  fits beta

Outputs:
    identified_params.json  (loadable in MATLAB via jsondecode)
    id_stage1.png
    id_stage2.png
"""

import sys
import time
import json
import threading
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import requests
from pathlib import Path
from datetime import datetime
from numpy.random import default_rng

# flush all prints immediately
print = lambda *a, **kw: __builtins__["print"](*a, **{**kw, "flush": True}) \
    if isinstance(__builtins__, dict) else \
    __import__("builtins").print(*a, **{**kw, "flush": True})

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
CFG = dict(
    url          = 'http://localhost:11434/api/generate',
    model        = 'qwen2.5:3b',
    # Use non-streaming with num_predict=1.
    # With 1 token output, end-to-end ≈ TTFT (generation term is ~50ms,
    # negligible vs prefill).  This matches the working MATLAB approach
    # and avoids streaming-buffer issues in Python.
    num_predict  = 1,
    timeout      = 20,      # s per request
    n_warmup     = 4,
    n_reps       = 5,       # repetitions per B in stage 1
    b_sweep      = [1, 2, 3, 4, 5, 6, 8, 10, 12],
    lambda_sweep = [2, 3, 5, 7, 8],
    settle_ticks = 25,
    measure_ticks= 15,
    b_max        = 12,
    b_min        = 1,
    q_max        = 20,
    out_dir      = Path(__file__).parent,
)

rng = default_rng(seed=42)


# -------------------------------------------------------------------------
# Single request latency (non-streaming, num_predict=1)
# -------------------------------------------------------------------------
def measure_latency(prompt: str) -> float:
    """
    Measure end-to-end latency for one request with num_predict=1.
    With 1 output token, end-to-end ≈ TTFT (prefill dominates).
    Returns wall-clock ms.  Returns penalty on any error.
    """
    body = {
        "model":   CFG["model"],
        "prompt":  prompt,
        "stream":  False,
        "options": {"num_predict": CFG["num_predict"]},
    }
    t0 = time.perf_counter()
    try:
        resp = requests.post(CFG["url"], json=body, timeout=CFG["timeout"])
        resp.raise_for_status()
        return (time.perf_counter() - t0) * 1000
    except Exception as e:
        return CFG["timeout"] * 1000   # penalty


def fire_concurrent(b: int, prompt: str = "What is 2+2?") -> list:
    """Fire b concurrent requests and return list of latencies [ms]."""
    results = [None] * b

    def worker(idx):
        results[idx] = measure_latency(prompt)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(b)]
    for t in threads: t.start()
    for t in threads: t.join()
    return results


# -------------------------------------------------------------------------
# Warm-up
# -------------------------------------------------------------------------
def warmup():
    print(f"Warming up {CFG['model']} ({CFG['n_warmup']} requests)...")
    lats = fire_concurrent(CFG["n_warmup"])
    valid = [l for l in lats if l < CFG["timeout"] * 900]
    print(f"  Latencies: {[f'{l:.0f}' for l in lats]} ms")
    if valid:
        print(f"  Mean warm latency: {np.mean(valid):.0f} ms")
    else:
        print("  WARNING: all warm-up requests timed out — is Ollama running?")
        sys.exit(1)
    print("  Warm-up complete.\n")


# =========================================================================
# STAGE 1
# =========================================================================
def stage1():
    print("=" * 60)
    print("STAGE 1: B sweep at q=0  (identifying alpha, gamma)")
    print("=" * 60 + "\n")

    b_vals  = CFG["b_sweep"]
    l_means = []
    l_stds  = []

    for b in b_vals:
        rep_means = []
        for rep in range(CFG["n_reps"]):
            lats  = fire_concurrent(b)
            valid = [l for l in lats if l < CFG["timeout"] * 900]
            rep_means.append(np.mean(valid) if valid else np.nan)
        mean_l = float(np.nanmean(rep_means))
        std_l  = float(np.nanstd(rep_means))
        l_means.append(mean_l)
        l_stds.append(std_l)
        print(f"  B={b:2d}:  mean = {mean_l:6.1f} ms  (std = {std_l:5.1f} ms)")

    l_means = np.array(l_means)
    b_arr   = np.array(b_vals, dtype=float)

    # l = alpha*B + gamma*B^2  ->  linear least squares
    A       = np.column_stack([b_arr, b_arr**2])
    params, _, _, _ = np.linalg.lstsq(A, l_means, rcond=None)
    alpha_id, gamma_id = float(params[0]), float(params[1])

    l_fit = A @ params
    ss_res = np.sum((l_means - l_fit)**2)
    ss_tot = np.sum((l_means - np.mean(l_means))**2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    print(f"\n  alpha = {alpha_id:.4f}  ms/req")
    print(f"  gamma = {gamma_id:.4f}  ms/req^2")
    print(f"  R^2   = {r2:.4f}\n")

    # Plot
    b_fine = np.linspace(1, max(b_vals), 200)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(b_vals, l_means, yerr=l_stds, fmt='bo',
                capsize=4, label='Measured (mean ± std)', zorder=5)
    ax.plot(b_fine, alpha_id*b_fine + gamma_id*b_fine**2, 'r-',
            linewidth=2,
            label=f'α·B + γ·B²  (α={alpha_id:.3f}, γ={gamma_id:.4f})')
    ax.set_xlabel('Batch size B  [requests]')
    ax.set_ylabel('Latency  [ms]')
    ax.set_title(f'Stage 1 — B sweep at q=0  (qwen2.5:3b)\nR²={r2:.4f}')
    ax.legend(); ax.grid(True); fig.tight_layout()
    out = CFG["out_dir"] / "id_stage1.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  Plot saved: {out}\n")

    return alpha_id, gamma_id, r2, {
        "b_sweep": b_vals,
        "l_means": l_means.tolist(),
        "l_stds":  l_stds,
    }


# =========================================================================
# STAGE 2
# =========================================================================
def stage2(alpha_id: float, gamma_id: float):
    print("=" * 60)
    print("STAGE 2: Sustained load sweep  (identifying beta)")
    print("=" * 60 + "\n")

    q_ss_list, b_ss_list, l_ss_list = [], [], []

    for lam in CFG["lambda_sweep"]:
        print(f"  lambda = {lam} req/tick ...")
        q_k     = float(lam)
        lat_buf = [250.0] * 20
        buf_idx = 0
        q_hist, b_hist, l_hist = [], [], []
        total   = CFG["settle_ticks"] + CFG["measure_ticks"]

        for tick in range(total):
            a_k = int(rng.poisson(lam))
            b_k = int(np.clip(round(q_k + a_k), CFG["b_min"], CFG["b_max"]))

            t_wall = time.perf_counter()
            lats   = fire_concurrent(b_k)
            wall   = (time.perf_counter() - t_wall) * 1000

            valid = [l for l in lats if l < CFG["timeout"] * 900]
            for l in valid:
                lat_buf[buf_idx % 20] = l
                buf_idx += 1
            l_meas = float(np.mean(lat_buf))

            q_k = float(np.clip(q_k + a_k - b_k, 0, CFG["q_max"]))
            q_hist.append(q_k); b_hist.append(b_k); l_hist.append(l_meas)

            if tick % 5 == 0:
                print(f"    tick {tick:3d}/{total}  q={q_k:.1f}  "
                      f"b={b_k}  l={l_meas:.0f}ms  wall={wall:.0f}ms")

        meas = slice(CFG["settle_ticks"], total)
        q_ss = float(np.mean(q_hist[meas]))
        b_ss = float(np.mean(b_hist[meas]))
        l_ss = float(np.mean(l_hist[meas]))
        q_ss_list.append(q_ss); b_ss_list.append(b_ss); l_ss_list.append(l_ss)
        print(f"    => q_ss={q_ss:.2f}  b_ss={b_ss:.2f}  l_ss={l_ss:.1f} ms\n")

    q_ss = np.array(q_ss_list)
    b_ss = np.array(b_ss_list)
    l_ss = np.array(l_ss_list)

    service  = alpha_id * b_ss + gamma_id * b_ss**2
    residual = l_ss - service
    valid    = q_ss > 0.1

    if valid.sum() > 0:
        beta_id = float(
            (q_ss[valid] @ residual[valid]) / (q_ss[valid] @ q_ss[valid])
        )
    else:
        beta_id = 2.0
        print("  WARNING: no valid q_ss > 0.1 — falling back to beta=2.0")

    l_fit2  = service + beta_id * q_ss
    ss_res2 = np.sum((l_ss[valid] - l_fit2[valid])**2)
    ss_tot2 = np.sum((l_ss[valid] - np.mean(l_ss[valid]))**2)
    r2      = float(1 - ss_res2 / ss_tot2) if ss_tot2 > 0 else 0.0

    print(f"  beta  = {beta_id:.4f}  ms/req")
    print(f"  R^2   = {r2:.4f}\n")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    ax.scatter(q_ss, residual, c='blue', s=70, zorder=5)
    q_fine = np.linspace(0, max(q_ss)*1.1 if len(q_ss) else 5, 100)
    ax.plot(q_fine, beta_id*q_fine, 'r-', lw=2,
            label=f'β·q  (β={beta_id:.3f})')
    ax.set_xlabel('q_ss  [requests]')
    ax.set_ylabel('l_ss − (α·B + γ·B²)  [ms]')
    ax.set_title(f'Stage 2 — Queuing term\nβ={beta_id:.4f} ms/req,  R²={r2:.4f}')
    ax.legend(); ax.grid(True)

    ax = axes[1]
    sc = ax.scatter(b_ss, l_ss, c=CFG["lambda_sweep"][:len(b_ss)],
                    cmap='viridis', s=80, zorder=5)
    plt.colorbar(sc, ax=ax, label='lambda')
    b_fine2 = np.linspace(1, max(b_ss)*1.1 if len(b_ss) else 6, 100)
    for qi in [0, 1, 2, 3]:
        ax.plot(b_fine2, alpha_id*b_fine2 + gamma_id*b_fine2**2 + beta_id*qi,
                '--', lw=1, label=f'q={qi}')
    ax.set_xlabel('B_ss  [requests]')
    ax.set_ylabel('l_ss  [ms]')
    ax.set_title('Identified surface l(B, q)')
    ax.legend(fontsize=8); ax.grid(True)
    fig.tight_layout()
    out = CFG["out_dir"] / "id_stage2.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  Plot saved: {out}\n")

    return beta_id, r2, {
        "lambda_sweep": CFG["lambda_sweep"],
        "q_ss": q_ss.tolist(), "b_ss": b_ss.tolist(), "l_ss": l_ss.tolist(),
    }


# =========================================================================
# Main
# =========================================================================
def main():
    print("=" * 64)
    print(f"  Plant identification: {CFG['model']}")
    print("=" * 64 + "\n")

    warmup()
    alpha_id, gamma_id, r2_s1, raw_s1 = stage1()
    beta_id,  r2_s2,    raw_s2        = stage2(alpha_id, gamma_id)

    dt, tau_out = 1.0, 30.0
    z_cl         = np.exp(-dt / tau_out)
    K_il_id      = (z_cl - 1) / beta_id
    K_il_assumed = (z_cl - 1) / 2.0

    print("=" * 64)
    print(f"  IDENTIFIED PARAMETERS  ({CFG['model']})")
    print("=" * 64)
    print(f"  alpha = {alpha_id:8.4f}  ms/req       (R² stage 1 = {r2_s1:.4f})")
    print(f"  gamma = {gamma_id:8.4f}  ms/req^2     (R² stage 1 = {r2_s1:.4f})")
    print(f"  beta  = {beta_id:8.4f}  ms/req       (R² stage 2 = {r2_s2:.4f})")
    print(f"  ---")
    print(f"  vs assumed:  alpha=0.1  gamma=0.8  beta=2.0")
    print(f"  K_il identified = {K_il_id:.6f}")
    print(f"  K_il assumed    = {K_il_assumed:.6f}")
    print(f"  Ratio           = {K_il_id/K_il_assumed:.2f}x")
    print("=" * 64 + "\n")

    result = {
        "model":      CFG["model"],
        "timestamp":  datetime.now().isoformat(),
        "alpha":      alpha_id,
        "gamma":      gamma_id,
        "beta":       beta_id,
        "r2_stage1":  r2_s1,
        "r2_stage2":  r2_s2,
        "K_il_identified": K_il_id,
        "K_il_assumed":    K_il_assumed,
        "raw_stage1": {k: (v.tolist() if hasattr(v,'tolist') else v)
                       for k,v in raw_s1.items()},
        "raw_stage2": {k: (v.tolist() if hasattr(v,'tolist') else v)
                       for k,v in raw_s2.items()},
    }

    out_json = CFG["out_dir"] / "identified_params.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {out_json}")
    print(f"\nTo use in setup_plant.m:")
    print(f"  id = jsondecode(fileread('{out_json}'));")
    print(f"  perturbed.alpha = id.alpha;   % {alpha_id:.4f}")
    print(f"  perturbed.gamma = id.gamma;   % {gamma_id:.4f}")
    print(f"  perturbed.beta  = id.beta;    % {beta_id:.4f}")


if __name__ == "__main__":
    main()
