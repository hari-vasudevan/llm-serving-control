#!/usr/bin/env python3
"""
run_controller.py  --  Chapter 5: Closed-loop CASCADE controller on live vLLM

LATENCY DEFINITION
------------------
l_total = t_first_token - t_enqueue  (queue wait + TTFT)

PLANT MODEL
-----------
l_total(B, q) = alpha*B + gamma*B^2 + (q/B)*dt*1000

CASCADE CONTROLLER  (signs verified)
--------------------------------------
Outer (l_total -> q_ref):
  K_il > 0.  CL pole z = 1 - beta_q * K_il = exp(-dt/tau_out)
  l > target (e_l < 0) -> xi_l ↓ -> q_ref = K_il*xi_l ↓
    -> inner sees q_ref < q_sw -> B ↑ -> queue drains -> queue_wait ↓  ✓

Inner (q_sw -> B):
  K_q > 0.  Law: dB = -(K_q*e_q + K_i*xi_q),  e_q = q_ref - q_sw
  q_sw > q_ref (e_q < 0) -> dB = -(K_q*neg) > 0 -> B ↑ -> drain ✓

Anti-windup:
  Outer: freeze xi_l when q_ref at bound AND e_l would push further into bound.
    K_il > 0: q_ref ↓ when xi_l ↓ (e_l < 0). Freeze at q_ref=0 when e_l<0.
              q_ref ↑ when xi_l ↑ (e_l > 0). Freeze at q_ref=q_max when e_l>0.
  Inner: freeze xi_q when B at bound AND e_q would push further into bound.
    K_q > 0: B ↓ when e_q > 0. Freeze at B=B_min when e_q>0.
             B ↑ when e_q < 0. Freeze at B=B_max when e_q<0.
"""

import argparse
import collections
import json
import math
import re
import statistics
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests

DEFAULT_PARAMS = Path(__file__).parent / "controller_params.json"
DEFAULT_URL    = "http://localhost:8001"
DEFAULT_MODEL  = "mlx-community/Qwen3-0.6B-4bit"

PROMPTS = [
    "What is 2+2?", "Name a colour.", "Capital of France?",
    "Days in a week?", "Name a planet.", "Speed of light?",
    "Name a mammal.", "10 times 10?", "Colour of the sky?",
    "Name a fruit.", "Hours in a day?", "What is 5 squared?",
]

PRESETS = {
    "rich": [
        {"ticks": 30, "lambda": 3, "L_target": 400, "label": "Steady"},
        {"ticks": 90, "lambda": 6, "L_target": 400, "label": "λ↑ Spike (90t)"},
        {"ticks": 40, "lambda": 3, "L_target": 400, "label": "Recovery"},
        {"ticks": 30, "lambda": 1, "L_target": 400, "label": "λ↓ Drop"},
        {"ticks": 30, "lambda": 3, "L_target": 400, "label": "Recovery"},
        {"ticks": 40, "lambda": 3, "L_target": 250, "label": "Target↓ (250ms)"},
        {"ticks": 30, "lambda": 3, "L_target": 400, "label": "Target restore"},
    ],
    "spike_only": [
        {"ticks": 20, "lambda": 3, "L_target": 400, "label": "Steady"},
        {"ticks": 90, "lambda": 6, "L_target": 400, "label": "λ↑ Spike (90t)"},
        {"ticks": 30, "lambda": 3, "L_target": 400, "label": "Recovery"},
    ],
}


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
def get_metrics(url):
    raw = requests.get(f"{url}/metrics", timeout=5).text
    out = {}
    for line in raw.splitlines():
        if line.startswith("#"):
            continue
        clean = re.sub(r"\{[^}]*\}", "", line).strip()
        parts = clean.split()
        if len(parts) < 2:
            continue
        try:
            out[parts[0]] = out.get(parts[0], 0.0) + float(parts[1])
        except ValueError:
            pass
    return out


def fire_batch_ltotal(url, model, prompts, sink, timeout=30):
    """Fire prompts concurrently; append l_total = (t_first_token - t_enqueue)*1000 to sink."""
    t_enqueue = time.perf_counter()

    def worker(prompt):
        body = json.dumps({"model": model, "prompt": prompt,
                           "max_tokens": 1, "stream": True})
        try:
            with requests.post(f"{url}/v1/completions", data=body,
                               headers={"Content-Type": "application/json"},
                               stream=True, timeout=timeout) as resp:
                resp.raise_for_status()
                for chunk in resp.iter_lines():
                    if chunk and chunk != b"data: [DONE]":
                        sink.append((time.perf_counter() - t_enqueue) * 1000)
                        return
        except Exception:
            pass

    threads = [threading.Thread(target=worker, args=(p,), daemon=True) for p in prompts]
    for t in threads: t.start()
    for t in threads: t.join()


class ResultsSink:
    def __init__(self):
        self._lock = threading.Lock()
        self._buf  = []

    def append(self, v):
        with self._lock:
            self._buf.append(v)

    def drain(self):
        with self._lock:
            out, self._buf = self._buf, []
        return out


# ---------------------------------------------------------------------------
# Cascade controller  (signs verified 2026-03-23)
# ---------------------------------------------------------------------------
class CascadeController:
    def __init__(self, p):
        oc = p["outer"]
        self.K_il     = oc["K_il"]        # > 0  (verified)
        self.xi_l     = 0.0
        self.xi_l_min = oc["xi_l_min"]
        self.xi_l_max = oc["xi_l_max"]
        self.q0       = oc["q0"]
        self.q_max    = oc["q_max"]

        ic = p["inner"]
        self.K_q      = ic["K_q"]          # > 0  (verified)
        self.K_i      = ic["K_i"]          # > 0  (verified)
        self.xi_q     = 0.0
        self.xi_q_min = ic["xi_q_min"]
        self.xi_q_max = ic["xi_q_max"]
        self.B0       = ic["B0"]
        self.B_min    = ic["B_min"]
        self.B_max    = ic["B_max"]
        self.L_target = p["L_target"]

    def step(self, l_meas, q_sw, L_tgt_override=None):
        L_tgt = L_tgt_override if L_tgt_override is not None else self.L_target

        # ── Outer loop: l_total → q_ref ───────────────────────────────────
        # K_il > 0:
        #   l > target (e_l < 0) → xi_l ↓ → q_ref = K_il*xi_l ↓
        #     → inner: q_ref < q_sw → e_q < 0 → B ↑ → drains queue
        #     → queue_wait ↓ → l_total ↓  ✓
        e_l      = L_tgt - l_meas
        xi_l_sat = max(self.xi_l_min, min(self.xi_l_max, self.xi_l))
        q_ref    = float(np.clip(self.q0 + self.K_il * xi_l_sat, 0, self.q_max))

        # Anti-windup (K_il > 0):
        #   at lower bound: e_l < 0 would push xi_l lower → q_ref below 0 → freeze
        #   at upper bound: e_l > 0 would push xi_l higher → q_ref above q_max → freeze
        at_lo_l = (q_ref <= 0)          and (e_l < 0)
        at_hi_l = (q_ref >= self.q_max) and (e_l > 0)
        if not (at_lo_l or at_hi_l):
            self.xi_l = max(self.xi_l_min, min(self.xi_l_max, xi_l_sat + e_l))
        else:
            self.xi_l = xi_l_sat

        # ── Inner loop: q_sw → B ──────────────────────────────────────────
        # K_q > 0:
        #   q_sw > q_ref (e_q < 0) → dB = -(K_q*neg) > 0 → B ↑ → drain ✓
        #   q_sw < q_ref (e_q > 0) → dB = -(K_q*pos) < 0 → B ↓ → queue builds ✓
        e_q      = q_ref - q_sw
        xi_q_sat = max(self.xi_q_min, min(self.xi_q_max, self.xi_q))
        dB       = -(self.K_q * e_q + self.K_i * xi_q_sat)
        B_cmd    = int(np.clip(round(self.B0 + dB), self.B_min, self.B_max))

        # Anti-windup (K_q > 0):
        #   at lower bound: e_q > 0 → dB < 0 → wants B below B_min → freeze
        #   at upper bound: e_q < 0 → dB > 0 → wants B above B_max → freeze
        at_lo_q = (B_cmd <= self.B_min) and (e_q > 0)
        at_hi_q = (B_cmd >= self.B_max) and (e_q < 0)
        if not (at_lo_q or at_hi_q):
            self.xi_q = max(self.xi_q_min, min(self.xi_q_max, xi_q_sat + e_q))
        else:
            self.xi_q = xi_q_sat

        return B_cmd, q_ref, e_l, e_q, xi_l_sat, xi_q_sat


# ---------------------------------------------------------------------------
# Experiment loop
# ---------------------------------------------------------------------------
def run(url, model, ctrl, schedule, dt, out_dir):
    total         = sum(s["ticks"] for s in schedule)
    tick_schedule = [s for s in schedule for _ in range(s["ticks"])]
    fifo          = collections.deque()
    prompt_idx    = 0
    sink          = ResultsSink()

    print(f"\n{'═'*72}")
    print(f"CASCADE RUN  {total} ticks  dt={dt}s")
    for s in schedule:
        print(f"  {s['ticks']:3d}t  λ={s['lambda']}  L_target={s['L_target']}ms  [{s['label']}]")
    print(f"{'═'*72}\n")
    print(f"{'tick':>5} {'λ':>3} {'L_tgt':>6} {'q_sw':>5} {'q_ref':>6} "
          f"{'B':>3} {'l_tot':>7} {'e_l':>7} {'e_q':>6}  label")
    print("-" * 72)

    log = {k: [] for k in ["tick","lambda","L_target","q_sw","q_ref",
                            "B","l_meas","e_l","e_q","xi_l","xi_q","label"]}

    for tick, seg in enumerate(tick_schedule):
        lam   = seg["lambda"]
        L_tgt = seg["L_target"]
        t0    = time.perf_counter()

        # 1. Collect l_total from completed requests
        completed = sink.drain()
        valid     = [x for x in completed if not math.isnan(x) and x > 0]
        l_meas    = statistics.mean(valid) if valid else L_tgt  # fallback

        # 2. Poisson arrivals → FIFO
        a_k = int(np.random.poisson(lam))
        for _ in range(a_k):
            fifo.append(PROMPTS[prompt_idx % len(PROMPTS)])
            prompt_idx += 1
        q_sw = len(fifo)

        # 3. Control
        B_cmd, q_ref, e_l, e_q, xi_l, xi_q = ctrl.step(l_meas, q_sw, L_tgt)

        # 4. Dispatch
        n_disp = min(B_cmd, len(fifo))
        batch  = [fifo.popleft() for _ in range(n_disp)]
        if batch:
            threading.Thread(target=fire_batch_ltotal,
                             args=(url, model, batch, sink),
                             daemon=True).start()

        # 5. Tick clock
        elapsed = time.perf_counter() - t0
        if elapsed < dt:
            time.sleep(dt - elapsed)

        print(f"{tick+1:5d} {lam:3.0f} {L_tgt:6.0f} {q_sw:5d} {q_ref:6.1f} "
              f"{B_cmd:3d} {l_meas:7.1f} {e_l:7.1f} {e_q:6.1f}  {seg['label']}")

        for k, v in [("tick",tick+1),("lambda",lam),("L_target",L_tgt),
                     ("q_sw",q_sw),("q_ref",q_ref),("B",B_cmd),
                     ("l_meas",l_meas),("e_l",e_l),("e_q",e_q),
                     ("xi_l",xi_l),("xi_q",xi_q),("label",seg["label"])]:
            log[k].append(v)

    return log


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_results(log, schedule, out_dir):
    ticks  = log["tick"]
    COLORS = ["#f0f0f0","#ffe8e8","#e8ffe8","#e8e8ff","#fff8e8","#ffe8ff","#e8f8ff"]

    fig, axes = plt.subplots(4, 1, figsize=(15, 11), sharex=True)
    fig.suptitle(
        "Chapter 5 — Cascade controller on live vLLM\n"
        "l_total = queue_wait + TTFT  |  Outer: l→q_ref (K_il>0)  |  Inner: q→B (K_q>0)",
        fontsize=12, fontweight="bold")

    seg_start = 0
    for ci, seg in enumerate(schedule):
        seg_end = seg_start + seg["ticks"]
        for ax in axes:
            ax.axvspan(seg_start+1, seg_end+1, alpha=0.22,
                       color=COLORS[ci % len(COLORS)], zorder=0)
        axes[0].text((seg_start+seg_end)/2+1, 0.97, seg["label"],
                     transform=axes[0].get_xaxis_transform(),
                     ha="center", va="top", fontsize=7.5, color="#333333")
        seg_start = seg_end

    axes[0].plot(ticks, log["l_meas"], "b-", lw=1.3, label="l_total [ms]", zorder=3)
    axes[0].step(ticks, log["L_target"], "k--", lw=1.5, where="post", label="L_target", zorder=3)
    axes[0].set_ylabel("l_total [ms]"); axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(True, alpha=0.4)

    axes[1].fill_between(ticks, log["q_sw"], step="post", color="orange", alpha=0.35, zorder=2)
    axes[1].step(ticks, log["q_sw"],  "darkorange", lw=1.2, where="post", label="q_sw (FIFO)", zorder=3)
    axes[1].step(ticks, log["q_ref"], "g--", lw=1.5, where="post", label="q_ref (outer cmd)", zorder=3)
    axes[1].set_ylabel("Queue [req]"); axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(True, alpha=0.4)

    axes[2].step(ticks, log["B"],      "m-",  lw=1.5, where="post", label="B (dispatch)", zorder=3)
    axes[2].step(ticks, log["lambda"], "k--", lw=1.0, where="post", label="λ (arrivals)", zorder=3)
    axes[2].set_ylabel("Req / tick"); axes[2].legend(loc="upper right", fontsize=9)
    axes[2].grid(True, alpha=0.4)

    ax3  = axes[3]
    ax3b = ax3.twinx()
    ax3.plot(ticks, log["xi_l"],  "g-",  lw=1.2, label="ξ_l (outer)", zorder=3)
    ax3b.plot(ticks, log["xi_q"], "r--", lw=1.0, label="ξ_q (inner)", zorder=3)
    ax3.axhline(0, color="k", lw=0.7, ls="--")
    ax3.set_ylabel("ξ_l", color="g"); ax3b.set_ylabel("ξ_q", color="r")
    ax3.set_xlabel("Tick [k]")
    l1, n1 = ax3.get_legend_handles_labels(); l2, n2 = ax3b.get_legend_handles_labels()
    ax3.legend(l1+l2, n1+n2, loc="upper left", fontsize=9); ax3.grid(True, alpha=0.4)

    fig.tight_layout()
    ts   = datetime.now().strftime("%H%M%S")
    path = out_dir / f"ch5_cascade_{ts}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {path}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params",      default=str(DEFAULT_PARAMS))
    ap.add_argument("--url",         default=DEFAULT_URL)
    ap.add_argument("--model",       default=DEFAULT_MODEL)
    ap.add_argument("--dt",          type=float, default=1.0)
    ap.add_argument("--preset",      default="rich", choices=list(PRESETS.keys()))
    ap.add_argument("--out_dir",     default=str(Path(__file__).parent))
    ap.add_argument("--n_ticks",     type=int,   default=None)
    ap.add_argument("--lambda_mean", type=float, default=None)
    ap.add_argument("--L_target",    type=float, default=None)
    args = ap.parse_args()

    if not Path(args.params).exists():
        sys.exit(f"Not found: {args.params}\nRun: python3 design_controller.py first.")
    with open(args.params) as f:
        params = json.load(f)
    if "inner" not in params or "outer" not in params:
        sys.exit("Params missing 'inner'/'outer' -- run design_controller.py first.")
    if args.L_target:
        params["L_target"] = args.L_target

    ic = params["inner"]; oc = params["outer"]
    assert ic["K_q"] > 0, f"K_q must be positive, got {ic['K_q']}"
    assert oc["K_il"] > 0, f"K_il must be positive, got {oc['K_il']}"
    print(f"Cascade controller:")
    print(f"  Inner: K_q={ic['K_q']:.4f} (>0 ✓)  K_i={ic['K_i']:.4f}  "
          f"B0={ic['B0']}  poles={ic.get('poles_cl','?')}")
    print(f"  Outer: K_il={oc['K_il']:.8f} (>0 ✓)  tau_out={oc.get('tau_out','?')}s  "
          f"beta_q={oc.get('beta_q','?')} ms/req")
    print(f"  L_target={params['L_target']:.0f} ms\n")

    try:
        requests.get(f"{args.url}/health", timeout=5).raise_for_status()
        print(f"vLLM healthy at {args.url}")
    except Exception as e:
        sys.exit(f"vLLM not reachable: {e}")

    schedule = PRESETS[args.preset] if not args.n_ticks else [
        {"ticks": args.n_ticks, "lambda": args.lambda_mean or 3,
         "L_target": args.L_target or params["L_target"], "label": "Steady"}]
    if args.L_target and not args.n_ticks:
        for s in schedule:
            s["L_target"] = args.L_target

    ctrl    = CascadeController(params)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log     = run(args.url, args.model, ctrl, schedule, args.dt, out_dir)

    print("\n=== Per-segment summary ===")
    seg_start = 0
    for seg in schedule:
        se    = seg_start + seg["ticks"]
        valid = [x for x in log["l_meas"][seg_start:se] if not math.isnan(x)]
        if valid:
            p95 = sorted(valid)[int(0.95*len(valid))]
            print(f"  {seg['label']:22s}  λ={seg['lambda']}  L_tgt={seg['L_target']}  "
                  f"l_mean={statistics.mean(valid):.0f}ms  p95={p95:.0f}ms  "
                  f"B_mean={statistics.mean(log['B'][seg_start:se]):.1f}  "
                  f"q_mean={statistics.mean(log['q_sw'][seg_start:se]):.1f}")
        seg_start = se

    ts   = datetime.now().strftime("%H%M%S")
    path = out_dir / f"ch5_cascade_log_{ts}.json"
    with open(path, "w") as f:
        json.dump({"params": params, "schedule": schedule, "log": log}, f, indent=2)
    print(f"\n  Log: {path}")
    plot_results(log, schedule, out_dir)


if __name__ == "__main__":
    main()
