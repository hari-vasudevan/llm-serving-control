# Chapter 11: Dispatch-Delay PI TTFT Controller (updated 2026-05-17)

## Summary

Closed-loop TTFT regulator using a dispatch-delay actuator. The controller
`time.sleep(delay_ms/1000)` before each request send; `t_send` is captured
BEFORE the sleep so the delay appears in the measured TTFT. The vLLM
scheduler stays in open_loop at fraction=1.0 — no internal coupling.

Plant: `TTFT = dispatch_delay + vLLM_natural_prefill (~130ms)`
Positive gain, unlimited upside, no load dependency.

## Architecture

```
run_load_step.py  (local)
  └─ POST /run_internal_load_step → Modal wrapper
       ├── feedback_loop thread (every 0.1s)
       │     rolling TTFT mean (window=W)
       │     e_norm = (target - measured) / target
       │     velocity-form PI → dispatch_delay_ms
       └── load_gen thread (per request)
             sleep(dispatch_delay_ms/1000)   ← t_send captured BEFORE this
             POST /v1/completions → vLLM
             record ttft_ms = t_first_token - t_send
```

## Phase Status

| Phase | Gains | Window | Run directory | Notes |
|-------|-------|--------|--------------|-------|
| 1 | open-loop | — | `budget_sweep_20260516_204118` | fraction sweep, qps=2 |
| 2 | kp=0.05 ki=0.005 | 10 | (no run dir saved) | per-target HTTP calls |
| 3 | kp=0.03 ki=0.002 | 10 | `load_step_20260517_212627` | chained, transient hidden |
| 3b | kp=0.03 ki=0.002 | 3 | `load_step_20260517_223115` | transient visible |

## Key Non-Obvious Bugs Fixed

1. `t_send` after sleep → delay invisible to feedback (fix: capture before sleep)
2. e_norm sign error with negative-gain plant (dispatch-delay is positive gain)
3. vLLM preemption crash from capping `max_num_running_reqs` in ttft mode
4. Video pending dict keyed by `sent_at_s` → ms collisions at qps=8 (fix: use index `i`)
5. Video answers fire at `recv_at_s` (full completion) → entries scroll off before answer renders (fix: fire at `sent_at_s + ttft_ms/1000`)
6. LaTeX crash in video from `$` in LLM answers (fix: escape `$` → `\$`)

## Gain Tuning

```
τ_total = window/(2×qps) + (window-1)/2×(1/qps)
PM = 180° - 90° - kp × feedback_rate × τ_total × (180/π)

window=10, qps=4: τ_total=2.375s → kp=0.03 → PM=49°  (well-damped)
window=3,  qps=4: τ_total=0.625s → kp=0.03 → PM=80°  (very stable)
```

kp_max for PM>45°: kp < π/(2 × τ_total × feedback_rate)

## Window=3 Finding

window=10 hides QPS-step transients because τ_total=2.375s smears and the
PI cancels the residual within the same window. window=3 drops τ_total to
0.625s — transient survives the MA filter. Evidence: std spike from 39ms
to 99ms at 200ms/qps=8. Controller still regulates correctly (mean within 5ms).

## Tooling

- `run_load_step.py` — chained multi-target, default ttft-window=3, kp=0.03, ki=0.002
- `plot_load_step.py` — 5 panels (Load, TTFT, Delay, Eff.Power, Energy/req); MATLAB absolute paths via `.resolve()`
- `make_video.py` — smooth QPS-proportional scroll speed; first-token answer timing; 4 right panels; `_clean()` LaTeX escape; enumeration uid

## Duty-Cycle Power Metrics

```python
active_ms = measured_ttft_ms - dispatch_delay_ms
duty      = min(qps * active_ms / 1000.0, 1.0)
eff_power = gpu_power_w * duty          # attribution, not actual GPU reduction
energy/req = gpu_power_w * active_ms / 1000.0
```

GPU stays at ~95-100% regardless — memory-bandwidth floor (T4 reads 6GB
Qwen2.5-3B weights per token step). These metrics attribute power to active
computation time only.

## Question Bank

491 unique questions across 14 topic categories. At qps=8 × 60s = 480 requests,
all unique — prevents repeated prompts and LaTeX science-formula crashes.

## Phase 3b Results (window=3)

Run: `python/results/load_step_20260517_223115`

| Target | QPS | Mean    | Std     |
|--------|-----|---------|---------|
| 200ms  | 4   | 198.6ms | 39.2ms  |
| 200ms  | 8   | 210.6ms | 99.7ms  | ← transient visible |
| 200ms  | 4   | 202.7ms | 41.6ms  |
| 350ms  | 4   | 349.6ms | 49.8ms  |
| 350ms  | 8   | 351.3ms | 28.7ms  |
| 500ms  | 8   | 503.2ms | 43.6ms  |

All means within 5ms of setpoint.
