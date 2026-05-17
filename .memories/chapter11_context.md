# Chapter 11 Context: Token Throughput Control for LLM Serving

## Lineage

Chapter 11 follows from the **failure of Chapter 10's queue-wait controller**
at the GPU batch scheduling level. The key insight: continuous batching
eliminates queue dynamics at the scheduler level, so the cascade controller
from Chapters 2b-9 has no queue to regulate. The correct control variable
at this layer is the **token throughput budget**, which directly shapes
TTFT, throughput, and GPU power.

## Chapter 10 Post-Mortem (condensed)

### What worked
- `--scheduler-cls` plugin point in vLLM 0.16 loads custom scheduler
- Overriding `schedule()` to temporarily cap `self.max_num_scheduled_tokens`
  and `self.max_num_running_reqs` before calling `super().schedule()` is a
  valid actuator — it genuinely constrains tokens per step
- With Qwen2.5-3B on T4, reducing token budget to 36% of nominal shifted
  TTFT from 318ms to 108ms and total latency from 1939ms to 1805ms

### What failed
- vLLM's continuous batching scheduler runs at GPU step frequency (~100+
  times/sec). Requests never accumulate in `self.waiting` — they get popped
  immediately. Queue wait stayed at 0.0ms across all experiments.
- The PI controller had no feedback signal: error was always maximal
  (target - 0 = target), integrator wound up monotonically, fraction
  decreased without equilibrium.
- This is fundamental to continuous batching, not a bug in our code.

## Chapter 11 Design

### Goal

Build closed-loop controllers that use the proven token budget actuator:

1. **Constant-TTFT controller**: regulate time-to-first-token to a target
2. **Constant-power controller**: regulate GPU power to a target (future)

### Architecture (dispatch-delay actuator, Phase 2+3)

```
                   ┌──────────────────────────────────────────────┐
                   │  Modal Container  (T4 GPU)                   │
                   │                                              │
  target_ttft      │  ┌────────────────────────┐                  │
  ────────────────▶│  │  vllm_modal_wrapper.py │                  │
                   │  │                        │ /v1/completions  │
  ┌──────────────┐ │  │  Feedback loop thread  │────────────────▶ │
  │ Local Machine│ │  │  (PI controller)        │◀──────────────── │
  │              │ │  │                        │   TTFT measured  │
  │run_load_step │ │  │  dispatch_delay_ms     │                  │
  │  .py sends   │ │  │  └─ time.sleep(d)──────┤                  │
  │  HTTP POST   │ │  │     before each send   │                  │
  │  with list   │ │  │                        │   ┌──────────┐   │
  │  of targets  │ │  │  load_gen thread       │   │  vLLM    │   │
  │              │ │  │  sends requests at QPS │──▶│ Engine   │   │
  │ Collects     │ │  │                        │◀──│ Qwen2.5  │   │
  │ timeseries,  │ │  └────────────────────────┘   │ -3B on T4│   │
  │ qa_log,      │ │                               └──────────┘   │
  │ plots, video │ │  POST /run_internal_load_step                 │
  └──────────────┘ └──────────────────────────────────────────────┘
          │                              ▲
          └──────────────────────────────┘
                HTTPS (Modal ingress)
```

### Control loop detail

The wrapper runs a `feedback_loop` thread alongside the load generator.
Every `feedback_period_s` (0.1s):

1. Computes rolling mean TTFT from last `ttft_window` responses
2. Computes normalized error: `e_norm = (target - measured) / target`
3. Runs velocity-form PI:
   ```python
   delay_xi[0] += e_norm * feedback_period_s  # integrator (with anti-windup)
   delta_ms = (kp * e_norm + ki * delay_xi[0]) * target_ttft[0]
   dispatch_delay_ms[0] = clip(dispatch_delay_ms[0] + delta_ms, 0, max_delay_ms)
   ```
4. Sleeps each outgoing request by `dispatch_delay_ms[0]` before sending to vLLM

TTFT is measured from `t_send` (captured **before** sleep) to first token received.

### Chained multi-target experiment (Phase 3)

All targets sent in one HTTP POST. Server loops over targets with shared PI state,
clearing the MA window at each setpoint switch. Phase labels encode target:
`step_{i}_qps{q}_t{int(tgt)}ms`.

## Phase Status and Commits

| Phase | Description | Commit | Gains |
|-------|-------------|--------|-------|
| 1 | Open-loop budget sweep (qps=2) | `b56e2b4` | N/A |
| 2 | Closed-loop TTFT PI, dispatch-delay, per-target calls | `1279536` | kp=0.05, ki=0.005 |
| 3 | Load-step disturbance rejection, chained targets | latest | kp=0.03, ki=0.002 |

## Phase 3 Results (2026-05-17)

Run: `python/results/load_step_20260517_212627`
Gains: kp=0.03, ki=0.002, window=10, feedback_period=0.1s
Load: qps=4 (60s) → qps=8 (60s) → qps=4 (60s) per target

| Target | QPS | Mean    | p95     | Std    |
|--------|-----|---------|---------|--------|
| 200ms  | 4→8→4 | ~199ms | ~240ms | 12-41ms |
| 350ms  | 4→8→4 | ~350ms | ~392ms | 11-50ms |
| 500ms  | 4→8→4 | ~500ms | ~549ms | 13-50ms |

Means within 2ms of setpoint across all load levels and targets.

## Gain Tuning Analysis

The effective feedback lag combines dead time and MA filter lag:

```
τ_dead = window / (2 × qps)
τ_MA   = (window - 1) / 2 × (1 / qps)
τ_total = τ_dead + τ_MA

At qps=4, window=10: τ_total = 1.25 + 1.125 = 2.375s

PM = 180° - 90° - kp × feedback_rate × τ_total × (180/π)

kp=0.05 → PM=22° (oscillatory, 13s cycle)
kp=0.03 → PM=49.2° (well-damped) ✓
kp=0.02 → PM=62.8° (conservative)
```

Rule: target PM > 45°. For these settings: kp_max ≈ 0.033.

## GPU Utilization Note (WIP)

At qps=4-8 with Qwen2.5-3B (6GB model), the T4 is always at ~95-100% utilization
regardless of dispatch delay. The GPU is memory-bandwidth bound: it must continuously
read the model weights (6GB) from HBM to generate each token. Dispatch delay affects
only the *inter-request timing*, not the per-step GPU activity. GPU utilization only
falls below this floor at extremely low QPS (<0.5 req/s). Multi-GPU sharding or
quantization would be needed to make GPU util a meaningful control variable.

## Key Files

```text
chapter_11/
  modal_vllm_wrapper.py              — Modal deployment entrypoint
  remote/
    vllm_modal_wrapper.py            — wrapper + PI controller + load generation
    ch11_vllm/
      controlled_scheduler.py        — token-budget scheduler hook (open_loop in Phase 3)
  python/
    run_load_step.py                 — local runner (kp=0.03, ki=0.002 defaults)
    plot_load_step.py                — SVG subplots with step-function reference + MATLAB
    make_video.py                    — scrolling QA replay video (uid-keyed pending dict)
    results/
      load_step_20260517_212627/     — Phase 3 canonical results
        timeseries.json              — merged flat timeseries
        qa_log.json                  — per-request log
        plots/subplot_200_350_500ms.svg
        view_figure.m                — MATLAB viewer
        qa_video.mp4                 — 5× speed replay video
```

## Modal Deployment

```text
URL:    https://hvasudevan--chapter-11-token-budget-serve.modal.run
Deploy: /Users/hvasudevan/.venvs/modal/bin/modal deploy chapter_11/modal_vllm_wrapper.py
Stop:   /Users/hvasudevan/.venvs/modal/bin/modal app stop chapter-11-token-budget --yes
```

Cold-start for Qwen2.5-3B on T4 takes ~3-4 min. Health check uses
`min(remaining, 300s)` timeout to survive it.

## Historical Notes

### Phase 2 early failures
1. Token-budget PI sign error → fraction stuck at 1.0 (e_norm sign flipped)
2. vLLM preemption crash at qps=8 in ttft mode (fixed: only cap scheduled_tokens)
3. t_send captured after sleep → delay invisible to feedback (fixed: capture before sleep)
4. Token-budget authority insufficient at qps=8 (floor → max TTFT ≈ 170ms; use dispatch-delay)

### Phase 3 oscillation fix
kp=0.05 gave PM=22° (oscillatory). Reducing to kp=0.03 → PM=49.2°. Key insight:
MA filter adds (N-1)/2×Ts lag on top of dead time — this is what the gain stability
analysis must use, not dead time alone.
