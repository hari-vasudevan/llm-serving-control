# LLM Serving Control: Project State (updated 2026-05-17)

## Repository

```text
path: /Users/hvasudevan/Library/Mobile Documents/com~apple~CloudDocs/git_src_personal/llm-serving-control
branch: chapter-10-experimental-not-for-merge-yet
```

## Chapter Progression

| Chapter | Topic | Status |
|---------|-------|--------|
| 2b-9 | Queue-based PI cascade controller | Complete (clean CUDA batch plant) |
| 10 | vLLM token-budget characterization | Complete, committed, post-mortem written |
| 11 | Phase 1: open-loop budget sweep | Complete (commit `b56e2b4`) |
| 11 | Phase 2: closed-loop TTFT PI — dispatch-delay actuator | Complete (commit `1279536`) |
| 11 | Phase 3: load-step disturbance rejection, chained targets | Complete (commit on this branch) |
| 11 | Phase 4: power controller or multi-GPU | Not started |

## Chapter 11 Phase 3 Status (2026-05-17, final)

Chained multi-target load-step sweep using dispatch-delay PI with gains kp=0.03,
ki=0.002, window=10, feedback_period=0.1s. Single HTTP POST sends all three targets;
server runs them sequentially with shared PI state so the resettling transient
between setpoints is visible in one continuous timeseries.

Key implementation files:
- `chapter_11/remote/vllm_modal_wrapper.py` — chained target loop, shared PI, phase labels
- `chapter_11/python/run_load_step.py` — single-call client, default gains kp=0.03 ki=0.002
- `chapter_11/python/plot_load_step.py` — SVG subplots + step-function target reference + MATLAB script
- `chapter_11/python/make_video.py` — scrolling QA replay video, uid-based pending dict

## Critical Lessons Learned

### 1. vLLM preemption crash in ttft mode
Never cap `max_num_running_reqs` dynamically in ttft mode. Only cap
`max_num_scheduled_tokens`. Capping running_reqs mid-execution forces
preemption and crashes vLLM at high load.

### 2. Plant gain sign depends on actuator
- Token-budget: NEGATIVE gain (fraction↑ → TTFT↓). e_norm = (measured - target)/target.
- Dispatch-delay: POSITIVE gain (delay↑ → TTFT↑). e_norm = (target - measured)/target.

### 3. Token-budget actuator authority is load-dependent
At qps=8, fraction=0.01 floor → max TTFT ≈ 170ms. Cannot regulate above
natural TTFT without pre-built queue. Use dispatch-delay instead.

### 4. Dispatch-delay: t_send must be captured BEFORE sleep
If t_send is captured after time.sleep(delay_s), the delay is invisible to the
rolling TTFT window — feedback sees ~130ms regardless of delay; PI saturates at ceiling.

### 5. Effective feedback lag = dead time + MA filter lag
lag = window/(2×qps) + (window-1)/2 × (1/qps).
At qps=4, window=10: dead_time=1.25s, MA_lag=1.125s, total=2.375s.
This is what limits kp — NOT dead time alone.

### 6. Phase margin formula
PM = 180° - 90° - ω_gc × τ_total × (180/π).
kp=0.05 → PM=22° (oscillatory). kp=0.03 → PM=49.2° (well-damped). Target PM > 45°.

### 7. Modal proxy timeout ~15 min
Chained 3-target run takes ~9.7 min — within limit. Per-target HTTP loop would exceed it.

### 8. GPU utilization is always high (WIP)
T4 always busy reading 3B model weights (6GB) from HBM. Dispatch delay does not
reduce GPU load. Only varies below ~0.5-1 QPS. Noted as WIP in README.

### 9. Modal container cold-start
Health check must use per-request timeout of `min(remaining, 300s)`.

## Modal Endpoint

```text
URL: https://hvasudevan--chapter-11-token-budget-serve.modal.run
App: chapter-11-token-budget
Model: Qwen/Qwen2.5-3B-Instruct
GPU: T4
Deploy cmd: /Users/hvasudevan/.venvs/modal/bin/modal deploy chapter_11/modal_vllm_wrapper.py
Stop cmd: /Users/hvasudevan/.venvs/modal/bin/modal app stop chapter-11-token-budget --yes
```

## Experimental Results

### Phase 1 (open-loop budget sweep, qps=2)
```
fraction=1.0  → TTFT=103ms, power=64W
fraction=0.10 → TTFT=310ms, power=65W  (queue artifact — not steady-state)
fraction=0.05 → TTFT=617ms, power=66W
```

### Phase 2 final (dispatch-delay PI, qps=8, kp=0.05, ki=0.005, ttft_window=10)
```
target=200ms: mean=203ms, p95=242ms, error_rate=0
target=350ms: mean=349ms, p95=393ms, error_rate=0
target=500ms: mean=498ms, p95=545ms, error_rate=0
```

### Phase 3 (load-step disturbance rejection, kp=0.03, ki=0.002, window=10)
Run directory: python/results/load_step_20260517_212627
Load steps: qps=4 (60s) → qps=8 (60s) → qps=4 (60s) per target

| Target | QPS | Mean    | p95     | Std    |
|--------|-----|---------|---------|--------|
| 200ms  | 4   | 198.4ms | 268.5ms | 41.4ms |
| 200ms  | 8   | 200.6ms | 220.2ms | 12.0ms |
| 200ms  | 4   | 200.9ms | 230.8ms | 16.9ms |
| 350ms  | 4   | 351.8ms | 437.3ms | 49.6ms |
| 350ms  | 8   | 348.9ms | 364.8ms | 10.7ms |
| 350ms  | 4   | 351.0ms | 375.8ms | 17.9ms |
| 500ms  | 4   | 502.1ms | 598.0ms | 49.9ms |
| 500ms  | 8   | 499.9ms | 520.4ms | 13.3ms |
| 500ms  | 4   | 499.5ms | 528.6ms | 20.0ms |

Means within 2ms of setpoint. Std drops 3-4× at qps=8 vs qps=4 (more samples per MA window).
