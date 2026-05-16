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

### The correct plant model

```
token_budget (u) --> tokens_per_step --> { TTFT (y1), throughput (y2), power (y3) }
```

Not:

```
token_budget --> queue_depth --> queue_wait  (this path does not exist)
```

## Chapter 11 Design

### Goal

Build two closed-loop controllers that use the proven token budget actuator:

1. **Constant-TTFT controller**: regulate time-to-first-token to a target
   (e.g., "keep TTFT under 200ms")
2. **Constant-power controller**: regulate GPU power to a target
   (e.g., "stay under 65W")

Both adjust `admission_fraction` (0.01 to 1.0), which scales
`max_num_scheduled_tokens` and `max_num_running_reqs` per schedule() call.

### Architecture

```
                 +---------------------------------------------+
  target ---> e  |  PI controller                              |
  (TTFT or   -+  |  admission_fraction = f(e, integral)        |
   power)     |  |                                             |
              |  +---------------+-----------------------------+
              |                  | admission_fraction
              |                  v
              |  +----------------------------------------------+
              |  |  ControlledScheduler.schedule()               |
              |  |  tok_cap = nominal * frac                     |
              |  |  run_cap = nominal * frac                     |
              |  |  super().schedule()                           |
              |  +---------------+------------------------------+
              |                  |
              |                  v
              |  +----------------------------------------------+
              |  |  vLLM engine + Qwen2.5-3B on T4              |
              |  |  generates tokens, serves requests            |
              |  +---------------+------------------------------+
              |                  |
              |    measurement   |
              |  +---------------+------------------------------+
              |  |  Wrapper measures:                            |
              |  |   - rolling TTFT (from vLLM response headers) |
              |  |   - GPU power (NVML)                          |
              |  |   - throughput (completed req/s)               |
              |  +---------------+------------------------------+
              |                  | measured TTFT or power
              +------------------+
```

### Feedback path detail

The wrapper (`vllm_modal_wrapper.py`) runs inside the same container as
vLLM. It proxies requests and measures timing. Every `control_period_s`
(0.25s default), it:

1. Computes rolling mean TTFT from the last N responses
2. Reads GPU power via NVML
3. Writes both to the control file (`/tmp/ch11_scheduler_control.json`):
   ```json
   {
     "mode": "ttft",
     "target_ttft_ms": 200,
     "measured_ttft_ms": 185.3,
     "measured_power_w": 62.4,
     "enabled": true
   }
   ```
4. The scheduler reads this file every `control_period_s` and runs the PI

This is the same control-file mechanism from Ch10, extended with the
measured TTFT and power fields.

### Step-by-step implementation plan

#### Phase 1: Open-loop plant characterization (budget sweep)

Set admission_fraction to fixed values [1.0, 0.75, 0.5, 0.25, 0.1, 0.05]
and at each point measure (at qps=8 with Qwen2.5-3B on T4):
  - TTFT mean and p95
  - Total latency mean and p95
  - Throughput (req/s completed)
  - GPU power mean and peak
  - Energy per request
  - Error rate

This gives the static plant characteristic curves:
  fraction -> TTFT(fraction)
  fraction -> power(fraction)
  fraction -> throughput(fraction)

These curves determine:
  - Whether the plant is monotonic (required for PI stability)
  - The gain (slope) at operating points
  - The feasible operating range

Implementation: modify `/run_internal_sweep` to accept a `fractions` list
instead of `target_wait_ms`. For each fraction, write it directly to the
control file (controller in open-loop mode), run the benchmark, collect.

#### Phase 2: Constant-TTFT controller

Measured variable: rolling mean TTFT (computed inside the wrapper from
vLLM response timing, written to control file).

Control law: PI on normalized error
  e = (target_ttft - measured_ttft) / target_ttft
  e > 0: TTFT is below target, can increase budget (admit more)
  e < 0: TTFT is above target, must decrease budget (admit less)

Note the sign is REVERSED from the Ch10 queue-wait controller:
  - Queue wait (Ch10): positive error -> reduce fraction (want more waiting)
  - TTFT (Ch11): positive error -> increase fraction (TTFT is good, serve more)

So: admission_fraction += delta  (not -= delta as in Ch10)

Anti-windup: conditional integration, integrator clamp.

Gain design: use the TTFT(fraction) plant curve from Phase 1 to pick
kp and ki that give stable response without oscillation.

#### Phase 3: Constant-power controller

Measured variable: rolling mean GPU power from NVML (already working).

Control law: PI on normalized error
  e = (target_power - measured_power) / target_power
  e > 0: power is below target, can increase budget
  e < 0: power is above target, must decrease budget

Same sign convention as TTFT controller.

NVML polling interval: wrapper samples power every 0.5s. The rolling
window should be ~2-5s to smooth out per-step fluctuations.

#### Phase 4: Demonstration and visualization

1. Summary plots from Phase 1: six panels showing budget -> metric curves
2. Step response plots: set target, show TTFT/power tracking over time
3. Load disturbance rejection: step the arrival rate, show controller
   adjusting budget to maintain the target
4. (Later) Video: side-by-side of queries/responses under native vLLM vs
   TTFT-controlled vs power-controlled

### Key differences from Chapter 10

| Aspect              | Ch10 (failed)          | Ch11                        |
|---------------------|------------------------|-----------------------------|
| Measured variable   | vLLM queue wait        | TTFT or GPU power           |
| Why it works        | (it didn't)            | These actually respond to   |
|                     |                        | the token budget actuator   |
| Feedback path       | Scheduler internal     | Wrapper -> control file ->  |
|                     | (microsecond loop)     | scheduler (250ms loop)      |
| Error sign          | positive = under-wait  | positive = under-target     |
|                     | -> reduce fraction     | -> increase fraction        |
| Plant dynamics      | No queue forms         | TTFT and power respond      |
|                     |                        | monotonically to budget     |

### Files to create

```
chapter_11/
  README.md
  modal_vllm_wrapper.py          (copy from ch10, model=Qwen2.5-3B)
  remote/
    vllm_modal_wrapper.py        (copy from ch10, add TTFT/power feedback)
    ch11_vllm/
      __init__.py
      controlled_scheduler.py    (from ch10, replace queue-wait PI with
                                  TTFT/power PI, read feedback from
                                  control file written by wrapper)
  python/
    run_budget_sweep.py          (open-loop sweep: fixed fractions)
    run_ttft_sweep.py            (closed-loop: TTFT targets)
    run_power_sweep.py           (closed-loop: power targets)
    plot_plant_curves.py         (visualization of plant characteristics)
```

### Measurement infrastructure already working (from Ch10)

- NVML GPU power: `gpu_snapshot()` in wrapper returns gpu_power_w
- vLLM metrics: `/metrics` endpoint, prometheus format
- In-container benchmark: `/run_internal_sweep` bypasses Modal ingress
- Control file hot-reload: scheduler reads JSON from /tmp every 250ms
- Wrapper timing: TTFT measured from vLLM response headers

### Platform notes (inherited from Ch10)

- Model: Qwen/Qwen2.5-3B-Instruct on T4 GPU (Modal)
- max_num_seqs=64, max_model_len=2048
- vllm>=0.16,<0.17, v1 engine default
- max_containers=1, scaledown_window=300
- /run_internal_sweep is the correct measurement path (bypasses Modal ingress)
- nominal_max_tokens=2048, nominal_max_running=64

### Scheduler attribute names (confirmed on vLLM 0.16)

```python
self.max_num_scheduled_tokens  # token budget per step (nominal: 2048)
self.max_num_running_reqs      # sequence count ceiling (nominal: 64)
```

Both are instance attributes on the Scheduler, writable, re-read each
schedule() call by the parent implementation.

### Working directory

Code lives at:
```
/Users/hvasudevan/Documents/MATLAB/llm-serving-control-ch10
```
on branch `chapter-10-experimental-not-for-merge-yet`.

The filesystem MCP has access to `/Users/hvasudevan/Documents/MATLAB/`.
The iCloud repo path is NOT MCP-accessible. Use git push/pull to sync.

### Experimental data from Chapter 10 (for reference)

#### Qwen2.5-3B on T4, qps=8, target=0ms (full budget, frac=1.0)
  - vllm_ttft_mean_ms: 318
  - total_mean_ms: 1939
  - gpu_power_mean_w: 62.4
  - gpu_power_peak_w: 73.3
  - energy_per_request_j: 9.8

#### Qwen2.5-3B on T4, qps=8, target=100ms (frac ramped to ~0.36)
  - vllm_ttft_mean_ms: 108
  - total_mean_ms: 1805
  - gpu_power_mean_w: 69.1
  - gpu_power_peak_w: 73.8
  - energy_per_request_j: 10.7

Note: TTFT *decreased* when budget was reduced. This is because with
fewer tokens scheduled per step, there is less contention for the first
prefill pass — a smaller batch means faster prefill per-request. This
is a key plant characteristic: the TTFT(fraction) curve may be non-monotonic.
The open-loop sweep in Phase 1 will characterize this properly.
