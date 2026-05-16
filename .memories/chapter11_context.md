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

---

## 2026-05-16 Update: Chapter 11 Phase 1 Completed and Committed

### High-level state

Chapter 11 has now moved from design notes into working code and measured
data. Phase 1, the open-loop token-budget plant characterization, is complete
and committed in the iCloud repo:

```text
/Users/hvasudevan/Library/Mobile Documents/com~apple~CloudDocs/git_src_personal/llm-serving-control
```

Current branch:

```text
chapter-10-experimental-not-for-merge-yet
```

Commit:

```text
b56e2b4 Add chapter 11 open-loop budget sweep
```

This commit includes:

- Chapter 11 Modal deployment.
- Chapter 11 vLLM scheduler hook.
- Chapter 11 wrapper endpoint.
- Open-loop budget sweep runner.
- Dependency-free SVG plotting.
- Successful Phase 1 result logs, summary JSON/CSV, and SVG plots.
- Chapter 11 README updated to embed all Phase 1 plots directly.

### Important repository note

The `.memories` directory being updated here lives in:

```text
/Users/hvasudevan/Documents/MATLAB/llm-serving-control-ch10/.memories
```

The active code work in this session happened in the iCloud repo path:

```text
/Users/hvasudevan/Library/Mobile Documents/com~apple~CloudDocs/git_src_personal/llm-serving-control
```

The user asked to keep memories updated before proceeding to Phase 2.

### Phase 1 implementation details

New/updated files committed in `b56e2b4`:

```text
chapter_11/README.md
chapter_11/modal_vllm_wrapper.py
chapter_11/remote/vllm_modal_wrapper.py
chapter_11/remote/ch11_vllm/__init__.py
chapter_11/remote/ch11_vllm/controlled_scheduler.py
chapter_11/python/run_budget_sweep.py
chapter_11/python/plot_budget_sweep.py
chapter_11/python/results/budget_sweep_20260516_204118/*
```

The scheduler file is:

```text
chapter_11/remote/ch11_vllm/controlled_scheduler.py
```

It currently implements Phase 1 open-loop operation in the committed version.
The scheduler reads:

```text
/tmp/ch11_scheduler_control.json
```

with payloads like:

```json
{
  "mode": "open_loop",
  "admission_fraction": 0.5,
  "enabled": true
}
```

It applies the fraction by temporarily capping both:

```python
self.max_num_scheduled_tokens
self.max_num_running_reqs
```

before calling `super().schedule()`, then restoring the original nominal
values afterward. This is the same actuator proven in Chapter 10, now used
directly rather than through a queue-wait PI loop.

The wrapper file is:

```text
chapter_11/remote/vllm_modal_wrapper.py
```

Important endpoints in the committed Phase 1 version:

```text
GET  /health
GET  /metrics
GET  /metrics/prom
GET  /power
POST /v1/completions
POST /control/admission_fraction
POST /run_internal_budget_sweep
```

`POST /run_internal_budget_sweep` runs the benchmark inside the Modal
container, close to vLLM and NVML, avoiding Modal ingress artifacts. It writes
each open-loop fraction to the scheduler control file, runs a load block,
samples power, collects vLLM Prometheus metrics, and returns summaries.

The local runner is:

```text
chapter_11/python/run_budget_sweep.py
```

It uses only Python standard-library networking after an early failure where
the local Python did not have `requests` installed. It writes:

```text
sweep_request.json
sweep_response.json
sweep_summary.json
sweep_summary.csv
plot_manifest.json
logs/run_budget_sweep.log
plots/*.svg
```

The plotter is:

```text
chapter_11/python/plot_budget_sweep.py
```

It intentionally does not depend on matplotlib. It generates SVG plots using
only the Python standard library so plots can be produced on the local machine
without network installs.

### Modal deployment details

The stable deployed endpoint used for the successful run was:

```text
https://hvasudevan--chapter-11-token-budget-serve.modal.run
```

The app name is:

```text
chapter-11-token-budget
```

The deployment file:

```text
chapter_11/modal_vllm_wrapper.py
```

uses:

```text
MODEL = Qwen/Qwen2.5-3B-Instruct
GPU = T4
vllm>=0.16,<0.17
max_num_seqs = 64
max_model_len = 2048
--scheduler-cls ch11_vllm.controlled_scheduler.ControlledScheduler
```

Important lesson: `modal serve` was not suitable for measurement once result
files were being written under `chapter_11/python/results/`, because the dev
server watched the chapter directory and hot-reloaded/killed the container
mid-run. For measurement, use `modal deploy`, not `modal serve`.

The command that succeeded:

```bash
.modal-venv/bin/modal deploy chapter_11/modal_vllm_wrapper.py
```

After the run, the app was stopped successfully with:

```bash
.modal-venv/bin/modal app stop chapter-11-token-budget --yes
```

At the end of the session there were no stale local `modal serve` or
`run_budget_sweep.py` processes.

### Failed / aborted run attempts before the successful Phase 1 run

There were two useful false starts:

1. Initial local runner failed because `requests` was not installed in either
   system Python or `.modal-venv` Python. The runner was patched to use
   `urllib.request` and `urllib.error` instead.

2. An initial full sweep attempt at:

   ```text
   offered_rate_qps = 8
   duration_s = 60
   warmup_s = 10
   max_tokens = 64
   prompt_repeat = 128
   fractions = [1.0, 0.75, 0.5, 0.25, 0.1, 0.05]
   ```

   overloaded vLLM even at `admission_fraction = 1.0`.
   Modal logs showed large vLLM waiting backlog, e.g. running 64 requests and
   waiting over 100 requests. That run would have measured runaway overload
   rather than the static token-budget plant curve. It was stopped intentionally.

The final successful run used lower load and shorter prompts/generations.

### Successful Phase 1 run

Successful run directory:

```text
chapter_11/python/results/budget_sweep_20260516_204118
```

Run configuration:

```text
admission_fractions = [1.0, 0.75, 0.5, 0.25, 0.1, 0.05]
offered_rate_qps = 2
duration_s = 45
warmup_s = 10
settle_s = 2
max_tokens = 32
prompt_repeat = 64
metric_period_s = 0.5
seed = 10
```

The command used:

```bash
python3 chapter_11/python/run_budget_sweep.py \
  --url https://hvasudevan--chapter-11-token-budget-serve.modal.run \
  --admission-fractions 1.0 0.75 0.5 0.25 0.1 0.05 \
  --offered-rate-qps 2 \
  --duration-s 45 \
  --warmup-s 10 \
  --settle-s 2 \
  --metric-period-s 0.5 \
  --max-tokens 32 \
  --prompt-repeat 64 \
  --out-dir chapter_11/python/results
```

Generated result files:

```text
chapter_11/python/results/budget_sweep_20260516_204118/NOTES.md
chapter_11/python/results/budget_sweep_20260516_204118/sweep_request.json
chapter_11/python/results/budget_sweep_20260516_204118/sweep_response.json
chapter_11/python/results/budget_sweep_20260516_204118/sweep_summary.json
chapter_11/python/results/budget_sweep_20260516_204118/sweep_summary.csv
chapter_11/python/results/budget_sweep_20260516_204118/plot_manifest.json
chapter_11/python/results/budget_sweep_20260516_204118/logs/run_budget_sweep.log
chapter_11/python/results/budget_sweep_20260516_204118/plots/phase1_dashboard.svg
chapter_11/python/results/budget_sweep_20260516_204118/plots/ttft_mean_ms.svg
chapter_11/python/results/budget_sweep_20260516_204118/plots/ttft_p95_ms.svg
chapter_11/python/results/budget_sweep_20260516_204118/plots/total_mean_ms.svg
chapter_11/python/results/budget_sweep_20260516_204118/plots/total_p95_ms.svg
chapter_11/python/results/budget_sweep_20260516_204118/plots/throughput_req_s.svg
chapter_11/python/results/budget_sweep_20260516_204118/plots/gpu_power_mean_w.svg
chapter_11/python/results/budget_sweep_20260516_204118/plots/gpu_power_peak_w.svg
chapter_11/python/results/budget_sweep_20260516_204118/plots/energy_per_request_j.svg
chapter_11/python/results/budget_sweep_20260516_204118/plots/error_rate.svg
chapter_11/python/results/budget_sweep_20260516_204118/plots/vllm_queue_wait_mean_ms.svg
```

Chapter 11 README embeds all of these plots directly using relative Markdown
image links, so they should render in GitHub.

### Successful run headline data

From `sweep_summary.csv`:

```text
frac   TTFT mean   TTFT p95    throughput   power mean   energy/req   vLLM queue wait
1.00   103 ms      148 ms      2.31 req/s   64.2 W       34.5 J       21.6 ms
0.75   117 ms      167 ms      1.80 req/s   64.0 W       43.6 J       0.14 ms
0.50   122 ms      173 ms      1.89 req/s   65.1 W       42.3 J       0.38 ms
0.25   118 ms      178 ms      2.20 req/s   66.0 W       37.1 J       0.15 ms
0.10   310 ms      1244 ms     2.07 req/s   65.3 W       39.9 J       173 ms
0.05   617 ms      3434 ms     1.73 req/s   66.3 W       51.2 J       466 ms
```

Exact summary values:

```text
fraction=1.0:
  requests_ok=104, error_rate=0
  throughput_req_s=2.311111111111111
  ttft_mean_ms=103.05267627884618
  ttft_p95_ms=147.79650099998776
  total_mean_ms=1668.257124423078
  total_p95_ms=2101.067180000001
  vllm_queue_wait_mean_ms=21.620860808333475
  vllm_ttft_mean_ms=255.99270264307657
  vllm_e2e_mean_ms=1882.8847249348958
  gpu_power_mean_w=64.19623636363636
  gpu_power_peak_w=78.125
  energy_per_request_j=34.50377439324092

fraction=0.75:
  requests_ok=81, error_rate=0
  throughput_req_s=1.8
  ttft_mean_ms=117.30393349382793
  ttft_p95_ms=167.23733800000673
  total_mean_ms=1735.8433253703702
  total_p95_ms=2137.6433690000167
  vllm_queue_wait_mean_ms=0.14162526804167222
  vllm_ttft_mean_ms=91.87905813000866
  vllm_e2e_mean_ms=1644.4492708776415
  gpu_power_mean_w=64.03308333333334
  gpu_power_peak_w=79.105
  energy_per_request_j=43.59788565017404

fraction=0.5:
  requests_ok=85, error_rate=0
  throughput_req_s=1.8888888888888888
  ttft_mean_ms=122.4547243058864
  ttft_p95_ms=173.43131900003073
  total_mean_ms=1873.6600544588268
  total_p95_ms=2135.7204459999934
  vllm_queue_wait_mean_ms=0.3849049622607159
  vllm_ttft_mean_ms=102.98381211622706
  vllm_e2e_mean_ms=1852.3916388457676
  gpu_power_mean_w=65.06337962962962
  gpu_power_peak_w=84.175
  energy_per_request_j=42.28939681468866

fraction=0.25:
  requests_ok=99, error_rate=0
  throughput_req_s=2.2
  ttft_mean_ms=118.24953434343529
  ttft_p95_ms=177.6821670000004
  total_mean_ms=1762.2570660505046
  total_p95_ms=2231.84833199997
  vllm_queue_wait_mean_ms=0.14862168907276108
  vllm_ttft_mean_ms=92.86395441584226
  vllm_e2e_mean_ms=1675.0964076579119
  gpu_power_mean_w=65.95971559633027
  gpu_power_peak_w=78.371
  energy_per_request_j=37.134507423153046

fraction=0.1:
  requests_ok=93, error_rate=0
  throughput_req_s=2.066666666666667
  ttft_mean_ms=310.4496951290352
  ttft_p95_ms=1243.7579730000152
  total_mean_ms=1885.7367484731212
  total_p95_ms=3218.5214059999794
  vllm_queue_wait_mean_ms=173.0569180263176
  vllm_ttft_mean_ms=258.6659925025806
  vllm_e2e_mean_ms=1784.7361313669305
  gpu_power_mean_w=65.26239285714286
  gpu_power_peak_w=80.128
  energy_per_request_j=39.89837657790033

fraction=0.05:
  requests_ok=78, error_rate=0
  throughput_req_s=1.7333333333333334
  ttft_mean_ms=616.5391024102561
  ttft_p95_ms=3433.5970709999515
  total_mean_ms=1799.7791207435898
  total_p95_ms=5284.445563000019
  vllm_queue_wait_mean_ms=466.20644847572896
  vllm_ttft_mean_ms=528.8571408651408
  vllm_e2e_mean_ms=1683.148972039084
  gpu_power_mean_w=66.32051260504201
  gpu_power_peak_w=74.657
  energy_per_request_j=51.245195691142676
```

### Interpretation of Phase 1

For this offered load and prompt shape, the useful TTFT-control operating
region appears to be:

```text
admission_fraction ~= 0.25 to 1.0
```

Fractions `1.0`, `0.75`, `0.5`, and `0.25` all have fairly similar TTFT
means and p95s. The strongest degradation starts below `0.25`.

At:

```text
admission_fraction = 0.1
```

TTFT mean jumps to ~310 ms and p95 to ~1.24 s.

At:

```text
admission_fraction = 0.05
```

TTFT mean jumps to ~617 ms and p95 to ~3.43 s.

Mean GPU power does NOT fall meaningfully with reduced fraction in this
experiment. It stays around 64-66 W across the sweep. This means the low
fractions mostly harm response time and energy/request rather than producing
a useful power reduction.

Energy/request is worst at `0.05` because throughput falls while power remains
roughly constant.

Queue wait remains close to zero through `0.25`, then grows at `0.1` and
`0.05`. This supports the Chapter 10 conclusion: queue wait is not the right
primary control variable for normal continuous batching, but it reappears only
when the token budget is pushed into an overloaded/throttled regime.

### Practical implication for Phase 2 TTFT control

Do NOT let the TTFT controller freely drive fraction all the way to 0.01.
Based on Phase 1, a safer initial clamp is:

```text
fraction_min = 0.25
fraction_max = 1.0
```

The sign convention for TTFT mode remains:

```text
e = (target_ttft_ms - measured_ttft_ms) / target_ttft_ms
```

Then:

```text
positive error -> measured TTFT below target -> increase admission_fraction
negative error -> measured TTFT above target -> decrease admission_fraction
```

This is the opposite sign from the failed Chapter 10 queue-wait controller.

Suggested first TTFT targets for Phase 2:

```text
150 ms
200 ms
300 ms
```

Because the open-loop TTFT mean is already around 100-120 ms for fractions
0.25-1.0, a 150 ms target may saturate high or hover near high fraction under
qps=2. A 200 ms target will likely be easy and may also saturate high unless
load is increased. A 300 ms target may reveal more controller movement if load
or prompt shape causes TTFT pressure.

Initial conservative gains:

```text
kp = 0.10 to 0.15
ki = 0.01 to 0.02
control_period_s = 0.25 or 0.5
```

Because TTFT samples are event-driven and noisy, wrapper-side smoothing should
use a rolling window. Avoid updating the scheduler PI on every GPU schedule
call with stale or empty samples.

### Phase 2 partial work after commit b56e2b4

After committing Phase 1, the user asked to "go to phase 2". Some partial
Phase 2 code edits were made, but the user interrupted and asked to update
memories before proceeding. These Phase 2 edits are NOT committed yet.

Current expected git state after memory update:

```text
branch: chapter-10-experimental-not-for-merge-yet
ahead of origin by 1 commit
working tree modified:
  chapter_11/modal_vllm_wrapper.py
  chapter_11/remote/ch11_vllm/controlled_scheduler.py
  chapter_11/remote/vllm_modal_wrapper.py
```

The partial uncommitted Phase 2 changes include:

1. `chapter_11/remote/ch11_vllm/controlled_scheduler.py`

   Started adding:

   - `mode == "ttft"` support.
   - TTFT PI state:

     ```python
     self.kp
     self.ki
     self.target_ttft_ms
     self.measured_ttft_ms
     self._xi
     self.fraction_min
     self.fraction_max
     ```

   - `fraction_min` default of `0.25`.
   - PI update method `_update_ttft_pi()`.
   - Status-file writing to `/tmp/ch11_scheduler_status.json`.
   - Actuator capping now applies for `mode in {"open_loop", "ttft"}`.

   Important: this code was not yet fully validated after the interruption.
   Before running Modal again, inspect it carefully and run `py_compile`.

2. `chapter_11/modal_vllm_wrapper.py`

   Started adding:

   ```text
   CH11_STATUS_FILE=/tmp/ch11_scheduler_status.json
   ```

3. `chapter_11/remote/vllm_modal_wrapper.py`

   Started adding:

   - `STATUS_FILE = "/tmp/ch11_scheduler_status.json"`
   - `CONTROL_WRITE_LOCK`
   - `scheduler_status()`
   - `write_scheduler_control(payload)`
   - Inclusion of scheduler status fields in `/metrics`
   - Refactor of Phase 1 write to use `write_scheduler_control(control)`

   This is not yet a complete Phase 2 wrapper. Still needed:

   - `POST /control/ttft_target`
   - `POST /run_internal_ttft_sweep`
   - Wrapper-side loop to compute rolling measured TTFT during a load run and
     repeatedly write `measured_ttft_ms` into the control file.
   - Per-sample logging of:

     ```text
     time
     target_ttft_ms
     measured_ttft_ms
     scheduler_admission_fraction
     scheduler_token_cap
     scheduler_running_cap
     gpu_power_w
     vllm waiting/running
     ```

   - A local runner, probably:

     ```text
     chapter_11/python/run_ttft_sweep.py
     ```

   - TTFT closed-loop plots:

     ```text
     target vs measured TTFT over time
     admission_fraction over time
     token_cap over time
     running_cap over time
     GPU power over time
     throughput / request completions
     ```

### Recommended next step when resuming

Before doing more Phase 2 work:

1. Inspect uncommitted diff:

   ```bash
   git diff -- chapter_11/modal_vllm_wrapper.py \
     chapter_11/remote/ch11_vllm/controlled_scheduler.py \
     chapter_11/remote/vllm_modal_wrapper.py
   ```

2. Decide whether to keep the partial Phase 2 edits or reset just those files
   to `b56e2b4` and reimplement more cleanly. Do NOT use destructive git
   commands without explicit user approval.

3. Run:

   ```bash
   python3 -m py_compile \
     chapter_11/modal_vllm_wrapper.py \
     chapter_11/remote/vllm_modal_wrapper.py \
     chapter_11/remote/ch11_vllm/controlled_scheduler.py
   ```

4. Finish Phase 2 in small steps:

   - Scheduler TTFT PI mode.
   - Wrapper feedback writer.
   - TTFT sweep endpoint.
   - Local TTFT runner.
   - Time-series SVG plots.
   - Short smoke run before a long run.

### Git / process notes

The Phase 1 commit succeeded with:

```text
b56e2b4 Add chapter 11 open-loop budget sweep
```

The repo is now ahead of origin by one commit.

Staging required escalation because the iCloud-backed repo path initially
failed to create `.git/index.lock` under sandbox permissions.

No push was performed.

No branch change was performed.

The final committed README embeds the Phase 1 SVG plots and summary table.
