# Chapter 10 — vLLM Admission Control: What We Found

## Intent

Chapter 10 returned to real vLLM serving after the Chapter 9 success with
a low-level GPU batch plant. The goal was to insert a queue-wait controller
*inside* vLLM's scheduler — one level above Chapter 9's explicit batch plant
but one level below the full top-level LLM API measured in Chapters 7 and 8.

The hypothesis: capping how many tokens vLLM schedules per step would create
a meaningful internal queue, and a PI controller could regulate queue-wait
time to a target.

---

## Architecture

```text
load generator (run_queue_wait_sweep.py)
  │
  ▼ HTTPS
Modal wrapper endpoint (vllm_modal_wrapper.py)
  │
  ├── POST /v1/completions     → proxied to local vLLM
  ├── GET  /metrics            → wrapper + vLLM + GPU snapshot
  ├── GET  /power              → NVML power/utilization
  └── POST /control/queue_wait_target → writes /tmp/ch10_scheduler_control.json
                                          ↓ read by ControlledScheduler
                                     vLLM (Qwen/Qwen2.5-3B-Instruct, T4 GPU)
                                       └── --scheduler-cls ControlledScheduler
                                              caps max_num_scheduled_tokens
                                              and max_num_running_reqs
```

The `ControlledScheduler` (in `remote/ch10_vllm/controlled_scheduler.py`)
subclasses vLLM's default scheduler via the `--scheduler-cls` hook. It reads
a target fraction from a control file every scheduling step and caps
`max_num_scheduled_tokens = int(frac × max_model_len)`.

---

## What Was Measured

An admission fraction sweep: hold the fraction fixed at each of
`[1.0, 0.75, 0.5, 0.25, 0.1, 0.05]` and run a fixed arrival process.

```text
offered_rate_qps = 2
duration_s = 45 per fraction
max_tokens = 32, prompt_repeat = 64
model = Qwen/Qwen2.5-3B-Instruct, T4 GPU
```

The sweep was run from the Chapter 11 runner (`run_budget_sweep.py`) after
the Chapter 11 infrastructure was built, because the Chapter 10 Python runner
was not yet complete. The results below are therefore recorded in the
Chapter 11 Phase 1 section, but they were the direct motivation for Chapter 11.

---

## Findings

### 1. Queue wait is always ~0ms

vLLM's continuous batching scheduler runs at GPU step frequency — roughly
once per token generation step, which at these load levels is hundreds of
times per second. Requests enter, are immediately batched into the next step,
and are never queued for any measurable duration. The target control variable
(external queue wait) did not exist.

The `ControlledScheduler` correctly capped tokens-per-step, but reducing the
token budget does not create queue wait — it increases per-request TTFT by
slowing down the prefill, while the scheduler still dispatches requests
immediately.

### 2. Token budget directly controls TTFT, not queue wait

| Fraction | TTFT mean | TTFT p95 | Throughput  | Power mean | Energy/req |
|----------|-----------|----------|-------------|------------|------------|
| 1.00     | 103 ms    | 148 ms   | 2.31 req/s  | 64.2 W     | 34.5 J     |
| 0.75     | 117 ms    | 167 ms   | 1.80 req/s  | 64.0 W     | 43.6 J     |
| 0.50     | 122 ms    | 173 ms   | 1.89 req/s  | 65.1 W     | 42.3 J     |
| 0.25     | 118 ms    | 178 ms   | 2.20 req/s  | 66.0 W     | 37.1 J     |
| 0.10     | 310 ms    | 1244 ms  | 2.07 req/s  | 65.3 W     | 39.9 J     |
| 0.05     | 617 ms    | 3434 ms  | 1.73 req/s  | 66.3 W     | 51.2 J     |

Key observations:
- TTFT is flat from fraction=1.0 down to fraction=0.25 (~100–120ms).
  Token budget in this range does not meaningfully limit the prefill.
- Below fraction=0.10, TTFT rises sharply (310ms at 0.10, 617ms at 0.05).
  The prefill is now genuinely token-starved across multiple steps.
- **GPU power is essentially flat** (~64–66W) across all fractions.
  The T4 reads Qwen2.5-3B's ~6GB weights from HBM every token step regardless;
  this is a memory-bandwidth floor, not a compute-limited workload.
  Dispatch delay (the Chapter 11 actuator) also does not move this floor —
  the GPU is at ~95–100% utilization whenever a token is being generated.
- Energy per request rises at low fractions because more GPU time is spent
  per request (slower prefill), not because the GPU draws more power.

### 3. The token budget has limited upside authority

At fraction=0.10, TTFT=310ms is a **transient artifact** (queue accumulated
during the step-down), not a steady-state setpoint. Once requests drain, the
scheduler dispatches the next request immediately and TTFT falls back toward
~130ms natural prefill. Targets above ~200ms are not achievable in steady
state via the token budget alone at these load levels.

---

## What This Led To

Chapter 10's failure defined Chapter 11's design:

1. **Wrong controlled variable**: queue wait cannot be regulated because it
   is always ~0ms in continuous batching. Switch to TTFT as the controlled
   variable — it responds monotonically to the actuator.

2. **Wrong actuator for wide range**: token budget has limited upside (TTFT
   floor ~130ms, ceiling ~200ms at reasonable fractions). Switch to
   dispatch delay — `sleep(d_ms/1000)` before the HTTP send — which has
   unlimited upside and a positive, linear plant gain.

3. **Right abstraction level**: the controller does not need to be *inside*
   vLLM's scheduler at all. Metering the arrival process externally gives a
   clean, decoupled plant. The scheduler stays in open-loop at fraction=1.0.

---

## Reproducing the Experiment

The admission fraction sweep that produced the table above was run using the
Chapter 11 infrastructure (the Ch10 Python runner was incomplete at the time).
To reproduce it with the Chapter 10 Modal wrapper:

### 1. Prerequisites

- Python 3.11+ with Modal account
- Modal CLI installed and authenticated (see root README for one-time setup)

### 2. Deploy to Modal

```bash
source .modal-venv/bin/activate
modal deploy chapter_10/modal_vllm_wrapper.py
```

Wait for the health check:

```bash
curl https://YOUR-ENDPOINT.modal.run/health
# expect: {"status":"ok","model":"Qwen/Qwen2.5-3B-Instruct",...}
```

### 3. Run a fraction sweep

The Chapter 10 runner accepts a sequence of queue-wait targets, but since
queue wait is always ~0ms in continuous batching, the meaningful parameter
to sweep is admission fraction via the `/control/queue_wait_target` endpoint.

Use the Chapter 11 budget sweep runner against the Chapter 10 endpoint:

```bash
python3 chapter_11/python/run_budget_sweep.py \
  --url https://YOUR-ENDPOINT.modal.run \
  --admission-fractions 1.0 0.75 0.5 0.25 0.1 0.05 \
  --offered-rate-qps 2 \
  --duration-s 45 \
  --warmup-s 10
```

Results are written to `chapter_11/python/results/budget_sweep_TIMESTAMP/`.

### 4. Observe the ControlledScheduler

The scheduler logs appear in Modal:

```bash
modal app logs chapter-10-vllm-admission
```

Look for lines like `CH10 ControlledScheduler loaded` and
`max_num_scheduled_tokens capped` to confirm the hook is active.

### 5. Tear down

```bash
modal app stop chapter-10-vllm-admission
```

## Files

```text
remote/vllm_modal_wrapper.py          Modal endpoint + /control + /power + /metrics
remote/ch10_vllm/controlled_scheduler.py   vLLM scheduler subclass (token-budget actuator)
python/run_queue_wait_sweep.py        Queue-wait target sweep runner (incomplete)
modal_vllm_wrapper.py                 Modal deployment entrypoint
matlab/                               Scaffold (unused — Python runner replaced MATLAB)
```

The `ControlledScheduler` scheduler hook is functional and correctly
subclasses vLLM's scheduler via `--scheduler-cls`. It is preserved as a
reference implementation for anyone wanting to cap token throughput at the
scheduler level.

---

## Model & Hardware

- **Model**: Qwen/Qwen2.5-3B-Instruct
- **GPU**: NVIDIA T4 (16GB) on Modal
- **vLLM**: 0.16.x, v1 engine
- **Scheduler**: `ControlledScheduler` via `--scheduler-cls`
