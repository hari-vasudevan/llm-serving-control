# Chapter 11: Token Throughput Control for LLM Serving

## Background

Chapter 10 attempted to insert a classical queue-wait controller at vLLM's
GPU batch scheduling level. The actuator worked (capping `max_num_scheduled_tokens`
genuinely constrains tokens per step), but the feedback variable failed:
vLLM's continuous batching scheduler processes waiting requests at GPU step
frequency (~100+ times/sec), so queue wait is always ~0ms regardless of the
token budget. There is no queue to control.

The key finding: **at the continuous-batching scheduler level, the plant
dynamics are token-throughput-based, not queue-based.** The token budget
directly shapes TTFT, throughput, and GPU power.

## This Chapter

We redesign the controller to regulate the variables that actually respond
to the token budget actuator:

1. **Constant-TTFT controller** — set a TTFT target (e.g., 200ms), and the
   controller adjusts the token budget to maintain it under varying load.

2. **Constant-power controller** — set a GPU power target (e.g., 65W), and
   the controller adjusts the token budget to stay within the power envelope.

Both controllers share the same inner actuator: the `schedule()` override
that caps `max_num_scheduled_tokens` and `max_num_running_reqs`.

## Experiment Plan

### Phase 1: Open-loop plant characterization

Fix `admission_fraction` at [1.0, 0.75, 0.5, 0.25, 0.1, 0.05] and measure
TTFT, throughput, power, energy/request at each point. This produces the
static gain curves needed to design PI gains.

Implemented files:

```text
chapter_11/modal_vllm_wrapper.py
chapter_11/remote/vllm_modal_wrapper.py
chapter_11/remote/ch11_vllm/controlled_scheduler.py
chapter_11/python/run_budget_sweep.py
chapter_11/python/plot_budget_sweep.py
```

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

The wrapper exposes:

```text
POST /control/admission_fraction
POST /run_internal_budget_sweep
```

Run the Modal app:

```bash
.modal-venv/bin/modal serve chapter_11/modal_vllm_wrapper.py
```

Then run the open-loop sweep:

```bash
python3 chapter_11/python/run_budget_sweep.py \
  --url https://YOUR-MODAL-URL.modal.run \
  --admission-fractions 1.0 0.75 0.5 0.25 0.1 0.05 \
  --offered-rate-qps 8 \
  --duration-s 60 \
  --warmup-s 10 \
  --metric-period-s 0.5
```

Each run writes a timestamped folder under `chapter_11/python/results/`:

```text
sweep_request.json
sweep_response.json
sweep_summary.json
sweep_summary.csv
plot_manifest.json
logs/run_budget_sweep.log
plots/phase1_dashboard.svg
plots/*.svg
```

#### Phase 1 results: 2026-05-16 sweep

This committed run used:

```text
offered_rate_qps = 2
duration_s = 45
warmup_s = 10
max_tokens = 32
prompt_repeat = 64
admission_fractions = [1.0, 0.75, 0.5, 0.25, 0.1, 0.05]
```

An earlier qps=8 attempt overloaded vLLM even at full budget, so this lower
load was used to capture the static plant curve rather than a runaway backlog.

![Phase 1 dashboard](python/results/budget_sweep_20260516_204118/plots/phase1_dashboard.svg)

![Mean TTFT](python/results/budget_sweep_20260516_204118/plots/ttft_mean_ms.svg)

![P95 TTFT](python/results/budget_sweep_20260516_204118/plots/ttft_p95_ms.svg)

![Mean total latency](python/results/budget_sweep_20260516_204118/plots/total_mean_ms.svg)

![P95 total latency](python/results/budget_sweep_20260516_204118/plots/total_p95_ms.svg)

![Throughput](python/results/budget_sweep_20260516_204118/plots/throughput_req_s.svg)

![Mean GPU power](python/results/budget_sweep_20260516_204118/plots/gpu_power_mean_w.svg)

![Peak GPU power](python/results/budget_sweep_20260516_204118/plots/gpu_power_peak_w.svg)

![Energy per request](python/results/budget_sweep_20260516_204118/plots/energy_per_request_j.svg)

![Error rate](python/results/budget_sweep_20260516_204118/plots/error_rate.svg)

![vLLM queue wait](python/results/budget_sweep_20260516_204118/plots/vllm_queue_wait_mean_ms.svg)

Headline result:

```text
frac   TTFT mean   TTFT p95    throughput   power mean   energy/req
1.00   103 ms      148 ms      2.31 req/s   64.2 W       34.5 J
0.75   117 ms      167 ms      1.80 req/s   64.0 W       43.6 J
0.50   122 ms      173 ms      1.89 req/s   65.1 W       42.3 J
0.25   118 ms      178 ms      2.20 req/s   66.0 W       37.1 J
0.10   310 ms      1244 ms     2.07 req/s   65.3 W       39.9 J
0.05   617 ms      3434 ms     1.73 req/s   66.3 W       51.2 J
```

For this load and prompt shape, the practical TTFT-control region appears to
be roughly `admission_fraction >= 0.25`. Below `0.1`, TTFT and queue wait
increase sharply while mean GPU power barely falls.

### Phase 2: Closed-loop TTFT control

PI controller with TTFT as measured variable and admission_fraction as actuator.

### Phase 3: Closed-loop power control

PI controller with GPU power as measured variable and admission_fraction as actuator.

### Phase 4: Demonstration

Side-by-side comparison of uncontrolled vs TTFT-controlled vs power-controlled
serving under varying load.

## Model & Hardware

- **Model**: Qwen/Qwen2.5-3B-Instruct
- **GPU**: NVIDIA T4 (16GB) on Modal
- **vLLM**: 0.16.x, v1 engine
- **Scheduler**: custom via `--scheduler-cls`
