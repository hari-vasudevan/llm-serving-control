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
