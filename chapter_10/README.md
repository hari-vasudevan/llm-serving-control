# Chapter 10 -- vLLM Admission, Power, and Top-Level Latency

Chapter 10 brings the project back up from the clean Chapter 9 GPU batching
plant to real LLM serving.

Chapter 9 answered the control-theory question: a cascade controller can work
when the plant has an exact batch actuator, a real carry-over queue, and
measured queue-plus-service latency. Chapter 10 is deliberately more
experimental. The goal is not to prove the Chapter 2 equations again. The goal
is to measure what happens when a Chapter 9-style admission layer sits in
front of vLLM/Qwen.

## Experiment Question

For a real Qwen model served by vLLM on a GPU:

```text
If we hold the external admission queue near different mean queue-wait targets,
what happens to:

  - top-level query latency,
  - TTFT,
  - throughput,
  - vLLM internal queueing,
  - GPU power,
  - energy per request?
```

The first version of this chapter should be a measurement study, not a full
new controller proof.

## Starting Point

This chapter was initially copied from Chapter 8 because Chapter 8 already has
the useful vLLM wrapper shape:

```text
Modal web endpoint
  -> wrapper queue
  -> local vLLM server
  -> Qwen model on NVIDIA GPU
```

The copied files are intentionally not trusted as final Chapter 10 code yet.
They are scaffolding to be reshaped.

Chapter 9 remains the conceptual reference for:

- defining queue wait explicitly,
- using a ticked admission decision,
- separating queue wait from service time,
- logging `q`, `B`, latency, and completion behavior.

## Proposed Architecture

```text
load generator
  -> Chapter 10 admission wrapper
      -> external FIFO queue
      -> queue-wait target or q_ref target
      -> release policy into vLLM
  -> vLLM /v1/completions
  -> Qwen on GPU
```

The wrapper should measure:

```text
external_queue_wait = t_release_to_vllm - t_enter_wrapper
ttft_external       = t_first_token - t_release_to_vllm
total_query_latency = t_first_token - t_enter_wrapper
```

When vLLM metrics are available, it should also record:

```text
vllm_num_requests_waiting
vllm_num_requests_running
vllm_request_queue_time_seconds
vllm_time_to_first_token_seconds
vllm_e2e_request_latency_seconds
```

For power, the first NVIDIA path should sample `nvidia-smi` or NVML inside the
Modal container if available:

```text
gpu_power_w
gpu_util_percent
gpu_memory_used_mb
energy_joules ~= integral(power_w dt)
energy_per_request = energy_joules / completed_requests
```

## First Experimental Sweep

Hold the load trace fixed and sweep the admission queue-wait target:

```text
target mean external queue wait:
  0 ms      baseline-like eager release
  50 ms
  100 ms
  200 ms
  400 ms
```

For each point:

```text
1. warm up vLLM/Qwen,
2. replay the same arrival process,
3. run for a fixed measurement window,
4. record latency, queue, throughput, and power,
5. summarize mean, p50, p95, and p99.
```

The baseline should be direct native vLLM with no external admission queue:

```text
load generator -> vLLM -> Qwen
```

The controlled/admission experiment should be:

```text
load generator -> wrapper queue -> controlled release -> vLLM -> Qwen
```

## Later Demo Idea

Later, a visual demo can show side-by-side query/response behavior:

```text
native vLLM under bursty load
vs.
admission-controlled vLLM under the same load
```

That video should not be part of the first implementation. The first goal is
to produce trustworthy measurements.

## Current Files

The initial `chapter_10/` folder was copied from Chapter 8:

- `modal_vllm_wrapper.py`
  Modal deployment entrypoint. This should become the Chapter 10 vLLM
  measurement service entrypoint.
- `remote/vllm_modal_wrapper.py`
  HTTP wrapper server. This should become the admission/measurement wrapper.
- `matlab/`
  Copied MATLAB scripts. These may be kept for controller continuity, but the
  first Chapter 10 experiment will likely be easier to drive from Python
  because power sampling, vLLM metrics, and arrival replay all live near the
  server process.

## Near-Term Implementation Plan

1. Rename the copied Chapter 8 service identity to Chapter 10.
2. Add a Python experiment runner that can run baseline and wrapper-admission
   sweeps.
3. Add wrapper metrics for external queue wait, TTFT, total query latency,
   throughput, and vLLM internal metrics.
4. Add GPU power sampling.
5. Produce one CSV/JSON result bundle per queue-wait target.
6. Generate summary plots:

```text
queue-wait target -> total latency p95
queue-wait target -> TTFT p95
queue-wait target -> throughput
queue-wait target -> mean GPU power
queue-wait target -> energy/request
```

