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
  -> Chapter 10 public wrapper endpoint
  -> vLLM /v1/completions
      -> experimental custom scheduler class
      -> queue-wait target control file
      -> Qwen execution
  -> Qwen on GPU
```

The first implementation inserts the controller at the lowest practical vLLM
layer by starting vLLM with:

```text
--scheduler-cls ch10_vllm.controlled_scheduler.ControlledScheduler
```

The scheduler hook is intentionally experimental because vLLM's scheduler
class is a private extension point. The hook:

- subclasses vLLM's default scheduler,
- observes the internal waiting queue when the private fields are available,
- estimates internal waiting time,
- adjusts the scheduler's active sequence budget from a queue-wait target,
- rereads `/tmp/ch10_scheduler_control.json` so target changes do not require a
  vLLM restart.

The Modal image pins vLLM to `vllm>=0.16,<0.17` for this first experiment so
the scheduler configuration surface is not floating underneath the code.

The public wrapper still matters. It exposes:

```text
/v1/completions              proxy to local vLLM, so clients send normal queries
/metrics                     combined wrapper, vLLM, and GPU power snapshot
/power                       current NVML power/utilization snapshot
/control/queue_wait_target   update scheduler target wait in ms
```

The client/runner should measure:

```text
ttft_client         = t_first_token - t_client_send
total_query_latency = t_done - t_client_send
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
load generator -> wrapper proxy -> vLLM custom scheduler -> Qwen
```

## Current Code

- `remote/ch10_vllm/controlled_scheduler.py`
  Experimental vLLM scheduler hook. This is the lowest-layer controller
  insertion point for Chapter 10.
- `remote/vllm_modal_wrapper.py`
  Public Modal endpoint. Proxies normal top-level `/v1/completions` calls to
  the local vLLM server, exposes target-control and power/metrics endpoints.
- `python/run_queue_wait_sweep.py`
  Top-level query benchmark. Sends ordinary streaming completion requests,
  scrapes metrics, computes latency/throughput/power summaries, and writes
  CSV/JSON results.

Example benchmark command after deploying Modal:

```bash
python chapter_10/python/run_queue_wait_sweep.py \
  --url https://YOUR-MODAL-URL.modal.run \
  --target-wait-ms 0 50 100 200 400 \
  --offered-rate-qps 4 \
  --duration-s 90 \
  --warmup-s 15
```

The key validation checks for the first run:

```text
1. Modal logs show "CH10 ControlledScheduler loaded".
2. POST /control/queue_wait_target updates the scheduler control file.
3. /metrics includes gpu_power_w on NVIDIA/Modal.
4. requests.csv has finite TTFT and total latency values.
5. sweep_summary.csv shows achieved vLLM queue wait moving with target.
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
