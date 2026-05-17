# Chapter 10: Post-Mortem — vLLM Queue-Wait Controller (updated 2026-05-17)

## What Was Tried

Queue-wait controller inserted at vLLM's GPU batch scheduling level using
the `--scheduler-cls` extension point. The `ControlledScheduler` subclasses
vLLM's default scheduler and caps `max_num_scheduled_tokens` and
`max_num_running_reqs` before each `super().schedule()` call.

## What Was Found

### Actuator fix (session 2)
The original actuator (mutating `scheduler_config.max_num_seqs`) was broken:
vLLM copies it to `self.max_num_running_reqs` at init and never re-reads.
Fixed by temporarily capping `self.max_num_scheduled_tokens` in `schedule()`.
This actuator **works** — it genuinely reduces tokens per step.

### Why queue-wait control failed
vLLM's continuous batching scheduler runs at GPU step frequency (~100+/sec).
Requests enter `self.waiting` and are popped in the very next scheduling step
(microseconds later). Queue wait is always ~0ms regardless of token budget.
The feedback variable does not exist.

The actuator affects TTFT and throughput via prefill chunking, NOT via queue
accumulation. The correct block diagram is:

```
token_budget → tokens_per_step → TTFT / throughput / GPU_power
```

Not:

```
token_budget → queue_depth → queue_wait  (this path does not exist)
```

### Admission fraction sweep (Ch11 Phase 1)

| Fraction | TTFT mean | TTFT p95  | Power mean | Energy/req |
|----------|-----------|-----------|------------|-----------|
| 1.00     | 103ms     | 148ms     | 64.2W      | 34.5J     |
| 0.25     | 118ms     | 178ms     | 66.0W      | 37.1J     |
| 0.10     | 310ms     | 1244ms    | 65.3W      | 39.9J     |
| 0.05     | 617ms     | 3434ms    | 66.3W      | 51.2J     |

GPU power flat ~64-66W across all fractions — memory-bandwidth floor.
The 310ms at fraction=0.10 is a transient artifact from settle, not steady-state.

### Key platform finding
`max_containers=1` is necessary. Without it Modal splits offered load across
multiple vLLM instances. Even then, public Modal web ingress serializes
requests (causing 26-29s client-observed TTFT at qps=8 while vLLM queue=0ms).
Solution: `/run_internal_sweep` endpoint runs benchmark inside the GPU container
against local vLLM, bypassing Modal ingress.

## What This Led To

Chapter 11: move the actuator entirely outside vLLM. `time.sleep(delay_ms/1000)`
before the HTTP send creates a clean, decoupled positive-gain plant.
Chapter 12 (planned): re-enter vLLM via `--scheduler-cls` with TTFT as the
feedback variable (not queue wait), which is measured externally in the wrapper.

## Deployed vLLM Version
- Modal image: `vllm>=0.16,<0.17`, v1 engine
- Scheduler class path: `vllm.v1.core.sched.scheduler.Scheduler` (primary)
- `--scheduler-cls` CLI flag: confirmed working
- Token budget cap: works as actuator; sequence count cap (max_num_running_reqs) less reliable
