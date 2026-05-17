# Chapter 12: Plan — Controller Inside vLLM via Exposed Interfaces

## Goal

Close the loop with the actuator **inside** vLLM's scheduler, using only the
extension points vLLM exposes — no fork required. The feedback variable is
TTFT measured externally in the wrapper (wrapper has `t_send` and `t_first_token`);
the actuator is the token-budget cap inside the scheduler.

This is distinct from Chapter 11's dispatch-delay approach:
- Ch11 metered arrival rate externally. GPU always ran at full token budget.
- Ch12 controls tokens-per-step inside vLLM. GPU work is directly throttled.

## Why No Fork Needed

vLLM exposes two relevant interfaces:

1. **`--scheduler-cls`** (confirmed working in Ch10/11):
   Loads a custom scheduler subclass at startup. Override `schedule()` to
   temporarily cap `self.max_num_scheduled_tokens` before calling
   `super().schedule()`, then restore. This is the only field the controller
   needs to touch.

2. **Wrapper TTFT measurement** (already in Ch11 wrapper):
   The wrapper has `t_send` and `t_first_token` for every request. No access
   to vLLM engine internals needed — TTFT is a client-observable quantity.

3. **Shared state between wrapper and scheduler** via a file or
   `multiprocessing.Value`/`mmap`. The simplest approach: the wrapper writes
   `current_token_budget_fraction` to a control file; the scheduler reads it
   each `schedule()` call. Same pattern as Ch10's `/tmp/ch10_scheduler_control.json`.

## Architecture

```
Wrapper feedback_loop thread (every 0.1s):
  rolling_ttft_mean ← last N completions
  e_norm = (target_ttft - measured_ttft) / target_ttft
  velocity-form PI → new_fraction
  write fraction to /tmp/ch12_scheduler_control.json

ControlledScheduler.schedule() (every GPU step, ~100+/sec):
  read fraction from control file (rate-limited: every ~10ms to avoid I/O overhead)
  tmp_budget = int(fraction × self.max_num_scheduled_tokens_nominal)
  saved = self.max_num_scheduled_tokens
  self.max_num_scheduled_tokens = tmp_budget
  result = super().schedule()
  self.max_num_scheduled_tokens = saved
  return result
```

## Key Design Points

- **Feedback variable**: TTFT, not queue wait. Queue wait is always ~0ms in
  continuous batching (Ch10 lesson). TTFT responds monotonically to token budget.
- **Controller location**: stays in the wrapper's feedback_loop thread (same as Ch11).
  The scheduler subclass is just the actuator read-path, not the control law.
- **Negative gain**: token_budget↑ → more tokens per step → shorter prefill → TTFT↓.
  Use `e_norm = (measured - target)/target` (opposite of Ch11 dispatch-delay sign).
- **Rate-limiting control file reads**: scheduler runs ~100+/sec. Re-reading the
  file every call adds I/O noise. Cache with a ~10ms TTL.
- **No `max_num_running_reqs` capping**: Ch10/11 lesson — capping running reqs
  causes preemption in ttft mode. Only cap `max_num_scheduled_tokens`.

## Why This Is Better Than Ch11 Dispatch-Delay

- **GPU load is actually shaped**: Ch11's dispatch-delay metered arrivals but the
  GPU always ran at full budget when requests were active. Ch12's token-budget cap
  directly reduces GPU work per step — more relevant for power control.
- **No artificial inter-request spacing**: Ch11 introduced artificial gaps between
  requests. Ch12 lets requests arrive naturally and shapes how many tokens execute.
- **Phase 4 power controller is a natural extension**: the same actuator
  (token_budget) can target GPU power (via NVML) instead of TTFT. Just change
  the feedback variable.

## Implementation Steps

1. Create `chapter_12/` scaffold from Ch11 (`modal_vllm_wrapper.py`, `remote/`)
2. Rewrite `ControlledScheduler` to:
   - Read fraction from control file with ~10ms TTL cache
   - Cap `self.max_num_scheduled_tokens` transiently before `super().schedule()`
   - Log admitted tokens vs nominal budget each step
3. Add TTFT measurement to feedback loop (already done in Ch11 wrapper)
4. Change PI sign: `e_norm = (measured - target)/target`
5. Run open-loop characterization: sweep fraction 0.05–1.0, measure TTFT/power at qps=4,8
6. Tune gains with PM formula (τ_total dominated by MA window + feedback period, not dead time)
7. Run chained load-step experiment as in Ch11 Phase 3b

## Expected Differences vs Ch11

- At the same TTFT setpoint, Ch12 will show **lower GPU power** than Ch11 (actual
  GPU throttling, not just inter-request spacing)
- Duty-cycle effective power metric will converge toward raw power (duty approaches 1.0
  since there's no artificial dispatch delay — requests run back-to-back)
- Transient response may be faster (shorter dead time — controller acts within GPU steps,
  not between dispatches)
- Token budget may expose KV-cache management behavior at low fractions (cache pressure)

## Files To Create

```text
chapter_12/
  modal_vllm_wrapper.py          Modal app entrypoint
  remote/
    vllm_modal_wrapper.py        Wrapper + PI controller + load gen
    ch12_vllm/
      __init__.py
      controlled_scheduler.py    Token-budget scheduler subclass
  python/
    run_load_step.py             (copy from Ch11, adjust sign)
    plot_load_step.py            (copy from Ch11)
    make_video.py                (copy from Ch11)
  README.md
```
