# Chapter 10 Context Handoff

Current branch:

```text
chapter-10-experimental-not-for-merge-yet
```

Repo:

```text
/Users/hvasudevan/Library/Mobile Documents/com~apple~CloudDocs/git_src_personal/llm-serving-control
```

## Project Story

Chapter 9 concluded the control-theory exploration by moving below vLLM to a
clean lower-level GPU batching plant:

```text
FIFO queue -> exact B dispatch -> fixed CUDA tensor batch -> measured service
```

There, `q` was the carry-over backlog after dispatch, and `L_mean` included
queue wait plus GPU service time. The cascade controller worked.

Chapter 10 now returns to real vLLM/Qwen serving, but as an experimental
measurement study:

```text
How does controlling low-level vLLM queue wait affect top-level query latency,
throughput, GPU power, and energy per request?
```

## User Requirements For Chapter 10

- Continue in the same repo as `chapter_10/`.
- Keep work on an experimental branch, not ready to merge.
- Preserve context in `.memories`.
- Insert the controller into the lowest practical vLLM layer.
- Test using normal top-level Qwen queries.
- Treat offered load as top-level queries per time unit.
- Treat controlled output as low-level queue latency in milliseconds.
- Measure enough to support conclusions, including power.
- Later, maybe generate a visual/video demo of query/response behavior, but
  not yet.

## Current Implementation

### vLLM Internal Scheduler Hook

File:

```text
chapter_10/remote/ch10_vllm/controlled_scheduler.py
```

Modal starts vLLM with:

```text
--scheduler-cls ch10_vllm.controlled_scheduler.ControlledScheduler
```

The custom scheduler subclasses vLLM's default scheduler and tries to:

- inspect the internal waiting queue,
- estimate request wait time,
- apply a PI-style control law against `target_queue_ms`,
- adjust `scheduler_config.max_num_seqs`,
- reload target/gain changes from:

```text
/tmp/ch10_scheduler_control.json
```

Important caveat: vLLM's scheduler API is private/unstable. This hook is
experimental and may need patching after first Modal deployment logs reveal
the actual vLLM version and object fields.

### Public Wrapper Endpoint

File:

```text
chapter_10/remote/vllm_modal_wrapper.py
```

Current useful endpoints:

```text
GET  /health
GET  /metrics
GET  /metrics/prom
GET  /power
POST /v1/completions
POST /control/queue_wait_target
```

`POST /v1/completions` proxies normal OpenAI-compatible completion requests to
local vLLM. This lets a top-level benchmark hit the Modal public URL while
vLLM remains bound to localhost inside the container.

`POST /control/queue_wait_target` writes:

```json
{"target_queue_ms": 100.0, "enabled": true}
```

to `/tmp/ch10_scheduler_control.json`, which the scheduler hook rereads.

`GET /metrics` combines wrapper/vLLM metrics and adds NVML GPU fields when
available:

```text
gpu_power_w
gpu_util_percent
gpu_mem_util_percent
gpu_memory_used_mb
gpu_temperature_c
```

### Benchmark Runner

File:

```text
chapter_10/python/run_queue_wait_sweep.py
```

Example:

```bash
python chapter_10/python/run_queue_wait_sweep.py \
  --url https://YOUR-MODAL-URL.modal.run \
  --target-wait-ms 0 50 100 200 400 \
  --offered-rate-qps 4 \
  --duration-s 90 \
  --warmup-s 15
```

It writes per-target:

```text
requests.csv
metrics.jsonl
summary.json
```

and whole-sweep:

```text
sweep_summary.json
sweep_summary.csv
```

Measured summary fields include:

```text
TTFT mean/p50/p95
total latency mean/p50/p95
throughput req/s
error rate
vLLM request queue mean
vLLM TTFT mean
vLLM e2e mean
GPU mean/peak power
energy/request
```

## Modal Entrypoint

File:

```text
chapter_10/modal_vllm_wrapper.py
```

Current app name:

```text
chapter-10-vllm-admission
```

It copies:

```text
chapter_10/remote/vllm_modal_wrapper.py -> /root/vllm_modal_wrapper.py
chapter_10/remote/ch10_vllm -> /root/ch10_vllm
```

and sets:

```text
PYTHONPATH=/root
CH10_TARGET_QUEUE_MS=0
CH10_SCHEDULER_ENABLED=1
CH10_CONTROL_FILE=/tmp/ch10_scheduler_control.json
```

## Next Practical Steps

1. Run static checks locally:

```bash
python3 -m py_compile \
  chapter_10/python/run_queue_wait_sweep.py \
  chapter_10/remote/vllm_modal_wrapper.py \
  chapter_10/remote/ch10_vllm/controlled_scheduler.py
```

2. Deploy Chapter 10 Modal service.

3. Watch logs for:

```text
CH10 ControlledScheduler loaded
```

4. If vLLM rejects `--scheduler-cls`, inspect installed vLLM CLI/version and
adjust the launch path or pin a version that exposes `scheduler_cls`.

5. If scheduler loads but does not observe queue fields, inspect Modal logs and
patch `_waiting_requests()` in `controlled_scheduler.py`.

6. Run a short sweep:

```bash
python chapter_10/python/run_queue_wait_sweep.py \
  --url MODAL_URL \
  --target-wait-ms 0 100 \
  --offered-rate-qps 2 \
  --duration-s 20 \
  --warmup-s 5
```

7. Only then run the full sweep.

## Unrelated Local File

`substack_llm_serving_control_draft.md` is currently untracked. Do not add it
unless the user explicitly asks.

