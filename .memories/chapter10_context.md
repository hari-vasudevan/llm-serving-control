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

## First Execution Results, May 16 2026

Deployment worked:

```text
https://hvasudevan--chapter-10-vllm-admission-serve.modal.run
```

The vLLM CLI accepted the internal scheduler hook:

```text
scheduler_cls='ch10_vllm.controlled_scheduler.ControlledScheduler'
```

Modal logs confirmed:

```text
CH10 ControlledScheduler loaded: enabled=True target_queue_ms=0.00 ...
```

The wrapper metrics exposed NVML power:

```text
gpu_power_w
gpu_util_percent
gpu_memory_used_mb
gpu_temperature_c
```

Important platform finding:

```text
max_containers=1 is necessary.
```

Without that, Modal can split offered load across multiple independent vLLM
containers, which invalidates the single-queue experiment. With
`max_containers=1`, public Modal web requests appear serialized enough that
client-side top-level latency becomes dominated by Modal ingress queueing. For
example, at qps=8 public-client TTFT became ~26-29 seconds while vLLM internal
queue remained below 1 ms. That is a Modal/web ingress artifact, not a vLLM
scheduler result.

To avoid that artifact, an in-container endpoint was added:

```text
POST /run_internal_sweep
```

It generates normal top-level `/v1/completions` requests inside the GPU
container against local vLLM, so the offered load hits one vLLM engine without
Modal ingress queueing.

Measured internal sweep results are saved in:

```text
chapter_10/python/results/internal_qps8_20260516.json
chapter_10/python/results/internal_qps30_20260516.json
```

At qps=8:

```text
target 0 ms:   vLLM queue wait 0.274 ms, total p95 374 ms, power mean 55.3 W
target 100 ms: vLLM queue wait 0.055 ms, total p95 461 ms, power mean 59.9 W
```

At qps=30:

```text
target 0 ms:   vLLM queue wait 0.118 ms, total p95 908 ms, power mean 65.4 W
target 100 ms: vLLM queue wait 0.075 ms, total p95 617 ms, power mean 64.3 W
```

The scheduler logs show the controller did clamp its computed budget:

```text
target_ms=100.00 budget=1 nominal=192
```

but vLLM's exported queue wait still stayed below 1 ms. Current interpretation:

```text
The hook loads and computes a control budget, but dynamically mutating
scheduler_config.max_num_seqs during schedule() is not an effective live
admission actuator in vLLM 0.16.
```

The apparent qps=30 improvement for target 100 ms should not be treated as a
validated queue-wait controller effect yet. It may be cache/order/noise, because
the measured queue wait did not move toward the 100 ms setpoint.

Best next implementation direction:

```text
Patch deeper into vLLM's schedule decision itself, where running/waiting
requests are selected, instead of mutating scheduler_config.max_num_seqs.
```

Alternative practical direction:

```text
Use a wrapper-level admission queue for controllable queue wait, while still
measuring vLLM internal queue/power. This is less "lowest layer" but gives a
real actuator immediately.
```

## Unrelated Local File

`substack_llm_serving_control_draft.md` is currently untracked. Do not add it
unless the user explicitly asks.



I’ll run the full path now: deploy the Chapter 10 Modal service, confirm the scheduler hook loads, run a short smoke sweep first, then run a more meaningful queue-wait sweep if the smoke looks healthy. If any vLLM scheduler API mismatch shows up, I’ll capture the exact failure and patch forward.


Ran 3 commands
There’s no existing Modal virtualenv in this checkout, so I’ll create one, install Modal, deploy Chapter 10, and then use the returned endpoint for the sweep.


Approved 2 requests, ran 3 commands
Modal is installed locally now. I’m deploying the Chapter 10 app; this will be the first real check of whether the pinned vLLM build accepts the scheduler class hook.


Approved request, ran 1 command
The Modal image build is underway and pulling the pinned vLLM/Torch stack. This is a heavy first build, so I’m letting it run; the next meaningful result will be whether vllm serve accepts the scheduler hook at container startup.

Deploy succeeded. Endpoint is https://hvasudevan--chapter-10-vllm-admission-serve.modal.run. I’ll now hit health/metrics; this will start the container and tell us whether vLLM actually booted with the scheduler hook.


Approved request, ran 2 commands
The first health call timed out after 10 seconds, which is expected if the web server is still starting Qwen/vLLM. I’ll poll with a longer timeout and also pull Modal logs to see whether the scheduler hook loaded or failed.


Approved request, ran 2 commands
The API server sees the non-default scheduler_cls, so the CLI accepted the hook. Health is still not returning after 60 seconds, so the question is now whether model init failed or just needs more time. I’m pulling fresh logs around startup.


Approved request, ran 2 commands
Health is green now. I’ll verify metrics and power fields, then run a very short two-point smoke sweep before doing anything longer.


Approved 2 requests, ran 3 commands
The smoke runner has been quiet longer than expected for a 20-second run. I’m checking Modal logs while it runs; likely the first streamed requests triggered additional compile/capture latency, or the proxy stream is hanging.

Ran .modal-venv/bin/modal app logs chapter-10-vllm-admission
The first smoke runner was launched without escalated network access and appears wedged before it even hit the Modal service. I’ll terminate that local runner and rerun the same smoke with network permission.


Approved 2 requests, ran 2 commands
The smoke ran cleanly with zero request errors and power data. The controlled target did not move vLLM queue wait yet: queue wait stayed around 0.4 ms for both 0 and 100 ms, which suggests this offered load is too light to expose the scheduler queue or the hook is not finding the internal waiting list under load. I’ll inspect logs and then run a higher-load short sweep to force queueing.

Ran .modal-venv/bin/modal app logs chapter-10-vllm-admission
The hook is definitely loading and running. Logs show it reading target 100 ms, but with the default gains it only reduced the admission budget from 192 to about 187, so no real queue formed. I’ll add benchmark flags for controller gains, then run an aggressive short test that clamps the internal sequence budget near 1 to see if vLLM queue latency becomes controllable at all.


Editing 1 file, explored 1 file, 1 search, approved request, ran 2 commands
Editing 1 file, explored 1 file, 1 search, approved request, ran 2 commands
I added --controller-kp and --controller-ki to the runner. Now I’ll run a more stressful short comparison: same offered query load, target 0 ms versus 100 ms, but with aggressive gains so the 100 ms case actually tries to constrain admission.


Approved request, ran 1 command
The higher-load run completed with no errors, but the measured vLLM queue time still stayed at roughly 0.30 ms for both targets. I’ll verify from logs whether the scheduler budget actually clamped; if it did, this tells us changing max_num_seqs after scheduler construction is not an effective actuator in vLLM 0.16.

Ran .modal-venv/bin/modal app logs chapter-10-vllm-admission | rg 'CH10 scheduler control|control file|target_ms=100|budget=|Waiting:'
The log explains the mismatch: Modal appears to have started more than one container, so the offered load was split across multiple independent vLLM engines. That defeats the queue experiment. I’m going to pin the web service to a single container and allow concurrent inputs into that one container, then redeploy and rerun a short high-load sweep.


Edited 1 file, approved request, ran 3 commands
I’ve patched Modal to max_containers=1 and max_inputs=256. I’ll redeploy so the next run is genuinely one queue, one vLLM engine, one GPU.


Approved request, ran 1 command
Modal’s current API only supports max_inputs=1, so I’m backing that part out and keeping the crucial constraint: max_containers=1. That should still prevent load splitting across multiple independent queues.


Edited 1 file, approved request, ran 1 command
Redeploy completed with a single-container cap. I’ll wait for the new container to come up, then rerun the high-load aggressive sweep.


Approved 2 requests, ran 3 commands
Single-container redeploy is healthy. I’ll rerun the aggressive qps=8 sweep now; this is the first valid one-queue/one-engine comparison.


Approved request, ran 1 command
This single-container run exposed a platform artifact: Modal is serializing public web inputs (max_inputs=1), so the huge 26-29s top-level latency is mostly Modal ingress queueing before the request reaches our wrapper. vLLM’s own queue stayed near zero. To test vLLM itself, I’m adding an in-container benchmark endpoint that generates top-level /v1/completions calls locally against vLLM, so the load hits one vLLM scheduler queue without Modal ingress becoming the bottleneck.


Edited 1 file, approved request, ran 2 commands
I added /run_internal_sweep, which runs the offered-load experiment inside the GPU container against vLLM’s normal /v1/completions API. I’ll redeploy and run a compact internal sweep.


Approved 3 requests, ran 4 commands
The internal-sweep service is healthy. I’m running the in-container qps=8 comparison now; this should avoid Modal public request queueing and measure vLLM-facing top-level completions directly.


Approved request, ran 1 command
The in-container benchmark worked and gives sane vLLM-level latencies. However, even with aggressive gains the internal queue wait is still sub-millisecond; the target is not being achieved because the actuator is not truly limiting vLLM admission. I’ll run one overload point at qps=30 to see whether queueing appears under pressure.


Approved request, ran 2 commands
I have enough experimental signal now: the scheduler hook loads and runs, but changing max_num_seqs dynamically is not a valid actuator for vLLM 0.16’s live scheduler. I’ll preserve the measurements and update the memory handoff with this result before I commit the test/fix scaffold.


Created 2 files, stopped creating 1 file, edited 1 file, explored 1 search, 1 list, approved request, ran 4 commands
I’ll keep the small summary JSON result files, but avoid committing the local virtualenv and bulky raw benchmark CSV/JSONL traces. I’m adding ignore rules for those generated artifacts, then I’ll commit the code and summarized results.


Edited 1 file, approved request, denied request, ran 6 commands
8 files changed
+454
-16
Undo

Review

chapter_10/python/run_queue_wait_sweep.py
    ap.add_argument("--metric-period-s", type=float, default=1.0)
    ap.add_argument("--controller-kp", type=float, default=None)
    ap.add_argument("--controller-ki", type=float, default=None)
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "results"))
def run_one_target(args: argparse.Namespace, target_wait_ms: float, out_dir: Path) -> dict[str, Any]:
    configure_target(args.url, target_wait_ms)
    configure_target(args.url, target_wait_ms, args.controller_kp, args.controller_ki)
    wait_for_health(args.url)

def configure_target(url: str, target_wait_ms: float) -> None:
def configure_target(url: str, target_wait_ms: float, kp: float | None = None, ki: float | None = None) -> None:
    # The Chapter 10 wrapper can expose this endpoint. Direct vLLM will return
    # 404, which is fine when the target is configured by environment/deploy.
    payload: dict[str, Any] = {"target_wait_ms": target_wait_ms}
    if kp is not None:
        payload["kp"] = kp
    if ki is not None:
        payload["ki"] = ki
    try:
            f"{url.rstrip('/')}/control/queue_wait_target",
            json={"target_wait_ms": target_wait_ms},
            json=payload,
            timeout=5,
chapter_10/modal_vllm_wrapper.py
    scaledown_window=300,
    max_containers=1,
    max_inputs=256,
)
    max_containers=1,
    max_inputs=256,
)
chapter_10/remote/vllm_modal_wrapper.py
import os
import random
import re

def metric_delta_mean_ms(before, after, stem):
    d_sum = after.get(f"{stem}_sum", 0.0) - before.get(f"{stem}_sum", 0.0)
    d_count = after.get(f"{stem}_count", 0.0) - before.get(f"{stem}_count", 0.0)
    if d_count <= 0:
        return None
    return 1000.0 * d_sum / d_count


def percentile(values, pct):
    vals = sorted(v for v in values if v is not None and math.isfinite(v))
    if not vals:
        return None
    idx = min(len(vals) - 1, max(0, round((pct / 100.0) * (len(vals) - 1))))
    return vals[idx]


def integrate_power(samples):
    if len(samples) < 2:
        return None
    total = 0.0
    for a, b in zip(samples, samples[1:]):
        p0 = a.get("gpu_power_w")
        p1 = b.get("gpu_power_w")
        if p0 is None or p1 is None:
            continue
        total += 0.5 * (float(p0) + float(p1)) * max(0.0, float(b["t"]) - float(a["t"]))
    return total


def make_benchmark_prompt(index, repeat):
    seeds = [
        "Explain queueing delay and model service time in one concise paragraph.",
        "Summarize how admission control changes latency and throughput.",
        "Describe why GPU power can change under different request schedules.",
        "Compare eager scheduling with controlled queue wait for LLM serving.",
    ]
    return " ".join([seeds[index % len(seeds)]] * max(1, repeat))


def run_internal_sweep(body):
    targets = [float(x) for x in body.get("target_wait_ms", [0, 100])]
    offered_rate = float(body.get("offered_rate_qps", 4.0))
    duration_s = float(body.get("duration_s", 30.0))
    warmup_s = float(body.get("warmup_s", 5.0))
    max_tokens = int(body.get("max_tokens", MAX_TOKENS_DEFAULT))
    prompt_repeat = int(body.get("prompt_repeat", 48))
    kp = body.get("kp")
    ki = body.get("ki")
    seed = int(body.get("seed", 10))
    random.seed(seed)

    summaries = []
    for target in targets:
        control = {"target_queue_ms": target, "enabled": True}
        if kp is not None:
            control["kp"] = float(kp)
        if ki is not None:
            control["ki"] = float(ki)
        with open(CONTROL_FILE, "w") as f:
            json.dump(control, f)

        before = fetch_backend_metrics()
        stop = threading.Event()
        sem = threading.Semaphore(int(body.get("max_outstanding", 256)))
        lock = threading.Lock()
        records = []
        power_samples = []

        def power_loop():
            while not stop.is_set():
                sample = gpu_snapshot()
                sample["t"] = time.perf_counter()
                power_samples.append(sample)
                time.sleep(float(body.get("metric_period_s", 1.0)))

        def one_request(i, measure):
            prompt = make_benchmark_prompt(i, prompt_repeat)
            t_send = time.perf_counter()
            t_first = None
            status = "ok"
            try:
                payload = {
                    "model": MODEL,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                    "stream": True,
                }
                with requests.post(
                    f"{BACKEND_URL}/v1/completions",
                    data=json.dumps(payload),
                    headers=headers(),
                    stream=True,
                    timeout=TIMEOUT,
                ) as resp:
                    resp.raise_for_status()
                    for chunk in resp.iter_lines():
                        if chunk and chunk != b"data: [DONE]" and t_first is None:
                            t_first = time.perf_counter()
            except Exception as exc:
                status = f"error:{exc!r}"
            finally:
                t_done = time.perf_counter()
                if measure:
                    with lock:
                        records.append(
                            {
                                "status": status,
                                "ttft_ms": 1000.0 * (t_first - t_send) if t_first else None,
                                "total_ms": 1000.0 * (t_done - t_send),
                            }
                        )
                sem.release()

        power_thread = threading.Thread(target=power_loop, daemon=True)
        power_thread.start()
        threads = []
        t_start = time.perf_counter()
        t_measure_start = t_start + warmup_s
        t_end = t_measure_start + duration_s
        next_arrival = t_start
        req_id = 0
        while time.perf_counter() < t_end:
            now = time.perf_counter()
            if now < next_arrival:
                time.sleep(min(0.01, next_arrival - now))
                continue
            if sem.acquire(timeout=0.1):
                req_id += 1
                measure = now >= t_measure_start
                thread = threading.Thread(target=one_request, args=(req_id, measure), daemon=True)
                thread.start()
                threads.append(thread)
            next_arrival += random.expovariate(offered_rate) if offered_rate > 0 else 1.0

        for thread in threads:
            thread.join(timeout=TIMEOUT)
        stop.set()
        power_thread.join(timeout=5)
        after = fetch_backend_metrics()

        ok = [r for r in records if r["status"] == "ok"]
        ttfts = [r["ttft_ms"] for r in ok if r["ttft_ms"] is not None]
        totals = [r["total_ms"] for r in ok]
        energy = integrate_power(power_samples)
        summaries.append(
            {
                "target_wait_ms": target,
                "control": control,
                "offered_rate_qps": offered_rate,
                "requests_measured": len(records),
                "requests_ok": len(ok),
                "error_rate": 1.0 - len(ok) / max(len(records), 1),
                "ttft_mean_ms": statistics.mean(ttfts) if ttfts else None,
                "ttft_p95_ms": percentile(ttfts, 95),
                "total_mean_ms": statistics.mean(totals) if totals else None,
                "total_p95_ms": percentile(totals, 95),
                "vllm_queue_wait_mean_ms": metric_delta_mean_ms(before, after, "vllm:request_queue_time_seconds"),
                "vllm_ttft_mean_ms": metric_delta_mean_ms(before, after, "vllm:time_to_first_token_seconds"),
                "vllm_e2e_mean_ms": metric_delta_mean_ms(before, after, "vllm:e2e_request_latency_seconds"),
                "gpu_power_mean_w": statistics.mean(
                    [float(x["gpu_power_w"]) for x in power_samples if "gpu_power_w" in x]
                )
                if any("gpu_power_w" in x for x in power_samples)
                else None,
                "gpu_power_peak_w": max(
                    [float(x["gpu_power_w"]) for x in power_samples if "gpu_power_w" in x],
                    default=None,
                ),
                "energy_j": energy,
                "energy_per_request_j": energy / len(ok) if energy is not None and ok else None,
            }
        )
    return {"status": "ok", "summaries": summaries}


def recent_arrival_rate():

        if parsed.path == "/run_internal_sweep":
            result = run_internal_sweep(body)
            self._send_json(200, result)
            return

        if parsed.path == "/enqueue":
chapter_10/python/results/internal_qps8_summary.json
{
  "note": "In-container benchmark against local vLLM /v1/completions after max_containers=1. Offered load avoids Modal public ingress queue.",
  "payload": {
    "target_wait_ms": [0, 100],
    "offered_rate_qps": 8,
    "duration_s": 30,
    "warmup_s": 5,
    "max_tokens": 24,
    "prompt_repeat": 48,
    "kp": 2.0,
    "ki": 0.02
  },
  "summaries": [
    {
      "target_wait_ms": 0.0,
      "requests_measured": 232,
      "requests_ok": 232,
      "error_rate": 0.0,
      "ttft_mean_ms": 37.37925856034448,
      "ttft_p95_ms": 65.72526400000811,
      "total_mean_ms": 290.6720562025861,
      "total_p95_ms": 374.4735569999875,
      "vllm_queue_wait_mean_ms": 0.2744551466165569,
      "vllm_ttft_mean_ms": 56.642852331462656,
      "vllm_e2e_mean_ms": 314.5272812448946,
      "gpu_power_mean_w": 55.32842857142857,
      "gpu_power_peak_w": 76.625,
      "energy_j": 1908.380010573912,
      "energy_per_request_j": 8.225775907646172
    },
    {
      "target_wait_ms": 100.0,
      "requests_measured": 255,
      "requests_ok": 255,
      "error_rate": 0.0,
      "ttft_mean_ms": 44.11538375294134,
      "ttft_p95_ms": 86.84557800000903,
      "total_mean_ms": 303.6981837686281,
.memories/chapter10_context.md

## First Execution Results, May 16 2026

Deployment worked:

```text
https://hvasudevan--chapter-10-vllm-admission-serve.modal.run
```

The vLLM CLI accepted the internal scheduler hook:

```text
scheduler_cls='ch10_vllm.controlled_scheduler.ControlledScheduler'
```

Modal logs confirmed:

```text
CH10 ControlledScheduler loaded: enabled=True target_queue_ms=0.00 ...
```

The wrapper metrics exposed NVML power:

```text
gpu_power_w
gpu_util_percent
gpu_memory_used_mb
gpu_temperature_c
```

Important platform finding:

```text
max_containers=1 is necessary.
```

Without that, Modal can split offered load across multiple independent vLLM
containers, which invalidates the single-queue experiment. With
`max_containers=1`, public Modal web requests appear serialized enough that
client-side top-level latency becomes dominated by Modal ingress queueing. For
example, at qps=8 public-client TTFT became ~26-29 seconds while vLLM internal
queue remained below 1 ms. That is a Modal/web ingress artifact, not a vLLM
scheduler result.

To avoid that artifact, an in-container endpoint was added:

```text
POST /run_internal_sweep
```

It generates normal top-level `/v1/completions` requests inside the GPU
container against local vLLM, so the offered load hits one vLLM engine without
Modal ingress queueing.

Measured internal sweep results are saved in:

```text
chapter_10/python/results/internal_qps8_20260516.json
chapter_10/python/results/internal_qps30_20260516.json
```

At qps=8:

```text
target 0 ms:   vLLM queue wait 0.274 ms, total p95 374 ms, power mean 55.3 W
target 100 ms: vLLM queue wait 0.055 ms, total p95 461 ms, power mean 59.9 W
```

At qps=30:

```text
target 0 ms:   vLLM queue wait 0.118 ms, total p95 908 ms, power mean 65.4 W
target 100 ms: vLLM queue wait 0.075 ms, total p95 617 ms, power mean 64.3 W
```

The scheduler logs show the controller did clamp its computed budget:

```text
target_ms=100.00 budget=1 nominal=192
```

but vLLM's exported queue wait still stayed below 1 ms. Current interpretation:

```text
The hook loads and computes a control budget, but dynamically mutating
scheduler_config.max_num_seqs during schedule() is not an effective live
admission actuator in vLLM 0.16.
```

The apparent qps=30 improvement for target 100 ms should not be treated as a
validated queue-wait controller effect yet. It may be cache/order/noise, because
the measured queue wait did not move toward the 100 ms setpoint.

Best next implementation direction:

```text
Patch deeper into vLLM's schedule decision itself, where running/waiting
requests are selected, instead of mutating scheduler_config.max_num_seqs.
```

Alternative practical direction:

```text
Use a wrapper-level admission queue for controllable queue wait, while still
measuring vLLM internal queue/power. This is less "lowest layer" but gives a
real actuator immediately.
```

## Unrelated Local File
unless the user explicitly asks.

chapter_10/python/results/internal_qps30_20260516.json
{
  "status": "ok",
  "summaries": [
    {
      "target_wait_ms": 0.0,
      "control": {"target_queue_ms": 0.0, "enabled": true, "kp": 2.0, "ki": 0.02},
      "offered_rate_qps": 30.0,
      "requests_measured": 596,
      "requests_ok": 596,
      "error_rate": 0.0,
      "ttft_mean_ms": 158.04351002684595,
      "ttft_p95_ms": 357.06936400001155,
      "total_mean_ms": 617.4543355721485,
      "total_p95_ms": 907.9016779999733,
      "vllm_queue_wait_mean_ms": 0.11827492503199487,
      "vllm_ttft_mean_ms": 65.28771992827674,
      "vllm_e2e_mean_ms": 517.500317559504,
      "gpu_power_mean_w": 65.37538461538462,
      "gpu_power_peak_w": 74.936,
      "energy_j": 1673.6144151983472,
      "energy_per_request_j": 2.8080778778495756
    },
    {
      "target_wait_ms": 100.0,
      "control": {"target_queue_ms": 100.0, "enabled": true, "kp": 2.0, "ki": 0.02},
      "offered_rate_qps": 30.0,
      "requests_measured": 595,
      "requests_ok": 595,
      "error_rate": 0.0,
      "ttft_mean_ms": 88.30920284033655,
      "ttft_p95_ms": 185.71531499998173,
      "total_mean_ms": 480.91891961848677,
      "total_p95_ms": 616.5236649999883,
      "vllm_queue_wait_mean_ms": 0.0750867440623615,
      "vllm_ttft_mean_ms": 44.7540953165623,
      "vllm_e2e_mean_ms": 437.1901788308941,
      "gpu_power_mean_w": 64.26483999999999,
      "gpu_power_peak_w": 74.594,
      "energy_j": 1570.6887839091933,
      "energy_per_request_j": 2.639813082200325
    }
  ]
}

chapter_10/python/results/internal_qps8_20260516.json
{
  "status": "ok",
  "summaries": [
    {
      "target_wait_ms": 0.0,
      "control": {"target_queue_ms": 0.0, "enabled": true, "kp": 2.0, "ki": 0.02},
      "offered_rate_qps": 8.0,
      "requests_measured": 232,
      "requests_ok": 232,
      "error_rate": 0.0,
      "ttft_mean_ms": 37.37925856034448,
      "ttft_p95_ms": 65.72526400000811,
      "total_mean_ms": 290.6720562025861,
      "total_p95_ms": 374.4735569999875,
      "vllm_queue_wait_mean_ms": 0.2744551466165569,
      "vllm_ttft_mean_ms": 56.642852331462656,
      "vllm_e2e_mean_ms": 314.5272812448946,
      "gpu_power_mean_w": 55.32842857142857,
      "gpu_power_peak_w": 76.625,
      "energy_j": 1908.380010573912,
      "energy_per_request_j": 8.225775907646172
    },
    {
      "target_wait_ms": 100.0,
      "control": {"target_queue_ms": 100.0, "enabled": true, "kp": 2.0, "ki": 0.02},
      "offered_rate_qps": 8.0,
      "requests_measured": 255,
      "requests_ok": 255,
      "error_rate": 0.0,
      "ttft_mean_ms": 44.11538375294134,
      "ttft_p95_ms": 86.84557800000903,
      "total_mean_ms": 303.6981837686281,
      "total_p95_ms": 460.9935539999981,
      "vllm_queue_wait_mean_ms": 0.05543157999862084,
      "vllm_ttft_mean_ms": 27.134448687235516,
      "vllm_e2e_mean_ms": 290.68155606587726,
      "gpu_power_mean_w": 59.87931428571429,
      "gpu_power_peak_w": 71.551,
      "energy_j": 2070.5131826734987,
      "energy_per_request_j": 8.119659539896073
    }
  ]
}

.gitignore
.venv/
.modal-venv/
# Chapter 10 generated benchmark traces can get large. Keep curated summary
# JSON files explicitly when useful.
chapter_10/python/results/**/
chapter_10/python/results/*.csv
chapter_10/python/results/*.jsonl
!chapter_10/python/results/
!chapter_10/python/results/internal_*.json
*.egg-info/
