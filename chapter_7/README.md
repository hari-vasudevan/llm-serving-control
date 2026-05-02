# Chapter 7 — Native vLLM On A Cheap Remote GPU

Chapter 7 asks a simple question:

> If Chapter 6 worked on a CPU queue server, what changes when we move the
> controller onto a *real GPU inference stack*?

The answer in this repo is a deliberately minimal architecture:

`local controller -> remote vLLM endpoint`

There is no extra software FIFO in the main experiment path. Instead:

- the plant is a real remote `vLLM` server,
- the controller reads `vLLM`'s own queueing metrics from `/metrics`,
- the control input is **client-side concurrency** `C`,
- the controlled output is **client-observed first-token latency**.

This chapter is intentionally experimental. It is not just a success story; it
documents what worked, what failed, and why the final control surface looks
different from Chapter 6.

## Why This Chapter Exists

Chapter 6 used a wrapper queue because the Intel Mac + Ollama setup did not
expose a meaningful native queue that we could regulate cleanly. On a GPU
serving stack, `vLLM` already has a scheduler, queue metrics, and batching.

So the Chapter 7 hypothesis was:

1. deploy a cheap real GPU endpoint,
2. use native `vLLM` metrics,
3. identify latency as a function of concurrency,
4. regulate latency with a single-loop controller.

That is exactly what the code in this folder now does.

## Recommendation

Use a **GPU VM or notebook runtime that allows long-running HTTP services**.
As of **May 2, 2026**, free Colab is a weak fit for the full chapter because
Google’s FAQ says free Colab resources are not guaranteed, GPU availability
varies, free runtimes can run for at most 12 hours, and free managed runtimes
disallow things like web service offerings and remote proxies. That makes
"serve vLLM in the cloud and drive it from your Mac" brittle on free Colab.

So the practical split is:

1. **Best technical path**: a normal Linux/NVIDIA VM or paid notebook runtime.
2. **Free Colab use**: quick smoke tests inside the notebook, not the final
   chapter architecture.

For this repo, the cheapest path that behaved like a real remotely accessible
service was **Modal on a T4**.

## Architecture

```text
Remote GPU host / Modal                    Local Mac
────────────────────────                  ─────────────────────────────
vLLM OpenAI-compatible server       <-     Chapter 7 Python controller
  /health                                  - characterise_remote.py
  /metrics                                 - design_controller.py
  /v1/completions                          - run_controller.py
```

The controller uses:
- `vllm:num_requests_waiting` for native queue visibility
- client first-token latency as the controlled output
- client concurrency `C` as the actuator

This is the key conceptual shift from Chapter 6:

- Chapter 6 actuator: wrapper-controlled batch size `B`
- Chapter 7 actuator: client-side concurrency `C`

## What Is In This Folder

- `python/vllm_native.py`
  Shared helper for direct native-vLLM experiments.
- `python/characterise_remote.py`
  Identifies `L(C)` where `C` is client concurrency.
- `python/design_controller.py`
  Designs a single-loop integral controller on concurrency.
- `python/run_controller.py`
  Runs the closed-loop experiment against the remote vLLM endpoint.
- `remote/start_vllm_linux.sh`
  Bootstrap script for an Ubuntu-like GPU host. Starts both `vllm serve` and
  the old wrapper queue server if you want that variant.
- `modal_vllm_server.py`
  Minimal Modal deployment for the cheap T4-based experiment path.

## Reading Guide

If someone opens this chapter cold on GitHub, the best order is:

1. read this `README`
2. inspect [identified_params.json](python/identified_params.json)
3. inspect [controller_params.json](python/controller_params.json)
4. look at the two plots:
   - [native concurrency sweep](python/ch7_native_concurrency_sweep.png)
   - [single-loop run](python/ch7_single_loop_215852.png)
5. then read:
   - [characterise_remote.py](python/characterise_remote.py)
   - [design_controller.py](python/design_controller.py)
   - [run_controller.py](python/run_controller.py)

## How To Run It

### 1. Deploy The Remote vLLM Endpoint

From the repo root:

```bash
python3 -m venv .modal-venv
source .modal-venv/bin/activate
pip install modal
modal setup
modal deploy chapter_7/modal_vllm_server.py
```

The Modal deployment used in this repo exposed:

`https://hvasudevan--chapter-7-vllm-serve.modal.run`

If you deploy from your own account, the hostname will differ.

### 2. Expect A Cold Start

The first request can take a while because `vLLM` must:

- start the container,
- load the model,
- compile kernels,
- warm up the engine.

In our run, engine initialization took on the order of a minute. A quick wakeup
request is helpful:

```bash
curl https://YOUR-ENDPOINT/health
```

### 3. Smoke Test The Endpoint

```bash
curl https://YOUR-ENDPOINT/health
curl https://YOUR-ENDPOINT/metrics
curl https://YOUR-ENDPOINT/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "prompt": "Explain in one sentence what a queue is in inference serving.",
    "max_tokens": 16,
    "stream": false,
    "temperature": 0.0
  }'
```

### 4. Create A Local Python Env For The Controller

```bash
python3 -m venv .chapter7-venv
source .chapter7-venv/bin/activate
pip install -r chapter_7/python/requirements.txt
```

### 5. Characterize The Plant

```bash
cd chapter_7/python
../../.chapter7-venv/bin/python characterise_remote.py \
  --url https://YOUR-ENDPOINT \
  --c-sweep 1 2 3 4 6 \
  --C0 3 \
  --n-reps 5 \
  --prompt-repeat 128 \
  --max-tokens 32
```

This writes:

- [identified_params.json](python/identified_params.json)
- [ch7_native_concurrency_sweep.png](python/ch7_native_concurrency_sweep.png)

### 6. Design The Controller

```bash
../../.chapter7-venv/bin/python design_controller.py
```

This writes:

- [controller_params.json](python/controller_params.json)

### 7. Run The Single-Loop Controller

```bash
../../.chapter7-venv/bin/python -u run_controller.py \
  --url https://YOUR-ENDPOINT \
  --background-scale 0.5 \
  --timeout 45
```

This writes:

- [single-loop plot](python/ch7_single_loop_215852.png)
- [single-loop log](python/ch7_single_loop_log_215852.json)

## Example Results

### 1. Native Concurrency Sweep

See [ch7_native_concurrency_sweep.png](python/ch7_native_concurrency_sweep.png).

This run produced:

- `L(C) = 9.34*C^2 + 121.65*C + 775.95`
- `R^2 = 0.9912`
- `C0 = 3`
- `beta_c ≈ 177.7 ms / concurrency`

Interpretation:

- client-observed latency increased cleanly with concurrency,
- the fit was strong,
- the endpoint was controllable from the client side,
- but most of the latency floor was *not* inside the raw model TTFT.

### 2. Single-Loop Controller

See [ch7_single_loop_215852.png](python/ch7_single_loop_215852.png).

That run produced:

- `L_mean ≈ 943.5 ms`
- `L_p95 ≈ 1145.9 ms`
- `C_mean ≈ 5.75`
- `q_wait_mean = 0.00`

Interpretation:

- the single-loop controller ran successfully,
- it increased concurrency when latency sat below target,
- but `vllm:num_requests_waiting` stayed near zero,
- so the endpoint never developed a meaningful server queue in this workload.

## What We Learned

This chapter produced a useful negative result:

1. **The remote endpoint works.**
   Modal + a T4 + `vLLM` can absolutely be driven from a local controller.

2. **The latency signal is real and controllable.**
   Client-observed first-token latency changed strongly with concurrency.

3. **The native vLLM queue signal did not become dominant in this setup.**
   `vllm:num_requests_waiting` stayed effectively zero during the main run.

4. **The platform adds a large fixed latency floor.**
   `vLLM`'s own TTFT metrics were typically tens of milliseconds, while the
   client saw roughly `0.9–1.8 s`.

5. **So this is not yet the clean queue-control story we wanted.**
   The system behaved more like “control serverless endpoint concurrency under
   fixed overhead” than “control a deeply queued GPU scheduler.”

That is still valuable. It narrows the next step:

- use a less serverless environment, or
- use a heavier model / prompt workload,

so that the native `vLLM` waiting queue becomes visible and worth regulating.

## Phase Plan

### Phase 1 — Native vLLM characterization

1. Deploy the remote vLLM endpoint.
2. Confirm `/health`, `/metrics`, and `/v1/completions`.
3. Sweep client concurrency `C`.
4. Fit the local slope `beta_c = dL/dC`.

### Phase 2 — Single-loop control

1. Choose an operating point `C0`.
2. Design an integral controller on measured latency.
3. Inject background load as a disturbance.
4. Regulate latency by changing client concurrency.

### Phase 3 — Write the chapter

The chapter should explicitly distinguish:

- why Chapter 6 used a wrapper queue on CPU,
- why native vLLM queue metrics are available on real GPU serving, and
- why the simplest stable experimental actuator here is client concurrency.

## Sources used for the hosting recommendation

- [Colab FAQ](https://research.google.com/colaboratory/faq.html?hl=en)
- [vLLM OpenAI-compatible server docs](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/)
- [vLLM metrics docs](https://docs.vllm.ai/en/stable/usage/metrics/)
- [Cloudflare Tunnel overview](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/)
