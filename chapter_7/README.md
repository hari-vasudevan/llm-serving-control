# Chapter 7 — Remote GPU plan

Chapter 7 should keep the Chapter 6 *shape*:

`local controller -> remote queue server -> model backend`

but replace the Intel Mac + Ollama backend with a Linux/NVIDIA host running
`vLLM`. This gives you a real GPU batching plant while preserving the same
local workflow from your Mac.

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

## Architecture

```text
GPU host (Linux/NVIDIA)                 Local Mac
──────────────────────────              ─────────────────────────────
vLLM (:8001, OpenAI-compatible)   <-    chapter_7/remote/vllm_queue_server.py
remote wrapper (:8002)            <-    MATLAB or Python controller
  /enqueue
  /control {"B": N}
  /metrics
```

This wrapper server is the key execution step in this scaffold. It preserves
the Chapter 6 remote-control API while swapping the backend to vLLM.

## What is in this folder

- `remote/vllm_queue_server.py`
  Remote queue server that dispatches requests in batches to vLLM and reports:
  `q_sw`, `B_current`, `ttft_recent_mean`, `l_total_mean`, plus selected
  upstream vLLM metrics such as `vllm_num_requests_waiting`.
- `remote/start_vllm_linux.sh`
  Bootstrap script for an Ubuntu-like GPU host. Starts both `vllm serve` and
  the wrapper queue server.
- `python/`
  Local controller-side utilities copied from the Chapter 5 Python flow as a
  starting point for direct-endpoint experiments.

## Phase plan

### Phase 1 — Repeat Chapter 6 on a GPU host

Goal: prove the remote GPU host works end-to-end with your existing control
workflow before changing controller architecture.

1. Start the GPU host with `remote/start_vllm_linux.sh`.
2. Confirm:
   - `curl http://HOST:8001/health`
   - `curl http://HOST:8002/health`
   - `curl http://HOST:8002/metrics`
3. Point your Chapter 6 controller at the Chapter 7 wrapper endpoint.
4. Re-run the single-loop TTFT experiment on GPU.

This de-risks networking, deployment, and measurement first.

### Phase 2 — Re-introduce cascade control

Once Phase 1 is stable:

1. Re-identify `TTFT(B)` on the GPU host.
2. Verify that larger `B` actually changes service behavior materially.
3. Run the cascade logic with queue depth as the regulated inner variable.
4. Compare:
   - wrapper FIFO depth `q_sw`
   - wrapper `ttft_recent_mean`
   - wrapper `l_total_mean`
   - upstream `vllm_num_requests_waiting`

### Phase 3 — Write the chapter

The chapter should explicitly distinguish:

- why Chapter 6 on CPU only supported a single-loop TTFT controller, and
- why the GPU-backed Chapter 7 architecture restores a meaningful queue-rate
  control handle.

## Suggested experiments

1. **Deployment smoke test**
   Show that the local Mac can enqueue prompts and change `B` remotely.
2. **B-sweep at low queue**
   Fit `TTFT(B) = alpha*B + gamma*B^2`.
3. **Queue buildup**
   Step the offered load above sustainable capacity and observe `q_sw`.
4. **Closed-loop recovery**
   Hold a steady load, inject a spike, and show recovery in TTFT and queue.

## Immediate next step

Use a GPU host that permits serving, then run:

```bash
cd chapter_7/remote
chmod +x start_vllm_linux.sh
./start_vllm_linux.sh
```

Then from your Mac:

```bash
curl http://GPU_HOST:8002/health
curl http://GPU_HOST:8002/metrics
curl -X POST http://GPU_HOST:8002/enqueue \
  -H "Content-Type: application/json" \
  -d '{"prompt":"What is 2+2?"}'
curl -X POST http://GPU_HOST:8002/control \
  -H "Content-Type: application/json" \
  -d '{"B":4}'
```

## Sources used for the hosting recommendation

- [Colab FAQ](https://research.google.com/colaboratory/faq.html?hl=en)
- [vLLM OpenAI-compatible server docs](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/)
- [vLLM metrics docs](https://docs.vllm.ai/en/stable/usage/metrics/)
- [Cloudflare Tunnel overview](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/)
