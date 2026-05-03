# LLM Inference Control

A blog series applying classical control theory to LLM inference serving systems. Each chapter explores a different controller architecture on progressively more realistic plant models.

---

## Chapter map

| Chapter | Plant | Architecture | Key result |
|---------|-------|-------------|------------|
| [Chapter 1](chapter_1/) | Simulated (MATLAB) | Single-loop LQR + pole placement | Proof of concept — discrete-time state-space design works on the simulated plant |
| [Chapter 2](chapter_2/) | Simulated (MATLAB) | Cascade (inner: B→q, outer: q_ref→l_p95) | Cascade architecture verified in simulation with Franklin augmented state-space and integral outer loop |
| [Chapter 3](chapter_3/) | Real Ollama on M-Mac (qwen2.5:3b) | Cascade attempt | Discovered q≈0 always — the real system has no persistent queue, so the cascade inner loop regulates a non-existent state |
| [Chapter 4](chapter_4/) | Real Ollama on M-Mac (qwen2.5:3b) | Single-loop integral on TTFT | Success — B directly controls TTFT via GPU concurrency; identified TTFT(B) curve, single integral controller regulates latency |
| [Chapter 5](chapter_5/) | vLLM on Apple Silicon (Qwen3-0.6B) | Cascade attempt | vLLM-metal's `num_requests_waiting` gauge is broken (accumulates monotonically); software FIFO queue fails because queue_wait is near-zero; motivates real queue server |
| [Chapter 6](chapter_6/) | Intel Mac queue server (qwen2.5:0.5b) | Single-loop integral on TTFT | Real FIFO queue server — requests genuinely wait; key lesson: l_total = queue_wait + TTFT, must use TTFT-only signal for stable control; CPU machine cannot support cascade |
| [Chapter 7](chapter_7/) | Modal + native vLLM on NVIDIA GPU | Remote single-loop / characterization | Remote GPU serving path works, but serverless/runtime effects hide a clean native queue signal; motivates a wrapper-based attempt |
| [Chapter 8](chapter_8/) | Modal wrapper queue + vLLM on NVIDIA GPU | MATLAB cascade attempt | End-to-end remote cascade plumbing works, but top-level LLM latency does not expose a trustworthy Chapter 2 outer plant; pivot to lower-level GPU batching for Chapter 9 |

---

## Architecture

```
Chapter 1–2: Simulation only
    MATLAB controller ──► simulated plant (llm_plant.m)

Chapter 3–4: Real hardware (M-Mac)
    MATLAB/Simulink ──► Ollama HTTP ──► GPU (Apple Silicon)

Chapter 5: vLLM on Apple Silicon
    Python controller ──► vLLM REST ──► GPU (Metal)
    [abandoned — broken Prometheus metrics]

Chapter 6: Real queue server (Intel Mac)
    MATLAB controller ──► queue_server.py HTTP ──► Ollama ──► CPU
    (controller on M-Mac, server on Intel Mac at 192.168.68.106:8002)

Chapter 7–8: Remote GPU serving on Modal
    MATLAB controller ──► Modal wrapper / vLLM ──► NVIDIA GPU
    Chapter 7: native remote serving characterization
    Chapter 8: wrapper queue + MATLAB cascade attempt
```

---

## Key learnings by chapter

**Ch1 → Ch2:** The cascade architecture (separate inner queue loop + outer latency loop) works cleanly in simulation because batch size B independently controls both queue drain rate and per-request latency.

**Ch3:** On real CPU/GPU hardware without backpressure, the queue is always near zero — the OS schedules requests immediately. The cascade inner loop has nothing to regulate.

**Ch4:** Single-loop integral on TTFT is the right architecture when the queue is always empty. B → TTFT is a stable monotone relationship on a GPU.

**Ch5:** vLLM's Apple Metal backend has a Prometheus metric bug (`num_requests_waiting` never decrements). Software FIFO queues don't help because requests are dispatched in the same tick they arrive. A real scheduler queue is needed to make the cascade valid.

**Ch6:** With a real queue server, `l_total = queue_wait + TTFT`. Using `l_total` as the control signal inverts the control sign at high queue depth (reducing B increases queue_wait faster than it reduces TTFT — positive feedback). Must use TTFT-only (`ttft_recent_mean`) as the control signal. The cascade is still not valid on Intel CPU because the CPU time-slices requests rather than batching them — service rate barely changes with B, so there's no independent queue-rate handle.

**Ch7:** A real remote NVIDIA/vLLM path is operational from the controller side, but a serverless-style deployment still does not expose the clean queue signal needed for a true Chapter 2 cascade story.

**Ch8:** Even with a wrapper FIFO and explicit per-tick batch dispatch on a real GPU, the top-level LLM latency signal is still too aggregated. The critical identification result was an unphysical outer fit:

`l_mean(q_mean) = -4.9228 q + 648.7647`

That negative slope is a warning that top-level request latency is not exposing a clean queueing plant. The correct pivot is to move down to a lower-level GPU batching experiment where batch size is a direct actuator and service time is measured per batch.

---

## Next: Chapter 9

Chapter 9 should pivot away from top-level HTTP LLM latency and toward a
lower-level GPU batching experiment:

- fixed model, prompt length, and output length,
- software queue outside the model runtime,
- exact batch-size control `B[k]`,
- direct measurement of batch service time,
- explicit queue evolution and completion latency.

That setup should match the Chapter 1/2 equations much more faithfully than a
whole-serving-stack latency experiment.

---

## Repo structure

```
chapter_1/          Simulation: LQR + pole placement
  src/              MATLAB scripts (setup_plant.m, design_controller.m, ...)
  simulink_model/   Simulink .slx model
  results_*.png     Final result plots

chapter_2/          Simulation: cascade controller
  src/
  simulink_model/

chapter_3/          Real Ollama: cascade attempt (q≈0 discovery)
  src/
  identification/   Plant identification scripts
  simulink_model/

chapter_4/          Real Ollama: single-loop integral (success)
  src/
  identification/
  simulink_model/

chapter_5/          vLLM Apple Silicon: cascade attempt (abandoned)
  python/           Python controller + characterise/design scripts
  start_vllm.sh

chapter_6/          Intel Mac queue server: single-loop on TTFT
  server/           queue_server.py + setup.sh (runs on Intel Mac)
  matlab/           characterise.m, design_controller.m, run_controller.m
  README.md

chapter_7/          Modal native vLLM remote experiment
  README.md

chapter_8/          Modal wrapper queue + MATLAB cascade attempt
  modal_vllm_wrapper.py
  remote/
  matlab/
  README.md
```
