# Chapter 8 — MATLAB Cascade Control Over A Modal GPU Wrapper

Chapter 8 returns to the Chapter 2 cascade architecture, but now runs against
a real GPU-backed remote service.

The plant for this chapter is:

`MATLAB controller -> Modal wrapper queue server -> local vLLM -> GPU`

This deliberately keeps **batch size `B` as the actuator**. Native vLLM does
not expose a clean per-tick batch-size control input, so the Modal wrapper is
responsible for:

- accepting requests from MATLAB,
- holding them in a software FIFO queue,
- dispatching exactly `B` requests per tick into vLLM,
- exposing queue and latency metrics for the cascade controller.

The chapter goal is:

1. identify the plant from MATLAB,
2. design the cascade controller in MATLAB,
3. run closed-loop control from MATLAB,
4. test both steady Poisson load and spike disturbances,
5. regulate `l_mean` only.

## Folder layout

- `modal_vllm_wrapper.py`
  Modal deployment entrypoint. Starts vLLM and the Chapter 8 wrapper server in
  the same GPU container.
- `remote/vllm_modal_wrapper.py`
  HTTP wrapper server with FIFO queue, per-tick batch dispatch, metrics, and
  verbose trace logging.
- `matlab/characterise_plant.m`
  MATLAB plant identification script.
- `matlab/design_controller.m`
  MATLAB cascade controller design script.
- `matlab/run_cascade_controller.m`
  MATLAB closed-loop run with steady and spiky arrival-rate segments.

## Runtime tracing

This chapter is intentionally noisy at runtime. The traces include:

- MATLAB side:
  - each HTTP request sent to Modal,
  - each reply received,
  - arrivals per tick,
  - queue and latency observations,
  - controller state and commanded `B`.
- Modal side:
  - each request received from the client,
  - prompt previews and payload summaries,
  - enqueue and dispatch timestamps,
  - queue size and approximate lambda,
  - TTFT, queue wait, end-to-end latency,
  - native vLLM metric snapshots.

## Hugging Face fast downloads

The Modal image installs `huggingface_hub[hf_xet]` and sets:

- `HF_XET_HIGH_PERFORMANCE=1`

That is the current preferred Hugging Face fast-transfer path. If a future
stack still logs the legacy `hf_transfer` message, install
`huggingface_hub[hf_transfer]` and set `HF_HUB_ENABLE_HF_TRANSFER=1`.

## High-level run flow

1. Deploy the Modal service:

   ```bash
   python3 -m venv .modal-venv
   source .modal-venv/bin/activate
   pip install modal
   modal setup
   modal deploy chapter_8/modal_vllm_wrapper.py
   ```

2. Set the resulting endpoint URL inside the MATLAB scripts.

3. In MATLAB:

   ```matlab
   cd chapter_8/matlab
   characterise_plant
   design_controller
   run_cascade_controller
   ```

4. Tail Modal logs:

   ```bash
   modal app logs chapter-8-vllm-wrapper
   ```

5. Review MATLAB logs and `.mat` outputs in `chapter_8/matlab`.
