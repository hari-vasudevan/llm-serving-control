# Chapter 5 — vLLM on Apple Silicon: Cascade Attempt (Broke)

## What This Is

Chapter 5 moves from Ollama to vLLM on Apple Silicon (Metal backend) and
replaces the MATLAB controller with pure Python. The cascade attempt fails
for two independent reasons:

1. **Broken Prometheus metric**: `vllm:num_requests_waiting` accumulates
   monotonically in the Apple Metal backend — it increments on request arrival
   but never decrements when a request is dispatched. The queue signal is
   unusable.
2. **Useless software FIFO**: even a workaround queue server is bypassed
   because vLLM dispatches requests in the same scheduler tick they arrive.
   Queue wait is always near zero.

## Prerequisites

- Apple Silicon Mac
- Python 3.11+ with a vLLM-metal virtualenv
- The vLLM Apple Metal build (`pip install vllm` in a Metal-enabled env)

Create the virtualenv once:

```bash
python3 -m venv ~/.venv-vllm-metal
source ~/.venv-vllm-metal/bin/activate
pip install vllm torch numpy scipy matplotlib requests
```

## How to Run

### 1. Start vLLM

```bash
cd chapter_5
./start_vllm.sh --bg          # background, logs to /tmp/vllm_chapter5.log
curl http://localhost:8001/health   # expect {"status":"ok"}
```

`start_vllm.sh` serves `mlx-community/Qwen3-0.6B-4bit` on port 8001 with
`--max-num-seqs 4` and `--max-model-len 256`.

To stop it:

```bash
pkill -f "vllm serve"
```

### 2. Identify the Plant

```bash
source ~/.venv-vllm-metal/bin/activate
cd chapter_5/python

# Full identification (~20 min):
python3 characterise.py

# Quick version — Stage 2 only (~5 min):
python3 characterise.py --skip_stage3 --skip_stage4
```

Outputs: `identified_params.json`, `ch5_stage2_b_sweep.png`.

### 3. Design the Controller

```bash
python3 design_controller.py
```

Reads `identified_params.json`, writes `controller_params.json`.

Override individual params if needed:

```bash
python3 design_controller.py --beta 200 --tau_out 20 --tau1 2 --tau2 3
```

### 4. Run the Closed-Loop Controller

```bash
# 60-tick steady-state test:
python3 run_controller.py --n_ticks 60 --lambda_mean 3

# With an arrival spike at tick 40:
python3 run_controller.py --n_ticks 120 --lambda_mean 3 --spike_on 40 --spike_off 80

# Different latency target:
python3 run_controller.py --n_ticks 60 --L_target 600
```

Outputs: `ch5_closed_loop_HHMMSS.png`, `ch5_cascade_log_HHMMSS.json`.

## Expected Outcome

The queue depth signal from vLLM metrics is unusable (monotonically
increasing). The software inflight counter used as a workaround shows
near-zero queue depth at all times. The cascade inner loop has nothing
to regulate. This is the expected (instructive) failure.

## MATLAB Simulink Path (Alternative)

If you prefer MATLAB + Simulink over the Python scripts:

```matlab
cd chapter_5
open_system('simulink_model/llm_inferencing_control.slx')
```

The `src/vllm_plant.m` and `src/vllm_ttft.m` System Objects connect the
Simulink model to the running vLLM server. The MATLAB path shows the same
failure: queue metric is broken.

## Files

| File | Purpose |
|------|---------|
| `start_vllm.sh` | Starts vLLM (Metal) with chapter-appropriate settings |
| `python/characterise.py` | Plant identification: B-sweep against live vLLM |
| `python/design_controller.py` | Cascade controller design from identified params |
| `python/run_controller.py` | Closed-loop experiment runner |
| `python/identified_params.json` | Saved identification results |
| `python/controller_params.json` | Saved controller design |
| `src/vllm_plant.m` | Simulink System Object for vLLM plant (MATLAB path) |
| `src/vllm_ttft.m` | TTFT measurement helper (MATLAB path) |
| `simulink_model/` | Simulink model (MATLAB path) |
