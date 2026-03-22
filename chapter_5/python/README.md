# Chapter 5 — Python implementation

All MATLAB identification, design, and control scripts replaced with clean Python.

## Prerequisites

```bash
source ~/.venv-vllm-metal/bin/activate
pip install numpy scipy matplotlib requests
```

## Step 1 — Start vLLM

```bash
cd /Users/hvasudevan/Documents/MATLAB/llm_control_v2/chapter_5
./start_vllm.sh --bg
curl http://localhost:8001/health   # confirm: {"status":"ok"}
```

## Step 2 — Identify the plant

```bash
cd python/

# Full identification (all stages, ~20 min):
python3 characterise.py

# Quick version (Stage 2 only, ~5 min, sufficient for controller design):
python3 characterise.py --skip_stage3 --skip_stage4
```

Outputs: `identified_params.json`, `ch5_stage2_b_sweep.png`

## Step 3 — Design the controller

```bash
python3 design_controller.py
```

Reads `identified_params.json`, outputs `controller_params.json`.

Override beta if not identified (default: analytical estimate):
```bash
python3 design_controller.py --beta 200 --tau_out 20 --tau1 2 --tau2 3
```

## Step 4 — Run the closed-loop controller

```bash
# Steady-state test (60 ticks, lambda=3):
python3 run_controller.py --n_ticks 60 --lambda_mean 3

# With step spike at tick 40, recovers by tick 80:
python3 run_controller.py --n_ticks 120 --lambda_mean 3 --spike_on 40 --spike_off 80

# Override latency target:
python3 run_controller.py --n_ticks 60 --L_target 600
```

Outputs: `ch5_closed_loop_HHMMSS.png`, `ch5_run_log_HHMMSS.json`

## What each file does

| File | Purpose |
|------|---------|
| `characterise.py` | Plant identification: Stage 2 B-sweep (α,γ), Stage 3 queue sweep (β), Stage 4 envelope |
| `design_controller.py` | Cascade controller design: inner loop pole placement, outer loop integral gain |
| `run_controller.py` | Closed-loop experiment: runs cascade against live vLLM, plots latency/queue/B |

## Known vllm-metal limitation

`vllm:num_requests_waiting` accumulates monotonically in vllm-metal
(Prometheus multiprocessing gauge bug). The scripts use a software inflight
counter as a queue proxy instead.
