# Chapter 3 — Real Ollama on M-Mac: Cascade Attempt (Broke)

## What This Is

The first attempt to run the Chapter 2 cascade on a real LLM inference stack.
The controller is in MATLAB/Simulink; the plant is Ollama running
`qwen2.5:3b` on an Apple Silicon Mac.

It broke: Ollama (and most serving frameworks in their default configuration)
does not maintain a persistent backpressure queue. The OS dispatches incoming
requests immediately into GPU threads. Queue depth is always near zero, so
the inner loop has nothing to regulate.

## Prerequisites

- MATLAB R2024b or newer with Control System Toolbox and Simulink
- Ollama installed on an Apple Silicon Mac
  ```bash
  brew install ollama
  ```
- Qwen2.5:3b model pulled
  ```bash
  ollama pull qwen2.5:3b
  ```

## How to Run

### 1. Start Ollama

```bash
ollama serve
# confirm it's up:
curl http://localhost:11434/api/tags
```

### 2. Identify the Plant

```matlab
cd chapter_3/identification
identify_plant      % sweeps batch size B, measures TTFT
```

This script sends requests to Ollama and measures first-token latency at
different batch sizes. It writes identification parameters used by the
controller design step.

### 3. Run the Simulink Controller

Open the Simulink model and run it with Ollama live:

```matlab
cd chapter_3
open_system('simulink_model/llm_inferencing_control.slx')
% set the Ollama URL inside the model (default: http://localhost:11434)
% click Run
```

Or drive the plant directly from the MATLAB script:

```matlab
cd chapter_3/src
setup_plant         % configures controller
% then run the Simulink model
```

## Expected Outcome

Queue depth `q` stays near zero regardless of arrival rate. The inner loop
has no signal to work with. The cascade does not regulate latency.

This is the expected (instructive) failure. See the root README for the full
explanation.

## Files

| File | Purpose |
|------|---------|
| `src/setup_plant.m` | Plant config and controller design |
| `src/ollama_plant.m` | Simulink System Object: Ollama HTTP plant |
| `src/ollama_ttft.m` | TTFT measurement helper |
| `identification/identify_plant.m` | B-sweep identification against live Ollama |
| `simulink_model/` | Simulink closed-loop model |
