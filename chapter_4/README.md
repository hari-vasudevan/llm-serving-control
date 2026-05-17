# Chapter 4 — Real Ollama on M-Mac: Single-Loop Integral on TTFT (Works)

## What This Is

Chapter 4 abandons the cascade and replaces it with a single-loop integral
controller that regulates TTFT directly using batch size `B` as the actuator.
This works because `B → TTFT` is a stable monotone relationship on a GPU:
larger batches increase per-request latency, giving the controller a
controllable plant.

## Prerequisites

- MATLAB R2024b or newer with Control System Toolbox and Simulink
- Ollama installed on Apple Silicon Mac
  ```bash
  brew install ollama
  ```
- Model pulled (choose one)
  ```bash
  ollama pull qwen2.5:3b        # larger, slower, more load
  ollama pull qwen2.5:0.5b      # faster, useful for quick tests
  ```

## How to Run

### 1. Start Ollama

```bash
ollama serve
curl http://localhost:11434/api/tags    # should list pulled models
```

### 2. Identify the Plant

This step measures how TTFT changes with batch size `B` at your operating point:

```matlab
cd chapter_4/identification
identify_plant      % sweeps B, measures TTFT, fits alpha and gamma
```

Outputs: `id_ttft_curve.png`, `id_stage1.png`, `id_stage2.png`.

### 3. Design the Controller

```matlab
cd chapter_4/src
setup_plant         % uses identification results, computes integral gain K_i
```

### 4. Run the Closed-Loop Controller

Open the Simulink model and run it with Ollama live:

```matlab
cd chapter_4
open_system('simulink_model/llm_inferencing_control.slx')
% click Run
```

The controller will adjust `B` each tick to keep TTFT near the target.

## Expected Output

TTFT converges to the target (e.g., 300ms) within a few ticks. Arrivals
can be stepped up or down as a disturbance — the integral controller
recovers. Queue depth stays near zero throughout (Ollama still dispatches
immediately), but TTFT is controllable directly.

## Files

| File | Purpose |
|------|---------|
| `src/setup_plant.m` | Entry point — plant params, single-loop design |
| `src/ollama_plant.m` | Simulink System Object: Ollama HTTP plant |
| `src/ollama_ttft.m` | TTFT measurement helper |
| `src/llm_plant.m` | Simulated plant (for offline testing) |
| `identification/identify_plant.m` | B-sweep plant ID against live Ollama |
| `simulink_model/` | Simulink closed-loop model |
