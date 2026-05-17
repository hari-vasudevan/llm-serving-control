# Chapter 2 — Simulation: Cascade Controller

## What This Is

Pure MATLAB simulation of a two-loop cascade controller on the same simulated
plant as Chapter 1. The cascade separates the inner batch-size actuator
(`B → q`) from the outer latency regulator (`q_ref → L_mean`).

This is the architecture that the rest of the project tries to apply to real
hardware — and progressively discovers where the real plant diverges from the
simulated one.

## Prerequisites

- MATLAB R2024b or newer
- Control System Toolbox
- Simulink (optional, for visual model)

## How to Run

```matlab
cd chapter_2/src
setup_plant        % designs cascade controller and runs simulation
```

The script defines the plant, designs both inner and outer loop gains, runs
the closed-loop simulation, and plots queue depth, latency, and batch size.

For a polished comparison plot showing `L_target` step changes:

```matlab
ch2_run_and_plot   % generates ch2_closed_loop_polished.png
```

## Simulink Model

```matlab
cd chapter_2
open_system('simulink_model/llm_inferencing_control.slx')
```

Run `setup_plant` first to populate the workspace, then click Run in Simulink.

## Expected Output

- Closed-loop plots: `q[k]` tracking `q_ref[k]`, `L_mean[k]` tracking
  `L_mean_target`, `B[k]` actuator trace.
- `ch2_closed_loop_polished.png` — publication-quality subplot with three
  `L_target` step changes.

![Closed-loop result](ch2_closed_loop_polished.png)

## Files

| File | Purpose |
|------|---------|
| `src/setup_plant.m` | Entry point — plant params, cascade design, simulation |
| `src/llm_plant.m` | Simulated plant (same as Chapter 1) |
| `src/design_controller.m` | Cascade gain design (inner + outer) |
| `src/ch2_run_and_plot.m` | Polished multi-step plot generator |
| `simulink_model/` | Equivalent Simulink model |
| `ch2_closed_loop_polished.png` | Final result plot |
