# Chapter 1 — Simulation: LQR and Pole Placement

## What This Is

Pure MATLAB simulation of a single-loop LQR and pole-placement controller on
a simulated LLM inference plant. No hardware, no network, no model download.
This is the control-theory proof of concept.

The plant is defined in `src/llm_plant.m`. The controller is designed by
`src/setup_plant.m`, which runs either LQR or pole-placement depending on the
`method` variable at the top of the script.

## Prerequisites

- MATLAB R2024b or newer (R2025b used in this project)
- Control System Toolbox
- Simulink (optional — scripts run standalone; Simulink model is for visualisation)

## How to Run

Open MATLAB, navigate to this chapter, and run the main script:

```matlab
cd chapter_1/src
setup_plant          % designs controller and runs the closed-loop simulation
```

That script:
1. defines the plant parameters (batch service time, queue dynamics),
2. computes the equilibrium operating point,
3. designs either an LQR or pole-placement controller (set `method` at the top),
4. runs the closed-loop simulation and plots the result.

To switch between LQR and pole placement, open `setup_plant.m` and change:

```matlab
method = 'lqr';          % or 'pole_placement'
```

then re-run.

## Simulink Model

The Simulink model at `simulink_model/llm_inferencing_control.slx` is a
visual version of the same experiment. Open it in Simulink after running
`setup_plant` (which populates the workspace) and click Run.

```matlab
cd chapter_1
open_system('simulink_model/llm_inferencing_control.slx')
```

## Expected Output

- A figure showing queue depth `q[k]`, latency `L_mean[k]`, and batch
  size `B[k]` over the simulation horizon.
- Console output with closed-loop pole locations and gain values.

## Files

| File | Purpose |
|------|---------|
| `src/setup_plant.m` | Entry point — defines plant, designs controller, runs simulation |
| `src/llm_plant.m` | Simulated plant: queue dynamics + latency model |
| `src/design_controller.m` | LQR and pole-placement design functions |
| `src/run_simulation.m` | Simulation loop (called by setup_plant) |
| `simulink_model/` | Equivalent Simulink model |
