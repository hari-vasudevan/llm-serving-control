# Project: llm-serving-control

## Environment

- macOS (Apple Silicon)
- MATLAB R2025b at `/Applications/MATLAB_R2025b.app`
- MATLAB MCP Core Server (`matlab-mcp-core-server-maca64`) available via Claude Desktop
- 90+ installed toolboxes including Control System, Robust Control, MPC, System Identification, Simulink, Simulink Control Design, Signal Processing, Deep Learning, Computer Vision, Communications, and Robotics

## Code Style

- Write clean, well-commented MATLAB code. One operation per line unless combining is clearly more readable.
- Use descriptive variable names: `G_plant` not `G1`, `K_lqr` not `K`, `T_settling` not `ts`.
- Include a header comment block in every script: purpose, date placeholder, and key assumptions.
- Prefer vectorized operations over loops.
- Use `s = tf('s')` for symbolic transfer function construction when it helps readability.
- SI units unless otherwise stated. Always label plot axes with units.

## MATLAB Conventions

- Prefer state-space (`ss`) as the canonical internal representation. Convert `tf`/`zpk` to `ss` early in any workflow.
- When discretizing, always specify the method explicitly: `c2d(sys, Ts, 'tustin')` or `'zoh'` — never rely on defaults silently.
- Close figures or use `figure('Name', '...')` to keep plot windows organized.
- Use `grid on` on all plots.

## Simulink Conventions

- Name all signals and subsystems descriptively.
- Use fixed-step solvers (`ode4`) for control design models unless there's a specific reason for variable-step.
- For linearization: `operpoint` / `findop` for operating points, then `linearize(mdl, op)`.

## Git

- Write concise commit messages in imperative mood: "Add PID tuning script for thermal plant" not "Added stuff."
- Do not commit generated data files (`.mat` over 10 MB) without asking first.
- Keep scripts and models in separate subdirectories where practical.

## What to Avoid

- Do not assume a system is SISO without checking dimensions.
- Do not present a controller design without verifying closed-loop stability (check poles or margins).
- Do not use continuous-time controllers for digital implementation without discretizing and noting the sample rate.
- Do not generate excessively long scripts. Break into functions if a script exceeds ~150 lines.
