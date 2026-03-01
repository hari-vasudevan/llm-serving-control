%% setup_plant.m
% Entry point. Defines plant parameters in 'perturbed', computes the
% equilibrium (B0, q0), designs the controller, then runs the simulation.
%
% Change 'method' below to switch between 'lqr' and 'pole_placement'.

clear; clc;
addpath(fileparts(mfilename('fullpath')));

% ── 1. Plant parameters ───────────────────────────────────────────────────────
perturbed.alpha = 10;    % ms  — linear service-time coefficient
perturbed.gamma = 0.8;   % ms  — quadratic service-time coefficient
perturbed.beta  = 2;     % ms  — queuing latency per waiting request
perturbed.delta = 15;    % ms  — p95 spread coefficient

perturbed.dt    = 0.1;   % s   — scheduling tick (discrete sample time)
perturbed.B_min = 1;     % –   — batch size lower bound (saturation)
perturbed.B_max = 32;    % –   — batch size upper bound (VRAM constraint)

% ── 2. Operating conditions ───────────────────────────────────────────────────
perturbed.lambda_mean  = 8;    % req/tick — mean arrival rate at equilibrium
perturbed.L_p95_target = 150;  % ms       — SLA target (determines q0)

% ── 3. Equilibrium (B0, q0) ───────────────────────────────────────────────────
%
% Condition 1 — queue balance at steady state:
%   q[k+1] = q[k]  →  lambda = B0  →  B0 = lambda_mean
%
% Condition 2 — SLA met at equilibrium:
%   L_p95(B0, q0) = L_p95_target  →  solve for q0:
%
%   q0 = ( L_p95_target - alpha*B0 - gamma*B0^2 - 1.645*delta/sqrt(B0) ) / beta

perturbed.B0 = perturbed.lambda_mean;

perturbed.q0 = ( perturbed.L_p95_target ...
               - perturbed.alpha * perturbed.B0 ...
               - perturbed.gamma * perturbed.B0^2 ...
               - 1.645 * perturbed.delta / sqrt(perturbed.B0) ) ...
               / perturbed.beta;

% Sanity check — evaluate nonlinear plant at equilibrium
[~, L_mean_eq, L_p95_eq] = llm_plant(perturbed.q0, perturbed.B0, ...
                                       perturbed.lambda_mean, perturbed);
fprintf('=== Equilibrium ===\n');
fprintf('  B0     = %.4f  req/tick\n', perturbed.B0);
fprintf('  q0     = %.4f  requests\n', perturbed.q0);
fprintf('  L_mean = %.2f  ms\n', L_mean_eq);
fprintf('  L_p95  = %.2f  ms  (target = %.0f ms)\n\n', L_p95_eq, perturbed.L_p95_target);

% ── 4. Design controller ──────────────────────────────────────────────────────
%   Change method here:  'lqr'  or  'pole_placement'
method     = 'lqr';
controller = design_controller(perturbed, method);

% ── 5. Run closed-loop simulation ─────────────────────────────────────────────
run_simulation(perturbed, controller);
