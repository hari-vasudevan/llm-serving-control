%% setup_plant.m  —  Chapter 2b: Cascade Control
% Entry point. Defines plant parameters, computes equilibrium, designs
% the cascade controller (inner: B->q, outer: q_ref->l_p95), then runs
% the MATLAB-only simulation.
%
% Change 'method' to switch between 'lqr' and 'pole_placement'.

clear; clc;
addpath(fileparts(mfilename('fullpath')));

% -- 1. Plant parameters ------------------------------------------------------
perturbed.alpha = 0.1;    % ms   -- linear service-time coefficient
perturbed.gamma = 0.8;   % ms   -- quadratic service-time coefficient
perturbed.beta  = 2;     % ms   -- queuing latency per waiting request
perturbed.delta = 15;    % ms   -- p95 spread coefficient

perturbed.dt    = 1.0;   % s    -- scheduling tick (1 s for real Ollama; warm latency ~500 ms/req)
perturbed.B_min = 1;     %      -- batch size lower bound
perturbed.B_max = 32*2;    %      -- batch size upper bound (VRAM)
perturbed.q_max = 3000;    %      -- queue depth upper bound

% -- 2. Operating conditions --------------------------------------------------
perturbed.lambda_mean  = 5;    % req/tick -- mean arrival rate
perturbed.L_p95_target = 100;  % ms       -- SLA target
perturbed.L_mean_target = 25;  % ms       -- SLA target

% -- 3. Rolling window (measurement, used upstream of outer controller) --------
perturbed.N_win = 20;    % samples -- l_p95 rolling window length (T_win = 20 s at dt=1s)

% -- 4. Equilibrium -----------------------------------------------------------
%
%   Queue balance:  q[k+1] = q[k]  =>  B0 = lambda_mean
%
%   SLA met at equilibrium:
%     q0 = ( L_p95_target - alpha*B0 - gamma*B0^2 - 1.645*delta/sqrt(B0) ) / beta

perturbed.B0 = perturbed.lambda_mean;

% perturbed.q0 = ( perturbed.L_p95_target ...
%                - perturbed.alpha  * perturbed.B0 ...
%                - perturbed.gamma  * perturbed.B0^2 ...
%                - 1.645 * perturbed.delta / sqrt(perturbed.B0) ) ...
%                / perturbed.beta;

perturbed.q0 = ( perturbed.L_mean_target ...
               - perturbed.alpha * perturbed.B0 ...
               - perturbed.gamma * perturbed.B0^2 ) ...
               / perturbed.beta;

[~, L_mean_eq, L_p95_eq] = llm_plant(perturbed.q0, perturbed.B0, ...
                                       perturbed.lambda_mean, perturbed);
fprintf('=== Equilibrium (computed from mean-latency target) ===\n');
fprintf('  B0      = %.4f  req/tick\n', perturbed.B0);
fprintf('  q0      = %.4f  requests\n', perturbed.q0);
fprintf('  L_mean  = %.2f  ms  (target = %.2f ms)\n', L_mean_eq, perturbed.L_mean_target);
fprintf('  L_p95   = %.2f  ms  (reported only; not used for equilibrium)\n\n', L_p95_eq);

% -- 5. Inner loop pole placement parameters (only used when method = 'pole_placement')
perturbed.pp_tau1 = 0.25;   % s  -- inner pole 1 time constant
perturbed.pp_tau2 = 0.5;    % s  -- inner pole 2 time constant
perturbed.pp_tau  = 1.0/3;  % s  -- inner oscillatory envelope (if pp_f > 0)
perturbed.pp_f    = 0;      % Hz -- inner damped frequency (0 = non-oscillatory)

% -- 6. Outer loop time constant -----------------------------------------------
%   Integral-only design. One parameter sets the closed-loop pole analytically:
%     K_il = (exp(-dt/tau_out) - 1) / beta
%   Must be >> tau_win (3 s) and >> tau_inner (~0.5 s).  10 s is a safe start.
perturbed.tau_out = 60;  % s  -- desired outer CL time constant (slow; respects 10s measurement lag)

% -- 7. Design cascade controller ---------------------------------------------
% method     = 'lqr';
method   = 'pole_placement';

controller = design_controller(perturbed, method);

% -- 8. Run MATLAB simulation -------------------------------------------------
% run_simulation(perturbed, controller);
