%% setup_plant.m  --  Chapter 5: Cascade Control on real vLLM inference
%
% Entry point for Chapter 5.  Loads identified plant parameters from
% characterise_plant.m, computes the equilibrium, and designs the cascade
% controller.
%
% Run this before pressing Run in Simulink.
%
% Requires:
%   identification/identified_params.mat   (run characterise_plant.m first)
%
% Plant model (identified from real hardware):
%   l_e2e[k] = alpha*B[k] + gamma*B[k]^2 + beta*q[k] + l_gen
%
%   where  alpha, gamma  capture GPU concurrency cost (from TTFT sweep)
%          beta          captures queueing overhead per waiting request
%          l_gen         is generation latency (constant, ~700ms for 20 tokens)
%          q[k]          is vllm:num_requests_waiting from /metrics
%
% Identified parameters (qwen3-0.6b-4bit, max_num_seqs=4, e2e_tokens=20):
%   alpha = 33.22 ms/req
%   gamma = -1.56 ms/req^2
%   beta  = 444.91 ms/req   (each queued request adds ~445ms to e2e latency)
%
% Operating regime (from Stage 4 envelope):
%   lambda <= 4 req/tick  -> q = 0,  latency driven by concurrency only
%   lambda >= 6 req/tick  -> q builds to ~4,  latency jumps to ~2000ms
%   We operate at lambda_mean = 3 (moderate load, queue-free equilibrium).
%
% Controller design:
%   Same cascade architecture as Chapter 2b:
%     Inner loop:  B -> q    (LQR / pole placement, Franklin augmented)
%     Outer loop:  q_ref -> l_e2e  (integral-only, analytical gain)
%   dt = 1.0s (real inference tick, not 0.1s simulation tick)

clear; clc;
addpath(fileparts(mfilename('fullpath')));

% -------------------------------------------------------------------------
% 1. Load identified parameters
% -------------------------------------------------------------------------
id_path = fullfile(fileparts(mfilename('fullpath')), ...
    '..', 'identification', 'identified_params.mat');

if ~exist(id_path, 'file')
    error(['identified_params.mat not found.\n' ...
           'Run chapter_5/identification/characterise_plant.m first.\n' ...
           'Expected: %s'], id_path);
end

load(id_path, 'identified');
fprintf('=== Loaded identified parameters (%s) ===\n', identified.model);
fprintf('  alpha = %.4f ms/req\n',   identified.alpha);
fprintf('  gamma = %.4f ms/req^2\n', identified.gamma);
fprintf('  beta  = %.4f ms/req\n\n', identified.beta);

% -------------------------------------------------------------------------
% 2. Plant parameters
% -------------------------------------------------------------------------
perturbed.alpha = identified.alpha;   % 33.2222 ms/req
perturbed.gamma = identified.gamma;   % -1.5605 ms/req^2
perturbed.beta  = identified.beta;    % 444.9068 ms/req

% Physical bounds (from characterisation)
perturbed.dt    = 1.0;   % s   -- real inference tick (not simulation)
perturbed.B_min = 1;     %     -- batch size lower bound
perturbed.B_max = 8;     %     -- max concurrent requests (tested safe cap)
perturbed.q_max = 4;     %     -- max observable queue (= max_num_seqs)

% -------------------------------------------------------------------------
% 3. Operating conditions
% -------------------------------------------------------------------------
% From the Stage 4 envelope:
%   lambda=3, B=3, q=0, l_e2e=790ms is a clean operating point.
%   Queue is zero, no overloading, wall-clock fits within the 1s tick.
%
% l_gen (generation latency) = l_e2e - TTFT at B=3, q=0
%   TTFT(B=3) = alpha*3 + gamma*9 = 99.67 - 14.04 = 85.6ms
%   l_e2e(B=3, q=0) = 790ms (measured)
%   l_gen ~ 790 - 86 = 704ms  (constant for 20 output tokens)

perturbed.lambda_mean   = 3;    % req/tick -- moderate load, q=0 at equilibrium
perturbed.L_mean_target = 800;  % ms       -- slightly above equilibrium latency

% p95 headroom: at q_max=4, beta contribution = 444.9*4 = 1780ms
perturbed.L_p95_target  = perturbed.L_mean_target + identified.beta * perturbed.q_max;
fprintf('=== Operating conditions ===\n');
fprintf('  lambda_mean    = %g req/tick\n', perturbed.lambda_mean);
fprintf('  L_mean_target  = %.0f ms\n',     perturbed.L_mean_target);
fprintf('  L_p95_target   = %.0f ms (target + beta*q_max)\n\n', perturbed.L_p95_target);

% -------------------------------------------------------------------------
% 4. Equilibrium
% -------------------------------------------------------------------------
% Queue balance:  q[k+1] = q[k]  =>  B0 = lambda_mean
% At q0 = 0:  l_e2e(B0, 0) = alpha*B0 + gamma*B0^2 + l_gen
%
% We set q0 = 0 since that is the natural operating point at lambda=3.
% The outer controller will command q_ref > 0 only if l_e2e > L_mean_target.

perturbed.B0 = perturbed.lambda_mean;   % = 3
perturbed.q0 = 0;                       % natural equilibrium at lambda=3

% Predicted equilibrium latency (TTFT component only; l_gen is additive constant)
l_ttft_eq = perturbed.alpha * perturbed.B0 + perturbed.gamma * perturbed.B0^2;
fprintf('=== Equilibrium ===\n');
fprintf('  B0           = %g req/tick\n', perturbed.B0);
fprintf('  q0           = %g req\n',      perturbed.q0);
fprintf('  TTFT(B0,q=0) = %.1f ms  (identified curve)\n\n', l_ttft_eq);

% -------------------------------------------------------------------------
% 5. Pole placement parameters (inner loop)
% -------------------------------------------------------------------------
% Inner loop dynamics: q[k+1] = q[k] - B[k] + arrivals
% dt=1s ticks. Queue settles in ~2-3 ticks -> tau1=2s, tau2=3s is appropriate.
perturbed.pp_tau1 = 2.0;   % s
perturbed.pp_tau2 = 3.0;   % s
perturbed.pp_f    = 0;     % non-oscillatory

% -------------------------------------------------------------------------
% 6. Outer loop time constant
% -------------------------------------------------------------------------
% Outer loop integrates latency error, commands q_ref.
% Must be >> inner settling time (~3s) and >> measurement lag.
% Use 20s: conservative, avoids oscillation given large beta.
perturbed.tau_out = 20.0;  % s

% -------------------------------------------------------------------------
% 7. Design cascade controller
% -------------------------------------------------------------------------
method     = 'pole_placement';
controller = design_controller(perturbed, method);

% -------------------------------------------------------------------------
% 8. Outer loop gain sanity check
% -------------------------------------------------------------------------
K_il = controller.outer_c.K_il;
z_cl = controller.outer_c.z_cl;
fprintf('=== Outer loop gain check ===\n');
fprintf('  K_il = %.8f\n', K_il);
fprintf('  z_cl = %.6f  (target %.6f from tau_out=%.0fs)\n', ...
    z_cl, exp(-perturbed.dt/perturbed.tau_out), perturbed.tau_out);
fprintf('  Stability: |z_cl| = %.4f < 1: %s\n\n', abs(z_cl), mat2str(abs(z_cl)<1));

% Each unit of q_ref changes latency by beta=444.9ms.
% So a K_il of this magnitude means the integrator must accumulate ~1/K_il
% ticks of error before commanding q_ref = 1.
fprintf('  Interpretation:\n');
fprintf('    A sustained latency error of 1ms for %.0f ticks commands q_ref += 1 req\n', ...
    round(1/abs(K_il)));
fprintf('    (beta=%.0fms/req, so q_ref=1 drives l_e2e up by %.0fms)\n\n', ...
    perturbed.beta, perturbed.beta);

% -------------------------------------------------------------------------
% 9. Summary table
% -------------------------------------------------------------------------
fprintf('╔══════════════════════════════════════════════════════════════╗\n');
fprintf('║  Chapter 5 -- Cascade Controller  (%s)\n', method);
fprintf('╠══════════════════════════════════════════════════════════════╣\n');
fprintf('║  Plant (identified)\n');
fprintf('║    alpha = %8.4f  ms/req\n',   perturbed.alpha);
fprintf('║    gamma = %8.4f  ms/req^2\n', perturbed.gamma);
fprintf('║    beta  = %8.4f  ms/req\n',   perturbed.beta);
fprintf('╠══════════════════════════════════════════════════════════════╣\n');
fprintf('║  Operating point\n');
fprintf('║    lambda = %g req/tick    B0 = %g    q0 = %g\n', ...
    perturbed.lambda_mean, perturbed.B0, perturbed.q0);
fprintf('║    L_target = %.0f ms    L_p95_target = %.0f ms\n', ...
    perturbed.L_mean_target, perturbed.L_p95_target);
fprintf('╠══════════════════════════════════════════════════════════════╣\n');
fprintf('║  Inner loop (B -> q)\n');
fprintf('║    K_q = %.4f    K_i = %.4f\n', ...
    controller.inner_c.K_q, controller.inner_c.K_i);
fprintf('║    CL poles: [%.4f, %.4f]\n', ...
    controller.inner_c.poles_cl(1), controller.inner_c.poles_cl(2));
fprintf('╠══════════════════════════════════════════════════════════════╣\n');
fprintf('║  Outer loop (q_ref -> l_e2e)\n');
fprintf('║    K_il = %.8f\n', controller.outer_c.K_il);
fprintf('║    z_cl = %.6f    tau_out = %.0f s\n', ...
    controller.outer_c.z_cl, controller.outer_c.tau_out);
fprintf('╚══════════════════════════════════════════════════════════════╝\n');
