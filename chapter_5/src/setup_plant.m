%% setup_plant.m  --  Chapter 5: Cascade Control on real vLLM inference
%
% Run this before pressing Run in Simulink.
%
% PLANT MODEL
% -----------
%   l_e2e[k] = alpha*B[k] + gamma*B[k]^2 + beta*q[k] + l_gen
%
%   alpha, gamma  -- GPU concurrency cost, identified from TTFT sweep:
%                    alpha=33.34 ms/req,  gamma=-1.36 ms/req^2,  R^2=0.979
%
%   beta          -- queueing delay per waiting request.
%                    NOT directly identified (vllm-metal gauge bug).
%                    Estimated analytically: see Section 4 below.
%
%   l_gen         -- generation latency, ~700ms for 20 tokens at B=3.
%                    Additive constant, does not affect control design.
%
%   q[k]          -- vllm:num_requests_waiting from /metrics at tick start.
%
% OPERATING REGIME (from Stage 4 envelope, e2e 20 tokens)
%   lambda=1: l_ss=259ms,  q=0,  wall=265ms
%   lambda=2: l_ss=523ms,  q=0,  wall=536ms
%   lambda=3: l_ss=741ms,  q=0,  wall=758ms   <- equilibrium
%   lambda=4: l_ss=1097ms, q=0,  wall=1125ms  (tick overrun begins)
%
% We operate at lambda_mean=3: wall-clock fits comfortably within dt=1s.

clear; clc;
addpath(fileparts(mfilename('fullpath')));

% -------------------------------------------------------------------------
% 1. Identified parameters (Stage 2, characterise_plant.m)
% -------------------------------------------------------------------------
perturbed.alpha = 33.3431;    % ms/req      R^2 = 0.979
perturbed.gamma = -1.3629;    % ms/req^2

fprintf('=== Stage 2 identification (qwen3-0.6b-4bit, TTFT) ===\n');
fprintf('  alpha = %.4f ms/req\n',   perturbed.alpha);
fprintf('  gamma = %.4f ms/req^2\n', perturbed.gamma);

% -------------------------------------------------------------------------
% 2. Beta estimation (analytical, vllm-metal gauge was unreliable)
% -------------------------------------------------------------------------
% Each request waiting in vLLM's scheduler queue must wait for currently
% running requests to complete before being scheduled.
%
% With max_num_seqs=4 running in parallel, the next batch starts as soon
% as all 4 complete.  e2e time at max concurrency (B=4):
%   l_e2e(B=4, q=0) = alpha*4 + gamma*16 + l_gen
%                   = 133.4 - 21.8 + 700 ~= 812ms  per-request mean
%
% But requests aren't served one at a time -- 4 complete simultaneously
% and the next 4 start.  So a request at position q in the queue waits
% ceil(q/4) batch completions.  Each batch takes ~l_e2e(B=4) = ~812ms.
%
% The incremental wait added by one extra request in the queue
% (averaged over position) is approximately l_e2e(B=4) / max_num_seqs:
%
%   beta = l_e2e(B=4, q=0) / max_num_seqs
%        = 812 / 4
%        ~= 200 ms/req
%
% This is a first-order estimate.  The cascade controller will correct
% for any bias via integral action on the latency error.  We round up
% slightly to 250 ms/req for conservatism.

max_num_seqs       = 4;
l_e2e_B4_approx    = perturbed.alpha*4 + perturbed.gamma*16 + 700;  % ~812ms
perturbed.beta     = l_e2e_B4_approx / max_num_seqs;

fprintf('\n=== Beta estimation (analytical) ===\n');
fprintf('  l_e2e(B=4, q=0) approx = %.0f ms\n', l_e2e_B4_approx);
fprintf('  max_num_seqs            = %d\n', max_num_seqs);
fprintf('  beta = l_e2e(B4) / max_num_seqs = %.1f ms/req\n\n', perturbed.beta);

% -------------------------------------------------------------------------
% 3. Physical bounds
% -------------------------------------------------------------------------
perturbed.dt    = 1.0;   % s  -- real inference tick
perturbed.B_min = 1;
perturbed.B_max = 8;     % tested safe upper bound from Stage 4
perturbed.q_max = 8;     % allow some queue headroom above max_num_seqs

% -------------------------------------------------------------------------
% 4. Operating conditions
% -------------------------------------------------------------------------
perturbed.lambda_mean   = 3;     % req/tick -- q=0 equilibrium, wall < 1s
perturbed.B0            = 3;     % = lambda_mean at equilibrium
perturbed.q0            = 0;     % natural equilibrium

% Targets: L_mean_target slightly above equilibrium latency
% l_e2e(B=3, q=0) from Stage 4 = 741ms
perturbed.L_mean_target = 800;   % ms
perturbed.L_p95_target  = perturbed.L_mean_target + perturbed.beta * 4;

fprintf('=== Operating conditions ===\n');
fprintf('  lambda_mean    = %g req/tick\n',  perturbed.lambda_mean);
fprintf('  B0             = %g\n',            perturbed.B0);
fprintf('  q0             = %g\n',            perturbed.q0);
fprintf('  L_mean_target  = %.0f ms\n',       perturbed.L_mean_target);
fprintf('  L_p95_target   = %.0f ms\n\n',     perturbed.L_p95_target);

% Equilibrium TTFT check
l_ttft_eq = perturbed.alpha * perturbed.B0 + perturbed.gamma * perturbed.B0^2;
fprintf('  TTFT(B0=%d, q=0) = %.1f ms  (from identified curve)\n\n', ...
    perturbed.B0, l_ttft_eq);

% -------------------------------------------------------------------------
% 5. Pole placement parameters
% -------------------------------------------------------------------------
% Inner loop: q[k+1] = q[k] + a[k] - B[k]
% At dt=1s and queue dynamics that are fast, use tight poles.
perturbed.pp_tau1 = 2.0;   % s -- inner loop closed-loop time constant
perturbed.pp_tau2 = 3.0;   % s -- second pole
perturbed.pp_f    = 0;

% Outer loop: must be >> inner settling time (~3s)
% beta=203ms/req means K_il is larger than Chapter 2b -- use tau_out=20s
perturbed.tau_out = 20.0;  % s

% -------------------------------------------------------------------------
% 6. Design cascade controller
% -------------------------------------------------------------------------
method     = 'pole_placement';
controller = design_controller(perturbed, method);

% -------------------------------------------------------------------------
% 7. Outer loop gain sanity check
% -------------------------------------------------------------------------
K_il = controller.outer_c.K_il;
fprintf('=== Outer loop ===\n');
fprintf('  K_il = %.6f\n', K_il);
fprintf('  z_cl = %.6f  (desired %.6f, tau_out=%.0fs)\n', ...
    controller.outer_c.z_cl, exp(-perturbed.dt/perturbed.tau_out), perturbed.tau_out);
fprintf('  Stable: %s\n\n', mat2str(abs(controller.outer_c.z_cl) < 1));

% -------------------------------------------------------------------------
% 8. Summary
% -------------------------------------------------------------------------
fprintf('╔══════════════════════════════════════════════════════════════╗\n');
fprintf('║  Chapter 5 Cascade Controller  [%s]\n', method);
fprintf('╠══════════════════════════════════════════════════════════════╣\n');
fprintf('║  alpha = %8.4f ms/req   gamma = %7.4f ms/req^2\n', perturbed.alpha, perturbed.gamma);
fprintf('║  beta  = %8.4f ms/req   (estimated analytically)\n', perturbed.beta);
fprintf('╠══════════════════════════════════════════════════════════════╣\n');
fprintf('║  lambda=%g  B0=%g  q0=%g  target=%.0fms\n', ...
    perturbed.lambda_mean, perturbed.B0, perturbed.q0, perturbed.L_mean_target);
fprintf('╠══════════════════════════════════════════════════════════════╣\n');
fprintf('║  Inner:  K_q=%.4f  K_i=%.4f\n', ...
    controller.inner_c.K_q, controller.inner_c.K_i);
fprintf('║          poles=[%.4f  %.4f]\n', ...
    controller.inner_c.poles_cl(1), controller.inner_c.poles_cl(2));
fprintf('╠══════════════════════════════════════════════════════════════╣\n');
fprintf('║  Outer:  K_il=%.6f   z_cl=%.4f\n', K_il, controller.outer_c.z_cl);
fprintf('╚══════════════════════════════════════════════════════════════╝\n');
