%% design_controller.m -- Chapter 9 cascade controller design
%
% Same terminology as Chapter 2:
%   controller.inner_c: B -> q
%   controller.outer_c: q_ref -> L_mean

clear; clc;
addpath(fileparts(mfilename('fullpath')));
load(fullfile(fileparts(mfilename('fullpath')), 'identified_params.mat'));

fprintf('=== Chapter 9 Cascade Design ===\n');

perturbed = struct();
perturbed.dt = DT;
perturbed.B0 = B0;
perturbed.q0 = q0;
perturbed.beta_q = beta_q;
perturbed.beta = beta;
perturbed.alpha = alpha;
perturbed.gamma = gamma;
perturbed.lambda_mean = lambda_mean;
perturbed.L_mean_target = L_mean_target;
perturbed.L_p95_target = L_p95_target;
perturbed.B_min = B_min;
perturbed.B_max = B_max;
perturbed.q_min = Q_REF_MIN;
perturbed.q_max = Q_MAX;
perturbed.tau_in = 2.0;
perturbed.tau_out = 10.0;

controller = design_cascade(perturbed);
save(fullfile(fileparts(mfilename('fullpath')), 'controller_params.mat'), ...
    'controller', 'perturbed', 'SERVER');
fprintf('[save] controller_params.mat\n');


function controller = design_cascade(perturbed)
B0 = perturbed.B0;
q0 = perturbed.q0;
dt = perturbed.dt;
beta_q = max(perturbed.beta_q, 1e-3);
beta = max(perturbed.beta, 1e-3);

% Inner loop, Chapter 2 form:
%   dq[k+1] = dq[k] - beta_q*dB[k]
%   dB[k] = -K_q*dq[k] - K_i_q*xi_q[k]
rho_in = exp(-dt / perturbed.tau_in);
K_q = (1 - rho_in) / beta_q;
K_i_q = 0.25 * K_q;

% Outer loop, Chapter 2 static-gain form:
%   Delta L = beta * Delta q_ref
%   q_ref = q0 + K_i_l * xi_l, with K_i_l < 0
rho_out = exp(-dt / perturbed.tau_out);
K_i_l = (rho_out - 1) / beta;

xi_q_max = (perturbed.B_max - B0) / max(abs(K_i_q), 1e-9);
xi_q_min = -(B0 - perturbed.B_min) / max(abs(K_i_q), 1e-9);
xi_l_max = (perturbed.q_max - q0) / max(abs(K_i_l), 1e-9);
xi_l_min = -(q0 - perturbed.q_min) / max(abs(K_i_l), 1e-9);

fprintf('Inner B->q: beta_q=%.4f K_q=%.6f K_i_q=%.6f rho=%.4f\n', beta_q, K_q, K_i_q, rho_in);
fprintf('Outer q_ref->L_mean: beta=%.4f K_i_l=%.6f rho=%.4f\n', beta, K_i_l, rho_out);

controller = struct();
controller.inner_c = struct( ...
    'K_q', K_q, ...
    'K_i_q', K_i_q, ...
    'B0', B0, ...
    'B_min', perturbed.B_min, ...
    'B_max', perturbed.B_max, ...
    'xi_min', xi_q_min, ...
    'xi_max', xi_q_max, ...
    'rho', rho_in);
controller.outer_c = struct( ...
    'K_i_l', K_i_l, ...
    'q0', q0, ...
    'q_min', perturbed.q_min, ...
    'q_max', perturbed.q_max, ...
    'L_mean_target', perturbed.L_mean_target, ...
    'L_p95_target', perturbed.L_p95_target, ...
    'xi_min', xi_l_min, ...
    'xi_max', xi_l_max, ...
    'rho', rho_out);
end
