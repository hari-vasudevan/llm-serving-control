%% design_controller.m  --  Chapter 8 cascade controller design in MATLAB

clear; clc;
addpath(fileparts(mfilename('fullpath')));

load(fullfile(fileparts(mfilename('fullpath')), 'identified_params.mat'));

fprintf('=== Chapter 8 Cascade Design ===\n');

perturbed.B0 = B0;
perturbed.q0 = q0;
perturbed.dt = DT;
perturbed.beta_q = beta_q;
perturbed.beta_l = beta_l;
perturbed.B_min = min(B_SWEEP);
perturbed.B_max = max(B_SWEEP);
perturbed.q_min = Q_REF_MIN;
perturbed.q_max = Q_MAX;
perturbed.L_mean_target = L_target;
perturbed.L0 = L0;
perturbed.lambda_mean = lambda_mean;
perturbed.tau_out = 30.0;
perturbed.tau_in = 6.0;
perturbed.zeta_in = 0.85;

controller = design_cascade(perturbed);
save(fullfile(fileparts(mfilename('fullpath')), 'controller_params.mat'), ...
    'controller', 'perturbed', 'SERVER', 'PROMPT_REPEAT', 'MAX_TOKENS', 'lambda_mean');
fprintf('[save] controller_params.mat\n');


function controller = design_cascade(perturbed)
B0 = perturbed.B0;
q0 = perturbed.q0;
dt = perturbed.dt;
beta_q = max(perturbed.beta_q, 0.5);
beta_l = max(perturbed.beta_l, 1.0);

rho_in = exp(-dt / perturbed.tau_in);
K_q = (1 - rho_in) / beta_q;
K_i_q = 0.35 * K_q;

rho_out = exp(-dt / perturbed.tau_out);
K_i_l = (1 - rho_out) / beta_l;

xi_q_max = (perturbed.B_max - B0) / max(abs(K_i_q), 1e-6);
xi_q_min = -(B0 - perturbed.B_min) / max(abs(K_i_q), 1e-6);

xi_l_max = (perturbed.q_max - q0) / max(abs(K_i_l), 1e-6);
xi_l_min = -(q0 - perturbed.q_min) / max(abs(K_i_l), 1e-6);

fprintf('Inner loop:\n');
fprintf('  beta_q=%.4f K_q=%.6f K_i_q=%.6f rho_in=%.4f\n', beta_q, K_q, K_i_q, rho_in);
fprintf('Outer loop:\n');
fprintf('  beta_l=%.4f K_i_l=%.6f rho_out=%.4f\n', beta_l, K_i_l, rho_out);

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
    'L0', perturbed.L0, ...
    'xi_min', xi_l_min, ...
    'xi_max', xi_l_max, ...
    'rho', rho_out);
end
