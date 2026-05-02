%% design_controller.m  --  Chapter 8 cascade controller design in MATLAB

clear; clc;
addpath(fileparts(mfilename('fullpath')));

load(fullfile(fileparts(mfilename('fullpath')), 'identified_params.mat'));

fprintf('=== Chapter 8 Cascade Design ===\n');

perturbed.B0 = B0;
perturbed.q0 = q0;
perturbed.dt = DT;
perturbed.beta = beta_q;
perturbed.B_min = min(B_SWEEP);
perturbed.B_max = max(B_SWEEP);
perturbed.q_max = q_max;
perturbed.L_mean_target = L_target;
perturbed.lambda_mean = lambda_mean;
perturbed.tau_out = 12.0;
perturbed.pp_tau1 = 1.5;
perturbed.pp_tau2 = 3.0;
perturbed.pp_tau = 2.0;
perturbed.pp_f = 0;

controller = design_cascade(perturbed);
save(fullfile(fileparts(mfilename('fullpath')), 'controller_params.mat'), ...
    'controller', 'perturbed', 'SERVER', 'PROMPT_REPEAT', 'MAX_TOKENS', 'lambda_mean');
fprintf('[save] controller_params.mat\n');


function controller = design_cascade(perturbed)
B0 = perturbed.B0;
q0 = perturbed.q0;
dt = perturbed.dt;
beta = perturbed.beta;

A_in = 1;
B_in = -1;
C_in = 1;
D_in = 0;
A_aug_in = [A_in, 0; C_in, 1];
B_aug_in = [B_in; D_in];
z1_in = exp(-dt / perturbed.pp_tau1);
z2_in = exp(-dt / perturbed.pp_tau2);
K_aug_in = acker(A_aug_in, B_aug_in, [z1_in, z2_in]);
A_cl_in = A_aug_in - B_aug_in * K_aug_in;
poles_in = eig(A_cl_in);

A_aug_out = 1;
B_aug_out = -beta;
z_out = exp(-dt / perturbed.tau_out);
K_aug_out = acker(A_aug_out, B_aug_out, z_out);
A_cl_out = A_aug_out - B_aug_out * K_aug_out;
poles_out = eig(A_cl_out);

K_i_in = K_aug_in(2);
xi_q_max = (perturbed.B_max - B0) / max(abs(K_i_in), 1e-6);
xi_q_min = -(B0 - perturbed.B_min) / max(abs(K_i_in), 1e-6);

K_il = -K_aug_out(1);
xi_l_max = (perturbed.q_max - q0) / max(abs(K_il), 1e-6);
xi_l_min = -(q0 - 0) / max(abs(K_il), 1e-6);

fprintf('Inner loop:\n');
fprintf('  K_q=%.6f K_i_q=%.6f poles=[%.4f %.4f]\n', K_aug_in(1), K_aug_in(2), poles_in(1), poles_in(2));
fprintf('Outer loop:\n');
fprintf('  K_i_l=%.6f pole=[%.4f]\n', K_il, poles_out(1));

controller = struct();
controller.inner_c = struct( ...
    'K_q', K_aug_in(1), ...
    'K_i_q', K_aug_in(2), ...
    'K_aug', K_aug_in, ...
    'B0', B0, ...
    'B_min', perturbed.B_min, ...
    'B_max', perturbed.B_max, ...
    'xi_min', xi_q_min, ...
    'xi_max', xi_q_max, ...
    'poles', poles_in);
controller.outer_c = struct( ...
    'K_i_l', K_il, ...
    'q0', q0, ...
    'q_max', perturbed.q_max, ...
    'L_mean_target', perturbed.L_mean_target, ...
    'xi_min', xi_l_min, ...
    'xi_max', xi_l_max, ...
    'pole', poles_out);
end
