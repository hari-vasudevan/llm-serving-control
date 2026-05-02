%% design_controller.m  --  Chapter 6: Cascade controller design
%
% Reads identified_params.mat and designs the two-loop cascade:
%
%   Inner loop: queue_sw -> B  (Franklin augmented pole placement)
%     q[k+1] = q[k] + a[k] - B[k],  e_q = q_ref - q_sw
%     A_aug = [1 0; 1 1],  B_aug = [1; 0]   (FIFO dynamics)
%     dB = -(K_q*e_q + K_i*xi_q)             K_q > 0
%
%   Outer loop: l_total -> q_ref  (integral, analytical gain)
%     beta_q = dt*1000/B0   [analytical: d(l_total)/d(q)]
%     K_il   = (1 - exp(-dt/tau_out)) / beta_q   [positive]
%     q_ref  = q0 + K_il * xi_l
%
% Outputs: controller_params.mat

clear; clc;
addpath(fileparts(mfilename('fullpath')));

% -------------------------------------------------------------------------
% Tuning parameters (adjust here)
% -------------------------------------------------------------------------
TAU1    = 2.0;    % inner CL time constant 1 [s]
TAU2    = 3.0;    % inner CL time constant 2 [s]
TAU_OUT = 25.0;   % outer CL time constant [s]  -- slower = more stable
B_MIN   = 1;
B_MAX   = 8;
Q_MAX   = 30;     % anti-windup ceiling on q_ref
Q0      = 0;      % nominal queue setpoint at equilibrium

% -------------------------------------------------------------------------
% Load identified parameters
% -------------------------------------------------------------------------
out_dir = fileparts(mfilename('fullpath'));
load(fullfile(out_dir, 'identified_params.mat'));

fprintf('╔══════════════════════════════════════════════════════════════╗\n');
fprintf('║  Chapter 6: Cascade Controller Design                        ║\n');
fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');

fprintf('=== Plant ===\n');
fprintf('  l_total(B,q) = %.4f*B + (%.4f)*B^2 + (q/B)*%.0f\n', alpha, gamma, DT*1000);
fprintf('  alpha   = %.4f ms/req\n', alpha);
fprintf('  gamma   = %.4f ms/req^2\n', gamma);
fprintf('  B0=%d  TTFT(B0)=%.2f ms\n', B0, ttft_B0);
fprintf('  beta_q  = dt*1000/B0 = %.0f/%.0f = %.4f ms/req  [ANALYTICAL]\n', ...
        DT*1000, B0, beta_q);
fprintf('  beta_eff= alpha+2*gamma*B0 = %.4f ms/req  [TTFT slope]\n\n', beta_eff);

% -------------------------------------------------------------------------
% Inner loop: Franklin augmented pole placement (FIFO plant)
% -------------------------------------------------------------------------
fprintf('=== Inner loop ===\n');

z1 = exp(-DT/TAU1);
z2 = exp(-DT/TAU2);

% FIFO plant: de_q[k+1] = de_q[k] + dB[k]  => A=[1 0;1 1], B=[1;0]
A_aug = [1 0; 1 1];
B_aug = [1; 0];

% Ackermann pole placement (use backslash, not inv)
C_ctrl = [B_aug, A_aug*B_aug];
e2     = [0 1];
p_A    = A_aug^2 - (z1+z2)*A_aug + z1*z2*eye(2);
K_pp   = (e2 / C_ctrl) * p_A;   % K = [K_q, K_i]  -- uses / (mrdivide) not inv
K_q    = K_pp(1);
K_i    = K_pp(2);

% Verify CL poles
A_cl   = A_aug - B_aug*K_pp;
poles  = eig(A_cl);
stable = all(abs(poles) < 1);

% Anti-windup bounds for xi_q
% At e_q=0: B = B0 - K_i*xi_q.  Bound: B in [B_MIN, B_MAX]
xi_q_min = (B0 - B_MAX) / K_i;
xi_q_max = (B0 - B_MIN) / K_i;
if xi_q_min > xi_q_max
    [xi_q_min, xi_q_max] = deal(xi_q_max, xi_q_min);
end

fprintf('  K_q = %.4f (>0: %s)\n', K_q, mat2str(K_q > 0));
fprintf('  K_i = %.4f (>0: %s)\n', K_i, mat2str(K_i > 0));
fprintf('  Desired poles: z1=%.4f  z2=%.4f\n', z1, z2);
fprintf('  Actual CL poles: %.4f  %.4f  (stable: %s)\n', poles(1), poles(2), mat2str(stable));
fprintf('  xi_q range: [%.2f, %.2f]\n\n', xi_q_min, xi_q_max);

assert(K_q > 0, sprintf('K_q=%.4f should be positive', K_q));
assert(stable,  'Inner loop is unstable');

% -------------------------------------------------------------------------
% Outer loop: integral gain from beta_q
% -------------------------------------------------------------------------
fprintf('=== Outer loop ===\n');

z_out = exp(-DT/TAU_OUT);
K_il  = (1 - z_out) / beta_q;   % positive

xi_l_min = (0    - Q0) / K_il;
xi_l_max = (Q_MAX - Q0) / K_il;

fprintf('  beta_q  = %.4f ms/req  (dt*1000/B0)\n', beta_q);
fprintf('  K_il    = %.8f (>0: %s)\n', K_il, mat2str(K_il > 0));
fprintf('  z_out   = %.6f  tau_out=%.0fs\n', z_out, TAU_OUT);
fprintf('  xi_l range: [%.2f, %.2f]\n\n', xi_l_min, xi_l_max);

assert(K_il > 0, sprintf('K_il=%.6f should be positive', K_il));

% -------------------------------------------------------------------------
% Summary
% -------------------------------------------------------------------------
fprintf('╔══════════════════════════════════════════════════════════════╗\n');
fprintf('║  CASCADE CONTROLLER  --  Chapter 6\n');
fprintf('╠══════════════════════════════════════════════════════════════╣\n');
fprintf('║  Plant:  l_total = %.4f*B + (%.4f)*B^2 + (q/B)*%.0f\n', alpha, gamma, DT*1000);
fprintf('║          beta_q = %.2f ms/req  [analytical]\n', beta_q);
fprintf('║\n');
fprintf('║  Outer:  K_il=%.8f  z_out=%.4f  tau_out=%.0fs\n', K_il, z_out, TAU_OUT);
fprintf('║          q_ref = q0 + K_il * xi_l\n');
fprintf('║\n');
fprintf('║  Inner:  K_q=%.4f  K_i=%.4f\n', K_q, K_i);
fprintf('║          CL poles ~ [%.4f  %.4f]\n', poles(1), poles(2));
fprintf('║          dB = -(K_q*e_q + K_i*xi_q)\n');
fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');

% -------------------------------------------------------------------------
% Save
% -------------------------------------------------------------------------
save(fullfile(out_dir, 'controller_params.mat'), ...
    'alpha','gamma','beta_q','beta_eff','B0','Q0', ...
    'DT','B_MIN','B_MAX','Q_MAX', ...
    'K_q','K_i','xi_q_min','xi_q_max','poles', ...
    'K_il','z_out','TAU_OUT','xi_l_min','xi_l_max', ...
    'SERVER');
fprintf('[save] controller_params.mat\n');
