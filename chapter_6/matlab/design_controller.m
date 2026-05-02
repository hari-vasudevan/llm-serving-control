%% design_controller.m  --  Chapter 6: Cascade controller design
%
% Reads identified_params.mat and designs the two-loop cascade.
%
% KEY PARAMETERS for Intel Mac with OLLAMA_NUM_PARALLEL=4:
%   B_max=4  -- Ollama processes max 4 requests truly concurrently
%   B0=2     -- nominal operating point (lambda_ss=2, comfortable headroom)
%   B_min=1  -- always dispatch at least 1
%   TAU_OUT  -- outer loop: slow enough that l_total measurement noise doesn't
%               destabilise it (l_total has ~200-400ms std on this hardware)

clear; clc;
addpath(fileparts(mfilename('fullpath')));

% ── Tuning ────────────────────────────────────────────────────────────────
TAU1    = 2.0;    % inner CL time constant 1 [s]
TAU2    = 3.0;    % inner CL time constant 2 [s]
TAU_OUT = 30.0;   % outer CL time constant [s] -- slow to handle noisy l_total
B_MIN   = 1;
B_MAX   = 4;      % matches OLLAMA_NUM_PARALLEL -- no benefit dispatching more
Q_MAX   = 20;
Q0      = 0;

% ── Load identified params ────────────────────────────────────────────────
out_dir = fileparts(mfilename('fullpath'));
load(fullfile(out_dir, 'identified_params.mat'));

% Override B0 with the identified lambda_ss if available
if exist('lambda_ss','var') && lambda_ss >= B_MIN && lambda_ss <= B_MAX
    B0 = lambda_ss;
    fprintf('Using identified lambda_ss=%d as B0\n', B0);
else
    B0 = 2;
    fprintf('Using default B0=2\n');
end

% Recompute derived quantities at the actual B0
ttft_B0  = alpha*B0 + gamma*B0^2;
beta_eff = alpha + 2*gamma*B0;
beta_q   = DT*1000 / B0;

fprintf('\n╔══════════════════════════════════════════════════════════════╗\n');
fprintf('║  Chapter 6: Cascade Controller Design                        ║\n');
fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');

fprintf('=== Plant at B0=%d ===\n', B0);
fprintf('  alpha   = %.4f ms/req\n', alpha);
fprintf('  gamma   = %.4f ms/req^2\n', gamma);
fprintf('  TTFT(B0=%d) = %.2f ms\n', B0, ttft_B0);
fprintf('  beta_q  = %.4f ms/req  [dt*1000/B0, analytical]\n', beta_q);
fprintf('  beta_eff= %.4f ms/req  [d(TTFT)/dB at B0]\n\n', beta_eff);

% ── Inner loop ────────────────────────────────────────────────────────────
fprintf('=== Inner loop ===\n');
z1 = exp(-DT/TAU1);
z2 = exp(-DT/TAU2);

A_aug  = [1 0; 1 1];
B_aug  = [1; 0];
C_ctrl = [B_aug, A_aug*B_aug];
e2     = [0 1];
p_A    = A_aug^2 - (z1+z2)*A_aug + z1*z2*eye(2);
K_pp   = (e2 / C_ctrl) * p_A;
K_q    = K_pp(1);
K_i    = K_pp(2);

A_cl   = A_aug - B_aug*K_pp;
poles  = eig(A_cl);
stable = all(abs(poles) < 1);

xi_q_min = (B0 - B_MAX) / K_i;
xi_q_max = (B0 - B_MIN) / K_i;
if xi_q_min > xi_q_max; [xi_q_min, xi_q_max] = deal(xi_q_max, xi_q_min); end

fprintf('  K_q=%.4f (>0:%s)  K_i=%.4f (>0:%s)\n', K_q, mat2str(K_q>0), K_i, mat2str(K_i>0));
fprintf('  Desired z1=%.4f z2=%.4f  Actual=[%.4f %.4f]  stable=%s\n', ...
    z1, z2, poles(1), poles(2), mat2str(stable));
assert(K_q > 0 && stable, 'Inner loop design failed');

% ── Outer loop ────────────────────────────────────────────────────────────
fprintf('\n=== Outer loop ===\n');
z_out = exp(-DT/TAU_OUT);
K_il  = (1 - z_out) / beta_q;   % positive

xi_l_min = (0     - Q0) / K_il;
xi_l_max = (Q_MAX - Q0) / K_il;

fprintf('  beta_q=%.4f ms/req\n', beta_q);
fprintf('  K_il=%.8f (>0:%s)  z_out=%.6f  tau_out=%.0fs\n', ...
    K_il, mat2str(K_il>0), z_out, TAU_OUT);
assert(K_il > 0, 'Outer loop K_il should be positive');

% ── Summary ───────────────────────────────────────────────────────────────
fprintf('\n╔══════════════════════════════════════════════════════════════╗\n');
fprintf('║  CASCADE CONTROLLER  --  Chapter 6\n');
fprintf('╠══════════════════════════════════════════════════════════════╣\n');
fprintf('║  B0=%d  B=[%d,%d]  Q_MAX=%d  tau_out=%.0fs\n', B0,B_MIN,B_MAX,Q_MAX,TAU_OUT);
fprintf('║  Inner: K_q=%.4f  K_i=%.4f  poles=[%.4f %.4f]\n', K_q,K_i,poles(1),poles(2));
fprintf('║  Outer: K_il=%.8f  z_out=%.4f\n', K_il, z_out);
fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');

% ── Save ──────────────────────────────────────────────────────────────────
save(fullfile(out_dir, 'controller_params.mat'), ...
    'alpha','gamma','beta_q','beta_eff','B0','Q0', ...
    'DT','B_MIN','B_MAX','Q_MAX', ...
    'K_q','K_i','xi_q_min','xi_q_max','poles', ...
    'K_il','z_out','TAU_OUT','xi_l_min','xi_l_max', ...
    'SERVER');
fprintf('[save] controller_params.mat\n');
